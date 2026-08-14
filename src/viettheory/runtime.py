"""Subject-agnostic production runtime assembly."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from viettheory.benchmark_generation import load_gemini_key
from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.pipeline.evidence_gate import GateThresholds
from viettheory.pipeline.generator import GeminiGenerator
from viettheory.pipeline.orchestrator import RagPipeline
from viettheory.recovery_v2 import EvidenceGuidedRecoveryRetriever, GeminiRecoveryPlanner
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hybrid import HybridRetriever
from viettheory.retrieval.parent import ParentChunkStore, ParentExpandedRetriever
from viettheory.retrieval.planned import PlannedRerankedRetriever
from viettheory.retrieval.reranker import PairScorer, QwenCrossEncoderScorer, Reranker
from viettheory.retrieval.retriever import (
    DenseFanoutRetriever,
    DenseRetriever,
    QueryEmbedder,
)
from viettheory.retrieval.sentence_transformer import SentenceTransformerEmbedder
from viettheory.schema import Chunk

DEFAULT_SUBJECT_CODE = "MLN111"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _integer_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def build_retrieval(
    project_root: Path | None = None,
    *,
    search_mode: SearchMode = SearchMode.WITHIN_SUBJECT,
    subject_code: str | None = DEFAULT_SUBJECT_CODE,
    embedder: QueryEmbedder | None = None,
    scorer: PairScorer | None = None,
) -> ParentExpandedRetriever | EvidenceGuidedRecoveryRetriever:
    """Build the frozen B0 retrieval path over one or all registered subjects."""
    root = (project_root or Path.cwd()).resolve()
    device = os.getenv("VIETTHEORY_DEVICE", "cuda")
    if embedder is None and device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
    catalog = UnifiedCorpusCatalog(root)
    corpora = catalog.resolve(search_mode, subject_code)
    model_paths = (
        root / "models" / "Qwen3-Embedding-0.6B",
        root / "models" / "Qwen3-Reranker-0.6B",
    )
    required = tuple(
        path
        for corpus in corpora
        for path in (
            corpus.children_path,
            corpus.parents_path,
            corpus.dense_index_dir / "manifest.json",
        )
    ) + (() if embedder is not None and scorer is not None else model_paths)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing runtime artifacts: " + ", ".join(missing))

    active_embedder = embedder or SentenceTransformerEmbedder(
        str(model_paths[0]),
        model_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        device=device,
    )
    chunks = tuple(
        chunk
        for chunk in catalog.load_children(search_mode, subject_code)
        if chunk.chapter is not None
    )
    lexical = BM25Retriever(chunks)
    dense_retrievers = tuple(
        DenseRetriever(corpus.dense_index_dir, corpus.children_path, active_embedder)
        for corpus in corpora
    )
    dense = (
        dense_retrievers[0]
        if len(dense_retrievers) == 1
        else DenseFanoutRetriever(dense_retrievers, active_embedder)
    )
    hybrid = HybridRetriever(
        lexical,
        dense,
        candidate_k=_integer_env("VIETTHEORY_HYBRID_CANDIDATE_K", 30),
    )
    active_scorer = scorer or QwenCrossEncoderScorer(
        str(model_paths[1]),
        device=device,
        max_length=_integer_env("VIETTHEORY_RERANK_MAX_LENGTH", 512),
    )
    child_retrieval = PlannedRerankedRetriever(
        hybrid,
        Reranker(active_scorer, batch_size=_integer_env("VIETTHEORY_RERANK_BATCH_SIZE", 4)),
        candidate_k=_integer_env("VIETTHEORY_RERANK_CANDIDATE_K", 12),
    )
    parents = (
        Chunk.model_validate_json(line)
        for corpus in corpora
        for line in corpus.parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    baseline = ParentExpandedRetriever(
        child_retrieval,
        ParentChunkStore(parents),
    )
    if os.getenv("VIETTHEORY_AGENTIC", "0").casefold() not in {"1", "true", "yes"}:
        return baseline
    return EvidenceGuidedRecoveryRetriever(
        baseline,
        GeminiRecoveryPlanner(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            api_key=load_gemini_key(root / ".env"),
        ),
        active_scorer,
        recovery_top_k=_integer_env("VIETTHEORY_RECOVERY_TOP_K", 5),
        support_margin=_float_env("VIETTHEORY_RECOVERY_SUPPORT_MARGIN", 0.0),
        scorer_batch_size=_integer_env("VIETTHEORY_RERANK_BATCH_SIZE", 4),
    )


def build_pipeline(
    project_root: Path | None = None,
    *,
    search_mode: SearchMode | None = None,
    subject_code: str | None = None,
) -> RagPipeline:
    """Load one shared five-subject pipeline and keep neural models resident."""
    root = (project_root or Path.cwd()).resolve()
    active_mode = search_mode or SearchMode(os.getenv("VIETTHEORY_SEARCH_MODE", "within_subject"))
    active_subject = subject_code or os.getenv("VIETTHEORY_SUBJECT", DEFAULT_SUBJECT_CODE)
    if active_mode is SearchMode.GLOBAL:
        active_subject = None
    retrieval = build_retrieval(root, search_mode=active_mode, subject_code=active_subject)
    active_subject_codes = (
        frozenset(UnifiedCorpusCatalog(root).subject_codes)
        if active_subject is None
        else frozenset({active_subject})
    )
    scope_label = (
        "hệ thống giáo trình VietTheory"
        if active_subject is None
        else f"giáo trình {active_subject}"
    )
    generator = GeminiGenerator(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=load_gemini_key(root / ".env"),
        corpus_label=scope_label,
    )
    # The original sufficient threshold was calibrated on MLN111 only. Raw
    # cross-encoder logits are not comparable enough across five corpora to
    # reuse that one-subject cutoff. In global mode, subject filtering,
    # pre-routing, grounded generation, and citation verification remain the
    # safety boundary; any evidence meeting the related threshold may proceed.
    default_sufficient = -5.0 if active_mode is SearchMode.GLOBAL else -1.0
    return RagPipeline(
        retrieval,
        generator,
        GateThresholds(
            sufficient_score=_float_env("VIETTHEORY_GATE_SUFFICIENT", default_sufficient),
            related_score=_float_env("VIETTHEORY_GATE_RELATED", -5.0),
        ),
        top_k=_integer_env("VIETTHEORY_TOP_K", 5),
        subject_codes=active_subject_codes,
        scope_label=scope_label,
    )
