"""Production runtime assembly for the MLN111 assistant."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from viettheory.benchmark_generation import load_gemini_key
from viettheory.pipeline.evidence_gate import GateThresholds
from viettheory.pipeline.generator import GeminiGenerator
from viettheory.pipeline.orchestrator import RagPipeline
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hybrid import HybridRetriever
from viettheory.retrieval.parent import ParentChunkStore, ParentExpandedRetriever
from viettheory.retrieval.planned import PlannedRerankedRetriever
from viettheory.retrieval.reranker import QwenCrossEncoderScorer, Reranker
from viettheory.retrieval.retriever import DenseRetriever
from viettheory.retrieval.sentence_transformer import SentenceTransformerEmbedder
from viettheory.schema import Chunk

SUBJECT_CODE = "MLN111"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _integer_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def build_pipeline(project_root: Path | None = None) -> RagPipeline:
    """Load the MLN111 corpus and keep embedding/reranker resident on the GPU."""
    root = (project_root or Path.cwd()).resolve()
    device = os.getenv("VIETTHEORY_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")

    structured = root / "data" / "processed" / SUBJECT_CODE / "structured_v1"
    children_path = structured / "children.jsonl"
    index_dir = structured / "dense_index"
    required = (
        children_path,
        structured / "parents.jsonl",
        index_dir / "manifest.json",
        root / "models" / "Qwen3-Embedding-0.6B",
        root / "models" / "Qwen3-Reranker-0.6B",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing MLN111 runtime artifacts: " + ", ".join(missing))

    embedder = SentenceTransformerEmbedder(
        str(root / "models" / "Qwen3-Embedding-0.6B"),
        model_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        device=device,
    )
    chunks = tuple(
        chunk
        for line in children_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in (Chunk.model_validate_json(line),)
        if chunk.chapter is not None
    )
    lexical = BM25Retriever(chunks)
    dense = DenseRetriever(index_dir, children_path, embedder)
    hybrid = HybridRetriever(
        lexical,
        dense,
        candidate_k=_integer_env("VIETTHEORY_HYBRID_CANDIDATE_K", 30),
    )
    scorer = QwenCrossEncoderScorer(
        str(root / "models" / "Qwen3-Reranker-0.6B"),
        device=device,
        max_length=_integer_env("VIETTHEORY_RERANK_MAX_LENGTH", 512),
    )
    child_retrieval = PlannedRerankedRetriever(
        hybrid,
        Reranker(scorer, batch_size=_integer_env("VIETTHEORY_RERANK_BATCH_SIZE", 4)),
        candidate_k=_integer_env("VIETTHEORY_RERANK_CANDIDATE_K", 12),
    )
    retrieval = ParentExpandedRetriever(
        child_retrieval,
        ParentChunkStore.from_jsonl(structured / "parents.jsonl"),
    )
    generator = GeminiGenerator(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=load_gemini_key(root / ".env"),
    )
    return RagPipeline(
        retrieval,
        generator,
        GateThresholds(
            sufficient_score=_float_env("VIETTHEORY_GATE_SUFFICIENT", -1.0),
            related_score=_float_env("VIETTHEORY_GATE_RELATED", -5.0),
        ),
        top_k=_integer_env("VIETTHEORY_TOP_K", 5),
    )
