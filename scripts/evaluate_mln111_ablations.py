"""Run reproducible retrieval ablations on the public MLN111 development set."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from viettheory.benchmark import BenchmarkQuestion, GoldEvidenceGroup
from viettheory.evaluation import RetrievalMetrics, evaluate_retrieval
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hybrid import HybridRetriever
from viettheory.retrieval.planned import comparison_query_variants
from viettheory.retrieval.reranker import QwenCrossEncoderScorer
from viettheory.retrieval.retriever import DenseRetriever
from viettheory.retrieval.sentence_transformer import SentenceTransformerEmbedder
from viettheory.runtime import EMBEDDING_MODEL_ID, EMBEDDING_REVISION
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


class CachedDenseRetriever:
    """Reuse one batched query-embedding pass across every dense ablation."""

    def __init__(self, dense: DenseRetriever, vectors: dict[str, NDArray[np.float32]]) -> None:
        self._dense = dense
        self._vectors = vectors

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        if subject_codes and "MLN111" not in subject_codes:
            return ()
        return self._dense.search_vector(self._vectors[query], top_k=top_k)


def _load_chunks(path: Path) -> tuple[Chunk, ...]:
    return tuple(
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _load_questions(path: Path) -> tuple[BenchmarkQuestion, ...]:
    return tuple(
        BenchmarkQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _span_key(span: SourceSpan) -> tuple[str, tuple[float, float, float, float], str]:
    return span.page_id, span.bbox, span.text


def _remap_to_fixed_chunks(
    questions: tuple[BenchmarkQuestion, ...],
    structured_chunks: tuple[Chunk, ...],
    fixed_chunks: tuple[Chunk, ...],
) -> tuple[BenchmarkQuestion, ...]:
    """Map structured gold IDs to fixed chunks through exact shared source lines."""
    fixed_by_span: dict[tuple[str, tuple[float, float, float, float], str], set[str]] = {}
    for chunk in fixed_chunks:
        for span in chunk.source_spans:
            fixed_by_span.setdefault(_span_key(span), set()).add(chunk.chunk_id)
    structured_by_id = {chunk.chunk_id: chunk for chunk in structured_chunks}

    remapped: list[BenchmarkQuestion] = []
    for question in questions:
        groups: list[GoldEvidenceGroup] = []
        for group in question.gold_evidence_groups:
            mapped: set[str] = set()
            for child_id in group.all_child_ids:
                child = structured_by_id[child_id]
                for span in child.source_spans:
                    mapped.update(fixed_by_span.get(_span_key(span), ()))
            if not mapped:
                raise ValueError(
                    f"no fixed-size evidence mapping for {question.id}/{group.group_id}"
                )
            groups.append(
                group.model_copy(
                    update={
                        "primary_child_ids": tuple(sorted(mapped)),
                        "acceptable_child_ids": (),
                    }
                )
            )
        remapped.append(question.model_copy(update={"gold_evidence_groups": tuple(groups)}))
    return tuple(remapped)


def _round_robin(
    rankings: tuple[tuple[RetrievedEvidence, ...], ...], *, top_k: int
) -> tuple[RetrievedEvidence, ...]:
    selected: list[RetrievedEvidence] = []
    seen: set[str] = set()
    depth = 0
    while len(selected) < top_k and any(depth < len(items) for items in rankings):
        for items in rankings:
            if depth < len(items) and items[depth].chunk.chunk_id not in seen:
                seen.add(items[depth].chunk.chunk_id)
                selected.append(items[depth])
                if len(selected) == top_k:
                    break
        depth += 1
    return tuple(selected)


def _precompute(
    searcher: Searcher, questions: tuple[BenchmarkQuestion, ...]
) -> dict[str, tuple[RetrievedEvidence, ...]]:
    return {
        question.question: searcher.search(
            question.question, top_k=10, subject_codes=frozenset({"MLN111"})
        )
        for question in questions
        if question.gold_evidence_groups
    }


def _rerank_all(
    questions: tuple[BenchmarkQuestion, ...],
    candidate_searcher: Searcher,
    scorer: QwenCrossEncoderScorer,
    *,
    planned: bool,
    candidate_k: int = 12,
    batch_size: int = 4,
) -> dict[str, tuple[RetrievedEvidence, ...]]:
    candidates_by_query: dict[str, tuple[RetrievedEvidence, ...]] = {}
    pairs: list[tuple[str, str]] = []
    offsets: dict[str, tuple[int, int]] = {}
    for question in questions:
        if not question.gold_evidence_groups:
            continue
        variants = comparison_query_variants(question.question) if planned else (question.question,)
        rankings = tuple(
            tuple(
                item
                for item in candidate_searcher.search(
                    variant,
                    top_k=candidate_k * 2,
                    subject_codes=frozenset({"MLN111"}),
                )
                if item.chunk.chapter is not None
            )
            for variant in variants
        )
        candidates = _round_robin(rankings, top_k=candidate_k)
        start = len(pairs)
        pairs.extend((question.question, item.chunk.text) for item in candidates)
        offsets[question.question] = start, len(pairs)
        candidates_by_query[question.question] = candidates

    scores = scorer.predict(pairs, batch_size=batch_size)
    results: dict[str, tuple[RetrievedEvidence, ...]] = {}
    for query, candidates in candidates_by_query.items():
        start, end = offsets[query]
        local_scores = scores[start:end]
        ordered = sorted(
            zip(candidates, local_scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].chunk.chunk_id),
        )[:10]
        results[query] = tuple(
            RetrievedEvidence(
                evidence_id=f"rerank_{item.chunk.chunk_id}",
                chunk=item.chunk,
                score=float(score),
                rank=rank,
                retrieval_method="qwen_reranker",
            )
            for rank, (item, score) in enumerate(ordered, start=1)
        )
    return results


def _evaluate_cached(
    questions: tuple[BenchmarkQuestion, ...],
    results: dict[str, tuple[RetrievedEvidence, ...]],
) -> RetrievalMetrics:
    metrics = evaluate_retrieval(questions, lambda query, top_k: results[query][:top_k])
    return replace(metrics, latency_p50_ms=0.0, latency_p95_ms=0.0)


def _per_query_results(
    questions: tuple[BenchmarkQuestion, ...],
    results: dict[str, tuple[RetrievedEvidence, ...]],
) -> list[dict[str, object]]:
    """Persist stable rankings so later delta analysis is CPU-only."""
    records: list[dict[str, object]] = []
    for question in questions:
        required = tuple(group for group in question.gold_evidence_groups if group.required)
        if not required:
            continue
        retrieved_ids = tuple(item.chunk.chunk_id for item in results[question.question])
        gold_ids = frozenset(child_id for group in required for child_id in group.all_child_ids)
        ranks = tuple(
            rank for rank, child_id in enumerate(retrieved_ids, start=1) if child_id in gold_ids
        )
        group_first_ranks = {
            group.group_id: next(
                (
                    rank
                    for rank, child_id in enumerate(retrieved_ids, start=1)
                    if child_id in group.all_child_ids
                ),
                None,
            )
            for group in required
        }
        records.append(
            {
                "question_id": question.id,
                "question": question.question,
                "difficulty": question.difficulty.value,
                "question_types": [kind.value for kind in question.question_types],
                "reasoning_scope": question.reasoning_scope.value,
                "gold_ids": sorted(gold_ids),
                "retrieved_ids": list(retrieved_ids),
                "first_gold_rank": min(ranks, default=None),
                "group_first_ranks": group_first_ranks,
                "full_evidence_at_5": all(
                    rank is not None and rank <= 5 for rank in group_first_ranks.values()
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path, default=Path("benchmark/v1.0/mln111_development.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("benchmark/reports/mln111_v1_ablations.json")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    structured_dir = root / "data/processed/MLN111/structured_v1"
    structured_path = structured_dir / "children.jsonl"
    fixed_path = root / "data/processed/MLN111/chunks.jsonl"
    questions = _load_questions(args.questions)
    structured_all = _load_chunks(structured_path)
    structured = tuple(chunk for chunk in structured_all if chunk.chapter is not None)
    fixed = _load_chunks(fixed_path)
    fixed_questions = _remap_to_fixed_chunks(questions, structured_all, fixed)

    variants = tuple(
        variant
        for question in questions
        if question.gold_evidence_groups
        for variant in comparison_query_variants(question.question)
    )
    unique_queries = tuple(dict.fromkeys(variants))
    embedder = SentenceTransformerEmbedder(
        str(root / "models/Qwen3-Embedding-0.6B"),
        model_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        device=args.device,
    )
    vectors = embedder.encode_queries(list(unique_queries), batch_size=8)
    vector_cache = {
        query: np.asarray(vectors[index : index + 1], dtype=np.float32, order="C")
        for index, query in enumerate(unique_queries)
    }

    fixed_bm25 = BM25Retriever(fixed)
    structured_bm25 = BM25Retriever(structured)
    fixed_dense = CachedDenseRetriever(
        DenseRetriever(root / "data/processed/MLN111/dense_index", fixed_path, embedder),
        vector_cache,
    )
    structured_dense = CachedDenseRetriever(
        DenseRetriever(structured_dir / "dense_index", structured_path, embedder),
        vector_cache,
    )
    fixed_hybrid = HybridRetriever(fixed_bm25, fixed_dense, candidate_k=30)
    structured_hybrid = HybridRetriever(structured_bm25, structured_dense, candidate_k=30)

    report: dict[str, object] = {
        "benchmark_version": questions[0].benchmark_version,
        "split": "development",
        "quality_only": True,
        "latency_note": "Quality was evaluated from precomputed rankings; latency is omitted.",
        "fixed_chunk_gold_mapping": "exact source-span overlap",
        "variants": {},
    }
    output = report["variants"]
    assert isinstance(output, dict)

    simple = (
        ("fixed_bm25", fixed_questions, fixed_bm25),
        ("fixed_dense", fixed_questions, fixed_dense),
        ("fixed_hybrid_rrf", fixed_questions, fixed_hybrid),
        ("structured_bm25", questions, structured_bm25),
        ("structured_dense", questions, structured_dense),
        ("structured_hybrid_rrf", questions, structured_hybrid),
    )
    for name, evaluation_questions, searcher in simple:
        started = time.perf_counter()
        cached = _precompute(searcher, evaluation_questions)
        output[name] = {
            "metrics": asdict(_evaluate_cached(evaluation_questions, cached)),
            "wall_seconds": time.perf_counter() - started,
            "per_query": _per_query_results(evaluation_questions, cached),
        }
        print(f"completed {name}", flush=True)

    scorer = QwenCrossEncoderScorer(
        str(root / "models/Qwen3-Reranker-0.6B"), device=args.device, max_length=512
    )
    for name, planned in (
        ("structured_hybrid_reranker", False),
        ("structured_hybrid_planner_reranker", True),
    ):
        started = time.perf_counter()
        cached = _rerank_all(questions, structured_hybrid, scorer, planned=planned)
        output[name] = {
            "metrics": asdict(_evaluate_cached(questions, cached)),
            "wall_seconds": time.perf_counter() - started,
            "per_query": _per_query_results(questions, cached),
        }
        print(f"completed {name}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "variants": list(output),
                "per_query_records": sum(len(variant["per_query"]) for variant in output.values()),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
