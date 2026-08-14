"""Evaluate five-subject B0 retrieval variants on the public development split."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hybrid import HybridRetriever
from viettheory.retrieval.reranker import QwenCrossEncoderScorer
from viettheory.retrieval.retriever import DenseRetriever, QueryEmbedder
from viettheory.retrieval.sentence_transformer import SentenceTransformerEmbedder
from viettheory.runtime import EMBEDDING_MODEL_ID, EMBEDDING_REVISION
from viettheory.schema import RetrievedEvidence


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


@dataclass(frozen=True)
class Metrics:
    evaluated_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    evidence_group_recall_at_5: float
    full_evidence_success_at_5: float
    latency_p50_ms: float
    latency_p95_ms: float


class CachedDenseFanout:
    def __init__(
        self,
        retrievers: tuple[DenseRetriever, ...],
        vectors: dict[str, NDArray[np.float32]],
    ) -> None:
        self._retrievers = retrievers
        self._vectors = vectors

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        vector = self._vectors[query]
        candidates = [
            item
            for retriever in self._retrievers
            if subject_codes is None or retriever.subject_code in subject_codes
            for item in retriever.search_vector(vector, top_k=top_k)
        ]
        candidates.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return tuple(
            item.model_copy(update={"rank": rank, "evidence_id": f"dense_cached_{rank}"})
            for rank, item in enumerate(candidates[:top_k], start=1)
        )


class CachedIdentityEmbedder:
    """Expose the pinned model identity without loading weights when vectors are cached."""

    model_id = EMBEDDING_MODEL_ID
    model_revision = EMBEDDING_REVISION

    def encode_queries(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        raise RuntimeError("cached evaluation must not request new query embeddings")


def _load_questions(path: Path) -> tuple[NaturalQuestionV2, ...]:
    return tuple(
        NaturalQuestionV2.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _group_ids(question: NaturalQuestionV2, *, parents: bool) -> tuple[frozenset[str], ...]:
    groups = tuple(group for group in question.required_evidence_groups if group.required)
    if parents:
        return tuple(frozenset(group.gold_parent_ids) for group in groups)
    return tuple(group.all_child_ids for group in groups)


def _evaluate(
    questions: tuple[NaturalQuestionV2, ...],
    rankings: dict[str, tuple[RetrievedEvidence, ...]],
    latencies: dict[str, float],
    *,
    parents: bool = False,
) -> tuple[Metrics, list[dict[str, object]]]:
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    group_hits_5 = 0
    total_groups = 0
    full_5 = 0
    per_query: list[dict[str, object]] = []
    for question in questions:
        groups = _group_ids(question, parents=parents)
        result_ids = tuple(item.chunk.chunk_id for item in rankings[question.id])
        gold = frozenset().union(*groups)
        hit_ranks = [rank for rank, item in enumerate(result_ids, 1) if item in gold]
        first_rank = min(hit_ranks, default=0)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in recalls:
            recalls[k] += int(0 < first_rank <= k)
        gains = [1 if item in gold else 0 for item in result_ids[:5]]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal = min(len(gold), 5)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        group_ranks = {
            f"g{index}": next(
                (rank for rank, item in enumerate(result_ids, 1) if item in group), None
            )
            for index, group in enumerate(groups, 1)
        }
        covered = sum(rank is not None and rank <= 5 for rank in group_ranks.values())
        group_hits_5 += covered
        total_groups += len(groups)
        complete = covered == len(groups)
        full_5 += int(complete)
        per_query.append(
            {
                "question_id": question.id,
                "subject_code": question.subject_code,
                "category": question.primary_category.value,
                "difficulty": question.difficulty.value,
                "reasoning_scope": question.reasoning_scope.value,
                "retrieved_ids": list(result_ids),
                "first_gold_rank": first_rank or None,
                "group_first_ranks": group_ranks,
                "full_evidence_at_5": complete,
                "latency_ms": latencies[question.id],
            }
        )
    count = len(questions)
    latency_values = list(latencies.values())
    return (
        Metrics(
            evaluated_questions=count,
            recall_at_1=recalls[1] / count,
            recall_at_3=recalls[3] / count,
            recall_at_5=recalls[5] / count,
            recall_at_10=recalls[10] / count,
            mrr=statistics.fmean(reciprocal_ranks),
            ndcg_at_5=statistics.fmean(ndcgs),
            evidence_group_recall_at_5=group_hits_5 / total_groups,
            full_evidence_success_at_5=full_5 / count,
            latency_p50_ms=_percentile(latency_values, 0.5),
            latency_p95_ms=_percentile(latency_values, 0.95),
        ),
        per_query,
    )


def _search_all(
    questions: tuple[NaturalQuestionV2, ...],
    searcher: Searcher,
    *,
    within_subject: bool,
    top_k: int = 10,
) -> tuple[dict[str, tuple[RetrievedEvidence, ...]], dict[str, float]]:
    rankings: dict[str, tuple[RetrievedEvidence, ...]] = {}
    latencies: dict[str, float] = {}
    for question in questions:
        started = time.perf_counter()
        rankings[question.id] = searcher.search(
            question.question,
            top_k=top_k,
            subject_codes=frozenset({question.subject_code}) if within_subject else None,
        )
        latencies[question.id] = (time.perf_counter() - started) * 1000.0
    return rankings, latencies


def _rerank_all(
    questions: tuple[NaturalQuestionV2, ...],
    hybrid: Searcher,
    scorer: QwenCrossEncoderScorer,
    *,
    within_subject: bool,
    candidate_k: int,
    batch_size: int,
) -> tuple[dict[str, tuple[RetrievedEvidence, ...]], dict[str, float]]:
    candidates: dict[str, tuple[RetrievedEvidence, ...]] = {}
    pairs: list[tuple[str, str]] = []
    offsets: dict[str, tuple[int, int]] = {}
    started_by_id: dict[str, float] = {}
    for question in questions:
        started_by_id[question.id] = time.perf_counter()
        items = hybrid.search(
            question.question,
            top_k=candidate_k,
            subject_codes=frozenset({question.subject_code}) if within_subject else None,
        )
        candidates[question.id] = items
        start = len(pairs)
        pairs.extend((question.question, item.chunk.text) for item in items)
        offsets[question.id] = start, len(pairs)
    scores = scorer.predict(pairs, batch_size=batch_size)
    finished = time.perf_counter()
    rankings: dict[str, tuple[RetrievedEvidence, ...]] = {}
    latencies: dict[str, float] = {}
    amortized_rerank_ms = (finished - min(started_by_id.values())) * 1000.0 / len(questions)
    for question in questions:
        start, end = offsets[question.id]
        ordered = sorted(
            zip(candidates[question.id], scores[start:end], strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].chunk.chunk_id),
        )[:10]
        rankings[question.id] = tuple(
            item.model_copy(
                update={
                    "score": float(score),
                    "rank": rank,
                    "evidence_id": f"rerank_{item.chunk.chunk_id}",
                    "retrieval_method": "qwen_reranker",
                }
            )
            for rank, (item, score) in enumerate(ordered, 1)
        )
        latencies[question.id] = amortized_rerank_ms
    return rankings, latencies


def _expand_parent_rankings(
    rankings: dict[str, tuple[RetrievedEvidence, ...]],
) -> dict[str, tuple[RetrievedEvidence, ...]]:
    expanded: dict[str, tuple[RetrievedEvidence, ...]] = {}
    for question_id, items in rankings.items():
        seen: set[str] = set()
        selected: list[RetrievedEvidence] = []
        for item in items:
            parent_id = item.chunk.parent_chunk_id
            if parent_id is None or parent_id in seen:
                continue
            seen.add(parent_id)
            selected.append(
                item.model_copy(
                    update={
                        "chunk": item.chunk.model_copy(
                            update={"chunk_id": parent_id, "chunk_kind": "parent"}
                        ),
                        "rank": len(selected) + 1,
                        "retrieval_method": "parent_expansion",
                    }
                )
            )
            if len(selected) == 10:
                break
        expanded[question_id] = tuple(selected)
    return expanded


def _slice_metrics(per_query: list[dict[str, object]]) -> dict[str, object]:
    dimensions = ("subject_code", "category", "difficulty", "reasoning_scope")
    output: dict[str, object] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in per_query:
            groups[str(row[dimension])].append(row)
        output[dimension] = {
            key: {
                "count": len(rows),
                "recall_at_5": sum(
                    isinstance(row["first_gold_rank"], int) and row["first_gold_rank"] <= 5
                    for row in rows
                )
                / len(rows),
                "full_evidence_at_5": sum(bool(row["full_evidence_at_5"]) for row in rows)
                / len(rows),
            }
            for key, rows in sorted(groups.items())
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--rerank-batch-size", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=12)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path.cwd().resolve()
    questions = tuple(
        question
        for question in _load_questions(args.questions)
        if question.required_evidence_groups
    )
    split_values = {question.split.value for question in questions}
    if len(split_values) != 1:
        raise ValueError(f"evaluation input mixes splits: {sorted(split_values)}")
    evaluation_split = next(iter(split_values))
    catalog = UnifiedCorpusCatalog(root)
    corpora = catalog.resolve(SearchMode.GLOBAL)
    chunks = tuple(
        chunk for chunk in catalog.load_children(SearchMode.GLOBAL) if chunk.chapter is not None
    )
    question_ids = [question.id for question in questions]
    embedder: QueryEmbedder
    if args.embedding_cache is not None and args.embedding_cache.exists():
        cached = np.load(args.embedding_cache, allow_pickle=False)
        if cached["question_ids"].tolist() != question_ids:
            raise ValueError("embedding cache question IDs do not match the benchmark")
        vectors = np.asarray(cached["vectors"], dtype=np.float32)
        embedder = CachedIdentityEmbedder()
        print(f"loaded embedding cache {args.embedding_cache}", flush=True)
    else:
        embedder = SentenceTransformerEmbedder(
            str(root / "models/Qwen3-Embedding-0.6B"),
            model_id=EMBEDDING_MODEL_ID,
            revision=EMBEDDING_REVISION,
            device=args.device,
        )
        vectors = embedder.encode_queries(
            [question.question for question in questions], batch_size=args.embedding_batch_size
        )
        if args.embedding_cache is not None:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.embedding_cache,
                question_ids=np.asarray(question_ids),
                vectors=vectors,
            )
            print(f"saved embedding cache {args.embedding_cache}", flush=True)
    vector_cache = {
        question.question: np.asarray(vectors[index : index + 1], dtype=np.float32)
        for index, question in enumerate(questions)
    }
    dense_retrievers = tuple(
        DenseRetriever(corpus.dense_index_dir, corpus.children_path, embedder) for corpus in corpora
    )
    dense = CachedDenseFanout(dense_retrievers, vector_cache)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(bm25, dense, candidate_k=30)

    report: dict[str, object]
    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report.get("benchmark_version") != questions[0].benchmark_version:
            raise ValueError("existing report benchmark version mismatch")
    else:
        report = {
            "benchmark_version": questions[0].benchmark_version,
            "split": evaluation_split,
            "device": args.device,
            "evaluated_answerable_questions": len(questions),
            "variants": {},
        }
    if report.get("split") != evaluation_split:
        raise ValueError("existing report split mismatch")
    variants = report["variants"]
    assert isinstance(variants, dict)

    def checkpoint() -> None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for mode, within_subject in (("within_subject", True), ("global", False)):
        for name, searcher in (("bm25", bm25), ("dense", dense), ("hybrid_rrf", hybrid)):
            variant_name = f"{mode}_{name}"
            if variant_name in variants:
                continue
            rankings, latencies = _search_all(questions, searcher, within_subject=within_subject)
            metrics, per_query = _evaluate(questions, rankings, latencies)
            variants[variant_name] = {
                "metrics": asdict(metrics),
                "slices": _slice_metrics(per_query),
                "per_query": per_query,
            }
            checkpoint()
            print(f"completed {variant_name}", flush=True)

    scorer = QwenCrossEncoderScorer(
        str(root / "models/Qwen3-Reranker-0.6B"), device=args.device, max_length=512
    )
    for mode, within_subject in (("within_subject", True), ("global", False)):
        reranker_name = f"{mode}_hybrid_reranker"
        parent_name = f"{mode}_parent_aware_b0"
        if reranker_name in variants and parent_name in variants:
            continue
        rankings, latencies = _rerank_all(
            questions,
            hybrid,
            scorer,
            within_subject=within_subject,
            candidate_k=args.candidate_k,
            batch_size=args.rerank_batch_size,
        )
        metrics, per_query = _evaluate(questions, rankings, latencies)
        variants[reranker_name] = {
            "metrics": asdict(metrics),
            "slices": _slice_metrics(per_query),
            "per_query": per_query,
        }
        parent_rankings = _expand_parent_rankings(rankings)
        parent_metrics, parent_per_query = _evaluate(
            questions, parent_rankings, latencies, parents=True
        )
        variants[parent_name] = {
            "metrics": asdict(parent_metrics),
            "slices": _slice_metrics(parent_per_query),
            "per_query": parent_per_query,
        }
        checkpoint()
        print(f"completed {mode}_reranker_and_parent", flush=True)

    checkpoint()
    print(json.dumps({"report": str(args.report), "variants": len(variants)}))


if __name__ == "__main__":
    main()
