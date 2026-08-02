"""Evidence-group retrieval metrics over stable child IDs."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from viettheory.benchmark import BenchmarkQuestion, GoldEvidenceGroup, ReviewStatus
from viettheory.schema import RetrievedEvidence


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    evaluated_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    group_recall_at_1: float
    group_recall_at_3: float
    group_recall_at_5: float
    group_recall_at_10: float
    partial_evidence_coverage_at_5: float
    full_evidence_success_at_5: float
    full_evidence_success_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _required_groups(question: BenchmarkQuestion) -> tuple[GoldEvidenceGroup, ...]:
    return tuple(group for group in question.gold_evidence_groups if group.required)


def _hit(group: GoldEvidenceGroup, retrieved_ids: Sequence[str], k: int) -> bool:
    return bool(group.all_child_ids.intersection(retrieved_ids[:k]))


def _coverage(groups: tuple[GoldEvidenceGroup, ...], ids: Sequence[str], k: int) -> float:
    if not groups:
        return 0.0
    return sum(_hit(group, ids, k) for group in groups) / len(groups)


def _ndcg_at_5(question: BenchmarkQuestion, retrieved_ids: Sequence[str]) -> float:
    primary = {
        child_id for group in question.gold_evidence_groups for child_id in group.primary_child_ids
    }
    acceptable = {
        child_id
        for group in question.gold_evidence_groups
        for child_id in group.acceptable_child_ids
    }
    gains = [
        2 if child_id in primary else 1 if child_id in acceptable else 0
        for child_id in retrieved_ids[:5]
    ]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted(([2] * len(primary) + [1] * len(acceptable)), reverse=True)[:5]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(
    questions: Sequence[BenchmarkQuestion],
    retrieve: Callable[[str, int], tuple[RetrievedEvidence, ...]],
    *,
    require_verified: bool = True,
) -> RetrievalMetrics:
    """Evaluate questions with gold groups without mutating the benchmark."""
    eligible = [question for question in questions if _required_groups(question)]
    if not eligible:
        raise ValueError("no questions with required evidence groups to evaluate")
    if require_verified and any(
        question.review_status is not ReviewStatus.VERIFIED for question in eligible
    ):
        raise ValueError("retrieval metrics require human-verified benchmark questions")

    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    group_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    coverages_5: list[float] = []
    full_5 = 0
    full_10 = 0
    total_groups = 0
    latencies: list[float] = []
    for question in eligible:
        groups = _required_groups(question)
        gold_ids = frozenset(child_id for group in groups for child_id in group.all_child_ids)
        started = time.perf_counter()
        results = retrieve(question.question, 10)
        latencies.append((time.perf_counter() - started) * 1000.0)
        result_ids = [result.chunk.chunk_id for result in results]
        ranks = [rank for rank, child_id in enumerate(result_ids, start=1) if child_id in gold_ids]
        first_rank = min(ranks, default=0)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in recalls:
            recalls[k] += int(0 < first_rank <= k)
            group_hits[k] += sum(_hit(group, result_ids, k) for group in groups)
        coverage_5 = _coverage(groups, result_ids, 5)
        coverages_5.append(coverage_5)
        full_5 += int(coverage_5 == 1.0)
        full_10 += int(_coverage(groups, result_ids, 10) == 1.0)
        total_groups += len(groups)
        ndcgs.append(_ndcg_at_5(question, result_ids))

    total = len(eligible)
    return RetrievalMetrics(
        evaluated_questions=total,
        recall_at_1=recalls[1] / total,
        recall_at_3=recalls[3] / total,
        recall_at_5=recalls[5] / total,
        recall_at_10=recalls[10] / total,
        mrr=statistics.fmean(reciprocal_ranks),
        ndcg_at_5=statistics.fmean(ndcgs),
        group_recall_at_1=group_hits[1] / total_groups,
        group_recall_at_3=group_hits[3] / total_groups,
        group_recall_at_5=group_hits[5] / total_groups,
        group_recall_at_10=group_hits[10] / total_groups,
        partial_evidence_coverage_at_5=statistics.fmean(coverages_5),
        full_evidence_success_at_5=full_5 / total,
        full_evidence_success_at_10=full_10 / total,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )
