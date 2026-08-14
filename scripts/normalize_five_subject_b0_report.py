"""Recompute B0 metrics on answerable Natural QA records from cached rankings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from viettheory.benchmark import Answerability
from viettheory.natural_benchmark import NaturalQuestionV2


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(
    rows: list[dict[str, Any]],
    questions: dict[str, NaturalQuestionV2],
    *,
    parents: bool,
) -> dict[str, float | int]:
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    group_hits = 0
    total_groups = 0
    full = 0
    latencies: list[float] = []
    for row in rows:
        question = questions[str(row["question_id"])]
        groups = tuple(group for group in question.required_evidence_groups if group.required)
        primary = (
            frozenset(parent for group in groups for parent in group.gold_parent_ids)
            if parents
            else frozenset(child for group in groups for child in group.primary_child_ids)
        )
        acceptable = (
            frozenset()
            if parents
            else frozenset(child for group in groups for child in group.acceptable_child_ids)
        )
        gold = primary | acceptable
        retrieved = tuple(str(item) for item in row["retrieved_ids"])
        hit_ranks = [rank for rank, item in enumerate(retrieved, 1) if item in gold]
        first_rank = min(hit_ranks, default=0)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in recalls:
            recalls[k] += int(0 < first_rank <= k)
        gains = [2 if item in primary else 1 if item in acceptable else 0 for item in retrieved[:5]]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal = sorted([2] * len(primary) + [1] * len(acceptable), reverse=True)[:5]
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        covered = 0
        for group in groups:
            group_ids = frozenset(group.gold_parent_ids) if parents else group.all_child_ids
            covered += int(bool(group_ids.intersection(retrieved[:5])))
        group_hits += covered
        total_groups += len(groups)
        full += int(covered == len(groups))
        latencies.append(float(row["latency_ms"]))
    count = len(rows)
    return {
        "evaluated_questions": count,
        "recall_at_1": recalls[1] / count,
        "recall_at_3": recalls[3] / count,
        "recall_at_5": recalls[5] / count,
        "recall_at_10": recalls[10] / count,
        "mrr": statistics.fmean(reciprocal_ranks),
        "ndcg_at_5": statistics.fmean(ndcgs),
        "evidence_group_recall_at_5": group_hits / total_groups,
        "full_evidence_success_at_5": full / count,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    questions = {
        question.id: question
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for question in (NaturalQuestionV2.model_validate_json(line),)
        if question.answerability is Answerability.ANSWERABLE
    }
    report = json.loads(args.input.read_text(encoding="utf-8"))
    for name, variant in report["variants"].items():
        rows = [row for row in variant["per_query"] if row["question_id"] in questions]
        variant["per_query"] = rows
        variant["metrics"] = _metrics(rows, questions, parents="parent_aware" in name)
    report["evaluated_answerable_questions"] = len(questions)
    report["evaluation_scope"] = "answerable records with required evidence groups"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
