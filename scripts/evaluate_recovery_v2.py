"""Evaluate bounded Recovery V2 against frozen B0 on public Natural QA development."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.recovery_v2 import RecoveryPlan
from viettheory.retrieval.reranker import QwenCrossEncoderScorer
from viettheory.runtime import build_retrieval
from viettheory.schema import Chunk


def _coverage(question: NaturalQuestionV2, ranking: list[str], top_k: int = 5) -> bool:
    selected = frozenset(ranking[:top_k])
    return all(
        bool(selected & frozenset(group.gold_parent_ids))
        for group in question.required_evidence_groups
        if group.required
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recovery-top-k", type=int, default=5)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    args = parser.parse_args()

    questions = {
        q.id: q
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for q in (NaturalQuestionV2.model_validate_json(line),)
    }
    plans = {
        plan.request_id: plan
        for line in args.plans.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for plan in (RecoveryPlan.model_validate_json(line),)
    }
    b0 = json.loads(args.b0_report.read_text(encoding="utf-8"))
    baseline_rows = b0["variants"]["within_subject_parent_aware_b0"]["per_query"]
    expected_ids = {row["question_id"] for row in baseline_rows}
    if plans.keys() != expected_ids:
        raise ValueError("Recovery plan checkpoint does not match B0 development IDs")

    catalog = UnifiedCorpusCatalog(Path.cwd())
    parents: dict[str, Chunk] = {
        chunk.chunk_id: chunk
        for corpus in catalog.resolve(SearchMode.GLOBAL)
        for line in corpus.parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in (Chunk.model_validate_json(line),)
    }
    retriever = build_retrieval(Path.cwd(), search_mode=SearchMode.GLOBAL, subject_code=None)
    scorer = QwenCrossEncoderScorer(
        str(Path.cwd() / "models/Qwen3-Reranker-0.6B"), device=args.device, max_length=512
    )

    pools: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    query_counts: dict[str, int] = {}
    for index, row in enumerate(baseline_rows, 1):
        question_id = row["question_id"]
        question = questions[question_id]
        plan = plans[question_id]
        original = list(row["retrieved_ids"][:10])
        pool = list(original)
        started = time.perf_counter()
        if plan.activate:
            for query in plan.targeted_queries:
                for item in retriever.search(
                    query,
                    top_k=args.recovery_top_k,
                    subject_codes=frozenset({question.subject_code}),
                ):
                    if item.chunk.chunk_id not in pool:
                        pool.append(item.chunk.chunk_id)
        pools[question_id] = pool
        query_counts[question_id] = len(plan.targeted_queries)
        latencies[question_id] = (time.perf_counter() - started) * 1000.0
        if plan.activate:
            print(f"retrieved {index}/{len(baseline_rows)} {question_id}", flush=True)

    active_pairs: list[tuple[str, str]] = []
    offsets: dict[str, tuple[int, int]] = {}
    for row in baseline_rows:
        question_id = row["question_id"]
        if not plans[question_id].activate:
            continue
        start = len(active_pairs)
        active_pairs.extend(
            (questions[question_id].question, parents[parent_id].text)
            for parent_id in pools[question_id]
        )
        offsets[question_id] = start, len(active_pairs)
    scores: list[float] = (
        scorer.predict(active_pairs, batch_size=args.rerank_batch_size).tolist()
        if active_pairs
        else []
    )

    rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        question_id = baseline["question_id"]
        plan = plans[question_id]
        baseline_ranking = list(baseline["retrieved_ids"])
        final_ranking = baseline_ranking
        if plan.activate:
            start, end = offsets[question_id]
            final_ranking = [
                parent_id
                for parent_id, _ in sorted(
                    zip(pools[question_id], scores[start:end], strict=True),
                    key=lambda pair: (-float(pair[1]), pair[0]),
                )
            ]
        before = _coverage(questions[question_id], baseline_ranking)
        after = _coverage(questions[question_id], final_ranking)
        rows.append(
            {
                "question_id": question_id,
                "activated": plan.activate,
                "targeted_queries": list(plan.targeted_queries),
                "baseline_full_evidence_at_5": before,
                "recovery_v2_full_evidence_at_5": after,
                "transition": "win"
                if after and not before
                else "loss"
                if before and not after
                else "tie",
                "final_parent_ids": final_ranking[:10],
                "recovery_latency_ms": latencies[question_id],
            }
        )
    wins = sum(row["transition"] == "win" for row in rows)
    losses = sum(row["transition"] == "loss" for row in rows)
    active = [row for row in rows if row["activated"]]
    residual = [row for row in rows if not row["baseline_full_evidence_at_5"]]
    report = {
        "schema_version": "1.0",
        "benchmark": "Natural QA v2 public development answerable",
        "hidden_accessed": False,
        "candidate": "B0 + evidence-guided bounded Recovery V2",
        "question_count": len(rows),
        "b0_full_evidence_at_5": sum(row["baseline_full_evidence_at_5"] for row in rows)
        / len(rows),
        "recovery_v2_full_evidence_at_5": sum(row["recovery_v2_full_evidence_at_5"] for row in rows)
        / len(rows),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "agent_activation_count": len(active),
        "agent_activation_rate": len(active) / len(rows),
        "residual_failure_recovery_count": sum(row["transition"] == "win" for row in residual),
        "residual_failure_count": len(residual),
        "average_recovery_queries_when_active": statistics.fmean(
            query_counts[row["question_id"]] for row in active
        )
        if active
        else 0.0,
        "recovery_latency_p50_ms_active": statistics.median(
            row["recovery_latency_ms"] for row in active
        )
        if active
        else 0.0,
        "acceptance_gate": {
            "beats_b0": wins > losses,
            "no_more_than_one_regression": losses <= 1,
            "accepted": wins > losses and losses <= 1,
        },
        "per_query": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "wins",
                    "losses",
                    "agent_activation_count",
                    "recovery_v2_full_evidence_at_5",
                    "acceptance_gate",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
