"""Evaluate a conservative missing-aspect insertion policy over Recovery V2 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.recovery_v2 import RecoveryPlan
from viettheory.retrieval.reranker import QwenCrossEncoderScorer
from viettheory.schema import Chunk


def _coverage(question: NaturalQuestionV2, ranking: list[str]) -> bool:
    selected = frozenset(ranking[:5])
    return all(
        bool(selected & frozenset(group.gold_parent_ids))
        for group in question.required_evidence_groups
        if group.required
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--support-margin", type=float, default=0.0)
    args = parser.parse_args()

    questions = {
        q.id: q
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for q in (NaturalQuestionV2.model_validate_json(line),)
    }
    plans = {
        p.request_id: p
        for line in args.plans.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for p in (RecoveryPlan.model_validate_json(line),)
    }
    b0_rows = json.loads(args.b0_report.read_text(encoding="utf-8"))["variants"][
        "within_subject_parent_aware_b0"
    ]["per_query"]
    baseline = {row["question_id"]: list(row["retrieved_ids"]) for row in b0_rows}
    v2_rows = json.loads(args.v2_report.read_text(encoding="utf-8"))["per_query"]
    v2 = {row["question_id"]: row for row in v2_rows}
    catalog = UnifiedCorpusCatalog(Path.cwd())
    parents: dict[str, Chunk] = {
        chunk.chunk_id: chunk
        for corpus in catalog.resolve(SearchMode.GLOBAL)
        for line in corpus.parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in (Chunk.model_validate_json(line),)
    }
    scorer = QwenCrossEncoderScorer(
        str(Path.cwd() / "models/Qwen3-Reranker-0.6B"), device=args.device, max_length=512
    )

    pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, str, str]] = []
    for question_id, plan in plans.items():
        if not plan.activate:
            continue
        original = baseline[question_id][:5]
        new = [
            parent_id
            for parent_id in v2[question_id]["final_parent_ids"]
            if parent_id not in baseline[question_id]
        ]
        for query in plan.targeted_queries:
            for parent_id in [*original, *new]:
                pairs.append((query, parents[parent_id].text))
                pair_keys.append((question_id, query, parent_id))
    predicted = scorer.predict(pairs, batch_size=args.batch_size).tolist()
    support = {key: float(score) for key, score in zip(pair_keys, predicted, strict=True)}

    rows: list[dict[str, Any]] = []
    for question_id, question in questions.items():
        if question_id not in baseline:
            continue
        plan = plans[question_id]
        original = baseline[question_id]
        accepted: list[str] = []
        decisions: list[dict[str, Any]] = []
        if plan.activate:
            new = [
                parent_id
                for parent_id in v2[question_id]["final_parent_ids"]
                if parent_id not in original
            ]
            for query in plan.targeted_queries:
                if not new:
                    break
                best_new = max(new, key=lambda parent_id: support[(question_id, query, parent_id)])
                best_new_score = support[(question_id, query, best_new)]
                baseline_score = max(
                    support[(question_id, query, parent_id)] for parent_id in original[:5]
                )
                passes = best_new_score > baseline_score + args.support_margin
                decisions.append(
                    {
                        "query": query,
                        "candidate_parent_id": best_new,
                        "candidate_score": best_new_score,
                        "baseline_best_score": baseline_score,
                        "accepted": passes,
                    }
                )
                if passes and best_new not in accepted:
                    accepted.append(best_new)
        accepted = accepted[:2]
        final = [*original[: 5 - len(accepted)], *accepted, *original[5:]]
        before = _coverage(question, original)
        after = _coverage(question, final)
        rows.append(
            {
                "question_id": question_id,
                "activated": plan.activate,
                "accepted_parent_ids": accepted,
                "support_decisions": decisions,
                "baseline_full_evidence_at_5": before,
                "recovery_v2_1_full_evidence_at_5": after,
                "transition": "win"
                if after and not before
                else "loss"
                if before and not after
                else "tie",
                "final_parent_ids": final[:10],
            }
        )
    wins = sum(row["transition"] == "win" for row in rows)
    losses = sum(row["transition"] == "loss" for row in rows)
    report = {
        "schema_version": "1.0",
        "benchmark": "Natural QA v2 public development answerable",
        "hidden_accessed": False,
        "candidate": "Recovery V2.1 conservative support-margin insertion",
        "support_margin_frozen_before_run": args.support_margin,
        "question_count": len(rows),
        "b0_full_evidence_at_5": sum(row["baseline_full_evidence_at_5"] for row in rows)
        / len(rows),
        "recovery_v2_1_full_evidence_at_5": sum(
            row["recovery_v2_1_full_evidence_at_5"] for row in rows
        )
        / len(rows),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "activation_count": sum(row["activated"] for row in rows),
        "insertion_count": sum(bool(row["accepted_parent_ids"]) for row in rows),
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
                    "insertion_count",
                    "recovery_v2_1_full_evidence_at_5",
                    "acceptance_gate",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
