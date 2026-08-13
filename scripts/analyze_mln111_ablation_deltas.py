"""Create a per-query delta report from cached MLN111 ablation rankings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rank_value(rank: int | None) -> int:
    return rank if rank is not None else 11


def _compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_rank = before["first_gold_rank"]
    after_rank = after["first_gold_rank"]
    delta = _rank_value(before_rank) - _rank_value(after_rank)
    before_full = bool(before["full_evidence_at_5"])
    after_full = bool(after["full_evidence_at_5"])
    rank_direction = 1 if delta > 0 else -1 if delta < 0 else 0
    evidence_direction = (
        1 if after_full and not before_full else -1 if before_full and not after_full else 0
    )
    directions = {direction for direction in (rank_direction, evidence_direction) if direction}
    if directions == {1}:
        outcome = "win"
    elif directions == {-1}:
        outcome = "loss"
    elif len(directions) > 1:
        outcome = "mixed"
    else:
        outcome = "tie"
    return {
        "question_id": before["question_id"],
        "question": before["question"],
        "difficulty": before["difficulty"],
        "question_types": before["question_types"],
        "reasoning_scope": before["reasoning_scope"],
        "before_rank": before_rank,
        "after_rank": after_rank,
        "rank_delta": delta,
        "before_full_evidence_at_5": before_full,
        "after_full_evidence_at_5": after_full,
        "outcome": outcome,
    }


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    variants = report["variants"]
    comparisons = (
        ("bm25_to_dense", "structured_bm25", "structured_dense"),
        ("dense_to_hybrid", "structured_dense", "structured_hybrid_rrf"),
        ("hybrid_to_reranker", "structured_hybrid_rrf", "structured_hybrid_reranker"),
        (
            "reranker_to_planner",
            "structured_hybrid_reranker",
            "structured_hybrid_planner_reranker",
        ),
    )
    output: dict[str, Any] = {"benchmark_version": report["benchmark_version"], "comparisons": {}}
    for label, before_name, after_name in comparisons:
        before = {item["question_id"]: item for item in variants[before_name]["per_query"]}
        after = {item["question_id"]: item for item in variants[after_name]["per_query"]}
        rows = [_compare(before[question_id], after[question_id]) for question_id in sorted(before)]
        outcomes = Counter(row["outcome"] for row in rows)
        output["comparisons"][label] = {
            "before": before_name,
            "after": after_name,
            "summary": {name: outcomes[name] for name in ("win", "loss", "mixed", "tie")},
            "queries": rows,
        }
    full = variants["structured_hybrid_planner_reranker"]["per_query"]
    residual = [
        item
        for item in full
        if item["first_gold_rank"] is None
        or item["first_gold_rank"] > 5
        or not item["full_evidence_at_5"]
    ]
    output["residual_failures"] = {
        "count": len(residual),
        "by_difficulty": dict(Counter(item["difficulty"] for item in residual)),
        "by_reasoning_scope": dict(Counter(item["reasoning_scope"] for item in residual)),
        "by_question_type": dict(
            Counter(kind for item in residual for kind in item["question_types"])
        ),
        "queries": residual,
    }
    return output


def _markdown(analysis: dict[str, Any]) -> str:
    lines = ["# MLN111 Development Per-query Delta Analysis", ""]
    for label, comparison in analysis["comparisons"].items():
        summary = comparison["summary"]
        lines.extend(
            [
                f"## {label.replace('_', ' ').title()}",
                "",
                f"**{comparison['before']} → {comparison['after']}**: "
                f"{summary['win']} wins, {summary['loss']} losses, {summary['mixed']} mixed, "
                f"{summary['tie']} ties.",
                "",
                "| ID | Outcome | Rank before → after | Full evidence@5 | Question |",
                "|---|---|---:|---|---|",
            ]
        )
        changed = [row for row in comparison["queries"] if row["outcome"] != "tie"]
        changed.sort(key=lambda row: (-abs(row["rank_delta"]), row["question_id"]))
        for row in changed:
            before_rank = row["before_rank"] if row["before_rank"] is not None else "miss"
            after_rank = row["after_rank"] if row["after_rank"] is not None else "miss"
            question = row["question"].replace("|", "\\|")
            evidence = f"{row['before_full_evidence_at_5']} → {row['after_full_evidence_at_5']}"
            lines.append(
                f"| {row['question_id']} | {row['outcome']} | {before_rank} → {after_rank} | "
                f"{evidence} | {question} |"
            )
        if not changed:
            lines.append("| — | tie | — | — | No per-query changes |")
        lines.append("")
    residual = analysis["residual_failures"]
    lines.extend(
        [
            "## Residual Full-pipeline Failures",
            "",
            f"{residual['count']} questions either miss all gold evidence in the top five or fail "
            "to cover every required evidence group.",
            "",
            f"- By difficulty: `{json.dumps(residual['by_difficulty'], ensure_ascii=False)}`",
            "- By reasoning scope: "
            f"`{json.dumps(residual['by_reasoning_scope'], ensure_ascii=False)}`",
            f"- By question type: `{json.dumps(residual['by_question_type'], ensure_ascii=False)}`",
            "",
            "| ID | First gold rank | Full evidence@5 | Question |",
            "|---|---:|---|---|",
        ]
    )
    for row in residual["queries"]:
        rank = row["first_gold_rank"] if row["first_gold_rank"] is not None else "miss"
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {row['question_id']} | {rank} | {row['full_evidence_at_5']} | {question} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("benchmark/reports/mln111_v1_ablations.json")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("benchmark/reports/mln111_v1_per_query_deltas.json"),
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/mln111_v1_per_query_deltas.md")
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(analysis) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
