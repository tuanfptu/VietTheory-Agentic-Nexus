"""Create per-query transition and residual-failure reports from five-subject B0 results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

TRANSITIONS = (
    ("within_subject_bm25", "within_subject_dense"),
    ("within_subject_dense", "within_subject_hybrid_rrf"),
    ("within_subject_hybrid_rrf", "within_subject_hybrid_reranker"),
    ("within_subject_hybrid_reranker", "within_subject_parent_aware_b0"),
    ("global_bm25", "global_dense"),
    ("global_dense", "global_hybrid_rrf"),
    ("global_hybrid_rrf", "global_hybrid_reranker"),
    ("global_hybrid_reranker", "global_parent_aware_b0"),
)


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in rows}


def _outcome(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_full = bool(before["full_evidence_at_5"])
    after_full = bool(after["full_evidence_at_5"])
    before_rank = before["first_gold_rank"]
    after_rank = after["first_gold_rank"]
    if after_full and not before_full:
        return "win"
    if before_full and not after_full:
        return "loss"
    if before_full and after_full:
        if isinstance(before_rank, int) and isinstance(after_rank, int):
            if after_rank < before_rank:
                return "win"
            if after_rank > before_rank:
                return "loss"
        return "tie"
    if isinstance(after_rank, int) and not isinstance(before_rank, int):
        return "mixed"
    if isinstance(before_rank, int) and not isinstance(after_rank, int):
        return "loss"
    return "tie"


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Five-subject B0 delta and failure analysis", "", "## Transitions", ""]
    lines.extend(
        (
            "| Transition | Win | Loss | Mixed | Tie |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name, counts in report["transitions"].items():
        lines.append(
            f"| {name} | {counts.get('win', 0)} | {counts.get('loss', 0)} | "
            f"{counts.get('mixed', 0)} | {counts.get('tie', 0)} |"
        )
    lines.extend(("", "## Residual parent-aware top-5 failures", ""))
    for mode, payload in report["residual_failures"].items():
        lines.extend((f"### {mode}", "", f"Total: **{payload['count']}**", ""))
        lines.append("| ID | Subject | Category | Difficulty | First gold rank |")
        lines.append("|---|---|---|---|---:|")
        for row in payload["queries"]:
            lines.append(
                f"| {row['question_id']} | {row['subject_code']} | {row['category']} | "
                f"{row['difficulty']} | {row['first_gold_rank'] or '—'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("json_output", type=Path)
    parser.add_argument("markdown_output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    variants = source["variants"]
    transitions: dict[str, dict[str, int]] = {}
    for before_name, after_name in TRANSITIONS:
        before = _index(variants[before_name]["per_query"])
        after = _index(variants[after_name]["per_query"])
        if before.keys() != after.keys():
            raise ValueError(f"query mismatch in {before_name} -> {after_name}")
        transitions[f"{before_name} -> {after_name}"] = dict(
            Counter(_outcome(before[question_id], after[question_id]) for question_id in before)
        )

    residual: dict[str, dict[str, Any]] = {}
    for mode in ("within_subject_parent_aware_b0", "global_parent_aware_b0"):
        failures = [row for row in variants[mode]["per_query"] if not row["full_evidence_at_5"]]
        residual[mode] = {
            "count": len(failures),
            "by_subject": dict(Counter(row["subject_code"] for row in failures)),
            "by_category": dict(Counter(row["category"] for row in failures)),
            "by_difficulty": dict(Counter(row["difficulty"] for row in failures)),
            "queries": failures,
        }
    report = {
        "benchmark_version": source["benchmark_version"],
        "split": source["split"],
        "transitions": transitions,
        "residual_failures": residual,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
