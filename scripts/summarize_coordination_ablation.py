"""Summarize single B0 versus role-separated B0+Graph development ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph_report.read_text(encoding="utf-8"))
    report = {
        "schema_version": "1.0",
        "split": "development",
        "baseline": "single_controller_frozen_B0",
        "candidate": "role_separated_retriever_graph_evidence_pool_v0",
        "fairness": {
            "same_B0_rankings": True,
            "same_top_k": 5,
            "candidate_extra_role": "adjacent_parent_graph",
            "benchmark_gold_used_in_coordination": False,
        },
        "hidden_accessed": False,
        "all": graph["all"],
        "relationship_multihop_slice": graph["relationship_multihop_slice"],
        "decision": "reject_candidate_keep_single_controller",
        "reason": (
            "The role-separated graph candidate caused more Full Evidence@5 losses than wins "
            "on both the full development set and its intended relationship/multi-hop slice."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"decision": report["decision"], "target": report["relationship_multihop_slice"]}
        )
    )


if __name__ == "__main__":
    main()
