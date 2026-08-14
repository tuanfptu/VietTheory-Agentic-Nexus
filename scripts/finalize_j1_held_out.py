"""Publish aggregate-only J1 held-out metrics and private artifact checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_cases", type=Path)
    parser.add_argument("private_checkpoint", type=Path)
    parser.add_argument("private_report", type=Path)
    parser.add_argument("public_summary", type=Path)
    args = parser.parse_args()
    report: dict[str, Any] = json.loads(args.private_report.read_text(encoding="utf-8"))
    if report.get("split") != "held_out_test" or report.get("case_count") != 16:
        raise ValueError("expected a complete 16-case held-out J1 report")
    if len(report.get("per_case", ())) != 16:
        raise ValueError("private J1 report is incomplete")
    checkpoint_count = sum(
        bool(line.strip())
        for line in args.private_checkpoint.read_text(encoding="utf-8").splitlines()
    )
    if checkpoint_count != 16:
        raise ValueError("private J1 checkpoint is incomplete")
    summary = {
        "schema_version": "1.0",
        "benchmark_version": report["benchmark_version"],
        "split": "held_out_test",
        "one_shot_after_j1_freeze": True,
        "judge": report["judge"],
        "model": report["model"],
        "prompt_version": report["prompt_version"],
        "case_count": report["case_count"],
        "gold_labels_not_sent_to_model": report["gold_labels_not_sent_to_model"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "f1_by_label": report["f1_by_label"],
        "confusion": report["confusion"],
        "predicted_distribution": report["predicted_distribution"],
        "per_case_data_public": False,
        "private_artifact_checksums": {
            "cases_sha256": _digest(args.private_cases),
            "checkpoint_sha256": _digest(args.private_checkpoint),
            "report_sha256": _digest(args.private_report),
        },
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}))


if __name__ == "__main__":
    main()
