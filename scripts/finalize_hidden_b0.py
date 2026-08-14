"""Validate the one-shot private hidden report and publish aggregate-only metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from viettheory.benchmark import BenchmarkSplit
from viettheory.natural_benchmark import NaturalQuestionV2


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("private_report", type=Path)
    parser.add_argument("public_summary", type=Path)
    args = parser.parse_args()
    questions = tuple(
        NaturalQuestionV2.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(questions) != 150 or {question.split for question in questions} != {
        BenchmarkSplit.HELD_OUT_TEST
    }:
        raise ValueError("expected exactly 150 held-out Natural QA questions")
    report: dict[str, Any] = json.loads(args.private_report.read_text(encoding="utf-8"))
    if len(report.get("variants", {})) != 10:
        raise ValueError("hidden B0 report must contain all 10 frozen variants")
    answerable = sum(bool(question.required_evidence_groups) for question in questions)
    if report.get("evaluated_answerable_questions") != answerable:
        raise ValueError("hidden report answerable count mismatch")
    # The original evaluator hard-coded `development`; correct metadata only, without rerunning or
    # changing any ranking/metric.
    report["split"] = BenchmarkSplit.HELD_OUT_TEST.value
    args.private_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    names = ("within_subject_parent_aware_b0", "global_parent_aware_b0")
    summary = {
        "schema_version": "1.0",
        "benchmark_version": report["benchmark_version"],
        "split": BenchmarkSplit.HELD_OUT_TEST.value,
        "one_shot_after_candidate_freeze": True,
        "question_count": len(questions),
        "evaluated_answerable_questions": answerable,
        "excluded_unanswerable_questions": len(questions) - answerable,
        "per_query_data_public": False,
        "variants": {name: report["variants"][name]["metrics"] for name in names},
        "private_artifact_checksums": {
            "questions_sha256": _digest(args.questions),
            "report_sha256": _digest(args.private_report),
        },
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["variants"]))


if __name__ == "__main__":
    main()
