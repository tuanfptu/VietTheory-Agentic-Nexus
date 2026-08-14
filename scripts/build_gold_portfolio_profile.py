"""Materialize the post-review Natural QA v2 Gold category profile for validation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from viettheory.natural_benchmark import (
    CategoryQuota,
    NaturalQuestionV2,
    PortfolioPlan,
    SubjectBatchPlan,
)
from viettheory.subjects import SUBJECTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    questions = tuple(
        NaturalQuestionV2.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    batches: list[SubjectBatchPlan] = []
    for number, subject in enumerate(SUBJECTS, 1):
        selected = tuple(item for item in questions if item.subject_code == subject.code)
        counts = Counter(item.primary_category for item in selected)
        batches.append(
            SubjectBatchPlan(
                subject_code=subject.code,
                batch_size=len(selected),
                batch_number=number,
                quotas=tuple(
                    CategoryQuota(category=category, target=count)
                    for category, count in sorted(counts.items(), key=lambda item: item[0].value)
                ),
            )
        )
    profile = PortfolioPlan(
        benchmark_version="natural_qa_v2_gold_v1.0_post_review_profile",
        natural_target_per_subject=100,
        cross_subject_target=75,
        evidence_sufficiency_target=120,
        pilot_batches=tuple(batches),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
