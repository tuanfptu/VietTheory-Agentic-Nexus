"""Write the one-pass 500-question Natural QA v2 draft portfolio plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from viettheory.natural_benchmark import (
    BenchmarkCategory,
    CategoryQuota,
    PortfolioPlan,
    SubjectBatchPlan,
)
from viettheory.subjects import SUBJECT_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v2/portfolio_plan_500.json"),
    )
    args = parser.parse_args()
    quotas = (
        CategoryQuota(category=BenchmarkCategory.DIRECT, target=26),
        CategoryQuota(category=BenchmarkCategory.EXPLANATION, target=20),
        CategoryQuota(category=BenchmarkCategory.COMPARISON_RELATIONSHIP, target=16),
        CategoryQuota(category=BenchmarkCategory.MULTI_CHUNK, target=14),
        CategoryQuota(category=BenchmarkCategory.SYNTHESIS, target=10),
        CategoryQuota(category=BenchmarkCategory.MULTI_HOP_CROSS_CHAPTER, target=6),
        CategoryQuota(category=BenchmarkCategory.NEGATIVE, target=8),
    )
    plan = PortfolioPlan(
        benchmark_version="natural_qa_v2_500_draft",
        natural_target_per_subject=100,
        cross_subject_target=75,
        evidence_sufficiency_target=120,
        pilot_batches=tuple(
            SubjectBatchPlan(
                subject_code=subject.code,
                batch_size=100,
                batch_number=1,
                quotas=quotas,
            )
            for subject in SUBJECT_BY_CODE.values()
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
