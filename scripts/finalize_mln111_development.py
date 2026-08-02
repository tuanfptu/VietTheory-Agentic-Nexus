"""Finalize the last MLN111 development review decision."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from viettheory.benchmark import (
    BenchmarkQuestion,
    BenchmarkReview,
    ReviewDecision,
    ReviewStatus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    drafts = [question for question in questions if question.review_status is ReviewStatus.DRAFT]
    if len(drafts) != 1 or drafts[0].id != "mln111_0071":
        raise ValueError("expected mln111_0071 to be the only remaining draft")
    original = drafts[0]
    finalized = [
        question.model_copy(update={"review_status": ReviewStatus.CHECKED})
        if question.id == original.id
        else question
        for question in questions
    ]
    if len(finalized) != 70 or any(
        question.review_status is not ReviewStatus.CHECKED for question in finalized
    ):
        raise ValueError("development set is not fully checked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(question.model_dump_json() + "\n" for question in finalized),
        encoding="utf-8",
    )
    digest = hashlib.sha256(original.model_dump_json().encode("utf-8")).hexdigest()
    review = BenchmarkReview(
        question_id=original.id,
        decision=ReviewDecision.CHECKED,
        reviewer_id="project_owner",
        reviewed_at=datetime(2026, 7, 24, tzinfo=UTC),
        benchmark_record_sha256=digest,
        notes=(
            "Question, gold, and full child evidence approved. "
            "single_chunk and explanation accepted."
        ),
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(review.model_dump_json() + "\n", encoding="utf-8")
    print("development=70 checked=70")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
