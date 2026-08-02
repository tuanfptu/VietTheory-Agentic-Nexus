"""Apply the owner's third-round decisions to the final 13 MLN111 draft items."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from viettheory.benchmark import (
    BenchmarkQuestion,
    BenchmarkReview,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
    ReviewDecision,
    ReviewStatus,
)
from viettheory.schema import Chunk

APPROVED = {
    "mln111_0001",
    "mln111_0003",
    "mln111_0021",
    "mln111_0034",
    "mln111_0050",
    "mln111_0051",
    "mln111_0054",
    "mln111_0059",
    "mln111_0083",
    "mln111_0088",
}
REVISED = {"mln111_0071", "mln111_0080", "mln111_0090"}


def _group(child: Chunk) -> GoldEvidenceGroup:
    return GoldEvidenceGroup(
        group_id="g1",
        subject_code="MLN111",
        role="direct_answer",
        primary_child_ids=(child.chunk_id,),
        gold_parent_ids=(child.parent_chunk_id,) if child.parent_chunk_id else (),
        gold_pdf_pages=tuple(sorted({span.pdf_page for span in child.source_spans})),
        gold_printed_pages=tuple(
            sorted({span.printed_page for span in child.source_spans if span.printed_page})
        ),
    )


def _apply(
    question: BenchmarkQuestion,
    children: dict[str, Chunk],
) -> BenchmarkQuestion:
    if question.id in APPROVED:
        return question.model_copy(update={"review_status": ReviewStatus.CHECKED})
    if question.id == "mln111_0071":
        return BenchmarkQuestion.model_validate(
            {
                **question.model_dump(),
                "gold_evidence_groups": (_group(children["child_c737633187ca0ad8eb15"]),),
                "question_types": (QuestionType.EXPLANATION,),
                "reasoning_scope": ReasoningScope.SINGLE_CHUNK,
                "review_status": ReviewStatus.DRAFT,
            },
            strict=False,
        )
    if question.id == "mln111_0080":
        group = next(
            group
            for group in question.gold_evidence_groups
            if "child_d8d7cb4635dae1726a07" in group.primary_child_ids
        ).model_copy(update={"group_id": "g1", "role": "direct_answer"})
        return BenchmarkQuestion.model_validate(
            {
                **question.model_dump(),
                "gold_evidence_groups": (group,),
                "reasoning_scope": ReasoningScope.SINGLE_CHUNK,
                "review_status": ReviewStatus.CHECKED,
            },
            strict=False,
        )
    if question.id == "mln111_0090":
        return question.model_copy(
            update={
                "question_types": (QuestionType.EXPLANATION,),
                "review_status": ReviewStatus.CHECKED,
            }
        )
    return question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    draft_ids = {
        question.id for question in questions if question.review_status is ReviewStatus.DRAFT
    }
    if draft_ids != APPROVED | REVISED:
        raise ValueError("round-3 IDs do not match the 13 draft questions")
    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    updated = [_apply(question, children) for question in questions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(question.model_dump_json() + "\n" for question in updated),
        encoding="utf-8",
    )

    reviewed_at = datetime(2026, 7, 24, tzinfo=UTC)
    audit: list[BenchmarkReview] = []
    for question in questions:
        if question.id not in draft_ids:
            continue
        digest = hashlib.sha256(question.model_dump_json().encode("utf-8")).hexdigest()
        decision = ReviewDecision.CHECKED if question.id in APPROVED else ReviewDecision.REVISE
        audit.append(
            BenchmarkReview(
                question_id=question.id,
                decision=decision,
                reviewer_id="project_owner",
                reviewed_at=reviewed_at,
                benchmark_record_sha256=digest,
                notes="Imported from MLN111 human review round 3.",
            )
        )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        "".join(item.model_dump_json() + "\n" for item in audit),
        encoding="utf-8",
    )
    print("round3=13 checked_decisions=10 revised_decisions=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
