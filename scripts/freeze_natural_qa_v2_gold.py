"""Merge the final 48-row human review and freeze Natural QA v2 Gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from viettheory.benchmark import (
    Answerability,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
    ReviewStatus,
)
from viettheory.natural_benchmark import (
    BenchmarkCategory,
    NaturalQuestionV2,
    NegativeType,
    ReviewGateAudit,
)
from viettheory.schema import Chunk

GOLD_VERSION = "natural_qa_v2_gold_v1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth(value: Any, *, field: str, question_id: str) -> bool:
    if value is True or str(value).strip().casefold() == "true":
        return True
    if value is False or str(value).strip().casefold() == "false":
        return False
    raise ValueError(f"{question_id}: invalid {field} value {value!r}")


def _parts(value: Any) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split("|") if part.strip())


def _pages(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in _parts(value))


def _question_types(
    current: tuple[QuestionType, ...], category: BenchmarkCategory
) -> tuple[QuestionType, ...]:
    required = {
        BenchmarkCategory.COMPARISON_RELATIONSHIP: QuestionType.COMPARISON,
        BenchmarkCategory.SYNTHESIS: QuestionType.SYNTHESIS,
        BenchmarkCategory.NEGATIVE: QuestionType.OUT_OF_SCOPE,
    }.get(category)
    if required is None or required in current:
        return current
    return (*current, required)


def _answerability_fields(
    answerability: Answerability, review_row: dict[str, Any]
) -> tuple[str | None, NegativeType | None, ExpectedBehavior]:
    if answerability is Answerability.ANSWERABLE:
        return None, None, ExpectedBehavior.ANSWER
    mapping = {
        Answerability.WRONG_SUBJECT: (
            NegativeType.WRONG_SUBJECT,
            ExpectedBehavior.ROUTE_TO_CORRECT_SUBJECT,
        ),
        Answerability.FALSE_PREMISE: (
            NegativeType.FALSE_PREMISE,
            ExpectedBehavior.CORRECT_PREMISE,
        ),
        Answerability.OUT_OF_SCOPE: (
            NegativeType.OUT_OF_SCOPE,
            ExpectedBehavior.REFUSE,
        ),
        Answerability.INSUFFICIENT_EVIDENCE: (
            NegativeType.INSUFFICIENT_EVIDENCE,
            ExpectedBehavior.REFUSE,
        ),
        Answerability.DETAIL_NOT_STATED: (
            NegativeType.INSUFFICIENT_EVIDENCE,
            ExpectedBehavior.REFUSE,
        ),
    }
    negative_type, behavior = mapping[answerability]
    reason = str(review_row.get("reviewer_notes") or "").strip()
    if not reason:
        reason = f"Human-verified negative case: {answerability.value}."
    return reason, negative_type, behavior


def freeze_gold(
    followup_path: Path,
    rows_path: Path,
    data_root: Path,
    *,
    reviewer_id: str,
) -> tuple[list[NaturalQuestionV2], dict[str, Any]]:
    records = {
        record.id: record
        for line in followup_path.read_text(encoding="utf-8").splitlines()
        if (record := NaturalQuestionV2.model_validate_json(line))
    }
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("final review export must contain exactly 48 rows")
    row_ids = [str(row["id"]) for row in rows]
    expected_ids = {
        record.id for record in records.values() if record.review_status is ReviewStatus.DRAFT
    }
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
        raise ValueError("final review IDs do not match the 48 pending records")

    children_by_subject: dict[str, dict[str, Chunk]] = {}
    for subject in sorted({record.subject_code for record in records.values()}):
        path = data_root / subject / "structured_v1" / "children.jsonl"
        children_by_subject[subject] = {
            chunk.chunk_id: chunk
            for line in path.read_text(encoding="utf-8").splitlines()
            if (chunk := Chunk.model_validate_json(line))
        }

    for row in rows:
        question_id = str(row["id"])
        if str(row.get("decision", "")).strip().casefold() != "approve":
            raise ValueError(f"{question_id}: final decision is not approve")
        gates = ReviewGateAudit(
            question_valid=_truth(
                row.get("question_valid"), field="question_valid", question_id=question_id
            ),
            gold_answer_valid=_truth(
                row.get("gold_answer_valid"),
                field="gold_answer_valid",
                question_id=question_id,
            ),
            parent_expanded_evidence_valid=_truth(
                row.get("evidence_valid"), field="evidence_valid", question_id=question_id
            ),
            difficulty_valid=_truth(
                row.get("difficulty_valid"),
                field="difficulty_valid",
                question_id=question_id,
            ),
            reviewer_id=reviewer_id,
            notes=str(row.get("reviewer_notes") or "").strip() or None,
        )
        if not gates.passed:
            raise ValueError(f"{question_id}: all four final review gates must pass")

        current = records[question_id]
        category = BenchmarkCategory(str(row["new_category"]).strip())
        answerability = Answerability(str(row["new_answerability"]).strip())
        child_ids = _parts(row.get("new_child_ids"))
        parent_ids = _parts(row.get("new_parent_ids"))
        pages = _pages(row.get("new_pages"))
        children = children_by_subject[current.subject_code]
        cited = [children[child_id] for child_id in child_ids]
        actual_parent_ids = tuple(
            dict.fromkeys(chunk.parent_chunk_id for chunk in cited if chunk.parent_chunk_id)
        )
        actual_pages = tuple(
            sorted({span.pdf_page for chunk in cited for span in chunk.source_spans})
        )
        if set(parent_ids) != set(actual_parent_ids):
            raise ValueError(f"{question_id}: reviewed parent IDs do not match corpus")
        if set(pages) != set(actual_pages):
            raise ValueError(f"{question_id}: reviewed pages do not match corpus")

        groups = current.required_evidence_groups
        current_children = tuple(
            child_id
            for group in current.required_evidence_groups
            for child_id in group.primary_child_ids
        )
        if child_ids != current_children:
            groups = tuple(
                GoldEvidenceGroup(
                    group_id=f"g{index}",
                    subject_code=current.subject_code,
                    role="human_verified_evidence",
                    required=True,
                    primary_child_ids=(child.chunk_id,),
                    gold_parent_ids=(child.parent_chunk_id,) if child.parent_chunk_id else (),
                    gold_pdf_pages=tuple(sorted({span.pdf_page for span in child.source_spans})),
                )
                for index, child in enumerate(cited, start=1)
            )

        unanswerable_reason, negative_type, expected_behavior = _answerability_fields(
            answerability, row
        )
        chapter_labels = (
            tuple(dict.fromkeys(chunk.chapter for chunk in cited if chunk.chapter))
            or current.chapter_labels
        )
        section_labels = (
            tuple(dict.fromkeys(chunk.section for chunk in cited if chunk.section))
            or current.section_labels
        )
        reasoning_scope = current.reasoning_scope
        if category in {BenchmarkCategory.MULTI_CHUNK, BenchmarkCategory.SYNTHESIS}:
            reasoning_scope = ReasoningScope.MULTI_CHUNK
        elif category is BenchmarkCategory.MULTI_HOP_CROSS_CHAPTER:
            reasoning_scope = ReasoningScope.MULTI_HOP
        elif len(child_ids) <= 1:
            reasoning_scope = ReasoningScope.SINGLE_CHUNK
        chapter_scope = (
            ChapterScope.MULTI_CHAPTER if len(chapter_labels) > 1 else ChapterScope.SINGLE_CHAPTER
        )

        records[question_id] = current.model_copy(
            update={
                "benchmark_version": GOLD_VERSION,
                "question": str(row["revised_question"]).strip(),
                "gold_answer": str(row.get("revised_gold_answer") or "").strip() or None,
                "primary_category": category,
                "answerability": answerability,
                "difficulty": Difficulty(str(row["new_difficulty"]).strip()),
                "question_types": _question_types(current.question_types, category),
                "reasoning_scope": reasoning_scope,
                "chapter_scope": chapter_scope,
                "chapter_labels": chapter_labels,
                "section_labels": section_labels,
                "required_evidence_groups": groups,
                "unanswerable_reason": unanswerable_reason,
                "negative_type": negative_type,
                "expected_behavior": expected_behavior,
                "review_status": ReviewStatus.VERIFIED,
                "review_notes": gates.notes,
                "review_gates": gates,
            }
        )

    frozen = [
        record.model_copy(update={"benchmark_version": GOLD_VERSION}) for record in records.values()
    ]
    for record in frozen:
        NaturalQuestionV2.model_validate(record.model_dump())
        if record.review_status is not ReviewStatus.VERIFIED:
            raise ValueError(f"{record.id}: Gold contains a non-verified record")
    manifest = {
        "schema_version": "1.0",
        "benchmark_version": GOLD_VERSION,
        "record_count": len(frozen),
        "review_status": dict(Counter(record.review_status.value for record in frozen)),
        "subjects": dict(Counter(record.subject_code for record in frozen)),
        "categories": dict(Counter(record.primary_category.value for record in frozen)),
        "answerability": dict(Counter(record.answerability.value for record in frozen)),
        "final_review_rows": len(rows),
        "final_review_approved": 48,
        "release_ready": True,
    }
    return frozen, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--review-rows", type=Path, required=True)
    parser.add_argument("--review-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--reviewer-id", default="human_final_review_2026_08_14")
    args = parser.parse_args()

    records, manifest = freeze_gold(
        args.followup,
        args.review_rows,
        args.data_root,
        reviewer_id=args.reviewer_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    manifest.update(
        {
            "gold_jsonl_sha256": _sha256(args.output),
            "followup_source_sha256": _sha256(args.followup),
            "final_review_workbook_sha256": _sha256(args.review_workbook),
        }
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.checksum.write_text(
        f"{manifest['gold_jsonl_sha256']}  {args.output.name}\n"
        f"{_sha256(args.manifest)}  {args.manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
