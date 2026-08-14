from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.import_natural_qa_v2_review import import_review

from viettheory.benchmark import ReviewStatus


def _question(question_id: str = "mln111_0001") -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "benchmark_version": "test",
        "id": question_id,
        "subject_code": "MLN111",
        "chapter_labels": ["Chapter"],
        "section_labels": [],
        "question": "Question?",
        "question_types": ["definition"],
        "primary_category": "direct",
        "difficulty": "easy",
        "reasoning_scope": "single_chunk",
        "chapter_scope": "single_chapter",
        "answerability": "answerable",
        "unanswerable_reason": None,
        "negative_type": None,
        "expected_behavior": "answer",
        "gold_answer": "Supported answer.",
        "required_evidence_groups": [
            {
                "group_id": "g1",
                "subject_code": "MLN111",
                "role": "main",
                "required": True,
                "primary_child_ids": ["child_1"],
                "acceptable_child_ids": [],
                "gold_parent_ids": ["parent_1"],
                "gold_pdf_pages": [1],
                "gold_printed_pages": [],
            }
        ],
        "required_concepts": ["concept"],
        "forbidden_claims": [],
        "split": "development",
        "generation": {
            "method": "human",
            "model": None,
            "prompt_version": None,
        },
        "artifact_manifest_ids": ["manifest:test"],
        "review_status": "draft",
        "review_notes": None,
        "review_gates": None,
    }


def _write_fixture(tmp_path: Path, *, decision: str, passed: bool = True) -> tuple[Path, Path]:
    draft = tmp_path / "draft.jsonl"
    draft.write_text(json.dumps(_question()) + "\n", encoding="utf-8")
    review = tmp_path / "review.csv"
    fields = [
        "id",
        "subject",
        "question",
        "decision",
        "question_valid",
        "gold_answer_valid",
        "evidence_valid",
        "difficulty_valid",
        "acceptable_chunk_notes",
        "reviewer_notes",
    ]
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "id": "mln111_0001",
                "subject": "MLN111",
                "question": "Question?",
                "decision": decision,
                "question_valid": str(passed).upper(),
                "gold_answer_valid": "TRUE",
                "evidence_valid": "TRUE",
                "difficulty_valid": "TRUE",
                "acceptable_chunk_notes": "Alternative child is acceptable.",
                "reviewer_notes": "Reviewed.",
            }
        )
    return draft, review


def test_import_review_promotes_only_approved_records(tmp_path: Path) -> None:
    draft, review = _write_fixture(tmp_path, decision="approve")
    records, manifest = import_review(draft, review, reviewer_id="reviewer")

    assert records[0].review_status is ReviewStatus.VERIFIED
    assert records[0].review_gates is not None
    assert records[0].review_gates.passed
    assert manifest["decision_counts"] == {"approve": 1}


def test_import_review_keeps_revise_pending(tmp_path: Path) -> None:
    draft, review = _write_fixture(tmp_path, decision="revise", passed=False)
    records, _ = import_review(draft, review, reviewer_id="reviewer")

    assert records[0].review_status is ReviewStatus.DRAFT


def test_import_review_rejects_failed_approve(tmp_path: Path) -> None:
    draft, review = _write_fixture(tmp_path, decision="approve", passed=False)
    with pytest.raises(ValueError, match="approve requires all review gates"):
        import_review(draft, review, reviewer_id="reviewer")
