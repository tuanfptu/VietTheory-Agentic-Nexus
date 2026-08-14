"""Tests for benchmark candidate safety guards."""

import pytest

from viettheory.benchmark_generation import (
    BenchmarkCandidate,
    CandidateEvidenceGroup,
    deduplicate_candidates,
    quota_category_plan,
    reject_unknown_evidence,
)
from viettheory.natural_benchmark import BenchmarkCategory


def _candidate(question: str, child_id: str = "child_1") -> BenchmarkCandidate:
    return BenchmarkCandidate.model_validate(
        {
            "question": question,
            "primary_category": "direct",
            "answerability": "answerable",
            "expected_behavior": "answer",
            "question_types": ("paraphrase",),
            "reasoning_scope": "single_chunk",
            "chapter_scope": "single_chapter",
            "difficulty": "medium",
            "gold_answer": "Câu trả lời tham chiếu.",
            "evidence_groups": (
                CandidateEvidenceGroup(
                    role="direct_answer",
                    child_ids=(child_id,),
                ),
            ),
        }
    )


def test_reject_unknown_evidence_blocks_hallucinated_ids() -> None:
    with pytest.raises(ValueError, match="unknown child IDs"):
        reject_unknown_evidence((_candidate("Một câu hỏi hợp lệ?", "invented"),), frozenset())


def test_deduplicate_candidates_ignores_case_accents_and_punctuation() -> None:
    first = _candidate("Vật chất là gì?")
    duplicate = _candidate("VAT CHAT LA GI")

    assert deduplicate_candidates((first, duplicate)) == (first,)


def test_quota_category_plan_accounts_for_existing_candidates() -> None:
    existing = (_candidate("Câu hỏi thứ nhất?"), _candidate("Câu hỏi thứ hai?"))
    plan = quota_category_plan(
        existing,
        {
            BenchmarkCategory.DIRECT: 3,
            BenchmarkCategory.EXPLANATION: 2,
        },
    )

    assert plan == (
        BenchmarkCategory.DIRECT,
        BenchmarkCategory.EXPLANATION,
        BenchmarkCategory.EXPLANATION,
    )
