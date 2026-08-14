from __future__ import annotations

from viettheory.benchmark import (
    Answerability,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
    ReviewStatus,
)
from viettheory.benchmark_split import assert_no_parent_leakage, split_subject
from viettheory.natural_benchmark import BenchmarkCategory, NaturalQuestionV2, ReviewGateAudit


def _question(index: int, parent: str) -> NaturalQuestionV2:
    return NaturalQuestionV2(
        benchmark_version="test",
        id=f"mln111_{index:04d}",
        subject_code="MLN111",
        chapter_labels=("Chapter",),
        question=f"Question {index}?",
        question_types=(QuestionType.DEFINITION,),
        primary_category=BenchmarkCategory.DIRECT,
        difficulty=Difficulty.EASY if index % 2 else Difficulty.MEDIUM,
        reasoning_scope=ReasoningScope.SINGLE_CHUNK,
        chapter_scope=ChapterScope.SINGLE_CHAPTER,
        answerability=Answerability.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        gold_answer="Supported answer.",
        required_evidence_groups=(
            GoldEvidenceGroup(
                group_id="g1",
                subject_code="MLN111",
                role="support",
                primary_child_ids=(f"child_{index}",),
                gold_parent_ids=(parent,),
                gold_pdf_pages=(index,),
            ),
        ),
        required_concepts=("concept",),
        split=BenchmarkSplit.DEVELOPMENT,
        generation=GenerationMetadata(method="human", model=None, prompt_version="test"),
        artifact_manifest_ids=("manifest",),
        review_status=ReviewStatus.VERIFIED,
        review_gates=ReviewGateAudit(
            question_valid=True,
            gold_answer_valid=True,
            parent_expanded_evidence_valid=True,
            difficulty_valid=True,
            reviewer_id="test",
        ),
    )


def test_split_subject_is_exact_deterministic_and_parent_safe() -> None:
    records = tuple(
        _question(index, "shared" if index in {1, 2, 3} else f"parent_{index}")
        for index in range(1, 11)
    )
    first = split_subject(records, hidden_size=4, seed="fixed", beam_width=128)
    second = split_subject(reversed(records), hidden_size=4, seed="fixed", beam_width=128)

    assert len(first.development) == 6
    assert len(first.hidden) == 4
    assert [record.id for record in first.hidden] == [record.id for record in second.hidden]
    assert_no_parent_leakage(first)
