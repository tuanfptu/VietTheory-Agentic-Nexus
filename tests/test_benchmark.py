"""Tests for benchmark v1 annotation contracts."""

import pytest
from pydantic import ValidationError

from viettheory.benchmark import (
    Answerability,
    BenchmarkQuestion,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
)


def _group(
    *,
    primary: tuple[str, ...] = ("child_1",),
    acceptable: tuple[str, ...] = (),
) -> GoldEvidenceGroup:
    return GoldEvidenceGroup(
        group_id="g1",
        subject_code="MLN111",
        role="definition",
        primary_child_ids=primary,
        acceptable_child_ids=acceptable,
        gold_parent_ids=("parent_1",),
        gold_pdf_pages=(12,),
    )


def _question(**overrides: object) -> BenchmarkQuestion:
    fields: dict[str, object] = {
        "benchmark_version": "1.0.0",
        "id": "mln111_0001",
        "subject_code": "MLN111",
        "question": "Khái niệm này được định nghĩa như thế nào?",
        "answerability": Answerability.ANSWERABLE,
        "expected_behavior": ExpectedBehavior.ANSWER,
        "question_types": (QuestionType.DEFINITION,),
        "reasoning_scope": ReasoningScope.SINGLE_CHUNK,
        "chapter_scope": ChapterScope.SINGLE_CHAPTER,
        "difficulty": Difficulty.EASY,
        "split": BenchmarkSplit.DEVELOPMENT,
        "gold_evidence_groups": (_group(),),
        "gold_answer": "Một câu trả lời có căn cứ.",
        "generation": GenerationMetadata(method="human"),
        "artifact_manifest_id": "mln111_manifest",
    }
    fields.update(overrides)
    return BenchmarkQuestion.model_validate(fields)


def test_answerable_question_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="require evidence groups"):
        _question(gold_evidence_groups=())


def test_unanswerable_question_requires_reason() -> None:
    with pytest.raises(ValidationError, match="require unanswerable_reason"):
        _question(
            answerability=Answerability.OUT_OF_SCOPE,
            expected_behavior=ExpectedBehavior.REFUSE,
            gold_evidence_groups=(),
            gold_answer=None,
        )


def test_primary_and_acceptable_ids_must_be_disjoint() -> None:
    with pytest.raises(ValidationError, match="must be disjoint"):
        _group(primary=("child_1",), acceptable=("child_1",))


def test_false_premise_may_keep_correction_evidence() -> None:
    question = _question(
        answerability=Answerability.FALSE_PREMISE,
        expected_behavior=ExpectedBehavior.CORRECT_PREMISE,
        unanswerable_reason="Tiền đề của câu hỏi trái với giáo trình.",
        gold_answer="Giáo trình nêu kết luận ngược lại.",
    )

    assert question.gold_evidence_groups[0].role == "definition"
