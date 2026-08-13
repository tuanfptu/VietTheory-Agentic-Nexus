import pytest
from pydantic import ValidationError

from viettheory.benchmark import (
    Answerability,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    ReasoningScope,
    ReviewStatus,
)
from viettheory.natural_benchmark import (
    BenchmarkCategory,
    NaturalQuestionV2,
    ReviewGateAudit,
    default_portfolio_plan,
    validate_natural_portfolio,
)


def _record(**overrides: object) -> NaturalQuestionV2:
    fields: dict[str, object] = {
        "benchmark_version": "natural_qa_v2_pilot",
        "id": "mln111_0001",
        "subject_code": "MLN111",
        "chapter_labels": ("Chương 1",),
        "question": "Khái niệm được định nghĩa như thế nào?",
        "primary_category": BenchmarkCategory.DIRECT,
        "difficulty": Difficulty.EASY,
        "reasoning_scope": ReasoningScope.SINGLE_CHUNK,
        "chapter_scope": ChapterScope.SINGLE_CHAPTER,
        "answerability": Answerability.ANSWERABLE,
        "expected_behavior": ExpectedBehavior.ANSWER,
        "gold_answer": "Câu trả lời dựa trên giáo trình.",
        "required_evidence_groups": (
            GoldEvidenceGroup(
                group_id="g1",
                subject_code="MLN111",
                role="definition",
                primary_child_ids=("child_1",),
                gold_parent_ids=("parent_1",),
                gold_pdf_pages=(1,),
            ),
        ),
        "split": BenchmarkSplit.DEVELOPMENT,
        "generation": GenerationMetadata(method="human"),
        "artifact_manifest_ids": ("mln111_manifest",),
    }
    fields.update(overrides)
    return NaturalQuestionV2.model_validate(fields)


def test_default_plan_covers_five_subjects_with_50_case_batches() -> None:
    plan = default_portfolio_plan()
    assert len(plan.pilot_batches) == 5
    assert all(sum(quota.target for quota in batch.quotas) == 50 for batch in plan.pilot_batches)
    assert plan.natural_target_per_subject == 250


def test_verified_record_requires_all_four_review_gates() -> None:
    failed = ReviewGateAudit(
        question_valid=True,
        gold_answer_valid=True,
        parent_expanded_evidence_valid=False,
        difficulty_valid=True,
        reviewer_id="reviewer_1",
    )
    with pytest.raises(ValidationError, match="all four review gates"):
        _record(review_status=ReviewStatus.VERIFIED, review_gates=failed)


def test_cross_subject_requires_two_evidence_subjects() -> None:
    with pytest.raises(ValidationError, match="at least two subjects"):
        _record(reasoning_scope=ReasoningScope.CROSS_SUBJECT)


def test_portfolio_validator_reports_distribution() -> None:
    report = validate_natural_portfolio((_record(),), default_portfolio_plan())
    assert report.valid
    assert report.distribution["subject"] == {"MLN111": 1}
    direct = report.subject_coverage["MLN111"].category["direct"]
    assert (direct.current, direct.target) == (1, 13)


def test_portfolio_validator_warns_about_near_duplicates_and_parent_overuse() -> None:
    records = tuple(
        _record(
            id=f"mln111_{index:04d}",
            question="Khái niệm cơ bản này được định nghĩa như thế nào?",
        )
        for index in range(1, 5)
    )
    report = validate_natural_portfolio(records, default_portfolio_plan())

    assert report.valid
    assert any("near-duplicate questions" in warning for warning in report.warnings)
    assert any("gold parent overused" in warning for warning in report.warnings)
