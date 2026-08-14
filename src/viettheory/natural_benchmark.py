"""Natural QA v2 contracts and deterministic portfolio validation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from viettheory.subjects import SUBJECT_BY_CODE, get_subject

NonEmpty = Annotated[str, Field(min_length=1)]


class NaturalBenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BenchmarkCategory(StrEnum):
    DIRECT = "direct"
    EXPLANATION = "explanation"
    COMPARISON_RELATIONSHIP = "comparison_relationship"
    MULTI_CHUNK = "multi_chunk"
    SYNTHESIS = "synthesis"
    MULTI_HOP_CROSS_CHAPTER = "multi_hop_cross_chapter"
    NEGATIVE = "negative"


class NegativeType(StrEnum):
    FALSE_PREMISE = "false_premise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    WRONG_SUBJECT = "wrong_subject"
    OUT_OF_SCOPE = "out_of_scope"


class ReviewGateAudit(NaturalBenchmarkModel):
    question_valid: bool
    gold_answer_valid: bool
    parent_expanded_evidence_valid: bool
    difficulty_valid: bool
    reviewer_id: NonEmpty
    notes: str | None = None

    @property
    def passed(self) -> bool:
        return all(
            (
                self.question_valid,
                self.gold_answer_valid,
                self.parent_expanded_evidence_valid,
                self.difficulty_valid,
            )
        )


class NaturalQuestionV2(NaturalBenchmarkModel):
    schema_version: Literal["2.0"] = "2.0"
    benchmark_version: NonEmpty
    id: Annotated[str, Field(pattern=r"^[a-z0-9]+_[0-9]{4}$")]
    subject_code: NonEmpty
    chapter_labels: tuple[NonEmpty, ...]
    section_labels: tuple[NonEmpty, ...] = ()
    question: NonEmpty
    question_types: tuple[QuestionType, ...]
    primary_category: BenchmarkCategory
    difficulty: Difficulty
    reasoning_scope: ReasoningScope
    chapter_scope: ChapterScope
    answerability: Answerability
    unanswerable_reason: str | None = None
    negative_type: NegativeType | None = None
    expected_behavior: ExpectedBehavior
    gold_answer: str | None = None
    required_evidence_groups: tuple[GoldEvidenceGroup, ...] = ()
    required_concepts: tuple[NonEmpty, ...] = ()
    forbidden_claims: tuple[NonEmpty, ...] = ()
    split: BenchmarkSplit
    generation: GenerationMetadata
    artifact_manifest_ids: tuple[NonEmpty, ...]
    review_status: ReviewStatus = ReviewStatus.DRAFT
    review_notes: str | None = None
    review_gates: ReviewGateAudit | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> NaturalQuestionV2:
        get_subject(self.subject_code)
        evidence_subjects = {group.subject_code for group in self.required_evidence_groups}
        unknown = evidence_subjects.difference(SUBJECT_BY_CODE)
        if unknown:
            raise ValueError(f"unknown evidence subjects: {sorted(unknown)}")
        if self.answerability is Answerability.ANSWERABLE:
            if not self.gold_answer or not self.required_evidence_groups:
                raise ValueError("answerable questions require gold answer and evidence")
            if self.negative_type is not None:
                raise ValueError("answerable questions cannot have negative_type")
            if self.unanswerable_reason is not None:
                raise ValueError("answerable questions cannot have unanswerable_reason")
        else:
            if self.primary_category is not BenchmarkCategory.NEGATIVE:
                raise ValueError("unanswerable questions must use the negative category")
            if self.negative_type is None:
                raise ValueError("unanswerable questions require negative_type")
            if not self.unanswerable_reason:
                raise ValueError("unanswerable questions require unanswerable_reason")
        if not self.question_types:
            raise ValueError("question_types must not be empty")
        if self.reasoning_scope is ReasoningScope.CROSS_SUBJECT:
            if len(evidence_subjects) < 2:
                raise ValueError(
                    "cross-subject questions require evidence from at least two subjects"
                )
        elif evidence_subjects.difference({self.subject_code}):
            raise ValueError("non-cross-subject evidence must stay in the declared subject")
        if not self.artifact_manifest_ids:
            raise ValueError("at least one artifact manifest is required")
        if self.review_status is ReviewStatus.VERIFIED:
            if self.review_gates is None or not self.review_gates.passed:
                raise ValueError("verified questions require all four review gates")
        return self


class CategoryQuota(NaturalBenchmarkModel):
    category: BenchmarkCategory
    target: Annotated[int, Field(gt=0)]


class SubjectBatchPlan(NaturalBenchmarkModel):
    subject_code: NonEmpty
    batch_size: Annotated[int, Field(gt=0)] = 50
    batch_number: Annotated[int, Field(gt=0)]
    quotas: tuple[CategoryQuota, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> SubjectBatchPlan:
        get_subject(self.subject_code)
        categories = [quota.category for quota in self.quotas]
        if len(categories) != len(set(categories)):
            raise ValueError("quota categories must be unique")
        if sum(quota.target for quota in self.quotas) != self.batch_size:
            raise ValueError("category quotas must sum to batch_size")
        return self


class PortfolioPlan(NaturalBenchmarkModel):
    schema_version: Literal["2.0"] = "2.0"
    benchmark_version: NonEmpty
    natural_target_per_subject: Annotated[int, Field(ge=50)] = 250
    cross_subject_target: Annotated[int, Field(ge=1)] = 125
    evidence_sufficiency_target: Annotated[int, Field(ge=1)] = 200
    pilot_batches: tuple[SubjectBatchPlan, ...]

    @model_validator(mode="after")
    def validate_subject_coverage(self) -> PortfolioPlan:
        codes = [batch.subject_code for batch in self.pilot_batches]
        if set(codes) != set(SUBJECT_BY_CODE) or len(codes) != len(SUBJECT_BY_CODE):
            raise ValueError("pilot plan requires exactly one batch for each registered subject")
        return self


class PortfolioValidation(NaturalBenchmarkModel):
    valid: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    distribution: dict[str, dict[str, int]]
    subject_coverage: dict[str, SubjectCoverage]


class QuotaProgress(NaturalBenchmarkModel):
    current: Annotated[int, Field(ge=0)]
    target: Annotated[int, Field(gt=0)]


class SubjectCoverage(NaturalBenchmarkModel):
    total: Annotated[int, Field(ge=0)]
    batch_target: Annotated[int, Field(gt=0)]
    category: dict[str, QuotaProgress]
    difficulty: dict[str, int]
    chapter: dict[str, int]


def _tokens(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def validate_natural_portfolio(
    records: tuple[NaturalQuestionV2, ...], plan: PortfolioPlan
) -> PortfolioValidation:
    issues: list[str] = []
    warnings: list[str] = []
    subject_coverage: dict[str, SubjectCoverage] = {}
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        issues.append("duplicate question IDs")

    question_tokens = [(record.id, _tokens(record.question)) for record in records]
    for index, (left_id, left_tokens) in enumerate(question_tokens):
        for right_id, right_tokens in question_tokens[index + 1 :]:
            similarity = _jaccard(left_tokens, right_tokens)
            if similarity >= 0.85:
                warnings.append(
                    f"near-duplicate questions: {left_id}/{right_id} (Jaccard={similarity:.2f})"
                )

    for batch in plan.pilot_batches:
        subject_records = [
            record for record in records if record.subject_code == batch.subject_code
        ]
        if len(subject_records) > batch.batch_size:
            issues.append(f"{batch.subject_code} exceeds pilot batch size")
        counts = Counter(record.primary_category for record in subject_records)
        quota_by_category = {quota.category: quota.target for quota in batch.quotas}
        for category, count in counts.items():
            if count > quota_by_category.get(category, 0):
                issues.append(f"{batch.subject_code} exceeds {category.value} quota")

        parent_counts: Counter[str] = Counter(
            parent_id
            for record in subject_records
            for group in record.required_evidence_groups
            for parent_id in group.gold_parent_ids
        )
        section_counts = Counter(
            section for record in subject_records for section in record.section_labels
        )
        overuse_limit = max(3, math.ceil(max(len(subject_records), 1) * 0.25))
        for parent_id, count in parent_counts.items():
            if count > overuse_limit:
                warnings.append(
                    f"{batch.subject_code} gold parent overused: {parent_id} ({count} records)"
                )
        for section, count in section_counts.items():
            if count > overuse_limit:
                warnings.append(
                    f"{batch.subject_code} section overrepresented: {section} ({count} records)"
                )
        for record in subject_records:
            if record.gold_answer:
                answer_words = len(record.gold_answer.split())
                if answer_words < 5 or answer_words > 250:
                    warnings.append(f"{record.id} answer length anomaly: {answer_words} words")
                overlap = _jaccard(_tokens(record.question), _tokens(record.gold_answer))
                if len(_tokens(record.question)) >= 5 and overlap >= 0.75:
                    warnings.append(
                        f"{record.id} high question/answer lexical overlap: {overlap:.2f}"
                    )

        subject_coverage[batch.subject_code] = SubjectCoverage(
            total=len(subject_records),
            batch_target=batch.batch_size,
            category={
                quota.category.value: QuotaProgress(
                    current=counts[quota.category], target=quota.target
                )
                for quota in batch.quotas
            },
            difficulty=dict(Counter(record.difficulty.value for record in subject_records)),
            chapter=dict(
                Counter(chapter for record in subject_records for chapter in record.chapter_labels)
            ),
        )
    return PortfolioValidation(
        valid=not issues,
        issues=tuple(issues),
        warnings=tuple(sorted(set(warnings))),
        distribution={
            "subject": dict(Counter(record.subject_code for record in records)),
            "category": dict(Counter(record.primary_category.value for record in records)),
            "difficulty": dict(Counter(record.difficulty.value for record in records)),
            "answerability": dict(Counter(record.answerability.value for record in records)),
            "review_status": dict(Counter(record.review_status.value for record in records)),
        },
        subject_coverage=subject_coverage,
    )


def default_portfolio_plan() -> PortfolioPlan:
    quotas = (
        CategoryQuota(category=BenchmarkCategory.DIRECT, target=13),
        CategoryQuota(category=BenchmarkCategory.EXPLANATION, target=10),
        CategoryQuota(category=BenchmarkCategory.COMPARISON_RELATIONSHIP, target=8),
        CategoryQuota(category=BenchmarkCategory.MULTI_CHUNK, target=7),
        CategoryQuota(category=BenchmarkCategory.SYNTHESIS, target=5),
        CategoryQuota(category=BenchmarkCategory.MULTI_HOP_CROSS_CHAPTER, target=3),
        CategoryQuota(category=BenchmarkCategory.NEGATIVE, target=4),
    )
    return PortfolioPlan(
        benchmark_version="natural_qa_v2_pilot",
        pilot_batches=tuple(
            SubjectBatchPlan(subject_code=subject.code, batch_number=1, quotas=quotas)
            for subject in SUBJECT_BY_CODE.values()
        ),
    )
