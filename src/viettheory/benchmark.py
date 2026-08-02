"""Versioned contracts for evidence-retrieval benchmarks."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class QuestionType(StrEnum):
    DEFINITION = "definition"
    ENUMERATION = "enumeration"
    PARAPHRASE = "paraphrase"
    EXPLANATION = "explanation"
    CAUSE_EFFECT = "cause_effect"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    DOCUMENT_POLICY = "document_policy"
    APPLICATION = "application"
    MISCONCEPTION = "misconception"
    SYNTHESIS = "synthesis"
    OUT_OF_SCOPE = "out_of_scope"


class ReasoningScope(StrEnum):
    SINGLE_CHUNK = "single_chunk"
    MULTI_CHUNK = "multi_chunk"
    MULTI_HOP = "multi_hop"
    CROSS_SUBJECT = "cross_subject"


class ChapterScope(StrEnum):
    SINGLE_CHAPTER = "single_chapter"
    MULTI_CHAPTER = "multi_chapter"
    WHOLE_SUBJECT = "whole_subject"
    CROSS_SUBJECT = "cross_subject"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BenchmarkSplit(StrEnum):
    DEVELOPMENT = "development"
    HELD_OUT_TEST = "held_out_test"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    CHECKED = "checked"
    VERIFIED = "verified"
    REJECTED = "rejected"
    STALE = "stale"


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    OUT_OF_SCOPE = "out_of_scope"
    WRONG_SUBJECT = "wrong_subject"
    FALSE_PREMISE = "false_premise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DETAIL_NOT_STATED = "detail_not_stated"


class ExpectedBehavior(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    ROUTE_TO_CORRECT_SUBJECT = "route_to_correct_subject"
    CORRECT_PREMISE = "correct_premise"


class EvidenceRequirementMode(StrEnum):
    ALL_REQUIRED = "all_required"
    ANY_REQUIRED = "any_required"
    MINIMUM_K = "minimum_k"


class EvidenceRequirement(BenchmarkModel):
    mode: EvidenceRequirementMode = EvidenceRequirementMode.ALL_REQUIRED
    minimum_groups: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_minimum(self) -> EvidenceRequirement:
        if self.mode is EvidenceRequirementMode.MINIMUM_K and self.minimum_groups is None:
            raise ValueError("minimum_k evidence requirement needs minimum_groups")
        if self.mode is not EvidenceRequirementMode.MINIMUM_K and self.minimum_groups is not None:
            raise ValueError("minimum_groups is only valid for minimum_k")
        return self


class GoldEvidenceGroup(BenchmarkModel):
    group_id: Annotated[str, Field(pattern=r"^g[1-9]\d*$")]
    subject_code: NonEmpty
    role: NonEmpty
    required: bool = True
    primary_child_ids: tuple[NonEmpty, ...]
    acceptable_child_ids: tuple[NonEmpty, ...] = ()
    gold_parent_ids: tuple[NonEmpty, ...]
    gold_pdf_pages: tuple[Annotated[int, Field(ge=0)], ...]
    gold_printed_pages: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_ids(self) -> GoldEvidenceGroup:
        primary = set(self.primary_child_ids)
        acceptable = set(self.acceptable_child_ids)
        if not primary and not acceptable:
            raise ValueError("evidence groups require at least one child ID")
        if primary.intersection(acceptable):
            raise ValueError("primary and acceptable child IDs must be disjoint")
        if len(primary) != len(self.primary_child_ids):
            raise ValueError("primary child IDs must be unique")
        if len(acceptable) != len(self.acceptable_child_ids):
            raise ValueError("acceptable child IDs must be unique")
        if not self.gold_parent_ids:
            raise ValueError("evidence groups require at least one parent ID")
        if not self.gold_pdf_pages:
            raise ValueError("evidence groups require at least one PDF page")
        return self

    @property
    def all_child_ids(self) -> frozenset[str]:
        return frozenset((*self.primary_child_ids, *self.acceptable_child_ids))


class GenerationMetadata(BenchmarkModel):
    method: Literal["human", "llm_assisted", "migration"]
    model: str | None = None
    prompt_version: str | None = None


class BenchmarkQuestion(BenchmarkModel):
    """One answerability and evidence-retrieval evaluation item."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: NonEmpty
    id: Annotated[str, Field(pattern=r"^[a-z0-9]+_[0-9]{4}$")]
    subject_code: NonEmpty
    question: NonEmpty
    answerability: Answerability
    unanswerable_reason: str | None = None
    expected_behavior: ExpectedBehavior
    question_types: tuple[QuestionType, ...]
    reasoning_scope: ReasoningScope
    chapter_scope: ChapterScope
    difficulty: Difficulty
    chapter_labels: tuple[str, ...] = ()
    split: BenchmarkSplit
    gold_evidence_groups: tuple[GoldEvidenceGroup, ...]
    evidence_requirement: EvidenceRequirement = EvidenceRequirement()
    gold_answer: str | None = None
    required_concepts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    generation: GenerationMetadata
    artifact_manifest_id: NonEmpty
    review_status: ReviewStatus = ReviewStatus.DRAFT
    notes: str | None = None

    @model_validator(mode="after")
    def validate_question(self) -> BenchmarkQuestion:
        if not self.question_types:
            raise ValueError("question_types must not be empty")
        group_ids = [group.group_id for group in self.gold_evidence_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("evidence group IDs must be unique")
        if self.answerability is Answerability.ANSWERABLE:
            if not self.gold_evidence_groups:
                raise ValueError("answerable questions require evidence groups")
            if not self.gold_answer:
                raise ValueError("answerable questions require gold_answer")
            if self.expected_behavior is not ExpectedBehavior.ANSWER:
                raise ValueError("answerable questions must expect an answer")
        elif not self.unanswerable_reason:
            raise ValueError("unanswerable questions require unanswerable_reason")
        required_count = sum(group.required for group in self.gold_evidence_groups)
        minimum = self.evidence_requirement.minimum_groups
        if minimum is not None and minimum > required_count:
            raise ValueError("minimum_groups exceeds required evidence groups")
        return self


class ArtifactManifest(BenchmarkModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_manifest_id: NonEmpty
    subject_code: NonEmpty
    source_artifact_sha256: Sha256
    chunk_artifact_sha256: Sha256
    chunking_config_sha256: Sha256
    chunk_schema_version: NonEmpty
    retrieval_corpus_version: NonEmpty


class ReviewDecision(StrEnum):
    CHECKED = "checked"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVISE = "revise"


class BenchmarkReview(BenchmarkModel):
    schema_version: Literal["1.0"] = "1.0"
    question_id: NonEmpty
    decision: ReviewDecision
    reviewer_id: NonEmpty
    reviewed_at: datetime
    benchmark_record_sha256: Sha256
    notes: str | None = None
