"""Contracts for controlled evidence-sufficiency evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]


class SufficiencyLabel(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    MISSING = "missing"
    WRONG_ASPECT = "wrong_aspect"
    CONTRADICTED = "contradicted"


class PerturbationType(StrEnum):
    NONE = "none"
    REMOVE_REQUIRED_GROUPS = "remove_required_groups"
    REMOVE_ALL_EVIDENCE = "remove_all_evidence"
    RELATED_WRONG_ASPECT = "related_wrong_aspect"
    NATURAL_CONTRADICTION = "natural_contradiction"


class SufficiencyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RequiredAspect(SufficiencyModel):
    aspect_id: Annotated[str, Field(pattern=r"^a[1-9]\d*$")]
    description: NonEmpty
    acceptable_parent_ids: tuple[NonEmpty, ...]


class ProvidedContext(SufficiencyModel):
    parent_id: NonEmpty
    subject_code: NonEmpty
    text: NonEmpty
    pdf_pages: tuple[Annotated[int, Field(ge=0)], ...]


class SufficiencyReview(SufficiencyModel):
    reviewer_id: NonEmpty
    label_valid: bool
    context_valid: bool
    natural_difficulty_valid: bool
    notes: str | None = None


class EvidenceSufficiencyCase(SufficiencyModel):
    schema_version: Literal["0.1"] = "0.1"
    benchmark_version: NonEmpty
    case_id: Annotated[str, Field(pattern=r"^es_[a-z0-9]+_[0-9]{4}_[a-z]+$")]
    source_question_id: NonEmpty
    subject_code: NonEmpty
    question: NonEmpty
    required_aspects: tuple[RequiredAspect, ...]
    provided_contexts: tuple[ProvidedContext, ...]
    expected_label: SufficiencyLabel
    perturbation: PerturbationType
    removed_aspect_ids: tuple[NonEmpty, ...] = ()
    generator_version: NonEmpty
    split_group: NonEmpty
    split: Literal["development", "held_out_test"] = "development"
    review_status: Literal["draft", "checked", "verified"] = "draft"
    review: SufficiencyReview | None = None

    @model_validator(mode="after")
    def validate_case(self) -> EvidenceSufficiencyCase:
        aspect_ids = {aspect.aspect_id for aspect in self.required_aspects}
        if not self.required_aspects:
            raise ValueError("at least one required aspect is required")
        if set(self.removed_aspect_ids).difference(aspect_ids):
            raise ValueError("removed aspects must exist in required_aspects")
        if self.expected_label is SufficiencyLabel.SUFFICIENT and self.removed_aspect_ids:
            raise ValueError("sufficient cases cannot remove required aspects")
        if self.expected_label is SufficiencyLabel.MISSING and self.provided_contexts:
            raise ValueError("controlled missing cases must provide no context")
        if self.review_status == "verified":
            if self.review is None or not all(
                (
                    self.review.label_valid,
                    self.review.context_valid,
                    self.review.natural_difficulty_valid,
                )
            ):
                raise ValueError("verified cases require a passed human review")
        return self
