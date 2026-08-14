"""Structured contracts for Evidence Judge decisions."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viettheory.evidence_sufficiency import SufficiencyLabel

NonEmpty = Annotated[str, Field(min_length=1)]


class JudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeDecision(JudgeModel):
    case_id: NonEmpty
    label: SufficiencyLabel
    required_aspects: tuple[NonEmpty, ...]
    covered_aspects: tuple[NonEmpty, ...]
    missing_aspects: tuple[NonEmpty, ...]
    rationale: NonEmpty

    @model_validator(mode="after")
    def validate_coverage(self) -> JudgeDecision:
        if self.label is SufficiencyLabel.SUFFICIENT and self.missing_aspects:
            raise ValueError("sufficient decisions cannot contain missing aspects")
        if self.label is SufficiencyLabel.MISSING and self.covered_aspects:
            raise ValueError("missing decisions cannot contain covered aspects")
        if not self.required_aspects:
            raise ValueError("judge must infer at least one required aspect")
        return self


class JudgeBatch(JudgeModel):
    decisions: tuple[JudgeDecision, ...]
