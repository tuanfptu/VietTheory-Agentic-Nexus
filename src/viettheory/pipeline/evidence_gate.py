"""Configurable evidence sufficiency decisions and dev-set calibration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viettheory.schema import RetrievedEvidence


class GateAction(StrEnum):
    GENERATE = "generate"
    REWRITE = "rewrite"
    REFUSE_INSUFFICIENT = "refuse_insufficient"
    REFUSE_OUT_OF_DOMAIN = "refuse_out_of_domain"


class GateThresholds(BaseModel):
    """Thresholds must be persisted from dev calibration, not hidden constants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficient_score: float = Field(ge=-1.0, le=1.0)
    related_score: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def ordered(self) -> GateThresholds:
        if self.related_score > self.sufficient_score:
            raise ValueError("related_score cannot exceed sufficient_score")
        return self


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: GateAction
    best_score: float | None
    retry_allowed: bool
    reason: str


def decide_evidence(
    evidence: tuple[RetrievedEvidence, ...],
    thresholds: GateThresholds,
    *,
    already_retried: bool = False,
) -> GateDecision:
    """Apply calibrated thresholds while enforcing at most one rewrite."""
    if not evidence:
        return GateDecision(
            action=GateAction.REFUSE_OUT_OF_DOMAIN,
            best_score=None,
            retry_allowed=False,
            reason="no evidence retrieved",
        )
    best_score = max(item.score for item in evidence)
    if best_score >= thresholds.sufficient_score:
        action = GateAction.GENERATE
        reason = "best evidence meets sufficient threshold"
    elif best_score >= thresholds.related_score and not already_retried:
        action = GateAction.REWRITE
        reason = "evidence is related but insufficient; allow one rewrite"
    elif best_score >= thresholds.related_score:
        action = GateAction.REFUSE_INSUFFICIENT
        reason = "evidence remained insufficient after rewrite"
    else:
        action = GateAction.REFUSE_OUT_OF_DOMAIN
        reason = "best evidence is below related threshold"
    return GateDecision(
        action=action,
        best_score=best_score,
        retry_allowed=action is GateAction.REWRITE,
        reason=reason,
    )


def calibrate_sufficient_threshold(samples: tuple[tuple[float, bool], ...]) -> float:
    """Choose the dev threshold maximizing balanced accuracy, deterministically."""
    positives = sum(label for _, label in samples)
    negatives = len(samples) - positives
    if not samples or positives == 0 or negatives == 0:
        raise ValueError("calibration requires positive and negative dev examples")
    candidates = sorted({score for score, _ in samples})
    best_threshold = candidates[0]
    best_metric = -1.0
    for threshold in candidates:
        true_positive = sum(score >= threshold and label for score, label in samples)
        true_negative = sum(score < threshold and not label for score, label in samples)
        balanced_accuracy = 0.5 * (true_positive / positives + true_negative / negatives)
        if balanced_accuracy > best_metric or (
            balanced_accuracy == best_metric and threshold > best_threshold
        ):
            best_metric = balanced_accuracy
            best_threshold = threshold
    return best_threshold
