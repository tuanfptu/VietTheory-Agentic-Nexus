from __future__ import annotations

import pytest
from pydantic import ValidationError

from viettheory.evidence_judge import JudgeDecision
from viettheory.evidence_sufficiency import SufficiencyLabel


def test_sufficient_decision_cannot_have_missing_aspects() -> None:
    with pytest.raises(ValidationError):
        JudgeDecision(
            case_id="case_1",
            label=SufficiencyLabel.SUFFICIENT,
            required_aspects=("definition",),
            covered_aspects=(),
            missing_aspects=("definition",),
            rationale="The required definition is absent.",
        )


def test_missing_decision_cannot_have_covered_aspects() -> None:
    with pytest.raises(ValidationError):
        JudgeDecision(
            case_id="case_2",
            label=SufficiencyLabel.MISSING,
            required_aspects=("relationship",),
            covered_aspects=("relationship",),
            missing_aspects=(),
            rationale="The relationship is present.",
        )


def test_partial_decision_tracks_covered_and_missing_aspects() -> None:
    decision = JudgeDecision(
        case_id="case_3",
        label=SufficiencyLabel.PARTIAL,
        required_aspects=("concept_a", "concept_b"),
        covered_aspects=("concept_a",),
        missing_aspects=("concept_b",),
        rationale="Only concept A is supported.",
    )

    assert decision.covered_aspects == ("concept_a",)
    assert decision.missing_aspects == ("concept_b",)
