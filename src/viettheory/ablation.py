"""Per-query comparison metrics for coordination candidates."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from viettheory.coordination import CoordinationOutcome
from viettheory.schema import NonEmptyText, VietTheoryModel


class Transition(StrEnum):
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"


class QueryAblation(VietTheoryModel):
    query_id: NonEmptyText
    baseline_hits: int = Field(ge=0)
    candidate_hits: int = Field(ge=0)
    transition: Transition
    baseline_latency_ms: float = Field(ge=0.0)
    candidate_latency_ms: float = Field(ge=0.0)
    baseline_llm_calls: int = Field(ge=0)
    candidate_llm_calls: int = Field(ge=0)


def compare_coordination(
    query_id: str,
    gold_evidence_ids: frozenset[str],
    baseline: CoordinationOutcome,
    candidate: CoordinationOutcome,
) -> QueryAblation:
    baseline_hits = len(gold_evidence_ids & {item.evidence_id for item in baseline.evidence})
    candidate_hits = len(gold_evidence_ids & {item.evidence_id for item in candidate.evidence})
    transition = (
        Transition.WIN
        if candidate_hits > baseline_hits
        else Transition.LOSS
        if candidate_hits < baseline_hits
        else Transition.TIE
    )
    return QueryAblation(
        query_id=query_id,
        baseline_hits=baseline_hits,
        candidate_hits=candidate_hits,
        transition=transition,
        baseline_latency_ms=baseline.total_latency_ms,
        candidate_latency_ms=candidate.total_latency_ms,
        baseline_llm_calls=baseline.llm_calls,
        candidate_llm_calls=candidate.llm_calls,
    )
