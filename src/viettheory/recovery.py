"""Bounded, missing-aspect targeted evidence recovery."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field, model_validator

from viettheory.evidence_judge import JudgeDecision
from viettheory.evidence_sufficiency import SufficiencyLabel
from viettheory.schema import NonEmptyText, RetrievedEvidence, VietTheoryModel


class EvidenceJudge(Protocol):
    def judge(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> JudgeDecision: ...


class RecoveryQueryWriter(Protocol):
    def write(
        self,
        question: str,
        missing_aspects: tuple[str, ...],
        *,
        previous_queries: tuple[str, ...],
    ) -> tuple[str, ...]: ...


class RecoveryRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


class RecoveryRound(VietTheoryModel):
    round_number: int = Field(gt=0)
    queries: tuple[NonEmptyText, ...]
    new_evidence_ids: tuple[str, ...]
    decision: JudgeDecision


class RecoveryOutcome(VietTheoryModel):
    activated: bool
    recovered: bool
    exhausted: bool
    initial_decision: JudgeDecision
    final_decision: JudgeDecision
    evidence: tuple[RetrievedEvidence, ...]
    rounds: tuple[RecoveryRound, ...]

    @model_validator(mode="after")
    def validate_flags(self) -> RecoveryOutcome:
        sufficient = self.final_decision.label is SufficiencyLabel.SUFFICIENT
        if self.recovered != (self.activated and sufficient):
            raise ValueError("recovered must mean activated and finally sufficient")
        if self.exhausted and sufficient:
            raise ValueError("a sufficient outcome cannot be exhausted")
        return self


class BoundedEvidenceRecovery:
    """Retry only partial/missing evidence; never loop on contradictions."""

    def __init__(
        self,
        retriever: RecoveryRetriever,
        judge: EvidenceJudge,
        writer: RecoveryQueryWriter,
        *,
        max_rounds: int = 2,
        max_queries_per_round: int = 2,
        top_k: int = 5,
    ) -> None:
        if not 1 <= max_rounds <= 2:
            raise ValueError("max_rounds must be 1 or 2")
        if not 1 <= max_queries_per_round <= 2:
            raise ValueError("max_queries_per_round must be 1 or 2")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._retriever = retriever
        self._judge = judge
        self._writer = writer
        self._max_rounds = max_rounds
        self._max_queries = max_queries_per_round
        self._top_k = top_k

    def run(
        self,
        question: str,
        initial_evidence: tuple[RetrievedEvidence, ...],
        *,
        subject_codes: frozenset[str] | None = None,
    ) -> RecoveryOutcome:
        initial = self._judge.judge(question, initial_evidence)
        if initial.label not in {SufficiencyLabel.PARTIAL, SufficiencyLabel.MISSING}:
            return RecoveryOutcome(
                activated=False,
                recovered=False,
                exhausted=False,
                initial_decision=initial,
                final_decision=initial,
                evidence=initial_evidence,
                rounds=(),
            )

        evidence = initial_evidence
        decision = initial
        rounds: list[RecoveryRound] = []
        previous_queries: tuple[str, ...] = ()
        for round_number in range(1, self._max_rounds + 1):
            queries = self._writer.write(
                question,
                decision.missing_aspects,
                previous_queries=previous_queries,
            )[: self._max_queries]
            if not queries:
                break
            before = {item.evidence_id for item in evidence}
            additions = tuple(
                item
                for query in queries
                for item in self._retriever.search(
                    query, top_k=self._top_k, subject_codes=subject_codes
                )
            )
            evidence = _merge_evidence(evidence, additions)
            decision = self._judge.judge(question, evidence)
            rounds.append(
                RecoveryRound(
                    round_number=round_number,
                    queries=queries,
                    new_evidence_ids=tuple(
                        item.evidence_id for item in evidence if item.evidence_id not in before
                    ),
                    decision=decision,
                )
            )
            previous_queries = (*previous_queries, *queries)
            if decision.label is SufficiencyLabel.SUFFICIENT:
                break
            if decision.label is SufficiencyLabel.CONTRADICTED:
                break

        recovered = decision.label is SufficiencyLabel.SUFFICIENT
        return RecoveryOutcome(
            activated=True,
            recovered=recovered,
            exhausted=not recovered,
            initial_decision=initial,
            final_decision=decision,
            evidence=evidence,
            rounds=tuple(rounds),
        )


def _merge_evidence(
    original: tuple[RetrievedEvidence, ...], additions: tuple[RetrievedEvidence, ...]
) -> tuple[RetrievedEvidence, ...]:
    by_id = {item.evidence_id: item for item in original}
    for item in additions:
        existing = by_id.get(item.evidence_id)
        if existing is None or item.score > existing.score:
            by_id[item.evidence_id] = item
    ordered = sorted(by_id.values(), key=lambda item: (-item.score, item.evidence_id))
    return tuple(item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered, 1))
