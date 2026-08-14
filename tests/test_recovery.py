from __future__ import annotations

from viettheory.evidence_judge import JudgeDecision
from viettheory.evidence_sufficiency import SufficiencyLabel
from viettheory.recovery import BoundedEvidenceRecovery
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _evidence(evidence_id: str, text: str, score: float) -> RetrievedEvidence:
    span = SourceSpan(page_id="page_1", pdf_page=1, bbox=(0.0, 0.0, 1.0, 1.0), text=text)
    chunk = Chunk(
        chunk_id=evidence_id,
        document_id="doc",
        subject_code="MLN111",
        text=text,
        token_count=4,
        source_spans=(span,),
    )
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk=chunk,
        score=score,
        rank=1,
        retrieval_method="test",
    )


class _Judge:
    def judge(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> JudgeDecision:
        del question
        relationship = any("quyết định" in item.chunk.text for item in evidence)
        label = SufficiencyLabel.SUFFICIENT if relationship else SufficiencyLabel.PARTIAL
        return JudgeDecision(
            case_id="runtime",
            label=label,
            required_aspects=("definition", "relationship"),
            covered_aspects=("definition", "relationship") if relationship else ("definition",),
            missing_aspects=() if relationship else ("relationship",),
            rationale="Deterministic test judge.",
        )


class _Writer:
    def write(
        self,
        question: str,
        missing_aspects: tuple[str, ...],
        *,
        previous_queries: tuple[str, ...],
    ) -> tuple[str, ...]:
        del question, previous_queries
        return tuple(f"target {aspect}" for aspect in missing_aspects)


class _Retriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        del query, top_k, subject_codes
        self.calls += 1
        return (_evidence("relationship", "Vật chất quyết định ý thức.", 0.9),)


def test_recovery_targets_missing_aspect_and_stops_when_sufficient() -> None:
    retriever = _Retriever()
    recovery = BoundedEvidenceRecovery(retriever, _Judge(), _Writer())

    outcome = recovery.run(
        "Mối quan hệ là gì?", (_evidence("definition", "Định nghĩa vật chất.", 0.8),)
    )

    assert outcome.activated and outcome.recovered and not outcome.exhausted
    assert len(outcome.rounds) == 1
    assert retriever.calls == 1
    assert outcome.rounds[0].queries == ("target relationship",)


def test_recovery_does_not_activate_for_sufficient_or_contradicted() -> None:
    retriever = _Retriever()
    recovery = BoundedEvidenceRecovery(retriever, _Judge(), _Writer())

    outcome = recovery.run(
        "Mối quan hệ là gì?", (_evidence("relationship", "Vật chất quyết định ý thức.", 0.9),)
    )

    assert not outcome.activated
    assert retriever.calls == 0
