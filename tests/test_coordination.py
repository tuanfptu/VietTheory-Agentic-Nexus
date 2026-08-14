from __future__ import annotations

from viettheory.ablation import Transition, compare_coordination
from viettheory.coordination import RoleSeparatedCoordinator, SingleController
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _evidence(evidence_id: str, score: float) -> RetrievedEvidence:
    span = SourceSpan(page_id="page", pdf_page=0, bbox=(0.0, 0.0, 1.0, 1.0), text=evidence_id)
    chunk = Chunk(
        chunk_id=evidence_id,
        document_id="doc",
        subject_code="MLN111",
        text=evidence_id,
        token_count=1,
        source_spans=(span,),
    )
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk=chunk,
        score=score,
        rank=1,
        retrieval_method="test",
    )


class _Retrieval:
    def retrieve(self, question: str) -> tuple[RetrievedEvidence, ...]:
        del question
        return (_evidence("lexical", 0.8),)


class _Graph:
    def retrieve_graph(self, question: str) -> tuple[RetrievedEvidence, ...]:
        del question
        return (_evidence("gold_relation", 0.9),)


class _Selection:
    def select(
        self, question: str, candidates: tuple[RetrievedEvidence, ...]
    ) -> tuple[RetrievedEvidence, ...]:
        del question
        return candidates


def test_role_separated_candidate_has_auditable_non_overlapping_steps() -> None:
    candidate = RoleSeparatedCoordinator(_Retrieval(), _Selection(), graph=_Graph()).run(
        "relationship question", use_graph=True
    )

    assert [step.action for step in candidate.steps] == [
        "retrieve",
        "graph_retrieve",
        "select_evidence",
    ]
    assert {item.evidence_id for item in candidate.evidence} == {"lexical", "gold_relation"}


def test_ablation_reports_per_query_quality_and_cost_delta() -> None:
    baseline = SingleController(_Retrieval(), _Selection()).run("question")
    candidate = RoleSeparatedCoordinator(_Retrieval(), _Selection(), graph=_Graph()).run(
        "question", use_graph=True
    )

    result = compare_coordination("q1", frozenset({"gold_relation"}), baseline, candidate)

    assert result.transition is Transition.WIN
    assert result.baseline_hits == 0
    assert result.candidate_hits == 1
