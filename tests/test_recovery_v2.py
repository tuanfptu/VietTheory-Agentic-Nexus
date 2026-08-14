from __future__ import annotations

import pytest

from viettheory.recovery_v2 import (
    EvidenceGuidedRecoveryRetriever,
    RecoveryPlan,
    reciprocal_rank_fuse,
)
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _item(chunk_id: str, rank: int, score: float = 1.0) -> RetrievedEvidence:
    span = SourceSpan(page_id="p1", pdf_page=0, bbox=(0.0, 0.0, 1.0, 1.0), text="evidence")
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        subject_code="MLN111",
        text="evidence",
        token_count=1,
        source_spans=(span,),
        chunk_kind="child",
        parent_chunk_id="parent",
        chapter="chapter",
    )
    return RetrievedEvidence(
        evidence_id=f"e_{chunk_id}",
        chunk=chunk,
        score=score,
        rank=rank,
        retrieval_method="test",
    )


def test_plan_enforces_bounded_activation() -> None:
    plan = RecoveryPlan(
        request_id="q1",
        activate=True,
        required_aspects=("A", "B"),
        missing_aspects=("B",),
        targeted_queries=("khía cạnh B",),
        rationale="B chưa được hỗ trợ.",
    )
    assert plan.targeted_queries == ("khía cạnh B",)
    with pytest.raises(ValueError):
        RecoveryPlan(
            request_id="q1",
            activate=True,
            required_aspects=("A",),
            missing_aspects=("A",),
            targeted_queries=(),
            rationale="missing",
        )


def test_rrf_rewards_evidence_found_by_multiple_queries() -> None:
    fused = reciprocal_rank_fuse(
        ((_item("a", 1), _item("shared", 2)), (_item("shared", 1), _item("b", 2))),
        top_k=3,
    )
    assert [item.chunk.chunk_id for item in fused] == ["shared", "a", "b"]
    assert [item.rank for item in fused] == [1, 2, 3]


class _Planner:
    def plan(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> RecoveryPlan:
        return RecoveryPlan(
            request_id="runtime_request",
            activate=True,
            required_aspects=("A", "B"),
            missing_aspects=("B",),
            targeted_queries=("missing B",),
            rationale="B is absent",
        )


class _Retriever:
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        ids = ("new",) if query == "missing B" else tuple(f"old_{i}" for i in range(10))
        return tuple(_item(chunk_id, rank) for rank, chunk_id in enumerate(ids[:top_k], 1))


class _Scorer:
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        return [0.1] * (len(pairs) - 1) + [0.9]


class _UnavailablePlanner:
    def plan(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> RecoveryPlan:
        raise RuntimeError("provider unavailable")


def test_recovery_v2_inserts_only_support_gated_new_parent() -> None:
    retriever = EvidenceGuidedRecoveryRetriever(_Retriever(), _Planner(), _Scorer())
    result = retriever.search("question", top_k=5, subject_codes=frozenset({"MLN111"}))
    assert [item.chunk.chunk_id for item in result] == [
        "old_0",
        "old_1",
        "old_2",
        "old_3",
        "new",
    ]


def test_recovery_v2_falls_back_to_b0_when_planner_is_unavailable() -> None:
    retriever = EvidenceGuidedRecoveryRetriever(_Retriever(), _UnavailablePlanner(), _Scorer())
    result = retriever.search("question", top_k=5)
    assert [item.chunk.chunk_id for item in result] == [f"old_{i}" for i in range(5)]
