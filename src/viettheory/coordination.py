"""Purposeful single-controller versus role-separated coordination contracts."""

from __future__ import annotations

from enum import StrEnum
from time import perf_counter
from typing import Protocol

from pydantic import Field, model_validator

from viettheory.schema import NonEmptyText, RetrievedEvidence, VietTheoryModel


class AgentRole(StrEnum):
    CONTROLLER = "controller"
    RETRIEVER = "retriever"
    GRAPH = "graph"
    EVIDENCE_JUDGE = "evidence_judge"
    CITATION_VERIFIER = "citation_verifier"


class CoordinationMode(StrEnum):
    SINGLE_CONTROLLER = "single_controller"
    ROLE_SEPARATED = "role_separated"


class CoordinationStep(VietTheoryModel):
    role: AgentRole
    action: NonEmptyText
    input_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    latency_ms: float = Field(ge=0.0)


class CoordinationOutcome(VietTheoryModel):
    mode: CoordinationMode
    evidence: tuple[RetrievedEvidence, ...]
    steps: tuple[CoordinationStep, ...]
    llm_calls: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_steps(self) -> CoordinationOutcome:
        if not self.steps:
            raise ValueError("coordination outcomes require an execution trace")
        return self


class RetrievalRole(Protocol):
    def retrieve(self, question: str) -> tuple[RetrievedEvidence, ...]: ...


class GraphRole(Protocol):
    def retrieve_graph(self, question: str) -> tuple[RetrievedEvidence, ...]: ...


class EvidenceSelectionRole(Protocol):
    def select(
        self, question: str, candidates: tuple[RetrievedEvidence, ...]
    ) -> tuple[RetrievedEvidence, ...]: ...


class SingleController:
    """Baseline that calls retrieval and selection within one controller boundary."""

    def __init__(self, retrieval: RetrievalRole, selection: EvidenceSelectionRole) -> None:
        self._retrieval = retrieval
        self._selection = selection

    def run(self, question: str) -> CoordinationOutcome:
        started = perf_counter()
        evidence = self._retrieval.retrieve(question)
        selected = self._selection.select(question, evidence)
        elapsed = (perf_counter() - started) * 1000
        return CoordinationOutcome(
            mode=CoordinationMode.SINGLE_CONTROLLER,
            evidence=selected,
            steps=(
                CoordinationStep(
                    role=AgentRole.CONTROLLER,
                    action="retrieve_and_select",
                    output_ids=tuple(item.evidence_id for item in selected),
                    latency_ms=elapsed,
                ),
            ),
            llm_calls=0,
            total_latency_ms=elapsed,
        )


class RoleSeparatedCoordinator:
    """Candidate with non-overlapping retrieval, optional graph, and selection roles."""

    def __init__(
        self,
        retrieval: RetrievalRole,
        selection: EvidenceSelectionRole,
        *,
        graph: GraphRole | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._graph = graph
        self._selection = selection

    def run(self, question: str, *, use_graph: bool = False) -> CoordinationOutcome:
        total_started = perf_counter()
        steps: list[CoordinationStep] = []

        started = perf_counter()
        vector = self._retrieval.retrieve(question)
        steps.append(
            CoordinationStep(
                role=AgentRole.RETRIEVER,
                action="retrieve",
                output_ids=tuple(item.evidence_id for item in vector),
                latency_ms=(perf_counter() - started) * 1000,
            )
        )

        graph_evidence: tuple[RetrievedEvidence, ...] = ()
        if use_graph:
            if self._graph is None:
                raise ValueError("graph role requested but not configured")
            started = perf_counter()
            graph_evidence = self._graph.retrieve_graph(question)
            steps.append(
                CoordinationStep(
                    role=AgentRole.GRAPH,
                    action="graph_retrieve",
                    output_ids=tuple(item.evidence_id for item in graph_evidence),
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )

        candidates = _merge(vector, graph_evidence)
        started = perf_counter()
        selected = self._selection.select(question, candidates)
        steps.append(
            CoordinationStep(
                role=AgentRole.EVIDENCE_JUDGE,
                action="select_evidence",
                input_ids=tuple(item.evidence_id for item in candidates),
                output_ids=tuple(item.evidence_id for item in selected),
                latency_ms=(perf_counter() - started) * 1000,
            )
        )
        return CoordinationOutcome(
            mode=CoordinationMode.ROLE_SEPARATED,
            evidence=selected,
            steps=tuple(steps),
            llm_calls=0,
            total_latency_ms=(perf_counter() - total_started) * 1000,
        )


def _merge(
    first: tuple[RetrievedEvidence, ...], second: tuple[RetrievedEvidence, ...]
) -> tuple[RetrievedEvidence, ...]:
    by_id = {item.evidence_id: item for item in (*first, *second)}
    return tuple(sorted(by_id.values(), key=lambda item: (-item.score, item.evidence_id)))
