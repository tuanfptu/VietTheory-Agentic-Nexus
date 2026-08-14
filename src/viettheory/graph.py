"""Selective graph retrieval with mandatory source provenance."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from pydantic import model_validator

from viettheory.schema import NonEmptyText, SourceSpan, VietTheoryModel


class GraphNodeType(StrEnum):
    CONCEPT = "concept"
    PERSON = "person"
    WORK = "work"
    EVENT = "event"
    ORGANIZATION = "organization"
    HISTORICAL_PERIOD = "historical_period"
    THEORY = "theory"


class GraphEdgeType(StrEnum):
    DEFINES = "defines"
    INFLUENCES = "influences"
    CONTRADICTS = "contradicts"
    DEVELOPS_FROM = "develops_from"
    CAUSES = "causes"
    IS_CONDITION_OF = "is_condition_of"
    APPLIED_IN = "applied_in"
    ASSOCIATED_WITH = "associated_with"


class GraphNode(VietTheoryModel):
    node_id: NonEmptyText
    label: NonEmptyText
    node_type: GraphNodeType
    subject_codes: frozenset[str]
    provenance: tuple[SourceSpan, ...]

    @model_validator(mode="after")
    def require_source(self) -> GraphNode:
        if not self.subject_codes or not self.provenance:
            raise ValueError("graph nodes require subject and source provenance")
        return self


class GraphEdge(VietTheoryModel):
    edge_id: NonEmptyText
    source_node_id: NonEmptyText
    target_node_id: NonEmptyText
    relation: GraphEdgeType
    claim: NonEmptyText
    subject_codes: frozenset[str]
    provenance: tuple[SourceSpan, ...]

    @model_validator(mode="after")
    def require_source(self) -> GraphEdge:
        if self.source_node_id == self.target_node_id:
            raise ValueError("self edges are not supported")
        if not self.subject_codes or not self.provenance:
            raise ValueError("graph edges require subject and source provenance")
        return self


class GraphPath(VietTheoryModel):
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    score: float
    provenance: tuple[SourceSpan, ...]

    @model_validator(mode="after")
    def validate_path(self) -> GraphPath:
        if not self.node_ids or len(self.node_ids) != len(self.edge_ids) + 1:
            raise ValueError("a path must contain exactly one more node than edge")
        if not self.provenance:
            raise ValueError("graph paths require provenance")
        return self


class ProvenanceGraph:
    """Small deterministic graph index; construction remains an offline step."""

    def __init__(self, nodes: tuple[GraphNode, ...], edges: tuple[GraphEdge, ...]) -> None:
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError("duplicate graph node IDs")
        self._edges = {edge.edge_id: edge for edge in edges}
        if len(self._edges) != len(edges):
            raise ValueError("duplicate graph edge IDs")
        self._adjacency: dict[str, list[GraphEdge]] = {node_id: [] for node_id in self._nodes}
        for edge in edges:
            if edge.source_node_id not in self._nodes or edge.target_node_id not in self._nodes:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            self._adjacency[edge.source_node_id].append(edge)
        for adjacency in self._adjacency.values():
            adjacency.sort(key=lambda edge: edge.edge_id)

    def search(
        self,
        query: str,
        *,
        subject_codes: frozenset[str] | None = None,
        max_hops: int = 2,
        top_k: int = 5,
    ) -> tuple[GraphPath, ...]:
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be between 1 and 3")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = frozenset(query.casefold().split())
        starts = sorted(
            (
                node
                for node in self._nodes.values()
                if (subject_codes is None or node.subject_codes & subject_codes)
                and terms.intersection(node.label.casefold().split())
            ),
            key=lambda node: node.node_id,
        )
        candidates: list[GraphPath] = []
        for start in starts:
            queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], tuple[SourceSpan, ...]]] = (
                deque([(start.node_id, (start.node_id,), (), start.provenance)])
            )
            while queue:
                current, node_ids, edge_ids, provenance = queue.popleft()
                if edge_ids:
                    coverage = len(
                        terms.intersection(
                            " ".join(
                                self._nodes[node_id].label.casefold() for node_id in node_ids
                            ).split()
                        )
                    )
                    candidates.append(
                        GraphPath(
                            node_ids=node_ids,
                            edge_ids=edge_ids,
                            score=float(coverage) / len(node_ids),
                            provenance=_deduplicate_spans(provenance),
                        )
                    )
                if len(edge_ids) >= max_hops:
                    continue
                for edge in self._adjacency[current]:
                    if edge.target_node_id in node_ids:
                        continue
                    if subject_codes is not None and not edge.subject_codes & subject_codes:
                        continue
                    target = self._nodes[edge.target_node_id]
                    queue.append(
                        (
                            target.node_id,
                            (*node_ids, target.node_id),
                            (*edge_ids, edge.edge_id),
                            (*provenance, *edge.provenance, *target.provenance),
                        )
                    )
        candidates.sort(key=lambda path: (-path.score, path.edge_ids, path.node_ids))
        return tuple(candidates[:top_k])


def _deduplicate_spans(spans: tuple[SourceSpan, ...]) -> tuple[SourceSpan, ...]:
    seen: set[tuple[str, tuple[float, float, float, float], str]] = set()
    result: list[SourceSpan] = []
    for span in spans:
        key = (span.page_id, span.bbox, span.text)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return tuple(result)
