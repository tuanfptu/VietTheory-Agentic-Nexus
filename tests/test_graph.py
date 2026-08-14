from __future__ import annotations

import pytest

from viettheory.graph import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ProvenanceGraph,
)
from viettheory.schema import SourceSpan


def _span(page: int, text: str) -> SourceSpan:
    return SourceSpan(
        page_id=f"page_{page}",
        pdf_page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        text=text,
    )


def test_graph_rejects_unprovenanced_node() -> None:
    with pytest.raises(ValueError):
        GraphNode(
            node_id="n1",
            label="Vật chất",
            node_type=GraphNodeType.CONCEPT,
            subject_codes=frozenset({"MLN111"}),
            provenance=(),
        )


def test_graph_path_preserves_edge_and_node_provenance() -> None:
    matter = GraphNode(
        node_id="matter",
        label="vật chất",
        node_type=GraphNodeType.CONCEPT,
        subject_codes=frozenset({"MLN111"}),
        provenance=(_span(74, "Vật chất là thực tại khách quan."),),
    )
    consciousness = GraphNode(
        node_id="consciousness",
        label="ý thức",
        node_type=GraphNodeType.CONCEPT,
        subject_codes=frozenset({"MLN111"}),
        provenance=(_span(89, "Ý thức là hình ảnh chủ quan."),),
    )
    edge = GraphEdge(
        edge_id="edge_1",
        source_node_id="matter",
        target_node_id="consciousness",
        relation=GraphEdgeType.INFLUENCES,
        claim="Vật chất quyết định ý thức.",
        subject_codes=frozenset({"MLN111"}),
        provenance=(_span(100, "Vật chất quyết định ý thức."),),
    )
    graph = ProvenanceGraph((matter, consciousness), (edge,))

    paths = graph.search("vật chất ý thức", subject_codes=frozenset({"MLN111"}))

    assert paths[0].edge_ids == ("edge_1",)
    assert {span.pdf_page for span in paths[0].provenance} == {74, 89, 100}


def test_within_subject_graph_search_filters_foreign_edges() -> None:
    first = GraphNode(
        node_id="first",
        label="cách mạng",
        node_type=GraphNodeType.CONCEPT,
        subject_codes=frozenset({"MLN111"}),
        provenance=(_span(1, "Cách mạng."),),
    )
    second = GraphNode(
        node_id="second",
        label="lịch sử",
        node_type=GraphNodeType.CONCEPT,
        subject_codes=frozenset({"VNR202"}),
        provenance=(_span(2, "Lịch sử."),),
    )
    edge = GraphEdge(
        edge_id="foreign",
        source_node_id="first",
        target_node_id="second",
        relation=GraphEdgeType.ASSOCIATED_WITH,
        claim="A cross-subject association.",
        subject_codes=frozenset({"VNR202"}),
        provenance=(_span(3, "Association."),),
    )

    graph = ProvenanceGraph((first, second), (edge,))

    assert graph.search("cách mạng", subject_codes=frozenset({"MLN111"})) == ()
