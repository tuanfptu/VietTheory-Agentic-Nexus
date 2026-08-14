"""Frozen deterministic baseline for typed tool selection."""

from __future__ import annotations

from viettheory.tools import ToolName


def route_tool(question: str, *, subject_code: str | None) -> ToolName:
    normalized = " ".join(question.casefold().split())
    if any(term in normalized for term in ("trang bao nhiêu", "trang nào", "nguồn pdf")):
        return ToolName.LOOKUP_SOURCE_PAGE
    if any(term in normalized for term in ("trích dẫn", "dẫn nguồn", "citation")):
        return ToolName.INSPECT_CITATION
    if any(
        term in normalized
        for term in ("mối quan hệ", "quan hệ giữa", "ảnh hưởng", "dẫn đến", "multi-hop")
    ):
        return ToolName.GRAPH_SEARCH
    if subject_code is None or any(
        term in normalized for term in ("liên môn", "cả năm môn", "giữa các môn")
    ):
        return ToolName.RETRIEVE_CROSS_SUBJECT
    return ToolName.RETRIEVE_SUBJECT
