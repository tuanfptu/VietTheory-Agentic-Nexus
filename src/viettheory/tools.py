"""Typed tool controller with deterministic budgets, errors, and traces."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from time import perf_counter
from typing import Any

from pydantic import Field

from viettheory.schema import NonEmptyText, VietTheoryModel


class ToolName(StrEnum):
    RETRIEVE_SUBJECT = "retrieve_subject"
    RETRIEVE_CROSS_SUBJECT = "retrieve_cross_subject"
    GRAPH_SEARCH = "graph_search"
    LOOKUP_SOURCE_PAGE = "lookup_source_page"
    INSPECT_CITATION = "inspect_citation"


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_INPUT = "invalid_input"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXECUTION_FAILED = "execution_failed"


class ToolCall(VietTheoryModel):
    trace_id: NonEmptyText
    tool: ToolName
    arguments: dict[str, Any]


class ToolResult(VietTheoryModel):
    trace_id: NonEmptyText
    tool: ToolName
    ok: bool
    output: dict[str, Any] | None = None
    error_code: ToolErrorCode | None = None
    error_message: str | None = None
    latency_ms: float = Field(ge=0.0)


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class BoundedToolController:
    def __init__(self, handlers: dict[ToolName, ToolHandler], *, max_calls: int = 4) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self._handlers = dict(handlers)
        self._max_calls = max_calls
        self._used_calls = 0

    @property
    def remaining_calls(self) -> int:
        return self._max_calls - self._used_calls

    def execute(self, call: ToolCall) -> ToolResult:
        started = perf_counter()
        if self._used_calls >= self._max_calls:
            return _error(
                call, ToolErrorCode.BUDGET_EXHAUSTED, "tool-call budget exhausted", started
            )
        self._used_calls += 1
        handler = self._handlers.get(call.tool)
        if handler is None:
            return _error(call, ToolErrorCode.UNKNOWN_TOOL, "no handler registered", started)
        try:
            output = handler(call.arguments)
        except (TypeError, ValueError, KeyError) as exc:
            return _error(call, ToolErrorCode.INVALID_INPUT, str(exc), started)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return _error(call, ToolErrorCode.EXECUTION_FAILED, type(exc).__name__, started)
        return ToolResult(
            trace_id=call.trace_id,
            tool=call.tool,
            ok=True,
            output=output,
            latency_ms=(perf_counter() - started) * 1000,
        )


def _error(call: ToolCall, code: ToolErrorCode, message: str, started: float) -> ToolResult:
    return ToolResult(
        trace_id=call.trace_id,
        tool=call.tool,
        ok=False,
        error_code=code,
        error_message=message,
        latency_ms=(perf_counter() - started) * 1000,
    )
