from __future__ import annotations

from viettheory.tools import (
    BoundedToolController,
    ToolCall,
    ToolErrorCode,
    ToolName,
)


def _call(trace_id: str) -> ToolCall:
    return ToolCall(
        trace_id=trace_id,
        tool=ToolName.RETRIEVE_SUBJECT,
        arguments={"query": "vật chất", "subject": "MLN111"},
    )


def test_controller_returns_typed_results_and_enforces_budget() -> None:
    controller = BoundedToolController(
        {ToolName.RETRIEVE_SUBJECT: lambda args: {"subject": args["subject"]}},
        max_calls=1,
    )

    first = controller.execute(_call("trace_1"))
    second = controller.execute(_call("trace_2"))

    assert first.ok and first.output == {"subject": "MLN111"}
    assert not second.ok and second.error_code is ToolErrorCode.BUDGET_EXHAUSTED


def test_controller_maps_validation_errors_without_leaking_exceptions() -> None:
    def invalid_handler(arguments: dict[str, object]) -> dict[str, object]:
        raise ValueError(f"missing field in {sorted(arguments)}")

    controller = BoundedToolController({ToolName.RETRIEVE_SUBJECT: invalid_handler})
    result = controller.execute(_call("trace_3"))

    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.trace_id == "trace_3"
