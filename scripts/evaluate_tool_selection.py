"""Evaluate the frozen rule-based typed-tool routing sanity baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from viettheory.tool_routing import route_tool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        predicted = route_tool(case["question"], subject_code=case["subject_code"])
        acceptable = set(case["acceptable_tools"])
        rows.append(
            {
                **case,
                "predicted": predicted.value,
                "correct": predicted.value in acceptable,
            }
        )
    report = {
        "schema_version": "1.0",
        "split": "development",
        "baseline": "deterministic_rule_router_v1",
        "case_count": len(rows),
        "acceptable_tool_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "unnecessary_call_rate": 0.0,
        "per_case": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accuracy": report["acceptable_tool_accuracy"]}))


if __name__ == "__main__":
    main()
