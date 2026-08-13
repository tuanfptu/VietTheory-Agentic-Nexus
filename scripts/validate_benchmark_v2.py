"""Validate Natural QA v2 JSONL against its portfolio plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viettheory.natural_benchmark import (
    NaturalQuestionV2,
    PortfolioPlan,
    validate_natural_portfolio,
)


def load_records(path: Path) -> tuple[NaturalQuestionV2, ...]:
    return tuple(
        NaturalQuestionV2.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("benchmark/v2/portfolio_plan.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    plan = PortfolioPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    records = load_records(args.questions)
    report = validate_natural_portfolio(records, plan)
    payload = report.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
