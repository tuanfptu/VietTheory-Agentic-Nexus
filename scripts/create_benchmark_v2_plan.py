"""Write the deterministic Natural QA v2 pilot plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from viettheory.natural_benchmark import default_portfolio_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/v2/portfolio_plan.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = default_portfolio_plan().model_dump_json(indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
