"""Resume generation until all five subjects have 100 Natural QA v2 drafts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("MLN111", "MLN122", "MLN131", "HCM202", "VNR202")


def main() -> int:
    generator = ROOT / "scripts" / "generate_benchmark_candidates.py"
    plan = ROOT / "benchmark" / "v2" / "portfolio_plan_500.json"
    for subject in SUBJECTS:
        print(f"=== {subject}: resuming to 100 drafts ===", flush=True)
        command = (
            sys.executable,
            str(generator),
            "--subject",
            subject,
            "--children",
            str(ROOT / "data" / "processed" / subject / "structured_v1" / "children.jsonl"),
            "--output",
            str(ROOT / "benchmark" / "v2" / "candidates" / f"{subject}_microbatch_01_raw.jsonl"),
            "--target",
            "100",
            "--batch-size",
            "5",
            "--candidates-per-batch",
            "5",
            "--requests-per-minute",
            "8",
            "--max-requests",
            "20",
            "--max-retries",
            "5",
            "--portfolio-plan",
            str(plan),
            "--dotenv",
            str(ROOT / ".env"),
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(
                f"Stopped safely at {subject}; rerun this command to resume checkpoints.",
                flush=True,
            )
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
