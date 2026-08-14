"""Write deterministic J1 development checksums after prompt/model freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "benchmark/evidence_sufficiency/pilot_v0.1/release/development_32.jsonl",
        "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_development.json",
        "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_error_analysis.md",
        "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_freeze.json",
        "scripts/run_evidence_judge.py",
        "src/viettheory/evidence_judge.py",
    )
    output = root / "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_sha256.json"
    payload = {
        "schema_version": "1.0",
        "hidden_or_heldout_included": False,
        "files": [{"path": path, "sha256": _digest(root / path)} for path in sorted(paths)],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
