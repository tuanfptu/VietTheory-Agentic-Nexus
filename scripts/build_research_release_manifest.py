"""Create deterministic checksums for candidate research implementation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = (
    "configs/research_candidate_v0.1.json",
    "src/viettheory/ablation.py",
    "src/viettheory/coordination.py",
    "src/viettheory/evidence_baselines.py",
    "src/viettheory/evidence_judge.py",
    "src/viettheory/evidence_sufficiency.py",
    "src/viettheory/graph.py",
    "src/viettheory/memory.py",
    "src/viettheory/recovery.py",
    "src/viettheory/tools.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"release inputs missing: {missing}")
    payload = {
        "schema_version": "1.0",
        "release": "research-candidate-v0.1",
        "hidden_or_heldout_included": False,
        "files": [{"path": name, "sha256": _sha256(root / name)} for name in sorted(FILES)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
