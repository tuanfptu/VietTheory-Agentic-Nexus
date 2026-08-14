"""Verify every public and private checksum in the final release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = (*manifest["public_files"], *manifest["private_artifact_checksums"])
    failures: list[str] = []
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"missing:{record['path']}")
        elif _digest(path) != record["sha256"]:
            failures.append(f"checksum:{record['path']}")
    if failures:
        raise RuntimeError(f"release verification failed: {failures}")
    print(json.dumps({"verified_files": len(records), "release": manifest["release"]}))


if __name__ == "__main__":
    main()
