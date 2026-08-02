"""Validate a benchmark JSONL against its current retrieval artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from viettheory.benchmark import ArtifactManifest, BenchmarkQuestion
from viettheory.benchmark_validation import validate_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    questions = tuple(
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    artifact = ArtifactManifest.model_validate_json(
        args.artifact_manifest.read_text(encoding="utf-8")
    )
    report = validate_benchmark(
        questions,
        artifact,
        pages_path=args.pages,
        structured_dir=args.structured_dir,
    )
    rendered = report.model_dump_json(indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
