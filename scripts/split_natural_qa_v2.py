"""Create a deterministic leakage-safe 350/150 Natural QA v2 release split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from viettheory.benchmark import BenchmarkSplit
from viettheory.benchmark_split import assert_no_parent_leakage, split_subject
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.subjects import SUBJECT_BY_CODE


def _read(path: Path) -> tuple[NaturalQuestionV2, ...]:
    return tuple(
        NaturalQuestionV2.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write(path: Path, records: list[NaturalQuestionV2]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(records: list[NaturalQuestionV2]) -> dict[str, dict[str, int]]:
    fields = {
        "subject": Counter(record.subject_code for record in records),
        "category": Counter(record.primary_category.value for record in records),
        "difficulty": Counter(record.difficulty.value for record in records),
        "answerability": Counter(record.answerability.value for record in records),
    }
    return {name: dict(sorted(values.items())) for name, values in fields.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--hidden-dir", type=Path, default=Path("benchmark_private/v2"))
    parser.add_argument("--seed", default="natural_qa_v2_gold_v1.0_split_v1")
    args = parser.parse_args()

    records = _read(args.source)
    if len(records) != 500 or Counter(record.subject_code for record in records) != Counter(
        {code: 100 for code in SUBJECT_BY_CODE}
    ):
        raise ValueError("Gold input must contain exactly 100 records for each of five subjects")

    development: list[NaturalQuestionV2] = []
    hidden: list[NaturalQuestionV2] = []
    component_report: dict[str, list[int]] = {}
    for subject_code in sorted(SUBJECT_BY_CODE):
        subject_records = [record for record in records if record.subject_code == subject_code]
        result = split_subject(subject_records, hidden_size=30, seed=f"{args.seed}:{subject_code}")
        assert_no_parent_leakage(result)
        development.extend(
            record.model_copy(update={"split": BenchmarkSplit.DEVELOPMENT})
            for record in result.development
        )
        hidden.extend(
            record.model_copy(update={"split": BenchmarkSplit.HELD_OUT_TEST})
            for record in result.hidden
        )
        component_report[subject_code] = list(result.component_sizes)

    development.sort(key=lambda record: record.id)
    hidden.sort(key=lambda record: record.id)
    development_path = args.output_dir / "natural_qa_v2_development_350.jsonl"
    hidden_path = args.hidden_dir / "natural_qa_v2_hidden_150.jsonl"
    _write(development_path, development)
    _write(hidden_path, hidden)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark_version": "natural_qa_v2_gold_v1.0",
        "split_version": "split_v1",
        "seed": args.seed,
        "policy": "shared gold-parent connected components; deterministic stratified beam",
        "source": args.source.name,
        "source_sha256": _sha256(args.source),
        "development": {
            "file": development_path.name,
            "count": len(development),
            "sha256": _sha256(development_path),
            "distribution": _counts(development),
        },
        "hidden": {
            "count": len(hidden),
            "sha256": _sha256(hidden_path),
            "location": "private",
        },
        "semantic_component_sizes": component_report,
        "leakage_checks": {"question_ids": "pass", "gold_parent_ids": "pass"},
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path = args.output_dir / "SHA256SUMS"
    checksum_path.write_text(
        f"{_sha256(development_path)}  {development_path.name}\n"
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    private_checksum_path = args.hidden_dir / "SHA256SUMS"
    private_checksum_path.write_text(
        f"{_sha256(hidden_path)}  {hidden_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
