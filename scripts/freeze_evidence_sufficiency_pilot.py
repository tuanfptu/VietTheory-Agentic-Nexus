"""Apply the human audit and freeze Evidence Sufficiency pilot v0.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from viettheory.evidence_sufficiency import EvidenceSufficiencyCase, SufficiencyReview

FINAL_CASE_ID = "es_mln131_0035_wrongaspect"
HELD_OUT_SOURCE_IDS = frozenset({"hcm202_0057", "mln111_0049", "mln122_0044", "mln131_0028"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth(value: str) -> bool:
    return value.strip().casefold() == "true"


def _write(path: Path, cases: list[EvidenceSufficiencyCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--held-out-dir",
        type=Path,
        default=Path("benchmark_private/evidence_sufficiency/pilot_v0.1"),
    )
    parser.add_argument("--owner-reviewer-id", default="owner_tuan_2026-08-14")
    args = parser.parse_args()

    cases = {
        case.case_id: case
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for case in (EvidenceSufficiencyCase.model_validate_json(line),)
    }
    with args.review.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = {row["case_id"]: row for row in csv.DictReader(handle)}
    if cases.keys() != reviewed.keys():
        raise ValueError("review IDs do not exactly match revised pilot cases")
    if reviewed[FINAL_CASE_ID]["decision"].strip().casefold() != "revise":
        raise ValueError("expected the adjudicated case to be marked revise in first review")

    frozen: list[EvidenceSufficiencyCase] = []
    for case_id, case in sorted(cases.items()):
        row = reviewed[case_id]
        if case_id == FINAL_CASE_ID:
            review = SufficiencyReview(
                reviewer_id=args.owner_reviewer_id,
                label_valid=True,
                context_valid=True,
                natural_difficulty_valid=True,
                notes=(
                    "Owner approved the adjudicated replacement parent in the Codex task on "
                    "2026-08-14."
                ),
            )
        else:
            if row["decision"].strip().casefold() != "approve":
                raise ValueError(f"non-final case is not approved: {case_id}")
            review = SufficiencyReview(
                reviewer_id=args.owner_reviewer_id,
                label_valid=_truth(row["label_valid"]),
                context_valid=_truth(row["context_valid"]),
                natural_difficulty_valid=_truth(row["difficulty_valid"]),
                notes=row["reviewer_notes"].strip() or None,
            )
        split = "held_out_test" if case.source_question_id in HELD_OUT_SOURCE_IDS else "development"
        frozen.append(
            case.model_copy(update={"split": split, "review_status": "verified", "review": review})
        )

    development = [case for case in frozen if case.split == "development"]
    held_out = [case for case in frozen if case.split == "held_out_test"]
    if len(development) != 32 or len(held_out) != 16:
        raise ValueError("pilot split must contain 32 development and 16 held-out cases")
    if {case.source_question_id for case in development} & {
        case.source_question_id for case in held_out
    }:
        raise ValueError("source-question leakage across pilot splits")

    development_path = args.output_dir / "development_32.jsonl"
    held_out_path = args.held_out_dir / "held_out_16.jsonl"
    _write(development_path, development)
    _write(held_out_path, held_out)
    manifest = {
        "benchmark_version": "mln_evidence_sufficiency_pilot_v0.1",
        "schema_version": "0.1",
        "status": "verified",
        "reviewed_cases": 48,
        "development": {
            "count": len(development),
            "source_questions": len({case.source_question_id for case in development}),
            "labels": dict(Counter(case.expected_label.value for case in development)),
            "sha256": _sha256(development_path),
        },
        "held_out": {
            "count": len(held_out),
            "location": "private",
            "source_questions": len({case.source_question_id for case in held_out}),
            "labels": dict(Counter(case.expected_label.value for case in held_out)),
            "sha256": _sha256(held_out_path),
        },
        "split_policy": "grouped by source_question_id; four perturbations remain together",
        "adjudicated_case": FINAL_CASE_ID,
        "source_review_sha256": _sha256(args.review),
        "source_cases_sha256": _sha256(args.cases),
        "contradiction_status": "exploratory; no synthetic contradiction cases",
    }
    manifest_path = args.output_dir / "manifest.json"
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
    private_checksum_path = args.held_out_dir / "SHA256SUMS"
    private_checksum_path.write_text(
        f"{_sha256(held_out_path)}  {held_out_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
