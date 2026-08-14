"""Build the public final-v1 manifest without exposing private benchmark content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PUBLIC_FILES = (
    "benchmark/evidence_sufficiency/pilot_v0.1/release/SHA256SUMS",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/development_32.jsonl",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_development.json",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_freeze.json",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_held_out_aggregate.json",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/j1_sha256.json",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/shortcut_baselines.json",
    "benchmark/evidence_sufficiency/pilot_v0.1/release/targeted_recovery_b0_development.json",
    "benchmark/memory/safety_contract_v1.json",
    "benchmark/tool_selection/development.jsonl",
    "benchmark/tool_selection/rule_router_development.json",
    "benchmark/v2/releases/v1.0/splits/natural_qa_v2_development_350.jsonl",
    "benchmark/v2/releases/v1.0/splits/split_manifest.json",
    "benchmark/v2/releases/v1.0/portfolio_profile_500_gold.json",
    "benchmark/v2/reports/b0_development_350_answerable.json",
    "benchmark/v2/reports/b0_hidden_150_aggregate.json",
    "benchmark/v2/reports/coordination_ablation_development.json",
    "benchmark/v2/reports/parent_graph_development.json",
    "benchmark/v2/reports/natural_qa_v2_gold_500_validation.json",
    "configs/final_candidate_v1.json",
    "docs/experimental-results-v1.md",
    "reports/five_subject_readiness_final.json",
)

PRIVATE_FILES = (
    "benchmark_private/v2/natural_qa_v2_hidden_150.jsonl",
    "benchmark_private/v2/b0_hidden_150_frozen.json",
    "benchmark_private/evidence_sufficiency/pilot_v0.1/held_out_16.jsonl",
    "benchmark_private/evidence_sufficiency/pilot_v0.1/j1_held_out_checkpoint.jsonl",
    "benchmark_private/evidence_sufficiency/pilot_v0.1/j1_held_out_report.json",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    all_paths = (*PUBLIC_FILES, *PRIVATE_FILES)
    missing = [path for path in all_paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"final release inputs missing: {missing}")
    payload = {
        "schema_version": "1.0",
        "release": "viettheory_final_candidate_v1_2026-08-15",
        "candidate_frozen_before_hidden": True,
        "natural_qa_hidden_evaluated_once": True,
        "evidence_sufficiency_held_out_evaluated": True,
        "public_files": [
            {"path": path, "sha256": _digest(root / path)} for path in sorted(PUBLIC_FILES)
        ],
        "private_artifact_checksums": [
            {"path": path, "sha256": _digest(root / path)} for path in sorted(PRIVATE_FILES)
        ],
    }
    output = root / "releases/v1.0/release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
