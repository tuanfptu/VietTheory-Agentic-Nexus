"""Evaluate frozen L0/L1 shortcut baselines on Evidence Sufficiency development."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Protocol

from viettheory.evidence_baselines import LexicalCoverageBaseline, TfidfSimilarityBaseline
from viettheory.evidence_sufficiency import EvidenceSufficiencyCase, SufficiencyLabel


class Baseline(Protocol):
    def score(self, case: EvidenceSufficiencyCase) -> float: ...

    def predict(self, case: EvidenceSufficiencyCase) -> SufficiencyLabel: ...


def _metrics(cases: tuple[EvidenceSufficiencyCase, ...], baseline: Baseline) -> dict[str, object]:
    labels = tuple(SufficiencyLabel)
    confusion = {gold.value: {pred.value: 0 for pred in labels} for gold in labels}
    rows: list[dict[str, object]] = []
    correct = 0
    for case in cases:
        predicted = baseline.predict(case)
        confusion[case.expected_label.value][predicted.value] += 1
        correct += int(predicted is case.expected_label)
        rows.append(
            {
                "case_id": case.case_id,
                "source_question_id": case.source_question_id,
                "gold": case.expected_label.value,
                "predicted": predicted.value,
                "score": baseline.score(case),
                "correct": predicted is case.expected_label,
            }
        )
    f1_by_label: dict[str, float] = {}
    for label in labels:
        name = label.value
        true_positive = confusion[name][name]
        false_positive = sum(confusion[gold.value][name] for gold in labels if gold is not label)
        false_negative = sum(confusion[name][pred.value] for pred in labels if pred is not label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_by_label[name] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    active = [label.value for label in labels if any(confusion[label.value].values())]
    return {
        "accuracy": correct / len(cases),
        "macro_f1": sum(f1_by_label[label] for label in active) / len(active),
        "f1_by_label": f1_by_label,
        "confusion": confusion,
        "predicted_distribution": dict(Counter(row["predicted"] for row in rows)),
        "per_case": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = tuple(
        EvidenceSufficiencyCase.model_validate_json(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    l0 = LexicalCoverageBaseline()
    l1 = TfidfSimilarityBaseline(cases)
    l0_metrics = _metrics(cases, l0)
    l1_metrics = _metrics(cases, l1)
    report = {
        "benchmark_version": cases[0].benchmark_version,
        "split": "development",
        "case_count": len(cases),
        "baselines_frozen_before_j1": True,
        "L0_lexical_coverage": {
            "configuration": {
                "sufficient_threshold": l0.sufficient_threshold,
                "partial_threshold": l0.partial_threshold,
            },
            **l0_metrics,
        },
        "L1_tfidf_semantic_lite": {
            "configuration": {
                "sufficient_threshold": l1.sufficient_threshold,
                "partial_threshold": l1.partial_threshold,
            },
            **l1_metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "L0_lexical_coverage": {
                    "accuracy": l0_metrics["accuracy"],
                    "macro_f1": l0_metrics["macro_f1"],
                },
                "L1_tfidf_semantic_lite": {
                    "accuracy": l1_metrics["accuracy"],
                    "macro_f1": l1_metrics["macro_f1"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
