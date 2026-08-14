"""Generate a deterministic 48-case Evidence Sufficiency pilot from B0 failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from viettheory.evidence_sufficiency import (
    EvidenceSufficiencyCase,
    PerturbationType,
    ProvidedContext,
    RequiredAspect,
    SufficiencyLabel,
)
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.schema import Chunk

SUBJECT_QUOTAS = {"HCM202": 3, "MLN111": 2, "MLN122": 3, "MLN131": 3, "VNR202": 1}
VARIANTS = ("sufficient", "partial", "missing", "wrongaspect")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(parent: Chunk) -> ProvidedContext:
    return ProvidedContext(
        parent_id=parent.chunk_id,
        subject_code=parent.subject_code,
        text=parent.text,
        pdf_pages=tuple(sorted({span.pdf_page for span in parent.source_spans})),
    )


def _case_id(question_id: str, variant: str) -> str:
    return f"es_{question_id}_{variant}"


def _select_sources(
    questions: dict[str, NaturalQuestionV2], failures: list[dict[str, Any]]
) -> tuple[NaturalQuestionV2, ...]:
    eligible: dict[str, list[NaturalQuestionV2]] = defaultdict(list)
    for failure in failures:
        question = questions[str(failure["question_id"])]
        required = tuple(group for group in question.required_evidence_groups if group.required)
        parent_sets = {tuple(group.gold_parent_ids) for group in required}
        if len(required) >= 2 and len(parent_sets) == len(required):
            eligible[question.subject_code].append(question)
    selected: list[NaturalQuestionV2] = []
    for subject_code, quota in SUBJECT_QUOTAS.items():
        ranked = sorted(
            eligible[subject_code],
            key=lambda question: (
                question.primary_category.value,
                question.difficulty.value,
                question.id,
            ),
        )
        if len(ranked) < quota:
            raise ValueError(f"not enough separated residual failures for {subject_code}")
        selected.extend(ranked[:quota])
    return tuple(sorted(selected, key=lambda question: question.id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--delta-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--recheck-csv", type=Path)
    args = parser.parse_args()

    adjudications: dict[str, str] = {}
    if args.adjudications is not None:
        adjudication_payload = json.loads(args.adjudications.read_text(encoding="utf-8"))
        adjudications = {
            str(case_id): str(parent_id)
            for case_id, parent_id in adjudication_payload["wrong_aspect_parent_overrides"].items()
        }

    questions = {
        question.id: question
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for question in (NaturalQuestionV2.model_validate_json(line),)
    }
    delta = json.loads(args.delta_report.read_text(encoding="utf-8"))
    failures = delta["residual_failures"]["within_subject_parent_aware_b0"]["queries"]
    sources = _select_sources(questions, failures)
    parents = {
        parent.chunk_id: parent
        for subject_code in SUBJECT_QUOTAS
        for line in (args.data_root / subject_code / "structured_v1" / "parents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
        for parent in (Chunk.model_validate_json(line),)
    }
    failures_by_id = {str(row["question_id"]): row for row in failures}
    cases: list[EvidenceSufficiencyCase] = []
    for question in sources:
        groups = tuple(group for group in question.required_evidence_groups if group.required)
        aspects = tuple(
            RequiredAspect(
                aspect_id=f"a{index}",
                description=group.role.replace("_", " "),
                acceptable_parent_ids=group.gold_parent_ids,
            )
            for index, group in enumerate(groups, 1)
        )
        all_gold_ids = tuple(
            dict.fromkeys(parent for group in groups for parent in group.gold_parent_ids)
        )
        gold_contexts = tuple(_context(parents[parent_id]) for parent_id in all_gold_ids)
        kept_ids = groups[0].gold_parent_ids
        partial_contexts = tuple(_context(parents[parent_id]) for parent_id in kept_ids)
        removed = tuple(f"a{index}" for index in range(2, len(groups) + 1))
        retrieved = failures_by_id[question.id]["retrieved_ids"]
        wrong_case_id = _case_id(question.id, "wrongaspect")
        wrong_id = adjudications.get(
            wrong_case_id, next((item for item in retrieved if item not in all_gold_ids), None)
        )
        if wrong_id is None:
            raise ValueError(f"no related wrong-aspect parent for {question.id}")
        common: dict[str, Any] = {
            "benchmark_version": "mln_evidence_sufficiency_pilot_v0.1",
            "source_question_id": question.id,
            "subject_code": question.subject_code,
            "question": question.question,
            "required_aspects": aspects,
            "generator_version": "deterministic_perturbation_v1",
            "split_group": question.id,
        }
        cases.extend(
            (
                EvidenceSufficiencyCase.model_validate(
                    common
                    | {
                        "case_id": _case_id(question.id, "sufficient"),
                        "provided_contexts": gold_contexts,
                        "expected_label": SufficiencyLabel.SUFFICIENT,
                        "perturbation": PerturbationType.NONE,
                    }
                ),
                EvidenceSufficiencyCase.model_validate(
                    common
                    | {
                        "case_id": _case_id(question.id, "partial"),
                        "provided_contexts": partial_contexts,
                        "expected_label": SufficiencyLabel.PARTIAL,
                        "perturbation": PerturbationType.REMOVE_REQUIRED_GROUPS,
                        "removed_aspect_ids": removed,
                    }
                ),
                EvidenceSufficiencyCase.model_validate(
                    common
                    | {
                        "case_id": _case_id(question.id, "missing"),
                        "provided_contexts": (),
                        "expected_label": SufficiencyLabel.MISSING,
                        "perturbation": PerturbationType.REMOVE_ALL_EVIDENCE,
                        "removed_aspect_ids": tuple(aspect.aspect_id for aspect in aspects),
                    }
                ),
                EvidenceSufficiencyCase.model_validate(
                    common
                    | {
                        "case_id": wrong_case_id,
                        "provided_contexts": (_context(parents[str(wrong_id)]),),
                        "expected_label": SufficiencyLabel.WRONG_ASPECT,
                        "perturbation": PerturbationType.RELATED_WRONG_ASPECT,
                        "removed_aspect_ids": tuple(aspect.aspect_id for aspect in aspects),
                    }
                ),
            )
        )

    cases.sort(key=lambda case: case.case_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "source_question_id",
                "subject_code",
                "expected_label",
                "question",
                "source_difficulty",
                "provided_parent_ids",
                "provided_context_texts",
                "decision",
                "label_valid",
                "context_valid",
                "difficulty_valid",
                "reviewer_notes",
            ),
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "source_question_id": case.source_question_id,
                    "subject_code": case.subject_code,
                    "expected_label": case.expected_label.value,
                    "question": case.question,
                    "source_difficulty": questions[case.source_question_id].difficulty.value,
                    "provided_parent_ids": "|".join(
                        context.parent_id for context in case.provided_contexts
                    ),
                    "provided_context_texts": "\n---\n".join(
                        context.text for context in case.provided_contexts
                    ),
                    "decision": "",
                    "label_valid": "",
                    "context_valid": "",
                    "difficulty_valid": "",
                    "reviewer_notes": "",
                }
            )
    if args.recheck_csv is not None:
        args.recheck_csv.parent.mkdir(parents=True, exist_ok=True)
        recheck_ids = set(adjudications)
        with args.review_csv.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        with args.recheck_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["case_id"] in recheck_ids)
    manifest = {
        "benchmark_version": "mln_evidence_sufficiency_pilot_v0.1",
        "case_count": len(cases),
        "source_question_count": len(sources),
        "source_question_ids": [question.id for question in sources],
        "labels": {
            label.value: sum(case.expected_label is label for case in cases)
            for label in SufficiencyLabel
        },
        "contradiction_status": "exploratory; no synthetic negation cases generated",
        "output_sha256": _sha256(args.output),
        "review_status": "draft_pending_human_audit",
        "adjudication_overrides": adjudications,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
