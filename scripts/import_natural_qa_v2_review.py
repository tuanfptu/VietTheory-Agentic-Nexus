"""Import a human CSV review into the canonical Natural QA v2 draft."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from viettheory.benchmark import ReviewStatus
from viettheory.natural_benchmark import NaturalQuestionV2, ReviewGateAudit

DECISION_STATUS = {
    "approve": ReviewStatus.VERIFIED,
    "revise": ReviewStatus.DRAFT,
    "reject": ReviewStatus.REJECTED,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_bool(value: str, *, field: str, question_id: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{question_id}: invalid {field} value {value!r}")


def _load_questions(path: Path) -> dict[str, NaturalQuestionV2]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = {
        question.id: question
        for question in (NaturalQuestionV2.model_validate_json(line) for line in lines)
    }
    if len(questions) != len(lines):
        raise ValueError("duplicate question IDs in canonical draft")
    return questions


def import_review(
    draft_path: Path,
    review_path: Path,
    *,
    reviewer_id: str,
) -> tuple[list[NaturalQuestionV2], dict[str, object]]:
    questions = _load_questions(draft_path)
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    review_ids = [row["id"].strip() for row in rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("duplicate question IDs in review CSV")
    if set(review_ids) != set(questions):
        missing = sorted(set(questions).difference(review_ids))
        extra = sorted(set(review_ids).difference(questions))
        raise ValueError(f"review/draft ID mismatch; missing={missing}, extra={extra}")

    imported: list[NaturalQuestionV2] = []
    decisions: Counter[str] = Counter()
    by_subject: dict[str, Counter[str]] = {}
    for row in rows:
        question_id = row["id"].strip()
        question = questions[question_id]
        decision = row["decision"].strip().casefold()
        if decision not in DECISION_STATUS:
            raise ValueError(f"{question_id}: unsupported decision {decision!r}")
        if row["subject"].strip() != question.subject_code:
            raise ValueError(f"{question_id}: subject differs between review and draft")
        if row["question"].strip() != question.question:
            raise ValueError(f"{question_id}: question text differs between review and draft")

        notes = row["reviewer_notes"].strip() or None
        acceptable = row["acceptable_chunk_notes"].strip()
        combined_notes = notes
        if acceptable:
            combined_notes = f"{notes or ''}\nAcceptable chunk notes: {acceptable}".strip()
        gates = ReviewGateAudit(
            question_valid=_parse_bool(
                row["question_valid"], field="question_valid", question_id=question_id
            ),
            gold_answer_valid=_parse_bool(
                row["gold_answer_valid"],
                field="gold_answer_valid",
                question_id=question_id,
            ),
            parent_expanded_evidence_valid=_parse_bool(
                row["evidence_valid"], field="evidence_valid", question_id=question_id
            ),
            difficulty_valid=_parse_bool(
                row["difficulty_valid"], field="difficulty_valid", question_id=question_id
            ),
            reviewer_id=reviewer_id,
            notes=combined_notes,
        )
        if decision == "approve" and not gates.passed:
            raise ValueError(f"{question_id}: approve requires all review gates to pass")

        imported.append(
            question.model_copy(
                update={
                    "review_status": DECISION_STATUS[decision],
                    "review_notes": combined_notes,
                    "review_gates": gates,
                }
            )
        )
        decisions[decision] += 1
        by_subject.setdefault(question.subject_code, Counter())[decision] += 1

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "reviewer_id": reviewer_id,
        "draft_source": {
            "path": draft_path.as_posix(),
            "sha256": _sha256(draft_path),
        },
        "review_source": {
            "path": review_path.as_posix(),
            "sha256": _sha256(review_path),
        },
        "record_count": len(imported),
        "decision_counts": dict(sorted(decisions.items())),
        "subject_decision_counts": {
            subject: dict(sorted(counts.items())) for subject, counts in sorted(by_subject.items())
        },
        "release_policy": {
            "approve": "verified",
            "revise": "draft_pending_correction_and_recheck",
            "reject": "rejected_excluded_from_gold",
        },
    }
    return imported, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path)
    parser.add_argument("--needs-action-output", type=Path)
    parser.add_argument("--reviewer-id", default="human_review_2026_08_14")
    args = parser.parse_args()

    records, manifest = import_review(args.draft, args.review, reviewer_id=args.reviewer_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    if args.verified_output:
        args.verified_output.parent.mkdir(parents=True, exist_ok=True)
        args.verified_output.write_text(
            "".join(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in records
                if record.review_status is ReviewStatus.VERIFIED
            ),
            encoding="utf-8",
        )
    if args.needs_action_output:
        args.needs_action_output.parent.mkdir(parents=True, exist_ok=True)
        args.needs_action_output.write_text(
            "".join(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in records
                if record.review_status is not ReviewStatus.VERIFIED
            ),
            encoding="utf-8",
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
