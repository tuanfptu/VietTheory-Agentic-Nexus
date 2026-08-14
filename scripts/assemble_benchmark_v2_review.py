"""Assemble validated Natural QA v2 drafts into a one-pass human review package."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from viettheory.natural_benchmark import (
    NaturalQuestionV2,
    PortfolioPlan,
    validate_natural_portfolio,
)
from viettheory.schema import Chunk

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("MLN111", "MLN122", "MLN131", "HCM202", "VNR202")
PLAN_PATH = ROOT / "benchmark" / "v2" / "portfolio_plan_500.json"
REVIEW_DIR = ROOT / "benchmark" / "v2" / "review"
COMBINED_PATH = REVIEW_DIR / "natural_qa_v2_500_draft.jsonl"
CSV_PATH = REVIEW_DIR / "natural_qa_v2_500_review.csv"
REPORT_PATH = ROOT / "reports" / "natural_qa_v2_500_draft_validation.json"
REPORT_MD_PATH = ROOT / "reports" / "natural_qa_v2_500_draft_validation.md"


def load_jsonl(path: Path, model: type[NaturalQuestionV2]) -> list[NaturalQuestionV2]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_children(subject: str) -> dict[str, Chunk]:
    path = ROOT / "data" / "processed" / subject / "structured_v1" / "children.jsonl"
    children: dict[str, Chunk] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunk = Chunk.model_validate_json(line)
            children[chunk.chunk_id] = chunk
    return children


def evidence_details(
    record: NaturalQuestionV2, children: dict[str, Chunk]
) -> tuple[str, str, str, str]:
    group_summaries: list[str] = []
    texts: list[str] = []
    child_ids: list[str] = []
    parent_ids: list[str] = []
    for group in record.required_evidence_groups:
        child_ids.extend(group.primary_child_ids)
        parent_ids.extend(group.gold_parent_ids)
        pages = ", ".join(str(page + 1) for page in group.gold_pdf_pages)
        group_summaries.append(
            f"{group.group_id}:{group.role} (PDF pages {pages}; required={group.required})"
        )
        for child_id in group.primary_child_ids:
            chunk = children[child_id]
            texts.append(f"[{group.group_id} | {child_id}]\n{chunk.text}")
    return (
        " | ".join(group_summaries),
        "\n\n".join(texts),
        " | ".join(dict.fromkeys(child_ids)),
        " | ".join(dict.fromkeys(parent_ids)),
    )


def write_subject_markdown(
    subject: str,
    records: list[NaturalQuestionV2],
    children: dict[str, Chunk],
) -> None:
    lines = [
        f"# {subject} Natural QA v2 - Human review (100 drafts)",
        "",
        "For every item, select approve/revise/reject and validate question, gold answer, "
        "parent-expanded evidence, and difficulty. Drafts are not benchmark gold.",
        "",
    ]
    for record in records:
        groups, evidence_text, _, _ = evidence_details(record, children)
        lines.extend(
            [
                f"## {record.id}",
                "",
                f"- Category: `{record.primary_category.value}`",
                f"- Difficulty: `{record.difficulty.value}`",
                f"- Reasoning: `{record.reasoning_scope.value}`",
                f"- Answerability: `{record.answerability.value}`",
                f"- Chapters: {', '.join(record.chapter_labels) or 'N/A'}",
                f"- Evidence groups: {groups or 'N/A'}",
                "",
                f"**Question:** {record.question}",
                "",
                f"**Gold answer:** {record.gold_answer or 'N/A'}",
                "",
                f"**Unanswerable reason:** {record.unanswerable_reason or 'N/A'}",
                "",
                "**Evidence text:**",
                "",
                "```text",
                evidence_text or "N/A",
                "```",
                "",
                "Decision: [ ] approve  [ ] revise  [ ] reject",
                "",
                "Gates: [ ] question  [ ] gold answer  [ ] evidence  [ ] difficulty",
                "",
                "Reviewer notes:",
                "",
                "---",
                "",
            ]
        )
    (REVIEW_DIR / f"{subject}_100_review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plan = PortfolioPlan.model_validate_json(PLAN_PATH.read_text(encoding="utf-8"))
    all_records: list[NaturalQuestionV2] = []
    children_by_subject: dict[str, dict[str, Chunk]] = {}
    completeness_issues: list[str] = []

    for subject in SUBJECTS:
        draft_path = ROOT / "benchmark" / "v2" / "drafts" / f"{subject}_100_draft.jsonl"
        records = load_jsonl(draft_path, NaturalQuestionV2)
        all_records.extend(records)
        children = load_children(subject)
        children_by_subject[subject] = children
        subject_plan = next(batch for batch in plan.pilot_batches if batch.subject_code == subject)
        counts = Counter(record.primary_category for record in records)
        if len(records) != subject_plan.batch_size:
            completeness_issues.append(
                f"{subject}: expected {subject_plan.batch_size}, got {len(records)}"
            )
        for quota in subject_plan.quotas:
            if counts[quota.category] != quota.target:
                completeness_issues.append(
                    f"{subject}/{quota.category.value}: expected {quota.target}, "
                    f"got {counts[quota.category]}"
                )

    report = validate_natural_portfolio(tuple(all_records), plan)
    issues = [*report.issues, *completeness_issues]
    payload = report.model_dump(mode="json")
    payload["valid"] = not issues
    payload["issues"] = issues
    payload["review_state"] = "draft_pending_human_verification"
    payload["combined_count"] = len(all_records)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_PATH.write_text(
        "".join(record.model_dump_json() + "\n" for record in all_records),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    columns = (
        "id",
        "subject",
        "category",
        "difficulty",
        "reasoning_scope",
        "answerability",
        "question",
        "gold_answer",
        "unanswerable_reason",
        "evidence_groups",
        "evidence_text",
        "child_ids",
        "parent_ids",
        "decision",
        "question_valid",
        "gold_answer_valid",
        "evidence_valid",
        "difficulty_valid",
        "acceptable_chunk_notes",
        "reviewer_notes",
    )
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in all_records:
            groups, text, child_ids, parent_ids = evidence_details(
                record, children_by_subject[record.subject_code]
            )
            writer.writerow(
                {
                    "id": record.id,
                    "subject": record.subject_code,
                    "category": record.primary_category.value,
                    "difficulty": record.difficulty.value,
                    "reasoning_scope": record.reasoning_scope.value,
                    "answerability": record.answerability.value,
                    "question": record.question,
                    "gold_answer": record.gold_answer or "",
                    "unanswerable_reason": record.unanswerable_reason or "",
                    "evidence_groups": groups,
                    "evidence_text": text,
                    "child_ids": child_ids,
                    "parent_ids": parent_ids,
                    "decision": "",
                    "question_valid": "",
                    "gold_answer_valid": "",
                    "evidence_valid": "",
                    "difficulty_valid": "",
                    "acceptable_chunk_notes": "",
                    "reviewer_notes": "",
                }
            )

    for subject in SUBJECTS:
        subject_records = [record for record in all_records if record.subject_code == subject]
        write_subject_markdown(subject, subject_records, children_by_subject[subject])

    summary_lines = [
        "# Natural QA v2 500-draft validation",
        "",
        f"- Status: **{'PASS' if not issues else 'FAIL'}**",
        f"- Drafts: **{len(all_records)}**",
        f"- Schema/portfolio issues: **{len(issues)}**",
        f"- Diversity warnings requiring human inspection: **{len(report.warnings)}**",
        "- Review state: **draft pending human verification**",
        "",
        "The JSONL, UTF-8 CSV, and five subject Markdown files form one review package. "
        "No record is checked or verified automatically.",
    ]
    REPORT_MD_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"records={len(all_records)} issues={len(issues)} warnings={len(report.warnings)}")
    print(CSV_PATH)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
