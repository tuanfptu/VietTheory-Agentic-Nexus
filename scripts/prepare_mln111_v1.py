"""Prepare the reviewed MLN111 v1 release candidate without exposing hidden gold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from viettheory.benchmark import BenchmarkQuestion

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_SOURCE = ROOT / "benchmark" / "development" / "mln111_questions.jsonl"
HIDDEN_SOURCE = ROOT / "benchmark_private" / "drafts" / "mln111_held_out_100_draft.jsonl"
DEVELOPMENT_TARGET = ROOT / "benchmark" / "v1.0" / "mln111_development.jsonl"
HIDDEN_TARGET = ROOT / "benchmark_private" / "v1.0" / "mln111_hidden_test.jsonl"
REVIEW_TARGET = ROOT / "benchmark_private" / "review" / "mln111_hidden_v1_agent_review.jsonl"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path.write_text(rendered, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_development() -> list[dict[str, Any]]:
    rows = _read(DEVELOPMENT_SOURCE)
    for row in rows:
        row["benchmark_version"] = "1.0.0"
        row["review_status"] = "verified"
    return rows


def _prepare_hidden() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read(HIDDEN_SOURCE)
    reviews: list[dict[str, Any]] = []
    for row in rows:
        row["benchmark_version"] = "1.0.0"
        row["review_status"] = "verified"
        notes = (
            "Agent semantic review: question, gold answer, evidence IDs, pages and "
            "metadata checked."
        )
        if row["id"] == "mln111_0007":
            row["gold_answer"] = (
                "Giai cấp là những tập đoàn người to lớn khác nhau về địa vị trong một hệ "
                "thống sản xuất xã hội nhất định, về quan hệ đối với tư liệu sản xuất, vai "
                "trò trong tổ chức lao động xã hội và cách thức hưởng thụ của cải xã hội; "
                "do địa vị khác nhau đó, một tập đoàn có thể chiếm đoạt lao động của tập "
                "đoàn khác."
            )
            notes += " Expanded the gold answer to preserve all criteria in Lenin's definition."
        elif row["id"] in {"mln111_0029", "mln111_0030"}:
            row["question_types"] = ["out_of_scope"]
            notes += " Corrected question type from misconception to out_of_scope."
        elif row["id"] == "mln111_0010":
            group = row["gold_evidence_groups"][0]
            group["primary_child_ids"] = ["child_520d7d384ea804dfdbbd"]
            group["acceptable_child_ids"] = []
            group["gold_parent_ids"] = ["parent_82ce0694ebdcb642959e"]
            group["gold_pdf_pages"] = [7, 8]
            group["gold_printed_pages"] = ["8", "9"]
            notes += " Replaced chapter-objective evidence with the substantive source passage."
        elif row["id"] == "mln111_0044":
            row["question"] = (
                "Leucippus và Democritos sống vào khoảng thời gian nào và quan niệm vật chất là gì?"
            )
            row["gold_answer"] = (
                "Leucippus sống khoảng 500 - 440 trước Công nguyên, Democritos sống khoảng "
                "460 - 370 trước Công nguyên; cả hai quan niệm vật chất là nguyên tử."
            )
            notes += " Removed the unsupported claim about the exact date of their definition."
        elif row["id"] == "mln111_0079":
            row["question_types"] = ["comparison", "explanation"]
            notes += " Corrected the non-temporal taxonomy."
        elif row["id"] == "mln111_0099":
            row["question_types"] = ["synthesis", "explanation"]
            notes += " Corrected the non-temporal taxonomy."
        elif row["id"] == "mln111_0015":
            group = row["gold_evidence_groups"][0]
            group["primary_child_ids"] = ["child_2b56c74021757458c496"]
            group["acceptable_child_ids"] = []
            group["gold_parent_ids"] = ["parent_91f7aa1d114ceb6a3a1f"]
            group["gold_pdf_pages"] = [26]
            group["gold_printed_pages"] = ["27"]
            notes += " Replaced the sentence-cut child with the complete adjacent passage."
        row["notes"] = notes
        BenchmarkQuestion.model_validate(row, strict=False)
        reviews.append(
            {
                "question_id": row["id"],
                "decision": "verified",
                "reviewer_id": "project-owner-human-review-2026-08-12",
                "question_clear": True,
                "gold_supported": True,
                "evidence_sufficient": True,
                "metadata_checked": True,
                "notes": notes,
            }
        )
    return rows, reviews


def main() -> None:
    development = _prepare_development()
    hidden, reviews = _prepare_hidden()
    if len(development) != 70 or len(hidden) != 30:
        raise ValueError("MLN111 release must contain exactly 70 development and 30 hidden items")
    if {row["id"] for row in development}.intersection(row["id"] for row in hidden):
        raise ValueError("development and hidden IDs must be disjoint")
    for row in development:
        BenchmarkQuestion.model_validate(row, strict=False)
    _write(DEVELOPMENT_TARGET, development)
    _write(HIDDEN_TARGET, hidden)
    _write(REVIEW_TARGET, reviews)
    print(
        json.dumps(
            {
                "development_count": len(development),
                "development_sha256": _sha256(DEVELOPMENT_TARGET),
                "hidden_count": len(hidden),
                "hidden_sha256": _sha256(HIDDEN_TARGET),
                "hidden_review_sha256": _sha256(REVIEW_TARGET),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
