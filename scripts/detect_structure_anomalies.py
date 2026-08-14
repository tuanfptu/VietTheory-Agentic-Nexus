"""Rank pages that merit selective Gemini structure/OCR auditing."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from viettheory.schema import Page

SUBJECTS = ("MLN111", "MLN122", "MLN131", "HCM202", "VNR202")
CHAPTER = re.compile(r"^chương\s+\d+\b", re.IGNORECASE)
NUMBERED = re.compile(r"^(?:[IVXLCDM]+\.|[1-9](?:\.\d+)*\.)\s+", re.IGNORECASE)
OCR_NOISE = re.compile(r"[¬§ð]|(?:^|\s)[|!](?:\s|$)|\ufffd")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _page_parent_coverage(parents: list[dict[str, Any]]) -> set[int]:
    return {span["pdf_page"] for parent in parents for span in parent.get("source_spans", [])}


def _page_heading_counts(headings: list[dict[str, Any]]) -> Counter[int]:
    return Counter(heading["pdf_page"] for heading in headings)


def _looks_heading_like(page: Page) -> bool:
    for block in page.blocks:
        text = " ".join(block.text.split())
        if len(text) <= 180 and (CHAPTER.match(text) or NUMBERED.match(text)):
            return True
    return False


def _score_page(
    page: Page,
    *,
    covered: set[int],
    heading_count: int,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    substantive = page.char_count >= 350
    if page.quality_score < 0.75:
        score += 5
        reasons.append("low_extraction_quality")
    elif page.quality_score < 0.9:
        score += 3
        reasons.append("moderate_extraction_quality")
    if substantive and page.pdf_page not in covered:
        score += 8
        reasons.append("substantive_page_without_parent")
    heading_like = _looks_heading_like(page)
    if heading_like and heading_count == 0:
        score += 6
        reasons.append("heading_like_text_not_detected")
    if heading_count >= 4:
        score += 2
        reasons.append("heading_dense_page")
    noise_hits = len(OCR_NOISE.findall(page.text))
    if noise_hits >= 3:
        score += min(4, noise_hits)
        reasons.append("ocr_noise_markers")
    if page.needs_ocr:
        score += 2
        reasons.append("extractor_still_flags_ocr")
    return score, reasons


def _select_portfolio(
    ranked: list[dict[str, Any]],
    *,
    per_subject: int,
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ranked:
        by_subject[record["subject"]].append(record)
    selected: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        subject_rows = by_subject[subject]
        reason_best: dict[str, dict[str, Any]] = {}
        for row in subject_rows:
            for reason in row["reasons"]:
                reason_best.setdefault(reason, row)
        seeds = sorted(
            {row["pdf_page"]: row for row in reason_best.values()}.values(),
            key=lambda row: (-row["score"], row["pdf_page"]),
        )
        chosen = {row["pdf_page"]: row for row in seeds[:per_subject]}
        for row in subject_rows:
            if len(chosen) >= per_subject:
                break
            chosen.setdefault(row["pdf_page"], row)
        selected.extend(chosen.values())
    return sorted(selected, key=lambda row: (SUBJECTS.index(row["subject"]), row["pdf_page"]))


def _select_contiguous_windows(
    ranked: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    *,
    windows_per_subject: int = 4,
    window_size: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose spaced anomaly anchors and expand each into one local page window."""
    windows: list[dict[str, Any]] = []
    selected_pages: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        page_count = summaries[subject]["page_count"]
        subject_rows = [row for row in ranked if row["subject"] == subject]
        anchors: list[dict[str, Any]] = []
        for row in subject_rows:
            if all(abs(row["pdf_page"] - anchor["pdf_page"]) >= window_size for anchor in anchors):
                anchors.append(row)
            if len(anchors) == windows_per_subject:
                break
        for anchor in sorted(anchors, key=lambda row: row["pdf_page"]):
            start = min(max(0, anchor["pdf_page"] - window_size // 2), page_count - window_size)
            page_numbers = list(range(start, start + window_size))
            windows.append(
                {
                    "subject": subject,
                    "anchor_pdf_page": anchor["pdf_page"],
                    "anchor_score": anchor["score"],
                    "anchor_reasons": anchor["reasons"],
                    "pdf_pages": page_numbers,
                }
            )
            selected_pages.extend(
                {
                    "subject": subject,
                    "pdf_page": number,
                    "pdf_page_human": number + 1,
                    "window_anchor_pdf_page": anchor["pdf_page"],
                }
                for number in page_numbers
            )
    return windows, selected_pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-subject", type=int, default=20)
    args = parser.parse_args()
    if args.per_subject < 1:
        parser.error("per-subject must be positive")
    root = args.root.resolve()
    ranked: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        base = root / "data" / "processed" / subject
        pages = [
            Page.model_validate_json(line)
            for line in (base / "pages.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        headings = _jsonl(base / "structured_v1" / "headings.jsonl")
        parents = _jsonl(base / "structured_v1" / "parents.jsonl")
        coverage = _page_parent_coverage(parents)
        heading_counts = _page_heading_counts(headings)
        rows: list[dict[str, Any]] = []
        for page in pages:
            score, reasons = _score_page(
                page,
                covered=coverage,
                heading_count=heading_counts[page.pdf_page],
            )
            if score:
                rows.append(
                    {
                        "subject": subject,
                        "pdf_page": page.pdf_page,
                        "pdf_page_human": page.pdf_page + 1,
                        "printed_page": page.printed_page,
                        "score": score,
                        "reasons": reasons,
                        "quality_score": page.quality_score,
                        "char_count": page.char_count,
                        "rotation": page.rotation,
                        "v1_heading_count": heading_counts[page.pdf_page],
                        "v1_parent_covered": page.pdf_page in coverage,
                    }
                )
        rows.sort(key=lambda row: (-row["score"], row["pdf_page"]))
        ranked.extend(rows)
        summaries[subject] = {
            "page_count": len(pages),
            "anomalous_page_count": len(rows),
            "substantive_without_parent": sum(
                "substantive_page_without_parent" in row["reasons"] for row in rows
            ),
            "heading_like_not_detected": sum(
                "heading_like_text_not_detected" in row["reasons"] for row in rows
            ),
            "low_or_moderate_quality": sum(
                bool(
                    {"low_extraction_quality", "moderate_extraction_quality"} & set(row["reasons"])
                )
                for row in rows
            ),
        }
    ranked.sort(key=lambda row: (-row["score"], SUBJECTS.index(row["subject"]), row["pdf_page"]))
    portfolio = _select_portfolio(ranked, per_subject=args.per_subject)
    windows, window_pages = _select_contiguous_windows(ranked, summaries)
    payload = {
        "schema_version": "1.0",
        "status": "local_detection_complete_gemini_not_called",
        "method": "deterministic_structure_anomaly_v1",
        "per_subject_limit": args.per_subject,
        "selected_page_count": len(portfolio),
        "estimated_batch5_requests": (len(portfolio) + 4) // 5,
        "subject_summaries": summaries,
        "selected_pages": portfolio,
        "all_ranked_anomalies": ranked,
        "contiguous_windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Five-subject selective Gemini audit portfolio",
        "",
        "Status: local detection complete; Gemini has not been called for this portfolio.",
        "",
        "| Subject | Selected PDF pages (human numbering) |",
        "|---|---|",
    ]
    for subject in SUBJECTS:
        numbers = [str(row["pdf_page_human"]) for row in portfolio if row["subject"] == subject]
        markdown.append(f"| {subject} | {', '.join(numbers)} |")
    markdown.extend(
        [
            "",
            f"Total: {len(portfolio)} pages; estimated "
            f"{payload['estimated_batch5_requests']} five-page requests.",
            "",
            "Selection signals:",
            "",
            "- substantive extracted text without any v1 parent coverage;",
            "- heading-like text not represented in v1 headings;",
            "- low or moderate extraction quality;",
            "- OCR noise markers;",
            "- unusually heading-dense pages.",
            "",
            "This is a repair/audit portfolio, not evidence that every selected page is wrong. "
            "Gemini output remains pending validation and human spot-checking before any v2 "
            "corpus promotion.",
        ]
    )
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    window_payload = {
        "schema_version": "1.0",
        "status": "local_window_selection_complete_gemini_not_called",
        "method": "four_spaced_anomaly_windows_per_subject_v1",
        "window_size": 5,
        "window_count": len(windows),
        "selected_page_count": len(window_pages),
        "estimated_requests": len(windows),
        "windows": windows,
        "selected_pages": window_pages,
    }
    window_path = args.output.with_name(f"{args.output.stem}_windows.json")
    window_path.write_text(
        json.dumps(window_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected={len(portfolio)} estimated_batch5_requests="
        f"{payload['estimated_batch5_requests']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
