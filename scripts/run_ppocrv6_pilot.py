"""Run an isolated 10-page PP-OCRv6 diagnostic pilot.

This script never modifies the canonical ``pages.jsonl`` files. It renders a
fixed, representative page portfolio, runs PP-OCRv6 on GPU, and writes raw
experimental output plus a Markdown/JSON comparison report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import paddle
import pypdfium2 as pdfium
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "ppocrv6_pilot"
REPORT_JSON = ROOT / "reports" / "ppocrv6_10_page_pilot.json"
REPORT_MD = ROOT / "reports" / "ppocrv6_10_page_pilot.md"

# Zero-based PDF pages. The portfolio covers chapter openings, body pages,
# low-quality OCR pages, heading-heavy pages, and late-book/review content.
PAGE_PORTFOLIO: tuple[dict[str, Any], ...] = (
    {"subject": "VNR202", "pdf_page": 21, "category": "chapter_opening"},
    {"subject": "VNR202", "pdf_page": 124, "category": "chapter_opening"},
    {"subject": "VNR202", "pdf_page": 180, "category": "body_control"},
    {"subject": "VNR202", "pdf_page": 225, "category": "late_review"},
    {"subject": "MLN131", "pdf_page": 46, "category": "low_quality"},
    {"subject": "MLN131", "pdf_page": 160, "category": "low_quality_heading"},
    {"subject": "MLN131", "pdf_page": 234, "category": "low_quality_heading"},
    {"subject": "HCM202", "pdf_page": 71, "category": "ocr_noise_heading"},
    {"subject": "HCM202", "pdf_page": 83, "category": "ocr_noise_heading"},
    {"subject": "HCM202", "pdf_page": 266, "category": "late_ocr_noise"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=len(PAGE_PORTFOLIO),
        help="Run only the first N portfolio pages (useful for a smoke test).",
    )
    parser.add_argument("--scale", type=float, default=2.5)
    return parser.parse_args()


def find_pdf(subject: str) -> Path:
    matches = sorted((ROOT / "Tài liệu").glob(f"*{subject}.pdf"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one PDF for {subject}, found {matches}")
    return matches[0]


def load_canonical_page(subject: str, pdf_page: int) -> dict[str, Any]:
    path = ROOT / "data" / "processed" / subject / "pages.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["pdf_page"] == pdf_page:
                return record
    raise RuntimeError(f"Canonical page not found: {subject} page {pdf_page}")


def render_page(pdf_path: Path, page_number: int, image_path: Path, scale: float) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf_path)
    try:
        if page_number >= len(document):
            raise IndexError(f"Page {page_number} outside {pdf_path.name}")
        bitmap = document[page_number].render(scale=scale)
        bitmap.to_pil().save(image_path)
    finally:
        document.close()


def result_payload(result: Any) -> dict[str, Any]:
    payload = result.json
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected PaddleOCR result JSON: {type(payload)!r}")
    return payload


def locate_recognition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("res", payload)
    if not isinstance(candidate, dict):
        raise TypeError("PaddleOCR result does not contain an object payload")
    return candidate


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def extract_lines(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    result = locate_recognition_payload(payload)
    texts = [str(item).strip() for item in as_list(result.get("rec_texts"))]
    scores = [float(item) for item in as_list(result.get("rec_scores"))]
    return [text for text in texts if text], scores


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower()
    return " ".join(normalized.split())


def vietnamese_diacritic_count(text: str) -> int:
    decomposed = unicodedata.normalize("NFD", text)
    return sum(unicodedata.combining(char) > 0 for char in decomposed)


def mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return round(statistics.fmean(materialized), 6) if materialized else None


def write_reports(records: list[dict[str, Any]], elapsed: float) -> None:
    pp_diacritics = sum(record["ppocrv6"]["vietnamese_diacritic_count"] for record in records)
    current_diacritics = sum(
        record["current_corpus"]["vietnamese_diacritic_count"] for record in records
    )
    summary = {
        "schema_version": "1.0",
        "status": "diagnostic_only_no_human_ground_truth",
        "engine": "PaddleOCR 3.7.0 / PP-OCRv6",
        "device": paddle.device.get_device(),
        "page_count": len(records),
        "total_runtime_seconds": round(elapsed, 3),
        "mean_runtime_seconds": mean_or_none(r["runtime_seconds"] for r in records),
        "mean_confidence": mean_or_none(
            score for record in records for score in record["ppocrv6"]["recognition_scores"]
        ),
        "ppocrv6_diacritic_count": pp_diacritics,
        "current_corpus_diacritic_count": current_diacritics,
        "ppocrv6_to_current_diacritic_ratio": round(pp_diacritics / max(1, current_diacritics), 6),
        "pilot_decision": "do_not_replace_current_vietnamese_recognizer",
        "pages": records,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# PP-OCRv6 10-page diagnostic pilot",
        "",
        "> This is a diagnostic comparison without typed human ground truth. "
        "Similarity and confidence are not CER/WER or proof of accuracy.",
        "",
        f"- Device: `{summary['device']}`",
        f"- Pages completed: **{len(records)}**",
        f"- Total runtime: **{summary['total_runtime_seconds']} s**",
        f"- Mean page runtime: **{summary['mean_runtime_seconds']} s**",
        f"- Mean PP-OCRv6 confidence: **{summary['mean_confidence']}**",
        f"- Vietnamese diacritic marks: **{pp_diacritics}** (PP-OCRv6) vs "
        f"**{current_diacritics}** (current corpus)",
        f"- Diacritic ratio vs current corpus: "
        f"**{summary['ppocrv6_to_current_diacritic_ratio']:.1%}**",
        "- Pilot decision: **do not replace the current Vietnamese recognizer**",
        "",
        "## Findings",
        "",
        "PP-OCRv6 produced cleaner line detection and removed much scan-border noise, "
        "but systematically dropped Vietnamese tone marks and occasionally letters on "
        "all inspected document types. Its mean recognition confidence remained high, "
        "so confidence is not a safe quality gate for this failure mode.",
        "",
        "The current Tesseract path is noisier on borders and marginal marks, but "
        "preserves substantially more Vietnamese orthography. PP-OCRv6 may still be "
        "evaluated later as a detector/layout component, or after Vietnamese-specific "
        "recognizer fine-tuning, but this pilot rejects it as a drop-in text recognizer.",
        "",
        "| Subject | PDF page | Category | PP chars | Current chars | "
        "Similarity | PP confidence | Runtime |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        pp = record["ppocrv6"]
        current = record["current_corpus"]
        lines.append(
            f"| {record['subject']} | {record['pdf_page_human']} | "
            f"{record['category']} | {pp['char_count']} | {current['char_count']} | "
            f"{record['normalized_similarity']:.3f} | "
            f"{pp['mean_confidence'] or 0:.3f} | {record['runtime_seconds']:.2f} s |"
        )
    lines.extend(["", "## Text samples", ""])
    for record in records:
        lines.extend(
            [
                f"### {record['subject']} - PDF page {record['pdf_page_human']}",
                "",
                "**PP-OCRv6**",
                "",
                "```text",
                record["ppocrv6"]["text"][:1800],
                "```",
                "",
                "**Current corpus (Tesseract path)**",
                "",
                "```text",
                record["current_corpus"]["text"][:1800],
                "```",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    portfolio = PAGE_PORTFOLIO[: max(0, min(args.limit, len(PAGE_PORTFOLIO)))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        lang="vi",
        ocr_version="PP-OCRv6",
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, selection in enumerate(portfolio, start=1):
        subject = selection["subject"]
        pdf_page = selection["pdf_page"]
        stem = f"{subject}_p{pdf_page + 1:03d}"
        image_path = OUTPUT_DIR / "images" / f"{stem}.png"
        raw_path = OUTPUT_DIR / "raw" / f"{stem}.json"
        render_page(find_pdf(subject), pdf_page, image_path, args.scale)

        page_started = time.perf_counter()
        results = list(ocr.predict(str(image_path)))
        runtime = time.perf_counter() - page_started
        if len(results) != 1:
            raise RuntimeError(f"Expected one OCR result for {image_path}, got {len(results)}")
        payload = result_payload(results[0])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pp_lines, scores = extract_lines(payload)
        pp_text = "\n".join(pp_lines)
        current = load_canonical_page(subject, pdf_page)
        current_text = str(current.get("text", ""))
        records.append(
            {
                "case_id": stem.lower(),
                "subject": subject,
                "pdf_page_zero_based": pdf_page,
                "pdf_page_human": pdf_page + 1,
                "category": selection["category"],
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "raw_result_path": raw_path.relative_to(ROOT).as_posix(),
                "runtime_seconds": round(runtime, 3),
                "normalized_similarity": round(
                    SequenceMatcher(
                        None, normalize_text(pp_text), normalize_text(current_text)
                    ).ratio(),
                    6,
                ),
                "ppocrv6": {
                    "text": pp_text,
                    "line_count": len(pp_lines),
                    "char_count": len(pp_text),
                    "vietnamese_diacritic_count": vietnamese_diacritic_count(pp_text),
                    "recognition_scores": scores,
                    "mean_confidence": mean_or_none(scores),
                },
                "current_corpus": {
                    "text": current_text,
                    "line_count": len(current_text.splitlines()),
                    "char_count": len(current_text),
                    "vietnamese_diacritic_count": vietnamese_diacritic_count(current_text),
                    "quality_score": current.get("quality_score"),
                    "extraction_method": current.get("extraction_method"),
                },
            }
        )
        write_reports(records, time.perf_counter() - started)
        print(
            f"[{index}/{len(portfolio)}] {subject} PDF page {pdf_page + 1}: "
            f"{len(pp_text)} chars, {runtime:.2f}s"
        )

    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
