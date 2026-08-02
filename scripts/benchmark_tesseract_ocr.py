"""Benchmark citation-preserving Tesseract OCR on representative scanned pages."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path

import pymupdf

from viettheory.extraction.tesseract_ocr import TesseractOcr
from viettheory.ids import stable_id


def _marked_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    marked = sum(
        bool(unicodedata.combining(part))
        for character in letters
        for part in unicodedata.normalize("NFD", character)[1:]
    )
    return marked / max(len(letters), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("Tài liệu"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/ocr_tesseract_sample.json"))
    parser.add_argument(
        "--tesseract",
        type=Path,
        default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    )
    parser.add_argument("--tessdata", type=Path, default=Path("models/tesseract"))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--pages-per-document", type=int, default=5)
    return parser


def _sample_pages(page_count: int, sample_count: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("pages-per-document must be positive")
    fractions = (0.05, 0.2, 0.4, 0.65, 0.9)
    return sorted(
        {
            min(page_count - 1, max(0, round((page_count - 1) * fraction)))
            for fraction in fractions[:sample_count]
        }
    )


def main() -> int:
    args = build_parser().parse_args()
    engine = TesseractOcr(args.tesseract, args.tessdata, scale=args.scale)
    records: list[dict[str, object]] = []
    elapsed_values: list[float] = []

    for subject in ("HCM202", "MLN131", "VNR202"):
        pdf_path = next(args.pdf_dir.glob(f"*{subject}.pdf"))
        with pymupdf.open(pdf_path) as source:  # type: ignore[no-untyped-call]
            for page_index in _sample_pages(len(source), args.pages_per_document):
                start = time.perf_counter()
                blocks, confidence = engine.recognize(
                    source[page_index],
                    page_id=stable_id("benchmark-page", subject, page_index),
                )
                elapsed = time.perf_counter() - start
                elapsed_values.append(elapsed)
                text = "\n\n".join(block.text for block in blocks)
                records.append(
                    {
                        "subject_code": subject,
                        "pdf_page": page_index,
                        "printed_page_estimate": page_index + 1,
                        "seconds": round(elapsed, 3),
                        "characters": len(text),
                        "blocks": len(blocks),
                        "lines": sum(len(block.lines) for block in blocks),
                        "mean_confidence": round(confidence, 4),
                        "marked_letter_ratio": round(_marked_ratio(text), 4),
                        "text_preview": text[:240],
                    }
                )
                print(
                    f"{subject} p{page_index}: {elapsed:.2f}s, "
                    f"{len(text)} chars, conf={confidence:.3f}"
                )

    payload = {
        "engine": "tesseract",
        "engine_version": "5.4.0.20240606",
        "language_model": "tessdata_best/vie",
        "scale": args.scale,
        "dpi": round(72 * args.scale),
        "page_segmentation_mode": 6,
        "sample_count": len(records),
        "mean_seconds_per_page": round(statistics.mean(elapsed_values), 3),
        "median_seconds_per_page": round(statistics.median(elapsed_values), 3),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
