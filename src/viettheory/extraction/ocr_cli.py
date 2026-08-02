"""Command-line OCR extraction for scanned PDFs with Tesseract provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from viettheory.extraction.pdf_extractor import read_document
from viettheory.extraction.tesseract_ocr import TesseractOcr, iter_ocr_pages
from viettheory.schema import ExtractionManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--tesseract",
        type=Path,
        default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    )
    parser.add_argument("--tessdata", type=Path, default=Path("models/tesseract"))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--end-page", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = TesseractOcr(
        args.tesseract,
        args.tessdata,
        scale=args.scale,
        page_segmentation_mode=args.psm,
    )
    document = read_document(args.pdf, args.subject)
    stop = min(args.end_page or document.page_count, document.page_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for page in iter_ocr_pages(
            args.pdf,
            args.subject,
            engine,
            start_page=args.start_page,
            end_page=stop,
        ):
            output.write(json.dumps(page.model_dump(mode="json"), ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            print(
                f"{args.subject} page {page.pdf_page + 1}/{stop}: "
                f"{page.char_count} chars, quality={page.quality_score:.3f}",
                flush=True,
            )

    if count == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError("Selected page range produced no OCR pages")

    manifest = ExtractionManifest(
        document=document,
        extractor="tesseract",
        extractor_version="5.4.0.20240606",
        postprocessors=("tessdata_best/vie", f"scale={args.scale}", f"psm={args.psm}"),
        artifact_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        start_page=args.start_page,
        end_page=stop,
        extracted_page_count=count,
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {count} OCR pages and {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
