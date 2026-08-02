"""Command-line interface for citation-preserving PDF extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pymupdf

from viettheory import __version__
from viettheory.extraction.pdf_extractor import iter_pdf_pages, read_document
from viettheory.schema import ExtractionManifest


def build_parser() -> argparse.ArgumentParser:
    """Build the extraction argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("--subject", required=True, help="Subject code, for example MLN111")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--manifest", type=Path, help="Companion manifest JSON path")
    parser.add_argument("--start-page", type=int, default=0, help="Zero-based first page")
    parser.add_argument("--end-page", type=int, help="Exclusive zero-based end page")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Extract pages to UTF-8 JSONL without loading the full document into memory."""
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    page_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for page in iter_pdf_pages(
            args.pdf,
            args.subject,
            start_page=args.start_page,
            end_page=args.end_page,
        ):
            output.write(json.dumps(page.model_dump(mode="json"), ensure_ascii=False) + "\n")
            page_count += 1

    if page_count == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError("Selected page range produced no pages")

    document = read_document(args.pdf, args.subject)
    artifact_digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    stop = min(args.end_page or document.page_count, document.page_count)
    manifest = ExtractionManifest(
        document=document,
        extractor_version=pymupdf.__version__,
        artifact_sha256=artifact_digest,
        start_page=args.start_page,
        end_page=stop,
        extracted_page_count=page_count,
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"Extracted {page_count} pages to {args.output} "
        f"(manifest: {manifest_path}, app: {__version__})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
