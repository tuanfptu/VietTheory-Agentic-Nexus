"""Render selected source pages with block and line bounding-box overlays."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pymupdf

from viettheory.schema import Page


def _parse_pages(value: str) -> tuple[int, ...]:
    pages = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not pages or any(page < 0 for page in pages):
        raise argparse.ArgumentTypeError("pages must be comma-separated non-negative indices")
    return pages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("pages_jsonl", type=Path)
    parser.add_argument("--pages", required=True, type=_parse_pages)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = {
        page.pdf_page: page
        for line in args.pages_jsonl.read_text(encoding="utf-8").splitlines()
        if (page := Page.model_validate_json(line)).pdf_page in args.pages
    }
    missing = set(args.pages) - records.keys()
    if missing:
        raise ValueError(f"Missing extracted page records: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = pymupdf.open()  # type: ignore[no-untyped-call]
    with pymupdf.open(args.pdf) as source:  # type: ignore[no-untyped-call]
        for page_index in args.pages:
            record = records[page_index]
            source_page = source[page_index]
            target = output.new_page(width=record.width, height=record.height)
            source_pixmap = source_page.get_pixmap(alpha=False)
            target.insert_image(target.rect, pixmap=source_pixmap)
            for block in record.blocks:
                target.draw_rect(block.bbox, color=(1.0, 0.0, 0.0), width=0.8, overlay=True)
                for line in block.lines:
                    target.draw_rect(line.bbox, color=(0.0, 0.3, 1.0), width=0.25, overlay=True)

    output.save(args.output, garbage=4, deflate=True)  # type: ignore[no-untyped-call]
    output.close()  # type: ignore[no-untyped-call]
    print(f"Rendered {len(args.pages)} overlay pages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
