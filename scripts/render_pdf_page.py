"""Render one zero-based PDF page to PNG for visual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # type: ignore[import-untyped]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("page", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.page < 0:
        parser.error("page must be zero-based and non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(args.pdf) as document:
        if args.page >= document.page_count:
            parser.error(f"page exceeds document page count ({document.page_count})")
        pixmap = document[args.page].get_pixmap(dpi=args.dpi, alpha=False)
        pixmap.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
