"""Generate a cached Gemini-assisted VNR202 structure pilot for selected pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pymupdf

from viettheory.benchmark_generation import load_gemini_key
from viettheory.extraction.gemini_structure import (
    GeminiPageStructure,
    GeminiStructureClient,
    validate_page_anchors,
)
from viettheory.extraction.structure_parser import parse_structure
from viettheory.schema import Page

DEFAULT_PAGES = (21, 68, 124, 180, 225)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_pages(path: Path) -> dict[int, Page]:
    return {
        page.pdf_page: page
        for page in (
            Page.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _render_png(document: pymupdf.Document, pdf_page: int, dpi: int) -> bytes:
    page = document[pdf_page]
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return pixmap.tobytes("png")


def _canonical_json(record: GeminiPageStructure) -> str:
    return json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pages-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", type=int, nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--requests-per-minute", type=float, default=4.0)
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.dpi < 72 or args.requests_per_minute <= 0 or args.max_requests < 1:
        parser.error("dpi must be >=72 and request budgets must be positive")
    selected = tuple(dict.fromkeys(args.pages))
    pages = _read_pages(args.pages_jsonl)
    missing = set(selected).difference(pages)
    if missing:
        parser.error(f"pages missing from JSONL: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "pages"
    records_dir.mkdir(parents=True, exist_ok=True)
    uncached = [
        pdf_page
        for pdf_page in selected
        if args.refresh or not (records_dir / f"page_{pdf_page:04d}.json").exists()
    ]
    if len(uncached) > args.max_requests:
        parser.error("selected uncached pages cannot exceed --max-requests")
    client: GeminiStructureClient | None = None
    if uncached:
        api_key = load_gemini_key(args.dotenv)
        if not api_key:
            parser.error("GEMINI_API_KEY is missing from the environment or selected dotenv file")
        client = GeminiStructureClient(
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    interval = 60.0 / args.requests_per_minute
    last_request: float | None = None
    requested = 0
    results: list[GeminiPageStructure] = []
    with pymupdf.open(args.pdf) as document:
        for pdf_page in selected:
            output_path = records_dir / f"page_{pdf_page:04d}.json"
            if output_path.exists() and not args.refresh:
                result = validate_page_anchors(
                    GeminiPageStructure.model_validate_json(
                        output_path.read_text(encoding="utf-8")
                    ),
                    pages[pdf_page],
                )
                output_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
                results.append(result)
                print(f"cache hit: pdf_page={pdf_page}")
                continue
            if requested >= args.max_requests:
                raise RuntimeError("request budget exhausted")
            if client is None:
                raise RuntimeError("Gemini client was not initialized")
            if last_request is not None:
                remaining = interval - (time.monotonic() - last_request)
                if remaining > 0:
                    time.sleep(remaining)
            last_request = time.monotonic()
            requested += 1
            result = client.analyze_page(
                pages[pdf_page],
                _render_png(document, pdf_page, args.dpi),
            )
            output_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
            results.append(result)
            print(f"checkpoint: pdf_page={pdf_page}; requests={requested}/{args.max_requests}")

    selected_pages = tuple(pages[pdf_page] for pdf_page in selected)
    rule_structure = parse_structure(selected_pages)
    rule_by_page: dict[int, list[dict[str, object]]] = {page: [] for page in selected}
    for heading in rule_structure.headings:
        rule_by_page[heading.pdf_page].append(
            {"level": heading.level, "text": heading.text, "block_id": heading.block_id}
        )
    comparison = {
        "schema_version": "1.0",
        "status": "pending_human_review",
        "scope": "five_page_gemini_structure_pilot",
        "subject": "VNR202",
        "model": args.model,
        "source_pdf_sha256": _sha256(args.pdf),
        "source_pages_sha256": _sha256(args.pages_jsonl),
        "pages": [
            {
                "pdf_page": result.pdf_page,
                "rule_based_headings": rule_by_page[result.pdf_page],
                "gemini_page_role": result.page_role,
                "gemini_elements": [element.model_dump(mode="json") for element in result.elements],
                "human_review": {
                    "page_role": "pending",
                    "heading_detection": "pending",
                    "heading_level": "pending",
                    "ocr_correction": "pending",
                    "notes": "",
                },
            }
            for result in results
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "pipeline_version": "gemini_structure_pilot_v1",
        "model": args.model,
        "dpi": args.dpi,
        "selected_pdf_pages_zero_based": list(selected),
        "source_pdf_sha256": _sha256(args.pdf),
        "source_pages_sha256": _sha256(args.pages_jsonl),
        "comparison_sha256": _sha256(args.output_dir / "comparison.json"),
        "page_record_sha256": {
            str(result.pdf_page): _sha256(records_dir / f"page_{result.pdf_page:04d}.json")
            for result in results
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote validated pilot to {args.output_dir}; API key was not persisted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
