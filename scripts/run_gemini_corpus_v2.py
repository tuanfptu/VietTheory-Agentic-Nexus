"""Run resumable Gemini correction/structure batches across the five-subject corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from viettheory.benchmark_generation import load_gemini_key
from viettheory.extraction.gemini_corpus_v2 import (
    CorrectedStructureBatch,
    GeminiCorpusV2Client,
)
from viettheory.extraction.gemini_structure import GeminiStructureError
from viettheory.schema import Page

OCR_SUBJECTS = ("MLN131", "HCM202", "VNR202")
NATIVE_SUBJECTS = ("MLN111", "MLN122")


@dataclass(frozen=True, slots=True)
class SubjectInput:
    code: str
    pdf: Path
    pages_jsonl: Path
    selected_pages: tuple[int, ...]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pages(path: Path) -> dict[int, Page]:
    return {
        page.pdf_page: page
        for page in (
            Page.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _native_audit_selection(pages: dict[int, Page], target: int) -> tuple[int, ...]:
    """Select heading-like and evenly spaced native pages deterministically."""
    candidates = [
        number
        for number, page in pages.items()
        if any(
            len(block.text) <= 180
            and (
                block.text.strip().casefold().startswith("chương ")
                or block.text.strip()[:3].rstrip(".").isnumeric()
                or block.text.strip().startswith(("I.", "II.", "III.", "IV."))
            )
            for block in page.blocks
        )
    ]
    selected = list(dict.fromkeys(candidates))[:target]
    if len(selected) < target:
        ordered = sorted(pages)
        step = max(1, len(ordered) // (target - len(selected)))
        for number in ordered[::step]:
            if number not in selected:
                selected.append(number)
            if len(selected) == target:
                break
    return tuple(sorted(selected[:target]))


def _subject_inputs(
    root: Path,
    native_pages_per_subject: int,
    selection_path: Path | None = None,
) -> tuple[SubjectInput, ...]:
    documents = root / "Tài liệu"
    selected_by_subject: dict[str, tuple[int, ...]] = {}
    if selection_path is not None:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[int]] = {
            subject: [] for subject in (*OCR_SUBJECTS, *NATIVE_SUBJECTS)
        }
        for row in selection["selected_pages"]:
            grouped[row["subject"]].append(row["pdf_page"])
        selected_by_subject = {
            subject: tuple(sorted(dict.fromkeys(numbers))) for subject, numbers in grouped.items()
        }
    inputs: list[SubjectInput] = []
    for code in (*OCR_SUBJECTS, *NATIVE_SUBJECTS):
        pages_path = root / "data" / "processed" / code / "pages.jsonl"
        pages = _load_pages(pages_path)
        selected = selected_by_subject.get(code, tuple(sorted(pages)))
        if selection_path is None and code in NATIVE_SUBJECTS:
            selected = _native_audit_selection(pages, native_pages_per_subject)
        pdf = next(documents.glob(f"*{code}.pdf"))
        inputs.append(
            SubjectInput(
                code=code,
                pdf=pdf,
                pages_jsonl=pages_path,
                selected_pages=selected,
            )
        )
    return tuple(inputs)


def _chunks(values: tuple[int, ...], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(values[start : start + size]) for start in range(0, len(values), size))


def _render(document: pymupdf.Document, pdf_page: int, dpi: int) -> bytes:
    return document[pdf_page].get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def _batch_path(output_root: Path, code: str, batch: tuple[int, ...]) -> Path:
    joined = "_".join(f"{number:04d}" for number in batch)
    return output_root / code / "batches" / f"pages_{joined}.json"


def _write_batch(path: Path, result: CorrectedStructureBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(result.model_dump(mode="json")) + "\n", encoding="utf-8")


def _completed_page_numbers(output_root: Path, code: str) -> set[int]:
    completed: set[int] = set()
    batches_dir = output_root / code / "batches"
    if not batches_dir.exists():
        return completed
    for path in batches_dir.glob("pages_*.json"):
        batch = CorrectedStructureBatch.model_validate_json(path.read_text(encoding="utf-8"))
        for page in batch.pages:
            if page.pdf_page in completed:
                raise RuntimeError(f"duplicate cached page {code}:{page.pdf_page}")
            completed.add(page.pdf_page)
    return completed


def _append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(event) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--native-pages-per-subject", type=int, default=50)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--requests-per-minute", type=float, default=2.0)
    parser.add_argument("--max-requests", type=int, default=220)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 5:
        parser.error("batch-size must be between 1 and 5")
    if args.requests_per_minute <= 0 or args.max_requests < 1 or args.dpi < 72:
        parser.error("rate, request budget and dpi must be positive")

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another corpus-v2 scheduler may be running; inspect {lock_path}"
        ) from exc
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    try:
        return _run(args, root, output_root)
    finally:
        lock_path.unlink(missing_ok=True)


def _run(args: argparse.Namespace, root: Path, output_root: Path) -> int:
    """Execute one locked scheduler run."""
    inputs = _subject_inputs(root, args.native_pages_per_subject, args.selection)
    if args.calibration_only:
        vnr = next(item for item in inputs if item.code == "VNR202")
        inputs = (
            SubjectInput(
                code=vnr.code,
                pdf=vnr.pdf,
                pages_jsonl=vnr.pages_jsonl,
                selected_pages=(21, 22, 23, 24, 25),
            ),
        )

    api_key = load_gemini_key(args.dotenv)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    client = GeminiCorpusV2Client(
        api_key=api_key,
        model=args.model,
        max_retries=0,
    )
    events_path = output_root / "run_events.jsonl"
    interval = 60.0 / args.requests_per_minute
    requests = 0
    last_request: float | None = None
    completed_pages = 0
    failed_singletons: list[tuple[str, int, str]] = []

    for subject in inputs:
        pages = _load_pages(subject.pages_jsonl)
        completed_for_subject = _completed_page_numbers(output_root, subject.code)
        queue = deque(_chunks(subject.selected_pages, args.batch_size))
        with pymupdf.open(subject.pdf) as document:
            while queue:
                batch = queue.popleft()
                batch = tuple(number for number in batch if number not in completed_for_subject)
                if not batch:
                    continue
                output_path = _batch_path(output_root, subject.code, batch)
                if output_path.exists():
                    cached = CorrectedStructureBatch.model_validate_json(
                        output_path.read_text(encoding="utf-8")
                    )
                    if tuple(page.pdf_page for page in cached.pages) != batch:
                        raise RuntimeError(f"invalid cached page coverage: {output_path}")
                    completed_pages += len(batch)
                    completed_for_subject.update(batch)
                    print(f"cache hit: {subject.code} {batch}", flush=True)
                    continue
                if requests >= args.max_requests:
                    print(f"request budget reached: {requests}/{args.max_requests}", flush=True)
                    break
                if last_request is not None:
                    remaining = interval - (time.monotonic() - last_request)
                    if remaining > 0:
                        time.sleep(remaining)
                targets = tuple(pages[number] for number in batch)
                context_number = batch[0] - 1
                context = (pages[context_number],) if context_number in pages else ()
                last_request = time.monotonic()
                requests += 1
                try:
                    result = client.analyze_batch(
                        targets,
                        tuple(_render(document, number, args.dpi) for number in batch),
                        context_pages=context,
                        context_images_png=tuple(
                            _render(document, page.pdf_page, args.dpi) for page in context
                        ),
                        structure_only=args.structure_only,
                    )
                except GeminiStructureError as exc:
                    _append_event(
                        events_path,
                        {
                            "event": "batch_rejected",
                            "subject": subject.code,
                            "pages": list(batch),
                            "error": str(exc),
                            "request_number": requests,
                        },
                    )
                    if len(batch) == 1:
                        failed_singletons.append((subject.code, batch[0], str(exc)))
                    else:
                        midpoint = (len(batch) + 1) // 2
                        queue.appendleft(batch[midpoint:])
                        queue.appendleft(batch[:midpoint])
                    print(f"rejected/split: {subject.code} {batch}: {exc}", flush=True)
                    continue
                _write_batch(output_path, result)
                completed_pages += len(batch)
                completed_for_subject.update(batch)
                _append_event(
                    events_path,
                    {
                        "event": "batch_completed",
                        "subject": subject.code,
                        "pages": list(batch),
                        "request_number": requests,
                    },
                )
                print(
                    f"checkpoint: {subject.code} {batch}; requests={requests}/{args.max_requests}",
                    flush=True,
                )
            if requests >= args.max_requests:
                break

    manifest = {
        "schema_version": "1.0",
        "pipeline_version": "gemini_corrected_structured_v2_batch5",
        "model": args.model,
        "batch_size": args.batch_size,
        "context_pages": 1,
        "dpi": args.dpi,
        "request_cap": args.max_requests,
        "structure_only": args.structure_only,
        "requests_this_run": requests,
        "completed_pages_observed_this_run": completed_pages,
        "failed_singletons": [
            {"subject": subject, "pdf_page": page, "error": error}
            for subject, page, error in failed_singletons
        ],
        "sources": {
            item.code: {
                "pdf_sha256": _sha256(item.pdf),
                "pages_sha256": _sha256(item.pages_jsonl),
                "selected_page_count": len(item.selected_pages),
            }
            for item in inputs
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"run complete: requests={requests}; completed_pages={completed_pages}; "
        f"failed_singletons={len(failed_singletons)}",
        flush=True,
    )
    return 0 if not failed_singletons else 2


if __name__ == "__main__":
    raise SystemExit(main())
