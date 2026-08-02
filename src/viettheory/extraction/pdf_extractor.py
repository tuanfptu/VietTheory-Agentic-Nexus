"""Extract text and source coordinates from text-layer PDFs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pymupdf

from viettheory.ids import stable_id
from viettheory.schema import BoundingBox, Document, ExtractionMethod, Page, TextBlock, TextLine


def _bbox(
    values: list[float] | tuple[float, ...],
    *,
    width: float,
    height: float,
) -> BoundingBox:
    """Normalize a PyMuPDF rectangle to an immutable four-value tuple."""
    if len(values) != 4:
        raise ValueError(f"Expected four bounding-box coordinates, received {len(values)}")
    x0, y0, x1, y1 = (float(value) for value in values)
    return (
        max(0.0, min(x0, width)),
        max(0.0, min(y0, height)),
        max(0.0, min(x1, width)),
        max(0.0, min(y1, height)),
    )


def _line_text(line: dict[str, Any]) -> str:
    """Combine spans without discarding meaningful intra-line spacing."""
    return "".join(str(span.get("text", "")) for span in line.get("spans", ())).strip()


def _extract_blocks(
    raw_page: dict[str, Any],
    *,
    page_id: str,
    width: float,
    height: float,
) -> tuple[TextBlock, ...]:
    blocks: list[TextBlock] = []
    for raw_block in raw_page.get("blocks", ()):
        if raw_block.get("type") != 0:
            continue

        raw_block_number = int(raw_block.get("number", len(blocks)))
        lines: list[TextLine] = []
        for line_index, raw_line in enumerate(raw_block.get("lines", ())):
            text = _line_text(raw_line)
            if not text:
                continue
            spans = raw_line.get("spans", ())
            font_sizes = [float(span["size"]) for span in spans if span.get("size")]
            font_flags = tuple(sorted({int(span.get("flags", 0)) for span in spans}))
            lines.append(
                TextLine(
                    line_id=stable_id("line", page_id, raw_block_number, line_index),
                    bbox=_bbox(raw_line["bbox"], width=width, height=height),
                    text=text,
                    font_size=max(font_sizes) if font_sizes else None,
                    font_flags=font_flags,
                )
            )
        if not lines:
            continue

        blocks.append(
            TextBlock(
                block_id=stable_id("block", page_id, raw_block_number),
                bbox=_bbox(raw_block["bbox"], width=width, height=height),
                text="\n".join(line.text for line in lines),
                lines=tuple(lines),
            )
        )
    return tuple(blocks)


_PAGE_NUMBER = re.compile(r"(?:[ivxlcdm]+|\d{1,4})", re.IGNORECASE)


def _printed_page(page: pymupdf.Page, blocks: tuple[TextBlock, ...]) -> str | None:
    """Infer printed identity from a non-default label or page-margin text."""
    label = str(page.get_label()).strip()  # type: ignore[no-untyped-call]
    if not label or label == str(page.number + 1):
        page_height = float(page.rect.height)
        margin_candidates = (
            line.text
            for block in blocks
            for line in block.lines
            if line.bbox[1] <= page_height * 0.15 or line.bbox[3] >= page_height * 0.85
        )
        return next((text for text in margin_candidates if _PAGE_NUMBER.fullmatch(text)), None)
    return str(label)


def _sha256(path: Path) -> str:
    """Hash a source document without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality(text: str) -> tuple[float, bool]:
    """Score native extraction and flag pages that should enter the OCR gate."""
    char_count = len(text)
    if not text:
        return 0.0, True
    replacement_ratio = text.count("\ufffd") / char_count
    length_score = min(char_count / 500.0, 1.0)
    quality_score = round(length_score * (1.0 - replacement_ratio), 4)
    return quality_score, char_count < 100 or replacement_ratio > 0.05


def read_document(pdf_path: str | Path, subject_code: str) -> Document:
    """Inspect immutable source identity and page count."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    with pymupdf.open(path) as source:  # type: ignore[no-untyped-call]
        page_count = len(source)
    return Document(
        document_id=stable_id("doc", digest),
        file_name=path.name,
        subject_code=subject_code,
        sha256=digest,
        page_count=page_count,
    )


def iter_pdf_pages(
    pdf_path: str | Path,
    subject_code: str,
    *,
    start_page: int = 0,
    end_page: int | None = None,
) -> Iterator[Page]:
    """Yield citation-ready pages from a PDF using zero-based page indices.

    ``end_page`` is exclusive. Image-only pages remain in the output with empty
    text and ``extraction_method='none'`` so downstream OCR can handle them.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if start_page < 0:
        raise ValueError("start_page must be non-negative")

    source_document = read_document(path, subject_code)
    document_id = source_document.document_id

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        stop = len(document) if end_page is None else min(end_page, len(document))
        if stop < start_page:
            raise ValueError("end_page must not be smaller than start_page")

        for page_index in range(start_page, stop):
            source_page = document[page_index]
            width = float(source_page.rect.width)
            height = float(source_page.rect.height)
            page_id = stable_id("page", document_id, page_index)
            raw_page = source_page.get_text("dict", sort=True)
            blocks = _extract_blocks(
                raw_page,
                page_id=page_id,
                width=width,
                height=height,
            )
            text = "\n\n".join(block.text for block in blocks)
            quality_score, needs_ocr = _quality(text)
            yield Page(
                page_id=page_id,
                document_id=document_id,
                pdf_file=path.name,
                subject_code=subject_code,
                pdf_page=page_index,
                printed_page=_printed_page(source_page, blocks),
                width=width,
                height=height,
                rotation=cast("Any", int(source_page.rotation)),
                text=text,
                extraction_method=(ExtractionMethod.PYMUPDF if text else ExtractionMethod.NONE),
                char_count=len(text),
                quality_score=quality_score,
                needs_ocr=needs_ocr,
                image_count=sum(
                    1 for block in raw_page.get("blocks", ()) if block.get("type") == 1
                ),
                blocks=blocks,
            )


def extract_pdf(
    pdf_path: str | Path,
    subject_code: str,
    *,
    start_page: int = 0,
    end_page: int | None = None,
) -> list[Page]:
    """Extract a bounded PDF range into memory."""
    return list(
        iter_pdf_pages(
            pdf_path,
            subject_code,
            start_page=start_page,
            end_page=end_page,
        )
    )
