"""Tesseract OCR fallback with citation-preserving PDF coordinates."""

from __future__ import annotations

import csv
import io
import subprocess
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TypeAlias

import pymupdf

from viettheory.extraction.pdf_extractor import read_document
from viettheory.ids import stable_id
from viettheory.schema import ExtractionMethod, Page, TextBlock, TextLine

CommandRunner: TypeAlias = Callable[[Sequence[str], bytes], str]
_LineKey: TypeAlias = tuple[int, int, int]


def _default_runner(command: Sequence[str], image: bytes) -> str:
    completed = subprocess.run(
        command,
        input=image,
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode("utf-8")


def _union(boxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def parse_tesseract_tsv(
    tsv: str,
    *,
    page_id: str,
    pixels_per_point: float,
    min_confidence: float = 0.0,
    page_width: float | None = None,
    page_height: float | None = None,
) -> tuple[tuple[TextBlock, ...], float]:
    """Convert word-level Tesseract TSV into ordered PDF-coordinate blocks."""
    if pixels_per_point <= 0:
        raise ValueError("pixels_per_point must be positive")

    grouped: dict[_LineKey, list[tuple[int, str, float, tuple[float, float, float, float]]]]
    grouped = defaultdict(list)
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        confidence = float(row["conf"])
        if confidence < min_confidence:
            continue
        left, top = int(row["left"]), int(row["top"])
        width, height = int(row["width"]), int(row["height"])
        x0, y0 = left / pixels_per_point, top / pixels_per_point
        x1, y1 = (left + width) / pixels_per_point, (top + height) / pixels_per_point
        if page_width is not None:
            x0, x1 = max(0.0, min(x0, page_width)), max(0.0, min(x1, page_width))
        if page_height is not None:
            y0, y1 = max(0.0, min(y0, page_height)), max(0.0, min(y1, page_height))
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        grouped[key].append(
            (
                int(row["word_num"]),
                text,
                confidence,
                (x0, y0, x1, y1),
            )
        )

    blocks: list[TextBlock] = []
    confidences: list[float] = []
    by_block: dict[int, list[TextLine]] = defaultdict(list)
    for key in sorted(grouped):
        words = sorted(grouped[key])
        boxes = [word[3] for word in words]
        confidences.extend(word[2] for word in words)
        block_number, paragraph_number, line_number = key
        by_block[block_number].append(
            TextLine(
                line_id=stable_id(
                    "line", page_id, "ocr", block_number, paragraph_number, line_number
                ),
                bbox=_union(boxes),
                text=" ".join(word[1] for word in words),
            )
        )

    for block_number in sorted(by_block):
        lines = by_block[block_number]
        blocks.append(
            TextBlock(
                block_id=stable_id("block", page_id, "ocr", block_number),
                bbox=_union([line.bbox for line in lines]),
                text="\n".join(line.text for line in lines),
                lines=tuple(lines),
            )
        )
    mean_confidence = sum(confidences) / (100.0 * len(confidences)) if confidences else 0.0
    return tuple(blocks), mean_confidence


def _quality(text: str, mean_confidence: float) -> tuple[float, bool]:
    letters = [character for character in text if character.isalpha()]
    marked = sum(
        bool(unicodedata.combining(part))
        for character in letters
        for part in unicodedata.normalize("NFD", character)[1:]
    )
    marked_ratio = marked / max(len(letters), 1)
    length_score = min(len(text) / 500.0, 1.0)
    language_score = min(marked_ratio / 0.08, 1.0)
    quality = round(length_score * mean_confidence * language_score, 4)
    return quality, len(text) < 100 or quality < 0.45


class TesseractOcr:
    """Run a pinned Tesseract binary and Vietnamese model without Python bindings."""

    def __init__(
        self,
        executable: str | Path,
        tessdata_dir: str | Path,
        *,
        language: str = "vie",
        scale: float = 2.0,
        page_segmentation_mode: int = 6,
        runner: CommandRunner = _default_runner,
    ) -> None:
        self.executable = Path(executable)
        self.tessdata_dir = Path(tessdata_dir)
        self.language = language
        self.scale = scale
        self.page_segmentation_mode = page_segmentation_mode
        self._runner = runner

    def recognize(self, page: pymupdf.Page, *, page_id: str) -> tuple[tuple[TextBlock, ...], float]:
        """Recognize one page and map pixel boxes back to PDF points."""
        matrix = pymupdf.Matrix(self.scale, self.scale)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        command = (
            str(self.executable),
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(self.tessdata_dir),
            "-l",
            self.language,
            "--psm",
            str(self.page_segmentation_mode),
            "-c",
            "tessedit_create_tsv=1",
        )
        image = pixmap.tobytes("png")  # type: ignore[no-untyped-call]
        tsv = self._runner(command, image)
        return parse_tesseract_tsv(
            tsv,
            page_id=page_id,
            pixels_per_point=self.scale,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
        )


def iter_ocr_pages(
    pdf_path: str | Path,
    subject_code: str,
    engine: TesseractOcr,
    *,
    start_page: int = 0,
    end_page: int | None = None,
) -> Iterator[Page]:
    """Yield OCR-backed pages for a bounded, zero-based PDF range."""
    path = Path(pdf_path)
    document = read_document(path, subject_code)
    with pymupdf.open(path) as source:  # type: ignore[no-untyped-call]
        stop = len(source) if end_page is None else min(end_page, len(source))
        if start_page < 0 or stop < start_page:
            raise ValueError("invalid OCR page range")
        for page_index in range(start_page, stop):
            source_page = source[page_index]
            page_id = stable_id("page", document.document_id, page_index)
            blocks, mean_confidence = engine.recognize(source_page, page_id=page_id)
            text = "\n\n".join(block.text for block in blocks)
            quality, needs_ocr = _quality(text, mean_confidence)
            yield Page(
                page_id=page_id,
                document_id=document.document_id,
                pdf_file=path.name,
                subject_code=subject_code,
                pdf_page=page_index,
                width=float(source_page.rect.width),
                height=float(source_page.rect.height),
                rotation=source_page.rotation,
                text=text,
                extraction_method=ExtractionMethod.OCR,
                char_count=len(text),
                quality_score=quality,
                needs_ocr=needs_ocr,
                image_count=len(source_page.get_images(full=True)),
                blocks=blocks,
            )
