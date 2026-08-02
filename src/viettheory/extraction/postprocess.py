"""Corpus-level page post-processing that preserves source provenance."""

from __future__ import annotations

import math
import re
from collections import Counter

from viettheory.ids import stable_id
from viettheory.schema import BlockRole, Page, TextBlock

_PAGE_NUMBER = re.compile(r"(?:[ivxlcdm]+|\d{1,4})", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def split_ocr_line_blocks(pages: tuple[Page, ...]) -> tuple[Page, ...]:
    """Promote OCR lines to blocks so marginal roles retain line-level geometry."""
    normalized: list[Page] = []
    for page in pages:
        if page.extraction_method.value != "ocr":
            normalized.append(page)
            continue
        blocks = tuple(
            TextBlock(
                block_id=stable_id("ocr-line-block", line.line_id),
                bbox=line.bbox,
                text=line.text,
                lines=(line,),
                role=block.role,
            )
            for block in page.blocks
            for line in block.lines
        )
        normalized.append(page.model_copy(update={"blocks": blocks}))
    return tuple(normalized)


def _normalized_marginal_text(text: str) -> str:
    normalized = _SPACE.sub(" ", text).strip().casefold()
    return _DIGITS.sub("#", normalized)


def _zone(block: TextBlock, page: Page, margin_ratio: float) -> BlockRole | None:
    if block.bbox[1] <= page.height * margin_ratio:
        return BlockRole.HEADER
    if block.bbox[3] >= page.height * (1.0 - margin_ratio):
        return BlockRole.FOOTER
    return None


def tag_marginal_roles(
    pages: tuple[Page, ...],
    *,
    repeat_ratio: float = 0.8,
    margin_ratio: float = 0.15,
) -> tuple[Page, ...]:
    """Tag page numbers and repeated marginal blocks across a document.

    Repetition is counted once per page, so a duplicated block on one page cannot
    falsely promote itself. Text and bounding boxes are never removed or modified.
    """
    if not pages:
        return ()
    if not 0.0 < repeat_ratio <= 1.0:
        raise ValueError("repeat_ratio must be in (0, 1]")
    if not 0.0 < margin_ratio < 0.5:
        raise ValueError("margin_ratio must be in (0, 0.5)")

    appearances: Counter[tuple[BlockRole, str]] = Counter()
    for page in pages:
        seen_on_page: set[tuple[BlockRole, str]] = set()
        for block in page.blocks:
            zone = _zone(block, page, margin_ratio)
            normalized = _normalized_marginal_text(block.text)
            if zone is not None and normalized:
                seen_on_page.add((zone, normalized))
        appearances.update(seen_on_page)

    minimum_pages = math.ceil(len(pages) * repeat_ratio)
    repeated = {key for key, count in appearances.items() if count >= minimum_pages}
    tagged_pages: list[Page] = []
    for page in pages:
        tagged_blocks: list[TextBlock] = []
        for block in page.blocks:
            zone = _zone(block, page, margin_ratio)
            normalized = _normalized_marginal_text(block.text)
            if zone is not None and _PAGE_NUMBER.fullmatch(block.text.strip()):
                role = BlockRole.PAGE_NUMBER
            elif zone is not None and (zone, normalized) in repeated:
                role = zone
            else:
                role = BlockRole.BODY
            tagged_blocks.append(block.model_copy(update={"role": role}))
        tagged_pages.append(page.model_copy(update={"blocks": tuple(tagged_blocks)}))
    return tuple(tagged_pages)
