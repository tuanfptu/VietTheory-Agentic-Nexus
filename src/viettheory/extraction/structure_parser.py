"""Heading detection and hierarchical section assignment for textbook pages."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from viettheory.ids import stable_id
from viettheory.schema import BlockRole, Page, TextBlock

_CHAPTER = re.compile(r"^chương\s+\d+\b(?P<remainder>.*)$", re.IGNORECASE)
_ROMAN_DASH = re.compile(r"^[IVXLCDM]+\s*[-\u2013]\s+\S", re.IGNORECASE)
_ROMAN_DOT = re.compile(r"^[IVXLCDM]+\.\s+\S", re.IGNORECASE)
_SECTION = re.compile(r"^\d+(?:\.\d+)*\.\s+\S")
_SUBSECTION = re.compile(r"^[a-zđ]\)\s+\S", re.IGNORECASE)
_BULLET_HEADING = re.compile(r"^\*\s+\S")
_UPPER_DIVISION = re.compile(r"^[A-ZĐ]\.\s+[A-ZÀ-Ỹ]")
_LEADING_OCR_NOISE = re.compile(r"^[|!¡:;,.„¬\s]+")
_OCR_CHAPTER_FIVE = re.compile(r"^(chương)\s+ð\b", re.IGNORECASE)


class HeadingRecord(BaseModel):
    """Flat tree record with a stable parent link and source position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading_id: str
    parent_heading_id: str | None
    level: int = Field(ge=1, le=5)
    text: str = Field(min_length=1)
    pdf_page: int = Field(ge=0)
    block_id: str


@dataclass(frozen=True, slots=True)
class SectionPath:
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedStructure:
    headings: tuple[HeadingRecord, ...]
    line_paths: dict[str, SectionPath]
    excluded_line_ids: frozenset[str]


def _is_bold(block: TextBlock) -> bool:
    return any(flags & 16 for line in block.lines for flags in line.font_flags)


def _clean(text: str) -> str:
    cleaned = _LEADING_OCR_NOISE.sub("", " ".join(text.split()))
    return _OCR_CHAPTER_FIVE.sub(r"\1 5", cleaned)


def detect_heading_level(block: TextBlock) -> int | None:
    """Return a hierarchy level only for short, heading-like blocks."""
    text = _clean(block.text)
    if not text or len(text) > 180:
        return None
    chapter_match = _CHAPTER.match(text)
    if chapter_match:
        remainder = chapter_match.group("remainder").strip(" :-")
        if not remainder or remainder.upper() == remainder:
            return 1
    if (
        _ROMAN_DASH.match(text)
        or (_ROMAN_DOT.match(text) and text.upper() == text)
        or (_UPPER_DIVISION.match(text) and text.upper() == text)
    ):
        return 2
    if _SECTION.match(text) and _is_bold(block):
        return 3
    if _SUBSECTION.match(text) and len(text) <= 120:
        return 4
    if _BULLET_HEADING.match(text) and len(text) <= 100:
        return 5
    return None


def parse_structure(pages: tuple[Page, ...]) -> ParsedStructure:
    """Build heading parent links and assign every body line to a section path."""
    tail_start = max(0, len(pages) - max(3, len(pages) // 10))
    tail_chapter_pages = [
        page.pdf_page
        for page in pages[tail_start:]
        if any(
            detect_heading_level(block) == 1
            for block in page.blocks
            if block.role is BlockRole.BODY
        )
    ]
    tail_chapter_counts = {
        page.pdf_page: sum(
            detect_heading_level(block) == 1
            for block in page.blocks
            if block.role is BlockRole.BODY
        )
        for page in pages[tail_start:]
    }
    back_matter_start = (
        min(tail_chapter_pages)
        if any(count >= 2 for count in tail_chapter_counts.values())
        else None
    )
    body_blocks = [
        (page, block) for page in pages for block in page.blocks if block.role is BlockRole.BODY
    ]
    headings: list[HeadingRecord] = []
    line_paths: dict[str, SectionPath] = {}
    active_ids: dict[int, str] = {}
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    reached_first_chapter = False
    reached_contents = False
    heading_block_indexes: set[int] = set()
    excluded_line_ids: set[str] = set()

    index = 0
    while index < len(body_blocks):
        page, block = body_blocks[index]
        heading_text = _clean(block.text)
        if back_matter_start is not None and page.pdf_page >= back_matter_start:
            reached_contents = True
            active_ids = {}
            chapter, section, subsection = None, None, None
        if heading_text.upper() == "MỤC LỤC":
            reached_contents = True
            active_ids = {}
            chapter, section, subsection = None, None, None
        detected_level = detect_heading_level(block)
        if (
            reached_contents
            and detected_level == 1
            and (back_matter_start is None or page.pdf_page < back_matter_start)
        ):
            reached_contents = False
        level = None if reached_contents else detected_level
        if reached_contents:
            excluded_line_ids.update(line.line_id for line in block.lines)
        if level == 1:
            reached_first_chapter = True
        elif not reached_first_chapter:
            level = None
        consumed_title: TextBlock | None = None
        if level == 1 and index + 1 < len(body_blocks):
            next_page, next_block = body_blocks[index + 1]
            next_text = _clean(next_block.text)
            if (
                next_page.page_id == page.page_id
                and next_text
                and len(next_text) <= 160
                and next_text.upper() == next_text
            ):
                heading_text = f"{heading_text}: {next_text}"
                consumed_title = next_block

        if level is not None:
            heading_block_indexes.add(index)
            parent_id = next(
                (
                    active_ids[parent_level]
                    for parent_level in range(level - 1, 0, -1)
                    if parent_level in active_ids
                ),
                None,
            )
            heading_id = stable_id(
                "heading", page.document_id, page.pdf_page, block.block_id, heading_text
            )
            headings.append(
                HeadingRecord(
                    heading_id=heading_id,
                    parent_heading_id=parent_id,
                    level=level,
                    text=heading_text,
                    pdf_page=page.pdf_page,
                    block_id=block.block_id,
                )
            )
            active_ids = {
                active_level: active_id
                for active_level, active_id in active_ids.items()
                if active_level < level
            }
            active_ids[level] = heading_id
            if level == 1:
                chapter, section, subsection = heading_text, None, None
            elif level in (2, 3):
                section, subsection = heading_text, None
            elif level == 4:
                subsection = heading_text

        path = SectionPath(chapter=chapter, section=section, subsection=subsection)
        for line in block.lines:
            line_paths[line.line_id] = path
        if consumed_title is not None:
            heading_block_indexes.add(index + 1)
            for line in consumed_title.lines:
                line_paths[line.line_id] = path
            index += 1
        index += 1

    next_body_path: SectionPath | None = None
    for block_index in range(len(body_blocks) - 1, -1, -1):
        _, block = body_blocks[block_index]
        if block_index in heading_block_indexes:
            if next_body_path is not None:
                for line in block.lines:
                    line_paths[line.line_id] = next_body_path
        elif block.lines:
            next_body_path = line_paths[block.lines[0].line_id]

    return ParsedStructure(
        headings=tuple(headings),
        line_paths=line_paths,
        excluded_line_ids=frozenset(excluded_line_ids),
    )
