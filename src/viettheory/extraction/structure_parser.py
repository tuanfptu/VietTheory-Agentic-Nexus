"""Heading detection and hierarchical section assignment for textbook pages."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from viettheory.ids import stable_id
from viettheory.schema import BlockRole, Page, TextBlock

_CHAPTER = re.compile(r"^chương\s+(?:\d+|nhập\s+môn)\b(?P<remainder>.*)$", re.IGNORECASE)
_ROMAN_DASH = re.compile(r"^[IVXLCDM]+\s*[-\u2013]\s+\S", re.IGNORECASE)
_ROMAN_DOT = re.compile(r"^[IVXLCDM]+\.\s+\S", re.IGNORECASE)
_ROMAN_SPACE = re.compile(r"^[IVXLCDM]+\s+[A-ZĐÀ-Ỹ]")
_SECTION = re.compile(r"^[1-9](?:\.\d+)*\.\s+\S")
_OCR_SECTION_NO_DOT = re.compile(r"^[1-9]\s+[A-ZĐÀ-Ỹ]\S*")
_SUBSECTION = re.compile(r"^[a-zđ]\)\s+\S", re.IGNORECASE)
_BULLET_HEADING = re.compile(r"^\*\s+\S")
_UPPER_DIVISION = re.compile(r"^[A-ZĐ]\.\s+[A-ZÀ-Ỹ]")
_LEADING_OCR_NOISE = re.compile(r"^[|!¡:;,.„¬\s]+")
_OCR_CHAPTER_FIVE = re.compile(r"^(chương)\s+ð\b", re.IGNORECASE)
_OCR_SECTION_ONE = re.compile(r"^1[IJl]\.\s+", re.IGNORECASE)
_REVIEW_SECTION = re.compile(
    r"^(nội dung|câu hỏi)\s+(ôn tập|thảo luận)|^bài tập\b",
    re.IGNORECASE,
)
_CONTENT_RESUME_SECTION = re.compile(
    r"^(?:k[ếé]t\s+luận|tổng\s+kết(?:\s+chương)?|phần\s+kết)\s*$",
    re.IGNORECASE,
)
_BACK_MATTER_SECTION = re.compile(
    r"^(?:tài\s+liệu\s+tham\s+khảo|phụ\s+lục|danh\s+mục\s+tài\s+liệu)",
    re.IGNORECASE,
)
_TERMINAL_PROSE = frozenset('.,;:!?…”"' + "\N{RIGHT SINGLE QUOTATION MARK}")


class HeadingRecord(BaseModel):
    """Flat tree record with a stable parent link and source position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading_id: str
    parent_heading_id: str | None
    level: int = Field(ge=1, le=5)
    text: str = Field(min_length=1)
    pdf_page: int = Field(ge=0)
    block_id: str


class HeadingOverride(BaseModel):
    """Reviewed, provenance-addressed heading correction for one source block."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pdf_page: int = Field(ge=0)
    block_id: str = Field(min_length=1)
    level: int = Field(ge=1, le=5)
    text: str = Field(min_length=1)
    decision: str = Field(min_length=1)


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


def _is_ocr_line(block: TextBlock) -> bool:
    return block.block_id.startswith("ocr-line-block_")


def _is_ocr_heading_candidate(block: TextBlock, text: str) -> bool:
    """Use conservative typography-free cues for one-line OCR blocks."""
    if not _is_ocr_line(block) or not text or text[-1] in _TERMINAL_PROSE:
        return False
    return len(text) <= 160


def _contains_inline_enumeration(text: str) -> bool:
    return bool(re.search(r"[:;]\s*[a-zđ]\)\s", text, re.IGNORECASE))


def _clean(text: str) -> str:
    cleaned = _LEADING_OCR_NOISE.sub("", " ".join(text.split()))
    cleaned = _OCR_CHAPTER_FIVE.sub(r"\1 5", cleaned)
    return _OCR_SECTION_ONE.sub("1. ", cleaned)


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
    if _CONTENT_RESUME_SECTION.match(text) and (_is_bold(block) or _is_ocr_line(block)):
        return 2
    if (
        _ROMAN_DASH.match(text)
        or (_ROMAN_SPACE.match(text) and _is_ocr_heading_candidate(block, text))
        or (
            _ROMAN_DOT.match(text)
            and (text.upper() == text or _is_ocr_heading_candidate(block, text))
        )
        or (_UPPER_DIVISION.match(text) and text.upper() == text)
    ):
        return 2
    standard_section = bool(_SECTION.match(text))
    ocr_section_without_dot = bool(
        _OCR_SECTION_NO_DOT.match(text) and _is_ocr_line(block) and len(text) >= 20
    )
    if (
        (standard_section or ocr_section_without_dot)
        and not _contains_inline_enumeration(text)
        and (
            _is_bold(block)
            or _is_ocr_heading_candidate(block, text)
            or (standard_section and _is_ocr_heading_candidate(block, text.removesuffix(".")))
        )
    ):
        return 3
    if (
        _SUBSECTION.match(text)
        and len(text) <= 120
        and (
            _is_bold(block)
            or not _is_ocr_line(block)
            or (
                _is_ocr_heading_candidate(block, text)
                and not re.search(r"\b[a-zđ]\)\s", text[3:], re.IGNORECASE)
            )
        )
    ):
        return 4
    if _BULLET_HEADING.match(text) and len(text) <= 100 and _is_bold(block):
        return 5
    return None


def parse_structure(
    pages: tuple[Page, ...],
    heading_overrides: tuple[HeadingOverride, ...] = (),
) -> ParsedStructure:
    """Build heading parent links and assign every body line to a section path."""
    override_by_source = {
        (override.pdf_page, override.block_id): override for override in heading_overrides
    }
    if len(override_by_source) != len(heading_overrides):
        raise ValueError("heading overrides must have unique page/block anchors")
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
    reached_review = False
    heading_block_indexes: set[int] = set()
    excluded_line_ids: set[str] = set()

    index = 0
    while index < len(body_blocks):
        page, block = body_blocks[index]
        override = override_by_source.get((page.pdf_page, block.block_id))
        heading_text = override.text if override is not None else _clean(block.text)
        if back_matter_start is not None and page.pdf_page >= back_matter_start:
            reached_contents = True
            active_ids = {}
            chapter, section, subsection = None, None, None
        if heading_text.upper() == "MỤC LỤC":
            reached_contents = True
            active_ids = {}
            chapter, section, subsection = None, None, None
        if _BACK_MATTER_SECTION.match(heading_text):
            reached_contents = True
            active_ids = {}
            chapter, section, subsection = None, None, None
        if _REVIEW_SECTION.match(heading_text):
            reached_review = True
            active_ids = {
                active_level: active_id
                for active_level, active_id in active_ids.items()
                if active_level == 1
            }
            section, subsection = None, None
        detected_level = override.level if override is not None else detect_heading_level(block)
        if (
            reached_contents
            and detected_level == 1
            and (back_matter_start is None or page.pdf_page < back_matter_start)
        ):
            reached_contents = False
        if reached_review and (detected_level == 1 or _CONTENT_RESUME_SECTION.match(heading_text)):
            reached_review = False
        level = None if reached_contents or reached_review else detected_level
        if reached_contents or reached_review:
            excluded_line_ids.update(line.line_id for line in block.lines)
        if level == 1:
            reached_first_chapter = True
        elif not reached_first_chapter:
            level = None
        consumed_title: TextBlock | None = None
        if level is not None and index + 1 < len(body_blocks):
            next_page, next_block = body_blocks[index + 1]
            next_text = _clean(next_block.text)
            chapter_title = (
                level == 1
                and next_page.page_id == page.page_id
                and next_text
                and len(next_text) <= 160
                and next_text.upper() == next_text
            )
            ocr_continuation = (
                level > 1
                and _is_ocr_line(block)
                and _is_ocr_line(next_block)
                and next_page.page_id == page.page_id
                and next_text
                and len(next_text) <= 100
                and next_block.bbox[1] - block.bbox[3] <= 12
                and (next_text[0].islower() or (heading_text.endswith("-") and next_text.isdigit()))
            )
            if chapter_title or ocr_continuation:
                if chapter_title:
                    heading_text = f"{heading_text}: {next_text}"
                elif heading_text.endswith("-"):
                    heading_text = f"{heading_text}{next_text}"
                else:
                    heading_text = f"{heading_text} {next_text}"
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
