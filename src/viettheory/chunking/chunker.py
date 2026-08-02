"""Fixed-size baseline chunker that never splits a source text line."""

from __future__ import annotations

import re
from dataclasses import dataclass

from viettheory.ids import stable_id
from viettheory.schema import BlockRole, Chunk, Page, SourceSpan, TextLine

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Versioned baseline chunking parameters."""

    target_tokens: int = 400
    overlap_tokens: int = 50
    version: str = "fixed_lines_v1"

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be in [0, target_tokens)")


@dataclass(frozen=True, slots=True)
class _Unit:
    page: Page
    line: TextLine
    tokens: int


def count_tokens(text: str) -> int:
    """Count deterministic Unicode word/punctuation tokens for baseline sizing."""
    return len(_TOKEN.findall(text))


def _units(pages: tuple[Page, ...]) -> list[_Unit]:
    return [
        _Unit(page=page, line=line, tokens=max(1, count_tokens(line.text)))
        for page in pages
        for block in page.blocks
        if block.role is BlockRole.BODY
        for line in block.lines
        if line.text.strip()
    ]


def _build_chunk(units: list[_Unit], config: ChunkingConfig) -> Chunk:
    first = units[0]
    last = units[-1]
    text = "\n".join(unit.line.text for unit in units)
    spans = tuple(
        SourceSpan(
            page_id=unit.page.page_id,
            pdf_page=unit.page.pdf_page,
            printed_page=unit.page.printed_page,
            bbox=unit.line.bbox,
            text=unit.line.text,
        )
        for unit in units
    )
    return Chunk(
        chunk_id=stable_id(
            "chunk",
            first.page.document_id,
            config.version,
            config.target_tokens,
            config.overlap_tokens,
            first.line.line_id,
            last.line.line_id,
        ),
        document_id=first.page.document_id,
        subject_code=first.page.subject_code,
        text=text,
        token_count=count_tokens(text),
        source_spans=spans,
    )


def chunk_pages(pages: tuple[Page, ...], config: ChunkingConfig | None = None) -> tuple[Chunk, ...]:
    """Chunk ordered pages while retaining complete line-level provenance."""
    active_config = config or ChunkingConfig()
    units = _units(pages)
    if not units:
        return ()

    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        end = start
        token_total = 0
        while end < len(units):
            next_tokens = units[end].tokens
            if end > start and token_total + next_tokens > active_config.target_tokens:
                break
            token_total += next_tokens
            end += 1

        chunks.append(_build_chunk(units[start:end], active_config))
        if end == len(units):
            break

        next_start = end
        overlap = 0
        while next_start > start + 1 and overlap < active_config.overlap_tokens:
            next_start -= 1
            overlap += units[next_start].tokens
        start = next_start

    return tuple(chunks)
