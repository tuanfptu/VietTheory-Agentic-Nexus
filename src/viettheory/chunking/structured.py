"""Heading-aware parent-child chunking with line-level provenance."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from viettheory.chunking.chunker import count_tokens
from viettheory.extraction.structure_parser import SectionPath, parse_structure
from viettheory.ids import stable_id
from viettheory.schema import BlockRole, Chunk, Page, SourceSpan, TextLine


@dataclass(frozen=True, slots=True)
class StructuredChunkingConfig:
    child_target_tokens: int = 400
    child_overlap_tokens: int = 50
    parent_target_tokens: int = 1500
    version: str = "heading_parent_child_v1"

    def __post_init__(self) -> None:
        if self.child_target_tokens <= 0 or self.parent_target_tokens <= 0:
            raise ValueError("chunk token targets must be positive")
        if not 0 <= self.child_overlap_tokens < self.child_target_tokens:
            raise ValueError("child overlap must be below the child target")
        if self.parent_target_tokens < self.child_target_tokens:
            raise ValueError("parent target must be at least the child target")


@dataclass(frozen=True, slots=True)
class StructuredChunks:
    parents: tuple[Chunk, ...]
    children: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class _Unit:
    page: Page
    line: TextLine
    path: SectionPath
    tokens: int


def _units(pages: tuple[Page, ...]) -> list[_Unit]:
    structure = parse_structure(pages)
    return [
        _Unit(
            page=page,
            line=line,
            path=structure.line_paths[line.line_id],
            tokens=max(1, count_tokens(line.text)),
        )
        for page in pages
        for block in page.blocks
        if block.role is BlockRole.BODY
        for line in block.lines
        if line.text.strip() and line.line_id not in structure.excluded_line_ids
    ]


def _split_by_target(units: list[_Unit], target: int) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    start = 0
    while start < len(units):
        end = start
        total = 0
        while end < len(units):
            if end > start and total + units[end].tokens > target:
                break
            total += units[end].tokens
            end += 1
        groups.append(units[start:end])
        start = end
    return groups


def _spans(units: list[_Unit]) -> tuple[SourceSpan, ...]:
    return tuple(
        SourceSpan(
            page_id=unit.page.page_id,
            pdf_page=unit.page.pdf_page,
            printed_page=unit.page.printed_page,
            bbox=unit.line.bbox,
            text=unit.line.text,
        )
        for unit in units
    )


def _parent(units: list[_Unit], config: StructuredChunkingConfig) -> Chunk:
    text = "\n".join(unit.line.text for unit in units)
    path = units[0].path
    return Chunk(
        chunk_id=stable_id(
            "parent",
            units[0].page.document_id,
            config.version,
            units[0].line.line_id,
            units[-1].line.line_id,
        ),
        document_id=units[0].page.document_id,
        subject_code=units[0].page.subject_code,
        text=text,
        token_count=count_tokens(text),
        source_spans=_spans(units),
        chunk_kind="parent",
        chapter=path.chapter,
        section=path.section,
        subsection=path.subsection,
    )


def _children(units: list[_Unit], parent: Chunk, config: StructuredChunkingConfig) -> list[Chunk]:
    children: list[Chunk] = []
    start = 0
    while start < len(units):
        end = start
        total = 0
        while end < len(units):
            if end > start and total + units[end].tokens > config.child_target_tokens:
                break
            total += units[end].tokens
            end += 1
        child_units = units[start:end]
        text = "\n".join(unit.line.text for unit in child_units)
        path = child_units[0].path
        children.append(
            Chunk(
                chunk_id=stable_id(
                    "child",
                    parent.chunk_id,
                    config.version,
                    child_units[0].line.line_id,
                    child_units[-1].line.line_id,
                ),
                document_id=parent.document_id,
                subject_code=parent.subject_code,
                text=text,
                token_count=count_tokens(text),
                source_spans=_spans(child_units),
                chunk_kind="child",
                parent_chunk_id=parent.chunk_id,
                chapter=path.chapter,
                section=path.section,
                subsection=path.subsection,
            )
        )
        if end == len(units):
            break
        next_start = end
        overlap = 0
        while next_start > start + 1 and overlap < config.child_overlap_tokens:
            next_start -= 1
            overlap += units[next_start].tokens
        start = next_start
    return children


def chunk_pages_structured(
    pages: tuple[Page, ...],
    config: StructuredChunkingConfig | None = None,
) -> StructuredChunks:
    """Create parent and child artifacts without crossing section boundaries."""
    active_config = config or StructuredChunkingConfig()
    units = _units(pages)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for _, section_units_iter in groupby(units, key=lambda unit: unit.path):
        section_units = list(section_units_iter)
        for parent_units in _split_by_target(section_units, active_config.parent_target_tokens):
            parent = _parent(parent_units, active_config)
            parents.append(parent)
            children.extend(_children(parent_units, parent, active_config))
    return StructuredChunks(parents=tuple(parents), children=tuple(children))
