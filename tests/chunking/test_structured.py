from tests.extraction.test_structure_parser import structured_page

from viettheory.chunking.structured import (
    StructuredChunkingConfig,
    chunk_pages_structured,
)
from viettheory.extraction.structure_parser import parse_structure


def test_parent_child_chunks_preserve_structure_and_provenance() -> None:
    page = structured_page()
    config = StructuredChunkingConfig(
        child_target_tokens=12,
        child_overlap_tokens=3,
        parent_target_tokens=24,
    )
    result = chunk_pages_structured((page,), config)
    assert result.parents
    assert result.children
    parent_ids = {parent.chunk_id for parent in result.parents}
    assert all(parent.chunk_kind == "parent" for parent in result.parents)
    assert all(child.chunk_kind == "child" for child in result.children)
    assert all(child.parent_chunk_id in parent_ids for child in result.children)
    assert all(child.chapter for child in result.children)

    source_line_ids = set(parse_structure((page,)).line_paths)
    parent_texts = {span.text for parent in result.parents for span in parent.source_spans}
    source_texts = {
        line.text
        for block in page.blocks
        for line in block.lines
        if line.line_id in source_line_ids
    }
    assert parent_texts == source_texts


def test_structured_chunk_ids_are_stable() -> None:
    config = StructuredChunkingConfig(
        child_target_tokens=12,
        child_overlap_tokens=3,
        parent_target_tokens=24,
    )
    assert chunk_pages_structured((structured_page(),), config) == chunk_pages_structured(
        (structured_page(),), config
    )
