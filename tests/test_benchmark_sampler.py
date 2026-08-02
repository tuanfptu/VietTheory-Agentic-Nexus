"""Tests for substantive benchmark evidence sampling."""

from viettheory.benchmark_generation import is_substantive_chunk
from viettheory.schema import Chunk, SourceSpan


def _chunk(*, chapter: str | None, section: str | None) -> Chunk:
    return Chunk(
        chunk_id="child_1",
        document_id="doc_1",
        subject_code="MLN111",
        text="Nội dung kiểm thử.",
        token_count=4,
        source_spans=(
            SourceSpan(
                page_id="page_1",
                pdf_page=1,
                bbox=(0.0, 0.0, 1.0, 1.0),
                text="Nội dung kiểm thử.",
            ),
        ),
        chunk_kind="child",
        parent_chunk_id="parent_1",
        chapter=chapter,
        section=section,
    )


def test_front_matter_is_not_sampled() -> None:
    assert not is_substantive_chunk(_chunk(chapter=None, section=None))


def test_learning_objective_is_not_sampled() -> None:
    chunk = _chunk(chapter="Chương 1", section="1. Về kiến thức")

    assert not is_substantive_chunk(chunk)


def test_substantive_section_is_sampled() -> None:
    chunk = _chunk(chapter="Chương 1", section="Nguồn gốc của triết học")

    assert is_substantive_chunk(chunk)
