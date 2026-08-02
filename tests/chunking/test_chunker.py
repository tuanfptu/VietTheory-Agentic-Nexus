"""Tests for fixed-size provenance-preserving chunking."""

from viettheory.chunking import ChunkingConfig, chunk_pages
from viettheory.schema import BlockRole, ExtractionMethod, Page, TextBlock, TextLine


def _pages() -> tuple[Page, ...]:
    pages: list[Page] = []
    for page_index in range(2):
        lines = tuple(
            TextLine(
                line_id=f"line_{page_index}_{line_index}",
                bbox=(10.0, 10.0 + line_index * 10, 190.0, 18.0 + line_index * 10),
                text=f"page {page_index} line {line_index} contains useful source text",
                font_size=10.0,
            )
            for line_index in range(4)
        )
        body = TextBlock(
            block_id=f"body_{page_index}",
            bbox=(10.0, 10.0, 190.0, 48.0),
            text="\n".join(line.text for line in lines),
            lines=lines,
        )
        number_line = TextLine(
            line_id=f"number_line_{page_index}",
            bbox=(95.0, 190.0, 105.0, 198.0),
            text=str(page_index + 1),
            font_size=8.0,
        )
        number = TextBlock(
            block_id=f"number_{page_index}",
            bbox=number_line.bbox,
            text=number_line.text,
            lines=(number_line,),
            role=BlockRole.PAGE_NUMBER,
        )
        text = body.text + "\n\n" + number.text
        pages.append(
            Page(
                page_id=f"page_{page_index}",
                document_id="doc_1",
                pdf_file="fixture.pdf",
                subject_code="TEST",
                pdf_page=page_index,
                width=200.0,
                height=200.0,
                text=text,
                extraction_method=ExtractionMethod.PYMUPDF,
                char_count=len(text),
                quality_score=1.0,
                needs_ocr=False,
                blocks=(body, number),
            )
        )
    return tuple(pages)


def test_chunker_preserves_all_body_lines_and_excludes_page_numbers() -> None:
    pages = _pages()

    chunks = chunk_pages(pages, ChunkingConfig(target_tokens=25, overlap_tokens=5))

    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.source_spans for chunk in chunks)
    assert all(span.text not in {"1", "2"} for chunk in chunks for span in chunk.source_spans)
    source_lines = {line.text for page in pages for line in page.blocks[0].lines}
    chunk_lines = {span.text for chunk in chunks for span in chunk.source_spans}
    assert source_lines == chunk_lines


def test_chunk_ids_are_stable() -> None:
    config = ChunkingConfig(target_tokens=25, overlap_tokens=5)

    assert chunk_pages(_pages(), config) == chunk_pages(_pages(), config)
