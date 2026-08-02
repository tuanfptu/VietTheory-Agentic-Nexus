"""Tests for non-destructive corpus-level page role detection."""

from viettheory.extraction.postprocess import split_ocr_line_blocks, tag_marginal_roles
from viettheory.schema import BlockRole, ExtractionMethod, Page, TextBlock, TextLine


def _block(block_id: str, text: str, bbox: tuple[float, float, float, float]) -> TextBlock:
    line = TextLine(line_id=f"line_{block_id}", bbox=bbox, text=text, font_size=10.0)
    return TextBlock(block_id=block_id, bbox=bbox, text=text, lines=(line,))


def _page(index: int) -> Page:
    blocks = (
        _block(f"header_{index}", "VietTheory Course", (20.0, 5.0, 180.0, 15.0)),
        _block(f"body_{index}", f"Body text {index}", (20.0, 50.0, 180.0, 80.0)),
        _block(f"number_{index}", str(index + 1), (90.0, 190.0, 100.0, 198.0)),
    )
    text = "\n\n".join(block.text for block in blocks)
    return Page(
        page_id=f"page_{index}",
        document_id="doc_1",
        pdf_file="fixture.pdf",
        subject_code="TEST",
        pdf_page=index,
        width=200.0,
        height=200.0,
        text=text,
        extraction_method=ExtractionMethod.PYMUPDF,
        char_count=len(text),
        quality_score=1.0,
        needs_ocr=False,
        blocks=blocks,
    )


def test_tag_marginal_roles_preserves_text_and_geometry() -> None:
    pages = tuple(_page(index) for index in range(5))

    tagged = tag_marginal_roles(pages)

    assert all(page.blocks[0].role is BlockRole.HEADER for page in tagged)
    assert all(page.blocks[1].role is BlockRole.BODY for page in tagged)
    assert all(page.blocks[2].role is BlockRole.PAGE_NUMBER for page in tagged)
    assert [page.text for page in tagged] == [page.text for page in pages]
    assert [block.bbox for page in tagged for block in page.blocks] == [
        block.bbox for page in pages for block in page.blocks
    ]


def test_tag_marginal_roles_rejects_invalid_threshold() -> None:
    try:
        tag_marginal_roles((_page(0),), repeat_ratio=0.0)
    except ValueError as error:
        assert "repeat_ratio" in str(error)
    else:
        raise AssertionError("Expected invalid repeat ratio to fail")


def test_split_ocr_line_blocks_is_provenance_preserving() -> None:
    page = _page(0)
    joined = TextBlock(
        block_id="ocr_block",
        bbox=(20.0, 5.0, 180.0, 80.0),
        text="\n".join(block.text for block in page.blocks[:2]),
        lines=(page.blocks[0].lines[0], page.blocks[1].lines[0]),
    )
    ocr_page = page.model_copy(
        update={
            "extraction_method": ExtractionMethod.OCR,
            "blocks": (joined,),
        }
    )

    split = split_ocr_line_blocks((ocr_page,))[0]

    assert len(split.blocks) == 2
    assert [block.text for block in split.blocks] == ["VietTheory Course", "Body text 0"]
    assert [block.lines[0].line_id for block in split.blocks] == ["line_header_0", "line_body_0"]
    assert split.text == ocr_page.text
