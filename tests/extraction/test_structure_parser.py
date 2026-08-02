from viettheory.extraction.structure_parser import detect_heading_level, parse_structure
from viettheory.schema import ExtractionMethod, Page, TextBlock, TextLine


def _block(index: int, text: str, flags: int = 4) -> TextBlock:
    line = TextLine(
        line_id=f"line-{index}",
        bbox=(10.0, 10.0 + index * 10, 190.0, 18.0 + index * 10),
        text=text,
        font_size=13.0,
        font_flags=(flags,),
    )
    return TextBlock(
        block_id=f"block-{index}",
        bbox=line.bbox,
        text=text,
        lines=(line,),
    )


def structured_page() -> Page:
    blocks = (
        _block(0, "Chương 1"),
        _block(1, "KHÁI LUẬN VỀ TRIẾT HỌC", 20),
        _block(2, "I- TRIẾT HỌC VÀ VẤN ĐỀ CƠ BẢN"),
        _block(3, "1. Khái lược về triết học", 20),
        _block(4, "Nội dung giải thích thứ nhất."),
        _block(5, "a) Nguồn gốc của triết học", 6),
        _block(6, "Nội dung giải thích thứ hai."),
    )
    text = "\n".join(block.text for block in blocks)
    return Page(
        page_id="page-1",
        document_id="doc-1",
        pdf_file="fixture.pdf",
        subject_code="MLN111",
        pdf_page=7,
        width=200.0,
        height=200.0,
        text=text,
        extraction_method=ExtractionMethod.PYMUPDF,
        char_count=len(text),
        quality_score=1.0,
        needs_ocr=False,
        blocks=blocks,
    )


def test_detects_document_heading_levels() -> None:
    page = structured_page()
    assert [detect_heading_level(block) for block in page.blocks] == [
        1,
        None,
        2,
        3,
        None,
        4,
        None,
    ]


def test_prose_starting_with_c_mac_is_not_a_roman_heading() -> None:
    assert detect_heading_level(_block(20, "C. Mác và Ph. Ăngghen khẳng định")) is None


def test_chapter_prose_is_not_a_chapter_heading() -> None:
    assert detect_heading_level(_block(21, "Chương 3 sẽ trình bày ba nội dung")) is None


def test_normalizes_tightly_scoped_ocr_chapter_noise() -> None:
    assert detect_heading_level(_block(22, "| Chương 3")) == 1
    assert detect_heading_level(_block(23, "Chương ð")) == 1


def test_builds_parent_links_and_assigns_paths() -> None:
    page = structured_page()
    parsed = parse_structure((page,))
    assert len(parsed.headings) == 4
    assert parsed.headings[0].text == "Chương 1: KHÁI LUẬN VỀ TRIẾT HỌC"
    assert parsed.headings[1].parent_heading_id == parsed.headings[0].heading_id
    assert parsed.headings[2].parent_heading_id == parsed.headings[1].heading_id
    body_path = parsed.line_paths["line-6"]
    assert body_path.chapter == "Chương 1: KHÁI LUẬN VỀ TRIẾT HỌC"
    assert body_path.section == "1. Khái lược về triết học"
    assert body_path.subsection == "a) Nguồn gốc của triết học"


def test_resumes_structure_after_front_matter_contents() -> None:
    page = structured_page()
    contents = _block(30, "MỤC LỤC")
    blocks = (contents, *page.blocks)
    with_contents = page.model_copy(
        update={
            "blocks": blocks,
            "text": "\n".join(block.text for block in blocks),
            "char_count": sum(len(block.text) for block in blocks) + len(blocks) - 1,
        }
    )

    parsed = parse_structure((with_contents,))

    assert parsed.headings
    assert parsed.headings[0].text == "Chương 1: KHÁI LUẬN VỀ TRIẾT HỌC"


def test_rejects_clustered_tail_chapters_as_contents() -> None:
    body = structured_page()
    pages = tuple(
        body.model_copy(
            update={
                "page_id": f"page-{index}",
                "pdf_page": index,
                "blocks": (
                    (_block(index * 10, "Chương 2"), _block(index * 10 + 1, "MỤC HAI"))
                    if index == 9
                    else (_block(index * 10, f"Nội dung trang {index}"),)
                ),
            }
        )
        for index in range(9)
    )
    contents = body.model_copy(
        update={
            "page_id": "page-9",
            "pdf_page": 9,
            "blocks": (_block(90, "Chương 3"), _block(91, "Chương 4")),
        }
    )

    parsed = parse_structure((*pages, contents))

    assert not parsed.headings
    assert "line-90" in parsed.excluded_line_ids
    assert "line-91" in parsed.excluded_line_ids
