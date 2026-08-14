from viettheory.extraction.structure_parser import (
    HeadingOverride,
    detect_heading_level,
    parse_structure,
)
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


def _ocr_block(index: int, text: str) -> TextBlock:
    block = _block(index, text, flags=0)
    return block.model_copy(update={"block_id": f"ocr-line-block_{index}"})


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


def test_detects_unnumbered_intro_chapter() -> None:
    assert detect_heading_level(_ocr_block(22, "Chương nhập môn")) == 1


def test_normalizes_tightly_scoped_ocr_chapter_noise() -> None:
    assert detect_heading_level(_block(22, "| Chương 3")) == 1
    assert detect_heading_level(_block(23, "Chương ð")) == 1


def test_detects_typography_free_ocr_headings_conservatively() -> None:
    assert detect_heading_level(_ocr_block(24, "II. Lãnh đạo công cuộc đổi mới")) == 2
    assert detect_heading_level(_ocr_block(24, "I Lãnh đạo cả nước xây dựng chủ nghĩa xã hội")) == 2
    assert detect_heading_level(_ocr_block(25, "2. Phong trào dân chủ 1936-1939")) == 3
    assert detect_heading_level(_ocr_block(25, "1. Phong trào cách mạng 1930-1931.")) == 3
    assert detect_heading_level(_ocr_block(25, "1J. Đổi mới toàn diện 1986-1996")) == 3
    assert detect_heading_level(_ocr_block(25, "5 Sự lãnh đạo đúng đắn của Đảng")) == 3
    assert detect_heading_level(_ocr_block(25, "8 Ấ")) is None
    assert (
        detect_heading_level(_ocr_block(25, "5 Hồ Chí Minh Toàn tập, Hà Nội, 2011, trang 312."))
        is None
    )
    assert detect_heading_level(_ocr_block(26, "2. Một mục liệt kê kết thúc;")) is None
    assert detect_heading_level(_ocr_block(27, "* Hồ Chí Minh Toàn tập")) is None


def test_excludes_review_questions_until_the_next_chapter() -> None:
    page = structured_page()
    blocks = (
        *page.blocks,
        _ocr_block(40, "NỘI DUNG ÔN TẬP VÀ THẢO LUẬN"),
        _ocr_block(41, "1. Phân tích nội dung chương"),
    )
    reviewed = page.model_copy(
        update={
            "blocks": blocks,
            "text": "\n".join(block.text for block in blocks),
            "char_count": sum(len(block.text) for block in blocks) + len(blocks) - 1,
        }
    )
    next_blocks = (_ocr_block(42, "Chương 2"), _ocr_block(43, "I. NỘI DUNG MỚI"))
    next_page = page.model_copy(
        update={
            "page_id": "page-2",
            "pdf_page": 8,
            "blocks": next_blocks,
            "text": "\n".join(block.text for block in next_blocks),
            "char_count": sum(len(block.text) for block in next_blocks) + 1,
        }
    )

    parsed = parse_structure((reviewed, next_page))

    assert "line-40" in parsed.excluded_line_ids
    assert "line-41" in parsed.excluded_line_ids
    assert any(heading.text.startswith("Chương 2") for heading in parsed.headings)


def test_resumes_body_after_review_at_conclusion_and_preserves_chapter_parent() -> None:
    page = structured_page()
    review_blocks = (
        *page.blocks,
        _ocr_block(40, "NỘI DUNG ÔN TẬP VÀ THẢO LUẬN"),
        _ocr_block(41, "1. Phân tích nội dung chương"),
        _ocr_block(42, "| KÉT LUẬN"),
        _ocr_block(43, "4. Kết hợp sức mạnh dân tộc với sức mạnh thời đại"),
        _ocr_block(44, "Nội dung kết luận cần được truy xuất."),
    )
    reviewed = page.model_copy(
        update={
            "blocks": review_blocks,
            "text": "\n".join(block.text for block in review_blocks),
            "char_count": sum(len(block.text) for block in review_blocks) + len(review_blocks) - 1,
        }
    )

    parsed = parse_structure((reviewed,))

    assert "line-40" in parsed.excluded_line_ids
    assert "line-41" in parsed.excluded_line_ids
    assert "line-42" not in parsed.excluded_line_ids
    assert "line-43" not in parsed.excluded_line_ids
    assert "line-44" not in parsed.excluded_line_ids
    conclusion = next(heading for heading in parsed.headings if heading.text == "KÉT LUẬN")
    chapter = parsed.headings[0]
    assert conclusion.level == 2
    assert conclusion.parent_heading_id == chapter.heading_id
    section = next(heading for heading in parsed.headings if heading.text.startswith("4."))
    assert section.parent_heading_id == conclusion.heading_id


def test_excludes_explicit_references_back_matter() -> None:
    page = structured_page()
    blocks = (
        *page.blocks,
        _ocr_block(50, "TÀI LIỆU THAM KHẢO CHỦ YẾU"),
        _ocr_block(51, "1. Giáo trình tham khảo"),
    )
    with_references = page.model_copy(
        update={
            "blocks": blocks,
            "text": "\n".join(block.text for block in blocks),
            "char_count": sum(len(block.text) for block in blocks) + len(blocks) - 1,
        }
    )

    parsed = parse_structure((with_references,))

    assert "line-50" in parsed.excluded_line_ids
    assert "line-51" in parsed.excluded_line_ids
    assert all(not heading.text.startswith("1. Giáo trình") for heading in parsed.headings)


def test_applies_reviewed_heading_override_by_stable_source_anchor() -> None:
    page = structured_page()
    plain_heading = _ocr_block(60, "Những bài học lớn về sự lãnh đạo của Đảng")
    body = _ocr_block(61, "Nội dung bài học được trình bày tại đây.")
    augmented = page.model_copy(
        update={
            "blocks": (*page.blocks, plain_heading, body),
            "text": f"{page.text}\n{plain_heading.text}\n{body.text}",
        }
    )
    override = HeadingOverride(
        pdf_page=page.pdf_page,
        block_id=plain_heading.block_id,
        level=3,
        text=plain_heading.text,
        decision="accepted_test_fixture",
    )

    parsed = parse_structure((augmented,), (override,))

    heading = next(item for item in parsed.headings if item.block_id == plain_heading.block_id)
    assert heading.level == 3
    assert parsed.line_paths[body.lines[0].line_id].section == plain_heading.text


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
