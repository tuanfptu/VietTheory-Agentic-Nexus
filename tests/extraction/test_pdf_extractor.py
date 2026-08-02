"""Tests for citation-preserving PDF extraction."""

from pathlib import Path

import pymupdf

from viettheory.extraction import extract_pdf, read_document


def _make_pdf(path: Path) -> None:
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    page = document.new_page(width=300, height=200)
    page.insert_text((40, 60), "VietTheory smoke page", fontsize=12)
    page.insert_text((145, 190), "42", fontsize=9)
    document.save(path)  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]


def test_extract_pdf_preserves_text_and_coordinates(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _make_pdf(pdf_path)

    pages = extract_pdf(pdf_path, "TEST")

    assert len(pages) == 1
    page = pages[0]
    assert page.pdf_file == "fixture.pdf"
    assert page.subject_code == "TEST"
    assert page.pdf_page == 0
    assert page.printed_page == "42"
    assert page.extraction_method == "pymupdf"
    assert "VietTheory smoke page" in page.text
    assert page.char_count == len(page.text)
    assert page.blocks
    assert page.blocks[0].lines
    assert len(page.blocks[0].bbox) == 4


def test_extract_pdf_validates_page_range(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _make_pdf(pdf_path)

    try:
        extract_pdf(pdf_path, "TEST", start_page=2, end_page=1)
    except ValueError as error:
        assert "end_page" in str(error)
    else:
        raise AssertionError("Expected invalid page range to fail")


def test_read_document_has_stable_content_identity(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _make_pdf(pdf_path)

    first = read_document(pdf_path, "TEST")
    second = read_document(pdf_path, "TEST")

    assert first == second
    assert first.page_count == 1
    assert len(first.sha256) == 64
    assert first.document_id.startswith("doc_")
    assert extract_pdf(pdf_path, "TEST")[0].document_id == first.document_id
