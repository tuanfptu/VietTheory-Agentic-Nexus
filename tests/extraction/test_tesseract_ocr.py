"""Tests for Tesseract TSV parsing and coordinate restoration."""

from viettheory.extraction.tesseract_ocr import parse_tesseract_tsv

_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
)
_TSV = (
    _TSV_HEADER
    + """\
5\t1\t1\t1\t1\t1\t20\t40\t60\t20\t95.0\tTư
5\t1\t1\t1\t1\t2\t90\t40\t100\t20\t85.0\ttưởng
5\t1\t1\t1\t2\t1\t20\t80\t80\t20\t90.0\tViệt
5\t1\t1\t1\t2\t2\t110\t80\t80\t20\t92.0\tNam
"""
)


def test_parse_tesseract_tsv_preserves_order_and_pdf_coordinates() -> None:
    blocks, confidence = parse_tesseract_tsv(
        _TSV,
        page_id="page_test",
        pixels_per_point=2.0,
    )

    assert len(blocks) == 1
    assert blocks[0].text == "Tư tưởng\nViệt Nam"
    assert blocks[0].bbox == (10.0, 20.0, 95.0, 50.0)
    assert blocks[0].lines[0].bbox == (10.0, 20.0, 95.0, 30.0)
    assert confidence == 0.905


def test_parse_tesseract_tsv_filters_rejected_words() -> None:
    blocks, confidence = parse_tesseract_tsv(
        _TSV.replace("95.0\tTư", "-1.0\tTư"),
        page_id="page_test",
        pixels_per_point=2.0,
    )

    assert blocks[0].lines[0].text == "tưởng"
    assert confidence > 0.0


def test_parse_tesseract_tsv_clamps_rounding_at_page_edge() -> None:
    blocks, _ = parse_tesseract_tsv(
        _TSV.replace("110\t80\t80\t20", "110\t80\t100\t40"),
        page_id="page_test",
        pixels_per_point=2.0,
        page_width=100.0,
        page_height=55.0,
    )

    assert blocks[0].bbox == (10.0, 20.0, 100.0, 55.0)


def test_parse_tesseract_tsv_treats_quotes_as_ocr_text() -> None:
    quoted = _TSV.replace("95.0\tTư", '95.0\t"Tư').replace("85.0\ttưởng", '85.0\ttưởng"')
    blocks, _ = parse_tesseract_tsv(
        quoted,
        page_id="page_test",
        pixels_per_point=2.0,
    )

    assert blocks[0].lines[0].text == '"Tư tưởng"'
    assert "\t" not in blocks[0].text
