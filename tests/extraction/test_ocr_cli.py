"""Tests for the OCR extraction CLI contract."""

from viettheory.extraction.ocr_cli import build_parser


def test_ocr_cli_defaults_are_reproducible() -> None:
    args = build_parser().parse_args(["scan.pdf", "--subject", "TEST", "--output", "pages.jsonl"])

    assert args.scale == 2.0
    assert args.psm == 6
    assert args.tessdata.name == "tesseract"
    assert args.start_page == 0
