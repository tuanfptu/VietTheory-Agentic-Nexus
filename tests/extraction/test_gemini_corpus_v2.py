from __future__ import annotations

import pytest

from viettheory.extraction.gemini_corpus_v2 import (
    CorrectedStructureBatch,
    validate_batch,
)
from viettheory.extraction.gemini_structure import GeminiStructureError
from viettheory.schema import BlockRole, ExtractionMethod, Page, TextBlock, TextLine


def _page(pdf_page: int) -> Page:
    text = f"Chương {pdf_page}"
    line = TextLine(
        line_id=f"line-{pdf_page}",
        bbox=(1.0, 1.0, 80.0, 10.0),
        text=text,
    )
    block = TextBlock(
        block_id=f"block-{pdf_page}",
        bbox=(1.0, 1.0, 80.0, 10.0),
        text=text,
        lines=(line,),
        role=BlockRole.BODY,
    )
    return Page(
        page_id=f"page-{pdf_page}",
        document_id="doc",
        pdf_file="source.pdf",
        subject_code="VNR202",
        pdf_page=pdf_page,
        width=100.0,
        height=100.0,
        text=text,
        extraction_method=ExtractionMethod.OCR,
        char_count=len(text),
        quality_score=0.8,
        needs_ocr=False,
        blocks=(block,),
    )


def _batch(page_numbers: tuple[int, ...]) -> CorrectedStructureBatch:
    return CorrectedStructureBatch.model_validate(
        {
            "pages": [
                {
                    "pdf_page": number,
                    "page_role": "chapter_opening",
                    "elements": [
                        {
                            "element_type": "chapter",
                            "level": 1,
                            "text": f"Chương {number}",
                            "source_block_ids": [f"block-{number}"],
                            "confidence": 0.9,
                            "rationale": "Visible heading",
                        }
                    ],
                    "corrections": [],
                }
                for number in page_numbers
            ]
        }
    )


def test_validate_batch_accepts_exact_page_coverage() -> None:
    pages = (_page(1), _page(2), _page(3), _page(4), _page(5))
    assert len(validate_batch(_batch((1, 2, 3, 4, 5)), pages).pages) == 5


def test_validate_batch_rejects_missing_page() -> None:
    with pytest.raises(GeminiStructureError, match="exact coverage"):
        validate_batch(_batch((1, 2)), (_page(1), _page(2), _page(3)))


def test_validate_batch_rejects_unknown_correction_anchor() -> None:
    result = CorrectedStructureBatch.model_validate(
        {
            "pages": [
                {
                    "pdf_page": 1,
                    "page_role": "body",
                    "elements": [],
                    "corrections": [
                        {
                            "block_id": "invented",
                            "corrected_text": "text",
                            "confidence": 0.9,
                            "correction_types": ["ocr_character"],
                            "rationale": "Visible mismatch",
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(GeminiStructureError, match="unknown block IDs"):
        validate_batch(result, (_page(1),))


def test_validate_batch_restores_unique_stable_id_prefix() -> None:
    result = CorrectedStructureBatch.model_validate(
        {
            "pages": [
                {
                    "pdf_page": 1,
                    "page_role": "body",
                    "elements": [],
                    "corrections": [
                        {
                            "block_id": "1",
                            "corrected_text": "text",
                            "confidence": 0.9,
                            "correction_types": ["ocr_character"],
                            "rationale": "Visible mismatch",
                        }
                    ],
                }
            ]
        }
    )
    validated = validate_batch(result, (_page(1),))
    assert validated.pages[0].corrections[0].block_id == "block-1"
