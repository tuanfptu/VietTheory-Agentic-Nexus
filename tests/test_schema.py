"""Contract tests for versioned pipeline schemas."""

import pytest
from pydantic import ValidationError

from viettheory.ids import stable_id
from viettheory.schema import (
    Answer,
    Citation,
    Claim,
    ExtractionMethod,
    Page,
    SourceSpan,
    TextBlock,
    TextLine,
)


def _line() -> TextLine:
    return TextLine(
        line_id="line_1",
        bbox=(10.0, 20.0, 100.0, 35.0),
        text="Nội dung kiểm thử",
        font_size=12.0,
        font_flags=(0,),
    )


def _page() -> Page:
    line = _line()
    block = TextBlock(
        block_id="block_1",
        bbox=(10.0, 20.0, 100.0, 35.0),
        text=line.text,
        lines=(line,),
    )
    return Page(
        page_id="page_1",
        document_id="doc_1",
        pdf_file="fixture.pdf",
        subject_code="TEST",
        pdf_page=0,
        printed_page="i",
        width=300.0,
        height=200.0,
        text=line.text,
        extraction_method=ExtractionMethod.PYMUPDF,
        char_count=len(line.text),
        quality_score=1.0,
        needs_ocr=False,
        blocks=(block,),
    )


def test_page_round_trip_preserves_schema() -> None:
    page = _page()

    restored = Page.model_validate_json(page.model_dump_json())

    assert restored == page
    assert restored.schema_version == "1.0"
    assert restored.printed_page == "i"


def test_page_rejects_bbox_outside_page() -> None:
    payload = _page().model_dump()
    payload["blocks"][0]["bbox"] = (10.0, 20.0, 301.0, 35.0)

    with pytest.raises(ValidationError, match="outside page bounds"):
        Page.model_validate(payload)


def test_answer_rejects_unknown_citation() -> None:
    claim = Claim(claim_id="claim_1", text="Một nhận định", citation_ids=("missing",))

    with pytest.raises(ValidationError, match="unknown citations"):
        Answer(
            answer_id="answer_1",
            question="Câu hỏi?",
            direct_answer="Câu trả lời",
            claims=(claim,),
            citations=(),
        )


def test_answer_accepts_claim_level_citation() -> None:
    span = SourceSpan(
        page_id="page_1",
        pdf_page=0,
        printed_page="1",
        bbox=(10.0, 20.0, 100.0, 35.0),
        text="Bằng chứng",
    )
    citation = Citation(citation_id="citation_1", evidence_id="evidence_1", source_span=span)
    claim = Claim(claim_id="claim_1", text="Một nhận định", citation_ids=("citation_1",))

    answer = Answer(
        answer_id="answer_1",
        question="Câu hỏi?",
        direct_answer="Câu trả lời",
        claims=(claim,),
        citations=(citation,),
    )

    assert answer.citations[0].source_span.printed_page == "1"


def test_citation_accepts_full_context_passage() -> None:
    span = SourceSpan(
        page_id="page_1",
        pdf_page=0,
        printed_page="1",
        bbox=(10.0, 20.0, 100.0, 35.0),
        text="Câu trực tiếp được dẫn.",
    )

    citation = Citation(
        citation_id="citation_1",
        evidence_id="evidence_1",
        source_span=span,
        context_text="Một đoạn giáo trình đầy đủ chứa câu trực tiếp được dẫn.",
    )

    assert citation.context_text is not None
    assert citation.context_text.startswith("Một đoạn giáo trình")


def test_stable_id_is_repeatable_and_namespaced() -> None:
    first = stable_id("chunk", "doc", 1, 2)

    assert first == stable_id("chunk", "doc", 1, 2)
    assert first != stable_id("chunk", "doc", 1, 3)
    assert first.startswith("chunk_")
