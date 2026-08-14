from __future__ import annotations

import json
from io import BytesIO
from urllib.request import Request

import pytest

from viettheory.extraction.gemini_structure import (
    GeminiPageStructure,
    GeminiStructureClient,
    GeminiStructureError,
    validate_page_anchors,
)
from viettheory.schema import BlockRole, ExtractionMethod, Page, TextBlock, TextLine


def _page() -> Page:
    line = TextLine(line_id="line-1", bbox=(1.0, 1.0, 80.0, 10.0), text="Chương 1")
    block = TextBlock(
        block_id="block-1",
        bbox=(1.0, 1.0, 80.0, 10.0),
        text="Chương 1",
        lines=(line,),
        role=BlockRole.BODY,
    )
    return Page(
        page_id="page-1",
        document_id="doc-1",
        pdf_file="source.pdf",
        subject_code="VNR202",
        pdf_page=7,
        width=100.0,
        height=100.0,
        text="Chương 1",
        extraction_method=ExtractionMethod.OCR,
        char_count=len("Chương 1"),
        quality_score=0.9,
        needs_ocr=False,
        blocks=(block,),
    )


def _result(*, block_id: str = "block-1", pdf_page: int = 7) -> GeminiPageStructure:
    return GeminiPageStructure.model_validate(
        {
            "schema_version": "1.0",
            "pdf_page": pdf_page,
            "page_role": "chapter_opening",
            "elements": [
                {
                    "element_type": "chapter",
                    "level": 1,
                    "text": "Chương 1",
                    "source_block_ids": [block_id],
                    "confidence": 0.99,
                    "rationale": "Tiêu đề chương xuất hiện rõ trên ảnh.",
                }
            ],
            "warnings": [],
        }
    )


def test_validate_page_anchors_accepts_existing_blocks() -> None:
    assert validate_page_anchors(_result(), _page()).pdf_page == 7


def test_validate_page_anchors_rejects_unknown_blocks() -> None:
    with pytest.raises(GeminiStructureError, match="unknown block IDs"):
        validate_page_anchors(_result(block_id="invented"), _page())


def test_validate_page_anchors_canonicalizes_element_order() -> None:
    page = _page()
    second_line = TextLine(
        line_id="line-2",
        bbox=(1.0, 20.0, 80.0, 30.0),
        text="Nội dung",
    )
    second_block = TextBlock(
        block_id="block-2",
        bbox=(1.0, 20.0, 80.0, 30.0),
        text="Nội dung",
        lines=(second_line,),
        role=BlockRole.BODY,
    )
    page = page.model_copy(update={"blocks": (*page.blocks, second_block)})
    result = GeminiPageStructure.model_validate(
        {
            "pdf_page": 7,
            "page_role": "body",
            "elements": [
                {
                    "element_type": "footnote",
                    "text": "Nội dung",
                    "source_block_ids": ["block-2"],
                    "confidence": 0.9,
                    "rationale": "Footnote",
                },
                {
                    "element_type": "chapter",
                    "level": 1,
                    "text": "Chương 1",
                    "source_block_ids": ["block-1"],
                    "confidence": 0.9,
                    "rationale": "Heading",
                },
            ],
        }
    )
    validated = validate_page_anchors(result, page)
    assert validated.elements[0].source_block_ids == ("block-1",)


def test_validate_page_anchors_enforces_numbering_hierarchy() -> None:
    result = GeminiPageStructure.model_validate(
        {
            "pdf_page": 7,
            "page_role": "chapter_opening",
            "elements": [
                {
                    "element_type": "section",
                    "level": 3,
                    "text": "I. Nội dung lớn",
                    "source_block_ids": ["block-1"],
                    "confidence": 0.9,
                    "rationale": "Roman-numbered heading",
                }
            ],
        }
    )
    element = validate_page_anchors(result, _page()).elements[0]
    assert element.element_type == "division"
    assert element.level == 2


def test_review_heading_can_be_a_nonhierarchical_exclusion_marker() -> None:
    element = GeminiPageStructure.model_validate(
        {
            "pdf_page": 7,
            "page_role": "review",
            "elements": [
                {
                    "element_type": "review_heading",
                    "text": "Câu hỏi ôn tập",
                    "source_block_ids": ["block-1"],
                    "confidence": 0.9,
                    "rationale": "Review-region marker",
                }
            ],
        }
    ).elements[0]
    assert element.level is None


def test_client_uses_inline_image_schema_and_never_places_key_in_body() -> None:
    captured: dict[str, object] = {}

    class Response(BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: Request, *, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        payload = {"candidates": [{"content": {"parts": [{"text": _result().model_dump_json()}]}}]}
        return Response(json.dumps(payload).encode("utf-8"))

    client = GeminiStructureClient(
        api_key="secret-test-key",
        model="gemini-test",
        urlopen=fake_urlopen,
    )
    assert client.analyze_page(_page(), b"png-bytes").pdf_page == 7
    request = captured["request"]
    assert isinstance(request, Request)
    request_data = request.data
    assert isinstance(request_data, bytes)
    assert b"secret-test-key" not in request_data
    body = json.loads(request_data)
    assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
