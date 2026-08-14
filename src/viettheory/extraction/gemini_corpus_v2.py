"""Validated Gemini batch contracts for corrected, structured corpus v2."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viettheory.extraction.gemini_structure import (
    GeminiPageStructure,
    GeminiStructureElement,
    GeminiStructureError,
    PageRole,
    validate_page_anchors,
)
from viettheory.schema import BlockRole, Page


class OcrCorrection(BaseModel):
    """A conservative text correction anchored to one existing source block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(min_length=1)
    corrected_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    correction_types: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=300)


class CorrectedPageStructure(BaseModel):
    """Structure plus sparse OCR corrections for one page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_page: int = Field(ge=0)
    page_role: PageRole
    elements: tuple[GeminiStructureElement, ...]
    corrections: tuple[OcrCorrection, ...]
    warnings: tuple[str, ...] = ()


class CorrectedStructureBatch(BaseModel):
    """One response covering one to three ordered pages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    pages: tuple[CorrectedPageStructure, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_pages(self) -> CorrectedStructureBatch:
        page_numbers = [page.pdf_page for page in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("batch pages must be unique")
        return self


UrlOpen = Callable[..., Any]


def _canonical_anchor(candidate: str, allowed: set[str]) -> str:
    """Restore a dropped stable-ID prefix only when the suffix is unique."""
    if candidate in allowed:
        return candidate
    matches = [
        block_id
        for block_id in allowed
        if block_id.endswith(candidate) or block_id.startswith(candidate)
    ]
    return matches[0] if len(matches) == 1 else candidate


def _page_payload(page: Page) -> dict[str, Any]:
    return {
        "pdf_page": page.pdf_page,
        "extraction_method": page.extraction_method,
        "blocks": [
            {
                "block_id": block.block_id,
                "bbox": list(block.bbox),
                "text": block.text,
            }
            for block in page.blocks
            if block.role is BlockRole.BODY
        ],
    }


def batch_prompt(
    pages: tuple[Page, ...],
    context_pages: tuple[Page, ...] = (),
    *,
    structure_only: bool = False,
) -> str:
    """Create a conservative multi-page correction and structure prompt."""
    expected = [page.pdf_page for page in pages]
    payload = [_page_payload(page) for page in pages]
    context_payload = [_page_payload(page) for page in context_pages]
    task_instruction = (
        "Đây là lượt STRUCTURE-ONLY: corrections bắt buộc là mảng rỗng cho mọi trang. "
        "Chỉ phát hiện page_role, heading và vùng footnote/page number/review. "
        if structure_only
        else "Đây là lượt structure và sparse OCR correction. "
    )
    return (
        "Bạn là bộ hiệu đính OCR và phân tích cấu trúc giáo trình tiếng Việt. "
        f"{task_instruction}"
        "Các ảnh được cung cấp theo đúng thứ tự của CONTEXT_PAGES_JSON rồi TARGET_PAGES_JSON. "
        "CONTEXT chỉ giúp kế thừa hierarchy và TUYỆT ĐỐI không được xuất record cho chúng. "
        "Hãy trả đúng một record "
        f"cho mỗi pdf_page trong {expected}, không thiếu và không thêm trang. "
        "Dùng ngữ cảnh các trang liền nhau để hiểu tiêu đề xuyên trang, nhưng mọi element "
        "và correction chỉ được trỏ đến block_id có thật trên chính trang đó. "
        "Chỉ tạo correction khi ảnh chứng minh rõ OCR sai; không diễn giải, tóm tắt, hiện đại "
        "hóa hay thêm kiến thức. Không tạo correction chỉ để đổi khoảng trắng hoặc dấu câu nếu "
        "không ảnh hưởng nội dung. corrected_text phải chứa toàn bộ block sau khi sửa. "
        "Giữ correction confidence thấp nếu ảnh khó đọc. "
        "Mỗi trang chỉ xuất tối đa 8 correction quan trọng nhất, ưu tiên lỗi làm đổi nghĩa, "
        "tên riêng, mốc thời gian và heading; không cố sửa mọi ký tự nhỏ. "
        "Phân cấp heading: chapter=1; division (I., II.)=2; section (1., 2.)=3; "
        "subsection (a), b))=4. Phân biệt body, review, contents, footnote và page_number. "
        "Không cần liệt kê mọi body paragraph vào elements; chỉ liệt kê heading và vùng "
        "phi nội dung cần loại.\nCONTEXT_PAGES_JSON:\n"
        f"{json.dumps(context_payload, ensure_ascii=False, separators=(',', ':'))}"
        "\nTARGET_PAGES_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def validate_batch(
    result: CorrectedStructureBatch,
    pages: tuple[Page, ...],
) -> CorrectedStructureBatch:
    """Validate exact page coverage and all source anchors."""
    expected = [page.pdf_page for page in pages]
    received = [page.pdf_page for page in result.pages]
    if set(received) != set(expected):
        raise GeminiStructureError(
            f"Gemini returned pages {received}, expected exact coverage {expected}"
        )
    source_by_page = {page.pdf_page: page for page in pages}
    validated: list[CorrectedPageStructure] = []
    for output in result.pages:
        source = source_by_page[output.pdf_page]
        body_ids = {block.block_id for block in source.blocks if block.role is BlockRole.BODY}
        normalized_elements: list[GeminiStructureElement] = []
        dropped_elements = 0
        for element in output.elements:
            anchors = tuple(
                _canonical_anchor(block_id, body_ids) for block_id in element.source_block_ids
            )
            if set(anchors).difference(body_ids):
                dropped_elements += 1
                continue
            normalized_elements.append(element.model_copy(update={"source_block_ids": anchors}))
        warnings = output.warnings
        if dropped_elements:
            warnings = (
                *warnings,
                f"validator_dropped_{dropped_elements}_cross_page_or_unknown_anchor_elements",
            )
        structure = validate_page_anchors(
            GeminiPageStructure(
                pdf_page=output.pdf_page,
                page_role=output.page_role,
                elements=tuple(normalized_elements),
                warnings=warnings,
            ),
            source,
        )
        normalized_corrections = tuple(
            correction.model_copy(
                update={"block_id": _canonical_anchor(correction.block_id, body_ids)}
            )
            for correction in output.corrections
        )
        correction_ids = [item.block_id for item in normalized_corrections]
        unknown = set(correction_ids).difference(body_ids)
        if unknown:
            raise GeminiStructureError(
                f"Gemini corrections cite unknown block IDs: {sorted(unknown)}"
            )
        if len(correction_ids) != len(set(correction_ids)):
            raise GeminiStructureError("Gemini returned duplicate correction block IDs")
        validated.append(
            output.model_copy(
                update={
                    "elements": structure.elements,
                    "corrections": normalized_corrections,
                    "warnings": warnings,
                },
            )
        )
    validated.sort(key=lambda item: expected.index(item.pdf_page))
    return result.model_copy(update={"pages": tuple(validated)})


class GeminiCorpusV2Client:
    """Multimodal Gemini client for batches of up to three textbook pages."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        urlopen: UrlOpen = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._urlopen = urlopen

    def analyze_batch(
        self,
        pages: tuple[Page, ...],
        images_png: tuple[bytes, ...],
        *,
        context_pages: tuple[Page, ...] = (),
        context_images_png: tuple[bytes, ...] = (),
        structure_only: bool = False,
    ) -> CorrectedStructureBatch:
        """Correct up to five targets with at most one preceding context page."""
        if not 1 <= len(pages) <= 5 or len(images_png) != len(pages):
            raise ValueError("target pages and images must have equal length between one and five")
        if len(context_pages) > 1 or len(context_images_png) != len(context_pages):
            raise ValueError("at most one context page and matching image are allowed")
        parts: list[dict[str, Any]] = [
            {"text": batch_prompt(pages, context_pages, structure_only=structure_only)}
        ]
        for page, image in zip(context_pages, context_images_png, strict=True):
            parts.extend(
                [
                    {"text": f"CONTEXT_IMAGE_FOR_PDF_PAGE={page.pdf_page}"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image).decode("ascii"),
                        }
                    },
                ]
            )
        for page, image in zip(pages, images_png, strict=True):
            parts.extend(
                [
                    {"text": f"SOURCE_IMAGE_FOR_PDF_PAGE={page.pdf_page}"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image).decode("ascii"),
                        }
                    },
                ]
            )
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseJsonSchema": CorrectedStructureBatch.model_json_schema(),
            },
        }
        quoted_model = urllib.parse.quote(self.model, safe="")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quoted_model}:generateContent"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        payload: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise GeminiStructureError(
                        f"Gemini request failed with HTTP {exc.code}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 5.0 * 2**attempt
                )
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise GeminiStructureError("Gemini network request failed") from exc
                time.sleep(5.0 * 2**attempt)
        if payload is None:
            raise GeminiStructureError("Gemini returned no response")
        try:
            raw_text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            finish_reason = payload.get("candidates", [{}])[0].get("finishReason", "unknown")
            raise GeminiStructureError(
                f"Gemini response had no text; finish_reason={finish_reason}"
            ) from exc
        try:
            result = CorrectedStructureBatch.model_validate_json(raw_text)
        except ValueError as exc:
            detail = " ".join(str(exc).split())[:500]
            raise GeminiStructureError(f"Gemini batch schema validation failed: {detail}") from exc
        return validate_batch(result, pages)
