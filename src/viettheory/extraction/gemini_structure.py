"""Gemini-assisted semantic structure extraction with deterministic guards."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viettheory.schema import BlockRole, Page


class PageRole(StrEnum):
    """Semantic role of one textbook page."""

    FRONT_MATTER = "front_matter"
    CONTENTS = "contents"
    CHAPTER_OPENING = "chapter_opening"
    BODY = "body"
    REVIEW = "review"
    REFERENCES = "references"
    BACK_MATTER = "back_matter"
    UNKNOWN = "unknown"


class StructureElementType(StrEnum):
    """Element classes needed by downstream hierarchy resolution."""

    CHAPTER = "chapter"
    DIVISION = "division"
    SECTION = "section"
    SUBSECTION = "subsection"
    BODY = "body"
    FOOTNOTE = "footnote"
    PAGE_NUMBER = "page_number"
    REVIEW_HEADING = "review_heading"
    OTHER = "other"


class GeminiStructureElement(BaseModel):
    """One semantic element anchored to existing OCR blocks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    element_type: StructureElementType
    level: int | None = Field(default=None, ge=1, le=5)
    text: str = Field(min_length=1)
    source_block_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_level(self) -> GeminiStructureElement:
        heading_types = {
            StructureElementType.CHAPTER,
            StructureElementType.DIVISION,
            StructureElementType.SECTION,
            StructureElementType.SUBSECTION,
        }
        if self.element_type in heading_types and self.level is None:
            raise ValueError("heading-like elements require a hierarchy level")
        if (
            self.element_type not in heading_types
            and self.element_type is not StructureElementType.REVIEW_HEADING
            and self.level is not None
        ):
            raise ValueError("non-heading elements cannot have a hierarchy level")
        return self


class GeminiPageStructure(BaseModel):
    """Schema-constrained response for one source page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    pdf_page: int = Field(ge=0)
    page_role: PageRole
    elements: tuple[GeminiStructureElement, ...]
    warnings: tuple[str, ...] = ()


class GeminiStructureError(RuntimeError):
    """Raised when Gemini cannot produce a validated structure result."""


UrlOpen = Callable[..., Any]

_ROMAN_PREFIX = re.compile(r"^[IVXLCDM]+\.\s+", re.IGNORECASE)
_NUMBERED_PREFIX = re.compile(r"^[1-9](?:\.\d+)*\.\s+")


def _canonicalize_heading(element: GeminiStructureElement) -> GeminiStructureElement:
    """Enforce textbook numbering invariants after semantic extraction."""
    text = " ".join(element.text.split())
    if text.casefold().startswith("chương "):
        return element.model_copy(update={"element_type": StructureElementType.CHAPTER, "level": 1})
    if _ROMAN_PREFIX.match(text):
        return element.model_copy(
            update={"element_type": StructureElementType.DIVISION, "level": 2}
        )
    if _NUMBERED_PREFIX.match(text):
        return element.model_copy(update={"element_type": StructureElementType.SECTION, "level": 3})
    return element


def structure_prompt(page: Page) -> str:
    """Build a strict, page-local prompt with immutable OCR anchors."""
    blocks = [
        {
            "block_id": block.block_id,
            "bbox": list(block.bbox),
            "text": block.text,
        }
        for block in page.blocks
        if block.role is BlockRole.BODY
    ]
    return (
        "Bạn là bộ phân tích cấu trúc giáo trình tiếng Việt. Hãy đối chiếu ẢNH TRANG "
        "với OCR_BLOCKS và chỉ mô tả cấu trúc thực sự nhìn thấy. Không bổ sung kiến thức. "
        "Mỗi element phải trỏ tới một hoặc nhiều block_id có thật trong OCR_BLOCKS. "
        "Giữ nguyên nội dung tiêu đề theo ảnh, chỉ sửa lỗi OCR rõ ràng trong trường text. "
        "Phân cấp: chapter=1, division=2, section=3, subsection=4; review_heading dùng "
        "cấp hiển thị tương ứng. Không gán body/footnote/page_number/other làm heading. "
        "Nếu trang chỉ là nội dung tiếp nối thì page_role=body và không cần tạo element "
        "cho mọi đoạn văn; ưu tiên các heading, footnote và vùng cần loại khỏi retrieval. "
        f"pdf_page bắt buộc bằng {page.pdf_page}.\nOCR_BLOCKS:\n"
        f"{json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))}"
    )


def validate_page_anchors(result: GeminiPageStructure, page: Page) -> GeminiPageStructure:
    """Reject invalid anchors and canonicalize page element order."""
    if result.pdf_page != page.pdf_page:
        raise GeminiStructureError(
            f"Gemini returned pdf_page={result.pdf_page}, expected {page.pdf_page}"
        )
    body_blocks = [block for block in page.blocks if block.role is BlockRole.BODY]
    block_order = {block.block_id: index for index, block in enumerate(body_blocks)}
    used_heading_blocks: set[str] = set()
    for element in result.elements:
        unknown = set(element.source_block_ids).difference(block_order)
        if unknown:
            raise GeminiStructureError(f"Gemini cited unknown block IDs: {sorted(unknown)}")
        indexes = [block_order[block_id] for block_id in element.source_block_ids]
        if indexes != sorted(indexes):
            raise GeminiStructureError("element source_block_ids are not in reading order")
        if element.level is not None:
            overlap = used_heading_blocks.intersection(element.source_block_ids)
            if overlap:
                raise GeminiStructureError(f"heading anchors reused: {sorted(overlap)}")
            used_heading_blocks.update(element.source_block_ids)
    canonical = (_canonicalize_heading(element) for element in result.elements)
    ordered = tuple(sorted(canonical, key=lambda item: block_order[item.source_block_ids[0]]))
    return result.model_copy(update={"elements": ordered})


class GeminiStructureClient:
    """Small REST client with schema constraints and bounded retry behavior."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 90.0,
        max_retries: int = 3,
        urlopen: UrlOpen = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries non-negative")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._urlopen = urlopen

    def analyze_page(self, page: Page, image_png: bytes) -> GeminiPageStructure:
        """Analyze one page image plus its existing OCR without persisting secrets."""
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(image_png).decode("ascii"),
                            }
                        },
                        {"text": structure_prompt(page)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseJsonSchema": GeminiPageStructure.model_json_schema(),
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
            result = GeminiPageStructure.model_validate_json(raw_text)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GeminiStructureError("Gemini returned an invalid structured response") from exc
        return validate_page_anchors(result, page)
