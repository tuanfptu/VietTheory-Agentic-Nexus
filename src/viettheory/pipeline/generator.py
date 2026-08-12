"""Structured generation adapters; credentials are read only at call time."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from viettheory.ids import stable_id
from viettheory.schema import Answer, Citation, RetrievedEvidence


class GenerationError(RuntimeError):
    """Raised when a provider cannot produce a valid grounded answer."""


class GeneratorAdapter(ABC):
    """Provider-neutral structured answer generator."""

    @abstractmethod
    def generate(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> Answer:
        """Generate an answer grounded only in supplied evidence."""


def _evidence_payload(evidence: tuple[RetrievedEvidence, ...]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "subject_code": item.chunk.subject_code,
            "text": item.chunk.text,
            "source_spans": [span.model_dump(mode="json") for span in item.chunk.source_spans],
        }
        for item in evidence
    ]


class GeminiGenerator(GeneratorAdapter):
    """Gemini REST adapter using JSON-schema constrained output."""

    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash-lite",
        api_key_env: str = "GEMINI_API_KEY",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> Answer:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not evidence:
            raise ValueError("evidence must not be empty")
        api_key = self._api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise GenerationError(f"missing required environment variable: {self.api_key_env}")

        prompt = (
            "Bạn là trợ lý học thuật chuyên giáo trình MLN111. "
            "Trả lời trực tiếp, tự nhiên và hữu ích bằng tiếng Việt, chỉ dựa trên evidence. "
            "Mỗi claim thực tế phải trỏ tới citation hợp lệ. "
            "Nếu evidence chỉ hỗ trợ một phần, hãy trả lời phần được hỗ trợ và nói ngắn gọn "
            "giới hạn còn lại; không từ chối toàn bộ khi đã có thể trả lời ý chính. "
            "Chỉ đặt refused=true khi evidence hoàn toàn không liên quan hoặc không thể tạo "
            "bất kỳ claim có căn cứ nào. Không nhắc tới từ 'evidence' trong câu trả lời.\n"
            f"Question: {question}\nEvidence JSON:\n"
            f"{json.dumps(_evidence_payload(evidence), ensure_ascii=False)}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": Answer.model_json_schema(),
                "temperature": 0.1,
            },
        }
        model = urllib.parse.quote(self.model, safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode())
            raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
            raw["answer_id"] = raw.get("answer_id") or stable_id("answer", question)
            raw["question"] = question
            # JSON has arrays, while the immutable internal contract uses tuples.
            # Relax coercion only at this untrusted provider boundary.
            answer = Answer.model_validate(raw, strict=False)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValidationError,
        ) as exc:
            raise GenerationError("Gemini returned no valid structured answer") from exc
        answer = _canonicalize_citations(answer, evidence)
        answer = _deduplicate_citations(answer)
        _require_retrieved_sources(answer, evidence)
        return answer


def _canonicalize_citations(
    answer: Answer,
    evidence: tuple[RetrievedEvidence, ...],
) -> Answer:
    """Replace provider-reconstructed geometry with the retrieved canonical span."""
    allowed = {item.evidence_id: item for item in evidence}
    citations: list[Citation] = []
    for citation in answer.citations:
        item = allowed.get(citation.evidence_id)
        if item is None:
            raise GenerationError("answer cites evidence outside retrieval")
        matches = tuple(
            span for span in item.chunk.source_spans if span.page_id == citation.source_span.page_id
        )
        if not matches:
            raise GenerationError("answer cites a page outside retrieved evidence")
        citations.append(
            citation.model_copy(
                update={
                    "source_span": matches[0],
                    # The span preserves exact page provenance. The parent
                    # passage gives readers enough surrounding textbook text
                    # to verify the answer after expanding a citation.
                    "context_text": item.chunk.text,
                }
            )
        )
    return answer.model_copy(update={"citations": tuple(citations)})


def _require_retrieved_sources(answer: Answer, evidence: tuple[RetrievedEvidence, ...]) -> None:
    allowed = {item.evidence_id: item for item in evidence}
    for citation in answer.citations:
        item = allowed.get(citation.evidence_id)
        if item is None or citation.source_span not in item.chunk.source_spans:
            raise GenerationError("answer cites a source outside retrieved evidence")


def _deduplicate_citations(answer: Answer) -> Answer:
    """Merge provider citations that resolve to the exact same source span.

    Gemini can emit several citation IDs for separate claims while pointing
    all of them at one retrieved page region. Keep the first citation and
    rewrite claim links so API clients receive one canonical source.
    """
    canonical_by_source: dict[tuple[object, ...], Citation] = {}
    aliases: dict[str, str] = {}
    citations: list[Citation] = []
    for citation in answer.citations:
        span = citation.source_span
        key = (
            citation.evidence_id,
            span.page_id,
            span.pdf_page,
            span.bbox,
            span.text,
        )
        canonical = canonical_by_source.get(key)
        if canonical is None:
            canonical_by_source[key] = citation
            citations.append(citation)
            aliases[citation.citation_id] = citation.citation_id
        else:
            aliases[citation.citation_id] = canonical.citation_id

    claims = tuple(
        claim.model_copy(
            update={
                "citation_ids": tuple(
                    dict.fromkeys(aliases[citation_id] for citation_id in claim.citation_ids)
                )
            }
        )
        for claim in answer.claims
    )
    return answer.model_copy(update={"claims": claims, "citations": tuple(citations)})
