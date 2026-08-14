"""Structured generation adapters; credentials are read only at call time."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from viettheory.ids import stable_id
from viettheory.schema import Answer, Citation, Claim, RetrievedEvidence


class GenerationError(RuntimeError):
    """Raised when a provider cannot produce a valid grounded answer."""


class GeneratedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    """Minimal provider contract; provenance is materialized locally."""

    model_config = ConfigDict(extra="forbid")

    direct_answer: str = Field(min_length=1)
    claims: tuple[GeneratedClaim, ...] = ()
    refused: bool = False
    refusal_reason: str | None = None


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
        corpus_label: str = "giáo trình MLN111",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        if not corpus_label.strip():
            raise ValueError("corpus_label must not be blank")
        self.corpus_label = corpus_label

    def generate(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> Answer:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not evidence:
            raise ValueError("evidence must not be empty")
        api_key = self._api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise GenerationError(f"missing required environment variable: {self.api_key_env}")

        prompt = (
            f"Bạn là trợ lý học thuật chuyên {self.corpus_label}. "
            "Trả lời trực tiếp, tự nhiên và hữu ích bằng tiếng Việt, chỉ dựa trên evidence. "
            "Mỗi claim thực tế phải liệt kê evidence_ids hợp lệ đúng như Evidence JSON. "
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
                "responseJsonSchema": GeneratedAnswer.model_json_schema(),
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
                response_text = response.read().decode()
        except urllib.error.HTTPError as exc:
            raise GenerationError(f"Gemini request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
            raise GenerationError("Gemini request failed") from exc
        try:
            payload = json.loads(response_text)
            generated_text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError("Gemini response contained no structured candidate") from exc
        try:
            raw = json.loads(generated_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise GenerationError("Gemini candidate was not JSON") from exc
        try:
            generated = GeneratedAnswer.model_validate(raw, strict=False)
        except ValidationError as exc:
            errors = [
                {"loc": error["loc"], "type": error["type"]}
                for error in exc.errors(include_input=False)
            ]
            raise GenerationError(
                "Gemini answer schema validation failed: " + json.dumps(errors, ensure_ascii=False)
            ) from exc
        answer = _materialize_answer(question, generated, evidence)
        _require_retrieved_sources(answer, evidence)
        return answer


def _normalize_schema_versions(value: Any) -> None:
    """Normalize provider-invented version aliases before strict validation."""
    if isinstance(value, dict):
        if "schema_version" in value:
            value["schema_version"] = "1.0"
        for child in value.values():
            _normalize_schema_versions(child)
    elif isinstance(value, list):
        for child in value:
            _normalize_schema_versions(child)


def _materialize_answer(
    question: str,
    generated: GeneratedAnswer,
    evidence: tuple[RetrievedEvidence, ...],
) -> Answer:
    """Build immutable claims/citations only from canonical retrieved evidence."""
    allowed = {item.evidence_id: item for item in evidence}
    used_ids = tuple(
        dict.fromkeys(
            evidence_id for claim in generated.claims for evidence_id in claim.evidence_ids
        )
    )
    unknown = set(used_ids) - allowed.keys()
    if unknown:
        raise GenerationError("generated answer cites evidence outside retrieval")
    citations = tuple(
        Citation(
            citation_id=stable_id("citation", question, evidence_id),
            evidence_id=evidence_id,
            source_span=allowed[evidence_id].chunk.source_spans[0],
            context_text=allowed[evidence_id].chunk.text,
        )
        for evidence_id in used_ids
    )
    citation_ids = {citation.evidence_id: citation.citation_id for citation in citations}
    claims = tuple(
        Claim(
            claim_id=stable_id("claim", question, index, claim.text),
            text=claim.text,
            citation_ids=tuple(citation_ids[evidence_id] for evidence_id in claim.evidence_ids),
        )
        for index, claim in enumerate(generated.claims, 1)
    )
    return Answer(
        answer_id=stable_id("answer", question, generated.direct_answer),
        question=question,
        direct_answer=generated.direct_answer,
        claims=claims,
        citations=citations,
        refused=generated.refused,
        refusal_reason=generated.refusal_reason,
    )


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
