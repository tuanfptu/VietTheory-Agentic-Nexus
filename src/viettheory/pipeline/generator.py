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
from viettheory.schema import Answer, RetrievedEvidence


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
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> Answer:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not evidence:
            raise ValueError("evidence must not be empty")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise GenerationError(f"missing required environment variable: {self.api_key_env}")

        prompt = (
            "Bạn là trợ lý học thuật. Chỉ dùng evidence được cung cấp. "
            "Mỗi claim phải trỏ tới citation hợp lệ. Nếu evidence không đủ, từ chối.\n"
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
            answer = Answer.model_validate(raw)
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
        _require_retrieved_sources(answer, evidence)
        return answer


def _require_retrieved_sources(answer: Answer, evidence: tuple[RetrievedEvidence, ...]) -> None:
    allowed = {item.evidence_id: item for item in evidence}
    for citation in answer.citations:
        item = allowed.get(citation.evidence_id)
        if item is None or citation.source_span not in item.chunk.source_spans:
            raise GenerationError("answer cites a source outside retrieved evidence")
