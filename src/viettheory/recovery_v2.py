"""Contracts and conservative fusion for evidence-guided Recovery V2."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Protocol

from pydantic import Field, model_validator

from viettheory.schema import NonEmptyText, RetrievedEvidence, VietTheoryModel


class RecoveryPlan(VietTheoryModel):
    """A bounded plan inferred from the question and currently missing aspects."""

    request_id: NonEmptyText
    activate: bool
    required_aspects: tuple[NonEmptyText, ...]
    missing_aspects: tuple[NonEmptyText, ...]
    targeted_queries: tuple[NonEmptyText, ...] = Field(max_length=2)
    rationale: NonEmptyText

    @model_validator(mode="after")
    def validate_activation(self) -> RecoveryPlan:
        if self.activate and not self.targeted_queries:
            raise ValueError("activated plans require at least one targeted query")
        if not self.activate and self.targeted_queries:
            raise ValueError("inactive plans cannot contain targeted queries")
        if self.activate and not self.missing_aspects:
            raise ValueError("activated plans require a missing aspect")
        return self


class RecoveryPlanBatch(VietTheoryModel):
    plans: tuple[RecoveryPlan, ...]


class Planner(Protocol):
    def plan(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> RecoveryPlan: ...


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


class SupportScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> Any: ...


class GeminiRecoveryPlanner:
    """Gold-free J1 + query planner for one production request."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 90.0,
    ) -> None:
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._model = model
        self._timeout = timeout_seconds

    def plan(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> RecoveryPlan:
        api_key = self._api_key or os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable: {self._api_key_env}")
        request_id = "runtime_request"
        cases = [
            {
                "request_id": request_id,
                "question": question,
                "contexts": [
                    {"context_id": item.chunk.chunk_id, "text": item.chunk.text}
                    for item in evidence[:5]
                ],
            }
        ]
        prompt = (
            "Judge evidence sufficiency for Vietnamese educational QA using only supplied "
            "contexts. Activate recovery only when a precise answer aspect is absent. If active, "
            "produce at most two concise, self-contained Vietnamese search queries, one per "
            "missing aspect. Preserve request_id. Never use outside knowledge.\nCases JSON:\n"
            + json.dumps(cases, ensure_ascii=False)
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": RecoveryPlanBatch.model_json_schema(),
                "temperature": 0.0,
            },
        }
        model = urllib.parse.quote(self._model, safe="")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode())
            raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
            raw["schema_version"] = "1.0"
            for plan in raw.get("plans", []):
                plan["schema_version"] = "1.0"
            batch = RecoveryPlanBatch.model_validate(raw, strict=False)
        except (OSError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Gemini returned no valid recovery plan") from exc
        if len(batch.plans) != 1 or batch.plans[0].request_id != request_id:
            raise RuntimeError("Gemini returned an invalid recovery plan ID")
        return batch.plans[0]


class EvidenceGuidedRecoveryRetriever:
    """B0 plus conservative, missing-aspect support-gated insertion."""

    def __init__(
        self,
        base: Retriever,
        planner: Planner,
        scorer: SupportScorer,
        *,
        recovery_top_k: int = 5,
        support_margin: float = 0.0,
        scorer_batch_size: int = 4,
    ) -> None:
        self._base = base
        self._planner = planner
        self._scorer = scorer
        self._recovery_top_k = recovery_top_k
        self._support_margin = support_margin
        self._batch_size = scorer_batch_size

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        original = self._base.search(query, top_k=max(10, top_k), subject_codes=subject_codes)
        try:
            plan = self._planner.plan(query, original[:5])
        except RuntimeError:
            return original[:top_k]
        if not plan.activate:
            return original[:top_k]
        new = _new_recovery_evidence(
            self._base,
            plan,
            original,
            top_k=self._recovery_top_k,
            subject_codes=subject_codes,
        )
        accepted = _support_gated_insertions(
            plan,
            original[:5],
            new,
            self._scorer,
            margin=self._support_margin,
            batch_size=self._batch_size,
        )
        keep = max(0, top_k - len(accepted))
        selected = (*original[:keep], *accepted)
        return tuple(
            item.model_copy(update={"rank": rank}) for rank, item in enumerate(selected, 1)
        )


def _new_recovery_evidence(
    retriever: Retriever,
    plan: RecoveryPlan,
    original: tuple[RetrievedEvidence, ...],
    *,
    top_k: int,
    subject_codes: frozenset[str] | None,
) -> tuple[RetrievedEvidence, ...]:
    original_ids = {item.chunk.chunk_id for item in original}
    unique: dict[str, RetrievedEvidence] = {}
    for query in plan.targeted_queries:
        for item in retriever.search(query, top_k=top_k, subject_codes=subject_codes):
            if item.chunk.chunk_id not in original_ids:
                unique.setdefault(item.chunk.chunk_id, item)
    return tuple(unique.values())


def _support_gated_insertions(
    plan: RecoveryPlan,
    original: tuple[RetrievedEvidence, ...],
    new: tuple[RetrievedEvidence, ...],
    scorer: SupportScorer,
    *,
    margin: float,
    batch_size: int,
) -> tuple[RetrievedEvidence, ...]:
    accepted: list[RetrievedEvidence] = []
    for query in plan.targeted_queries:
        candidates = (*original, *new)
        if not new:
            break
        scores = scorer.predict(
            [(query, item.chunk.text) for item in candidates], batch_size=batch_size
        )
        values = [float(score) for score in scores]
        baseline_best = max(values[: len(original)])
        new_scores = values[len(original) :]
        best_index = max(range(len(new)), key=new_scores.__getitem__)
        candidate = new[best_index]
        if new_scores[best_index] > baseline_best + margin and candidate not in accepted:
            accepted.append(candidate)
    return tuple(accepted[:2])


def reciprocal_rank_fuse(
    rankings: tuple[tuple[RetrievedEvidence, ...], ...],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> tuple[RetrievedEvidence, ...]:
    """Fuse original and recovery rankings without trusting incomparable raw scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = {}
    evidence: dict[str, RetrievedEvidence] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            evidence.setdefault(chunk_id, item)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return tuple(
        evidence[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
                "rank": rank,
                "evidence_id": f"recovery_v2_{chunk_id}",
                "retrieval_method": "recovery_v2_rrf",
            }
        )
        for rank, chunk_id in enumerate(ordered, 1)
    )
