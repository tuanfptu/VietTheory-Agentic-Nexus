"""Resident cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder

from viettheory.schema import RetrievedEvidence

DEFAULT_RERANK_INSTRUCTION = (
    "Given a Vietnamese political-theory question, retrieve textbook passages "
    "that directly support an answer"
)


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> NDArray[np.float32]: ...


class CandidateRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


class QwenCrossEncoderScorer:
    """Load-once Qwen reranker using FP16 on CUDA."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_length: int = 512,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> None:
        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self._model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
            model_kwargs=model_kwargs,
            prompts={"viettheory": instruction},
            default_prompt_name="viettheory",
        )

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> NDArray[np.float32]:
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return cast(
            "NDArray[np.float32]",
            np.asarray(cast(Any, scores), dtype=np.float32),
        )


class Reranker:
    """Rerank candidates without losing source provenance."""

    def __init__(self, scorer: PairScorer, *, batch_size: int = 4) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._scorer = scorer
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: tuple[RetrievedEvidence, ...],
        *,
        top_k: int = 5,
    ) -> tuple[RetrievedEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return ()
        scores = self._scorer.predict(
            [(query, item.chunk.text) for item in candidates],
            batch_size=self._batch_size,
        )
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("reranker returned invalid scores")
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk.chunk_id),
        )[:top_k]
        return tuple(
            RetrievedEvidence(
                evidence_id=f"rerank_{item.chunk.chunk_id}",
                chunk=item.chunk,
                score=float(score),
                rank=rank,
                retrieval_method="qwen_reranker",
            )
            for rank, (item, score) in enumerate(ordered, start=1)
        )


class RerankedRetriever:
    """Adapter that makes hybrid retrieval plus reranking pipeline-compatible."""

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        reranker: Reranker,
        *,
        candidate_k: int = 12,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        candidates = self._candidate_retriever.search(
            query,
            top_k=self._candidate_k,
            subject_codes=subject_codes,
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)
