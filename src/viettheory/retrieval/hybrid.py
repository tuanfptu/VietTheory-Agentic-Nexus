"""Hybrid retrieval using Reciprocal Rank Fusion and chunk-level deduplication."""

from __future__ import annotations

from typing import Protocol

from viettheory.schema import RetrievedEvidence


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


def reciprocal_rank_fusion(
    rankings: tuple[tuple[RetrievedEvidence, ...], ...],
    *,
    top_k: int = 20,
    rank_constant: int = 60,
    subject_codes: frozenset[str] | None = None,
) -> tuple[RetrievedEvidence, ...]:
    """Fuse ranked lists by stable chunk ID, independent of raw score scales."""
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k and rank_constant must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedEvidence] = {}
    for ranking in rankings:
        for item in ranking:
            if subject_codes and item.chunk.subject_code not in subject_codes:
                continue
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + item.rank)
            chunks.setdefault(chunk_id, item)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return tuple(
        RetrievedEvidence(
            evidence_id=f"hybrid_{chunk_id}",
            chunk=chunks[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            retrieval_method="rrf",
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    )


class HybridRetriever:
    """Run lexical and dense search, then fuse and filter the candidates."""

    def __init__(
        self,
        lexical: Searcher,
        dense: Searcher,
        *,
        candidate_k: int = 20,
        rank_constant: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._lexical = lexical
        self._dense = dense
        self._candidate_k = candidate_k
        self._rank_constant = rank_constant

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        lexical = self._lexical.search(query, top_k=self._candidate_k, subject_codes=subject_codes)
        dense = self._dense.search(query, top_k=self._candidate_k, subject_codes=subject_codes)
        return reciprocal_rank_fusion(
            (lexical, dense),
            top_k=top_k,
            rank_constant=self._rank_constant,
            subject_codes=subject_codes,
        )
