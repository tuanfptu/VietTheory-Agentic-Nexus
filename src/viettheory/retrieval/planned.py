"""MLN111 query planning and reranking."""

from __future__ import annotations

import re

from viettheory.retrieval.reranker import CandidateRetriever, Reranker
from viettheory.schema import RetrievedEvidence


class PlannedRerankedRetriever:
    """Retrieve one or both sides of an MLN111 question, then rerank once."""

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
        if subject_codes and subject_codes != frozenset({"MLN111"}):
            return ()
        rankings = tuple(
            tuple(
                item
                for item in self._candidate_retriever.search(
                    variant,
                    top_k=self._candidate_k * 2,
                    subject_codes=frozenset({"MLN111"}),
                )
                if item.chunk.chapter is not None
            )
            for variant in comparison_query_variants(query)
        )
        candidates = _round_robin_unique(rankings, top_k=self._candidate_k)
        return self._reranker.rerank(query, candidates, top_k=top_k)


def comparison_query_variants(query: str) -> tuple[str, ...]:
    """Split a simple comparison so both concepts receive retrieval candidates."""
    normalized = " ".join(query.split()).strip(" .?!")
    match = re.match(r"(?i)^so sánh\s+(.+?)\s+(?:và|với)\s+(.+)$", normalized)
    if match is None:
        return (query,)
    left, right = (part.strip(" ,;:") for part in match.groups())
    if len(left) < 8 or len(right) < 8:
        return (query,)
    return (left, right)


def _round_robin_unique(
    rankings: tuple[tuple[RetrievedEvidence, ...], ...], *, top_k: int
) -> tuple[RetrievedEvidence, ...]:
    selected: list[RetrievedEvidence] = []
    seen: set[str] = set()
    depth = 0
    while len(selected) < top_k and any(depth < len(ranking) for ranking in rankings):
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            item = ranking[depth]
            if item.chunk.chunk_id not in seen:
                seen.add(item.chunk.chunk_id)
                selected.append(item)
                if len(selected) == top_k:
                    break
        depth += 1
    return tuple(selected)
