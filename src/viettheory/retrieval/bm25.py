"""Dependency-free Vietnamese-friendly BM25 retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from viettheory.schema import Chunk, RetrievedEvidence

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize Vietnamese text while retaining diacritics."""
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


class BM25Retriever:
    """In-memory Okapi BM25 index with optional subject filtering."""

    def __init__(self, chunks: Iterable[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("BM25 requires at least one chunk")
        self._term_frequencies = tuple(Counter(tokenize(chunk.text)) for chunk in self._chunks)
        self._lengths = tuple(sum(freq.values()) for freq in self._term_frequencies)
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(self._chunks)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._k1 = k1
        self._b = b

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        scored: list[tuple[float, Chunk]] = []
        for chunk, frequencies, length in zip(
            self._chunks, self._term_frequencies, self._lengths, strict=True
        ):
            if subject_codes and chunk.subject_code not in subject_codes:
                continue
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                score += self._idf[term] * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            RetrievedEvidence(
                evidence_id=f"bm25_{chunk.chunk_id}",
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="bm25",
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )
