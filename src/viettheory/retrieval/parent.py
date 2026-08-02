"""Expand ranked child evidence to bounded parent context."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from viettheory.schema import Chunk, RetrievedEvidence


class ParentChunkStore:
    """Validated in-memory parent lookup."""

    def __init__(self, parents: Iterable[Chunk]) -> None:
        parent_list = tuple(parents)
        if any(parent.chunk_kind != "parent" for parent in parent_list):
            raise ValueError("parent store accepts only parent chunks")
        self._parents = {parent.chunk_id: parent for parent in parent_list}
        if len(self._parents) != len(parent_list):
            raise ValueError("parent chunk IDs must be unique")

    @classmethod
    def from_jsonl(cls, path: Path) -> ParentChunkStore:
        return cls(
            Chunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )

    def get(self, parent_chunk_id: str) -> Chunk:
        try:
            return self._parents[parent_chunk_id]
        except KeyError as exc:
            raise ValueError(f"unknown parent chunk: {parent_chunk_id}") from exc


def expand_to_parents(
    child_evidence: tuple[RetrievedEvidence, ...],
    store: ParentChunkStore,
    *,
    top_k: int = 5,
) -> tuple[RetrievedEvidence, ...]:
    """Map ranked children to unique parents, retaining first/highest-ranked score."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    expanded: list[tuple[Chunk, float]] = []
    seen: set[str] = set()
    for evidence in child_evidence:
        parent_id = evidence.chunk.parent_chunk_id
        if evidence.chunk.chunk_kind != "child" or parent_id is None:
            raise ValueError("parent expansion requires child evidence")
        if parent_id in seen:
            continue
        seen.add(parent_id)
        expanded.append((store.get(parent_id), evidence.score))
        if len(expanded) == top_k:
            break
    return tuple(
        RetrievedEvidence(
            evidence_id=f"parent_{parent.chunk_id}",
            chunk=parent,
            score=score,
            rank=rank,
            retrieval_method="parent_expansion",
        )
        for rank, (parent, score) in enumerate(expanded, start=1)
    )
