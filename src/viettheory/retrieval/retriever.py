"""Integrity-checked dense retrieval over persisted FAISS artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from numpy.typing import NDArray

from viettheory.retrieval.models import IndexManifest, VectorMapping
from viettheory.schema import Chunk, RetrievedEvidence


class QueryEmbedder(Protocol):
    model_id: str
    model_revision: str

    def encode_queries(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DenseRetriever:
    """Load-once exact cosine retriever with validated vector-to-chunk mapping."""

    def __init__(
        self,
        index_dir: Path,
        chunks_path: Path,
        embedder: QueryEmbedder,
    ) -> None:
        self._manifest = IndexManifest.model_validate_json(
            (index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if embedder.model_id != self._manifest.model_id:
            raise ValueError("query embedder model does not match index model")
        if embedder.model_revision != self._manifest.model_revision:
            raise ValueError("query embedder revision does not match index revision")
        index_path = index_dir / "index.faiss"
        mapping_path = index_dir / "mapping.jsonl"
        if _sha256(index_path) != self._manifest.index_sha256:
            raise ValueError("FAISS index checksum mismatch")
        if _sha256(mapping_path) != self._manifest.mapping_sha256:
            raise ValueError("vector mapping checksum mismatch")
        if _sha256(chunks_path) != self._manifest.chunk_artifact_sha256:
            raise ValueError("chunk artifact checksum mismatch")

        self._index = faiss.read_index(str(index_path))
        mappings = tuple(
            VectorMapping.model_validate_json(line)
            for line in mapping_path.read_text(encoding="utf-8").splitlines()
        )
        chunks = tuple(
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
        )
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(mappings) != self._manifest.vector_count or self._index.ntotal != len(mappings):
            raise ValueError("index, manifest, and mapping vector counts differ")
        self._chunks = tuple(chunk_by_id[mapping.chunk_id] for mapping in mappings)
        self._embedder = embedder

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievedEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(
            self._embedder.encode_queries([query], batch_size=1),
            dtype=np.float32,
            order="C",
        )
        if vector.shape != (1, self._manifest.dimension):
            raise ValueError("query embedder returned an invalid shape")
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("query embedding must be finite and non-zero")
        faiss.normalize_L2(vector)
        limit = min(top_k, len(self._chunks))
        scores, vector_ids = self._index.search(vector, limit)
        return tuple(
            RetrievedEvidence(
                evidence_id=f"dense_{int(vector_id)}",
                chunk=self._chunks[int(vector_id)],
                score=float(score),
                rank=rank,
                retrieval_method="dense",
            )
            for rank, (score, vector_id) in enumerate(
                zip(scores[0], vector_ids[0], strict=True), start=1
            )
            if vector_id >= 0
        )
