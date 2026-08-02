"""Build reproducible cosine-similarity FAISS indexes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from numpy.typing import NDArray

from viettheory.retrieval.models import IndexManifest, VectorMapping
from viettheory.schema import Chunk

FloatMatrix = NDArray[np.float32]


class Embedder(Protocol):
    model_id: str
    model_revision: str

    def encode_documents(self, texts: list[str], *, batch_size: int) -> FloatMatrix: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_faiss_index(
    chunks: tuple[Chunk, ...],
    embedder: Embedder,
    output_dir: Path,
    *,
    chunk_artifact_sha256: str,
    batch_size: int = 8,
) -> IndexManifest:
    """Embed chunks once and persist normalized vectors plus deterministic mapping."""
    if not chunks:
        raise ValueError("cannot index an empty chunk collection")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vectors = np.asarray(
        embedder.encode_documents([chunk.text for chunk in chunks], batch_size=batch_size),
        dtype=np.float32,
        order="C",
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] <= 0:
        raise ValueError("embedder returned an invalid matrix shape")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("embeddings contain zero vectors")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    mapping_path = output_dir / "mapping.jsonl"
    faiss.write_index(index, str(index_path))
    with mapping_path.open("w", encoding="utf-8", newline="\n") as output:
        for vector_id, chunk in enumerate(chunks):
            mapping = VectorMapping(vector_id=vector_id, chunk_id=chunk.chunk_id)
            output.write(mapping.model_dump_json() + "\n")

    manifest = IndexManifest(
        model_id=embedder.model_id,
        model_revision=embedder.model_revision,
        dimension=int(vectors.shape[1]),
        vector_count=len(chunks),
        batch_size=batch_size,
        chunk_artifact_sha256=chunk_artifact_sha256,
        index_sha256=_sha256(index_path),
        mapping_sha256=_sha256(mapping_path),
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
