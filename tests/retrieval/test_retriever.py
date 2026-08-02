"""Tests for dense retrieval ranking and artifact integrity."""

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viettheory.retrieval.indexer import build_faiss_index
from viettheory.retrieval.retriever import DenseRetriever
from viettheory.schema import Chunk, SourceSpan


class DirectionalEmbedder:
    model_id = "fake/model"
    model_revision = "rev-1"

    def encode_documents(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        del batch_size
        return np.asarray(
            [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts], dtype=np.float32
        )

    def encode_queries(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        return self.encode_documents(texts, batch_size=batch_size)


def _write_chunks(path: Path) -> tuple[Chunk, ...]:
    chunks = tuple(
        Chunk(
            chunk_id=f"chunk_{name}",
            document_id="doc_1",
            subject_code="TEST",
            text=f"{name} evidence",
            token_count=2,
            source_spans=(
                SourceSpan(page_id="page_1", pdf_page=0, bbox=(1.0, 1.0, 2.0, 2.0), text=name),
            ),
        )
        for name in ("alpha", "beta")
    )
    path.write_text("".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8")
    return chunks


def test_dense_retriever_returns_ranked_chunk(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = _write_chunks(chunks_path)
    index_dir = tmp_path / "index"
    embedder = DirectionalEmbedder()
    build_faiss_index(
        chunks,
        embedder,
        index_dir,
        chunk_artifact_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    )

    results = DenseRetriever(index_dir, chunks_path, embedder).search("alpha question", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["chunk_alpha", "chunk_beta"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score
