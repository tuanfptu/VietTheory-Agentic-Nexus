"""Tests for deterministic FAISS index artifacts."""

import hashlib
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from viettheory.retrieval.indexer import build_faiss_index
from viettheory.retrieval.models import IndexManifest, VectorMapping
from viettheory.schema import Chunk, SourceSpan


class FakeEmbedder:
    model_id = "fake/model"
    model_revision = "test-revision"

    def encode_documents(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        del batch_size
        return np.asarray(
            [[len(text), index + 1, 1] for index, text in enumerate(texts)], dtype=np.float32
        )


def _chunks() -> tuple[Chunk, ...]:
    return tuple(
        Chunk(
            chunk_id=f"chunk_{index}",
            document_id="doc_1",
            subject_code="TEST",
            text=f"text {index}",
            token_count=2,
            source_spans=(
                SourceSpan(
                    page_id="page_1", pdf_page=0, bbox=(1.0, 1.0, 2.0, 2.0), text=f"text {index}"
                ),
            ),
        )
        for index in range(3)
    )


def test_build_faiss_index_writes_normalized_searchable_artifacts(tmp_path: Path) -> None:
    chunks = _chunks()
    digest = hashlib.sha256(b"chunks").hexdigest()

    manifest = build_faiss_index(
        chunks, FakeEmbedder(), tmp_path, chunk_artifact_sha256=digest, batch_size=2
    )

    restored = IndexManifest.model_validate_json((tmp_path / "manifest.json").read_text())
    mappings = [
        VectorMapping.model_validate_json(line)
        for line in (tmp_path / "mapping.jsonl").read_text().splitlines()
    ]
    index = faiss.read_index(str(tmp_path / "index.faiss"))
    assert restored == manifest
    assert index.ntotal == 3
    assert index.d == 3
    assert [mapping.chunk_id for mapping in mappings] == [chunk.chunk_id for chunk in chunks]


def test_build_faiss_index_rejects_zero_vectors(tmp_path: Path) -> None:
    class ZeroEmbedder(FakeEmbedder):
        def encode_documents(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
            return np.zeros((len(texts), 3), dtype=np.float32)

    try:
        build_faiss_index(_chunks(), ZeroEmbedder(), tmp_path, chunk_artifact_sha256="0" * 64)
    except ValueError as error:
        assert "zero vectors" in str(error)
    else:
        raise AssertionError("Expected zero vectors to fail")
