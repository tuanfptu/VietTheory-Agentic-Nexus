"""Sentence Transformers adapter with explicit model identity."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str, *, model_id: str, revision: str, device: str = "cpu"):
        self.model_id = model_id
        self.model_revision = revision
        self._model = SentenceTransformer(model_path, device=device)

    def _encode(self, method: str, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        encoder = cast(Any, getattr(self._model, method))
        vectors = encoder(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        return cast("NDArray[np.float32]", np.asarray(cast(Any, vectors), dtype=np.float32))

    def encode_documents(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        return self._encode("encode_document", texts, batch_size=batch_size)

    def encode_queries(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        return self._encode("encode_query", texts, batch_size=batch_size)

    def encode(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        """Compatibility wrapper for index jobs started before query/document separation."""
        return self.encode_documents(texts, batch_size=batch_size)
