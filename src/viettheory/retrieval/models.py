"""Validated retrieval artifact metadata."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from viettheory.schema import NonEmptyText, VietTheoryModel


class VectorMapping(VietTheoryModel):
    vector_id: Annotated[int, Field(ge=0)]
    chunk_id: NonEmptyText


class IndexManifest(VietTheoryModel):
    model_id: NonEmptyText
    model_revision: NonEmptyText
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    normalized: Literal[True] = True
    dimension: Annotated[int, Field(gt=0)]
    vector_count: Annotated[int, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    chunk_artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    index_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    mapping_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
