"""Reproducibility manifest for heading-aware parent-child artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from viettheory.chunking.structured import StructuredChunkingConfig

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StructuredArtifactManifest(BaseModel):
    """Immutable identity of one structured retrieval corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    chunk_schema_version: Literal["1.0"] = "1.0"
    chunking_version: str
    parent_target_tokens: int = Field(gt=0)
    child_target_tokens: int = Field(gt=0)
    child_overlap_tokens: int = Field(ge=0)
    source_pages_sha256: Sha256
    headings_sha256: Sha256
    parents_sha256: Sha256
    children_sha256: Sha256
    config_sha256: Sha256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha256(config: StructuredChunkingConfig) -> str:
    """Hash canonical JSON rather than an implementation-specific repr."""
    payload = {
        "child_overlap_tokens": config.child_overlap_tokens,
        "child_target_tokens": config.child_target_tokens,
        "parent_target_tokens": config.parent_target_tokens,
        "version": config.version,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_structured_manifest(
    *,
    source_pages: Path,
    output_dir: Path,
    config: StructuredChunkingConfig,
) -> StructuredArtifactManifest:
    return StructuredArtifactManifest(
        chunking_version=config.version,
        parent_target_tokens=config.parent_target_tokens,
        child_target_tokens=config.child_target_tokens,
        child_overlap_tokens=config.child_overlap_tokens,
        source_pages_sha256=sha256_file(source_pages),
        headings_sha256=sha256_file(output_dir / "headings.jsonl"),
        parents_sha256=sha256_file(output_dir / "parents.jsonl"),
        children_sha256=sha256_file(output_dir / "children.jsonl"),
        config_sha256=config_sha256(config),
    )
