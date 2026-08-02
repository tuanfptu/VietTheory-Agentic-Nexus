"""Tests for deterministic structured artifact manifests."""

from pathlib import Path

from viettheory.chunking.manifest import build_structured_manifest, config_sha256
from viettheory.chunking.structured import StructuredChunkingConfig


def test_config_hash_is_deterministic_and_sensitive() -> None:
    baseline = StructuredChunkingConfig()
    changed = StructuredChunkingConfig(child_target_tokens=300)

    assert config_sha256(baseline) == config_sha256(StructuredChunkingConfig())
    assert config_sha256(baseline) != config_sha256(changed)


def test_manifest_hashes_all_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "pages.jsonl"
    source.write_text("page\n", encoding="utf-8")
    for name in ("headings.jsonl", "parents.jsonl", "children.jsonl"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    manifest = build_structured_manifest(
        source_pages=source,
        output_dir=tmp_path,
        config=StructuredChunkingConfig(),
    )

    assert len(manifest.config_sha256) == 64
    assert len(manifest.children_sha256) == 64
    assert manifest.child_target_tokens == 400
