"""Validate five-subject corpus artifacts and emit a deterministic readiness report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from viettheory.chunking.manifest import StructuredArtifactManifest
from viettheory.retrieval.models import IndexManifest, VectorMapping
from viettheory.schema import Chunk, ExtractionManifest
from viettheory.subjects import SUBJECTS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, model: type[Chunk] | type[VectorMapping]) -> tuple[Any, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def audit_subject(root: Path, code: str) -> dict[str, Any]:
    spec = next(subject for subject in SUBJECTS if subject.code == code)
    corpus = root / "data" / "processed" / code
    structured = corpus / "structured_v1"
    index_dir = structured / "dense_index"
    paths = {
        "pages": corpus / "pages.jsonl",
        "pages_manifest": corpus / "pages.manifest.json",
        "headings": structured / "headings.jsonl",
        "parents": structured / "parents.jsonl",
        "children": structured / "children.jsonl",
        "structured_manifest": structured / "manifest.json",
        "index": index_dir / "index.faiss",
        "mapping": index_dir / "mapping.jsonl",
        "index_manifest": index_dir / "manifest.json",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        return {"subject": code, "ready": False, "missing_artifacts": missing}

    extraction = ExtractionManifest.model_validate_json(
        paths["pages_manifest"].read_text(encoding="utf-8")
    )
    chunk_manifest = StructuredArtifactManifest.model_validate_json(
        paths["structured_manifest"].read_text(encoding="utf-8")
    )
    index_manifest = IndexManifest.model_validate_json(
        paths["index_manifest"].read_text(encoding="utf-8")
    )
    parents = _jsonl(paths["parents"], Chunk)
    children = _jsonl(paths["children"], Chunk)
    mappings = _jsonl(paths["mapping"], VectorMapping)
    heading_rows = tuple(
        json.loads(line)
        for line in paths["headings"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    chapter_count = sum(row.get("level") == 1 for row in heading_rows)
    parent_ids = {chunk.chunk_id for chunk in parents}
    child_ids = {chunk.chunk_id for chunk in children}
    mapped_ids = {mapping.chunk_id for mapping in mappings}
    checks = {
        "subject_identity": extraction.document.subject_code == code,
        "page_count": extraction.extracted_page_count == spec.expected_pages,
        "extraction_mode": extraction.extractor
        == ("pymupdf" if spec.extraction_mode.value == "native" else "tesseract"),
        "pages_checksum": _sha256(paths["pages"]) == extraction.artifact_sha256,
        "structured_source_checksum": (
            chunk_manifest.source_pages_sha256 == extraction.artifact_sha256
        ),
        "headings_checksum": _sha256(paths["headings"]) == chunk_manifest.headings_sha256,
        "parents_checksum": _sha256(paths["parents"]) == chunk_manifest.parents_sha256,
        "children_checksum": _sha256(paths["children"]) == chunk_manifest.children_sha256,
        "index_chunk_checksum": (
            index_manifest.chunk_artifact_sha256 == chunk_manifest.children_sha256
        ),
        "index_checksum": _sha256(paths["index"]) == index_manifest.index_sha256,
        "mapping_checksum": _sha256(paths["mapping"]) == index_manifest.mapping_sha256,
        "vector_count": len(mappings) == index_manifest.vector_count,
        "mapping_coverage": mapped_ids == child_ids,
        "parent_links": all(child.parent_chunk_id in parent_ids for child in children),
        "subject_purity": all(chunk.subject_code == code for chunk in (*parents, *children)),
        "chapter_count": chapter_count == spec.expected_chapters,
    }
    return {
        "subject": code,
        "name": spec.name_en,
        "extraction_mode": spec.extraction_mode.value,
        "ready": all(checks.values()),
        "checks": checks,
        "counts": {
            "pages": extraction.extracted_page_count,
            "headings": len(heading_rows),
            "chapters": chapter_count,
            "parents": len(parents),
            "children": len(children),
            "vectors": index_manifest.vector_count,
        },
        "model": {
            "id": index_manifest.model_id,
            "revision": index_manifest.model_revision,
            "dimension": index_manifest.dimension,
        },
    }


def build_report(root: Path) -> dict[str, Any]:
    subjects = [audit_subject(root, subject.code) for subject in SUBJECTS]
    return {
        "schema_version": "1.0",
        "registry_version": "five_subject_v1",
        "all_subjects_ready": all(subject["ready"] for subject in subjects),
        "subjects": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/five_subject_readiness.json"))
    args = parser.parse_args()
    report = build_report(Path.cwd().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    if not report["all_subjects_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
