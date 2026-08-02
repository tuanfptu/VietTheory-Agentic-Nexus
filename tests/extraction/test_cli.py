"""End-to-end tests for extraction artifacts and manifests."""

import hashlib
from pathlib import Path

import pymupdf

from viettheory.extraction.cli import main
from viettheory.schema import ExtractionManifest, Page


def test_cli_writes_linked_page_artifact_and_manifest(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    page = document.new_page()
    page.insert_text((50, 50), "Manifest fixture")
    document.save(pdf_path)  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]
    output = tmp_path / "pages.jsonl"

    result = main([str(pdf_path), "--subject", "TEST", "--output", str(output)])

    assert result == 0
    extracted_page = Page.model_validate_json(output.read_text(encoding="utf-8").strip())
    manifest_path = tmp_path / "pages.manifest.json"
    manifest = ExtractionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.document.document_id == extracted_page.document_id
    assert manifest.extracted_page_count == 1
    assert manifest.artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
