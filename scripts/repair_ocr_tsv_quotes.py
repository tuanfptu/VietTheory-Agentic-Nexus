"""Re-OCR pages corrupted by TSV quote interpretation and refresh manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from viettheory.extraction.tesseract_ocr import TesseractOcr, iter_ocr_pages
from viettheory.schema import ExtractionManifest, Page


def _is_corrupted(page: Page) -> bool:
    return "\t1\t" in page.text or "\r\n5\t" in page.text or page.char_count > 5_000


def main() -> int:
    root = Path.cwd()
    engine = TesseractOcr(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        root / "models" / "tesseract",
    )
    for subject in ("HCM202", "MLN131", "VNR202"):
        directory = root / "data" / "processed" / subject
        artifact = directory / "pages.jsonl"
        manifest_path = directory / "pages.manifest.json"
        pages = [
            Page.model_validate_json(line)
            for line in artifact.read_text(encoding="utf-8").splitlines()
        ]
        affected = [page.pdf_page for page in pages if _is_corrupted(page)]
        if not affected:
            print(f"{subject}: no corrupted pages")
            continue

        pdf_path = next((root / "Tài liệu").glob(f"*{subject}.pdf"))
        replacements = {
            page_index: next(
                iter_ocr_pages(
                    pdf_path,
                    subject,
                    engine,
                    start_page=page_index,
                    end_page=page_index + 1,
                )
            )
            for page_index in affected
        }
        repaired = [replacements.get(page.pdf_page, page) for page in pages]
        temporary = artifact.with_suffix(".repairing")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for page in repaired:
                output.write(json.dumps(page.model_dump(mode="json"), ensure_ascii=False) + "\n")
        temporary.replace(artifact)

        manifest = ExtractionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        postprocessors = (*manifest.postprocessors, "tsv_quote_repair_v1")
        refreshed = manifest.model_copy(
            update={
                "postprocessors": postprocessors,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
        manifest_path.write_text(refreshed.model_dump_json(indent=2), encoding="utf-8")
        print(f"{subject}: repaired pages {affected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
