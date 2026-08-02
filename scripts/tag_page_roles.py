"""Tag repeated header/footer and page-number blocks in extracted JSONL pages."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from viettheory.extraction.postprocess import split_ocr_line_blocks, tag_marginal_roles
from viettheory.schema import ExtractionManifest, Page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repeat-ratio", type=float, default=0.8)
    args = parser.parse_args()

    pages = tuple(
        Page.model_validate_json(line)
        for line in args.pages_jsonl.read_text(encoding="utf-8").splitlines()
    )
    normalized = split_ocr_line_blocks(pages)
    tagged = tag_marginal_roles(normalized, repeat_ratio=args.repeat_ratio)
    destination = args.output or args.pages_jsonl
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for page in tagged:
            output.write(page.model_dump_json() + "\n")
    temporary.replace(destination)

    manifest_path = args.manifest or destination.with_suffix(".manifest.json")
    if manifest_path.is_file():
        manifest = ExtractionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        processors_to_add: tuple[str, ...] = ("marginal_roles_v1",)
        if any(page.extraction_method.value == "ocr" for page in pages):
            processors_to_add = ("ocr_line_blocks_v1", *processors_to_add)
        processors = tuple(dict.fromkeys((*manifest.postprocessors, *processors_to_add)))
        updated_manifest = manifest.model_copy(
            update={
                "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "postprocessors": processors,
            }
        )
        manifest_path.write_text(updated_manifest.model_dump_json(indent=2), encoding="utf-8")

    roles = Counter(block.role.value for page in tagged for block in page.blocks)
    print(
        f"Tagged {len(tagged)} pages: {dict(sorted(roles.items()))} "
        f"(manifest_updated={manifest_path.is_file()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
