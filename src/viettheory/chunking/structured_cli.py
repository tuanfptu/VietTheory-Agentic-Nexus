"""Create heading tree and parent-child chunks from extracted page JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from viettheory.chunking.manifest import build_structured_manifest
from viettheory.chunking.structured import (
    StructuredChunkingConfig,
    chunk_pages_structured,
)
from viettheory.extraction.structure_parser import HeadingOverride, parse_structure
from viettheory.schema import Page


def _write_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-target-tokens", type=int, default=400)
    parser.add_argument("--child-overlap-tokens", type=int, default=50)
    parser.add_argument("--parent-target-tokens", type=int, default=1500)
    parser.add_argument("--version", default="heading_parent_child_v1")
    parser.add_argument("--structure-overrides", type=Path)
    args = parser.parse_args()
    pages = tuple(
        Page.model_validate_json(line)
        for line in args.pages_jsonl.read_text(encoding="utf-8").splitlines()
    )
    config = StructuredChunkingConfig(
        child_target_tokens=args.child_target_tokens,
        child_overlap_tokens=args.child_overlap_tokens,
        parent_target_tokens=args.parent_target_tokens,
        version=args.version,
    )
    overrides = (
        tuple(
            HeadingOverride.model_validate(item)
            for item in json.loads(args.structure_overrides.read_text(encoding="utf-8"))
        )
        if args.structure_overrides is not None
        else ()
    )
    structure = parse_structure(pages, overrides)
    chunks = chunk_pages_structured(pages, config, structure=structure)
    _write_jsonl(args.output_dir / "headings.jsonl", structure.headings)
    _write_jsonl(args.output_dir / "parents.jsonl", chunks.parents)
    _write_jsonl(args.output_dir / "children.jsonl", chunks.children)
    manifest = build_structured_manifest(
        source_pages=args.pages_jsonl,
        output_dir=args.output_dir,
        config=config,
        structure_overrides=args.structure_overrides,
    )
    (args.output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(structure.headings)} headings, {len(chunks.parents)} parents "
        f"and {len(chunks.children)} children to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
