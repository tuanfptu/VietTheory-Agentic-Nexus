"""Create baseline chunks from versioned page JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from viettheory.chunking.chunker import ChunkingConfig, chunk_pages
from viettheory.schema import Page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    args = parser.parse_args()

    pages = tuple(
        Page.model_validate_json(line)
        for line in args.pages_jsonl.read_text(encoding="utf-8").splitlines()
    )
    config = ChunkingConfig(args.target_tokens, args.overlap_tokens)
    chunks = chunk_pages(pages, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(chunk.model_dump_json() + "\n")
    print(f"Wrote {len(chunks)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
