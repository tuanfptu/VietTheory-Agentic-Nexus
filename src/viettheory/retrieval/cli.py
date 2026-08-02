"""Build a normalized FAISS index from chunk JSONL."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from viettheory.retrieval.hardware import detect_runtime
from viettheory.retrieval.indexer import build_faiss_index
from viettheory.retrieval.sentence_transformer import SentenceTransformerEmbedder
from viettheory.schema import Chunk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the hardware-aware document batch size.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    runtime = detect_runtime(args.device)
    batch_size = args.batch_size or runtime.document_batch_size

    chunks = tuple(
        Chunk.model_validate_json(line)
        for line in args.chunks_jsonl.read_text(encoding="utf-8").splitlines()
    )
    digest = hashlib.sha256(args.chunks_jsonl.read_bytes()).hexdigest()
    embedder = SentenceTransformerEmbedder(
        args.model_path,
        model_id=args.model_id,
        revision=args.revision,
        device=runtime.device,
    )
    manifest = build_faiss_index(
        chunks,
        embedder,
        args.output_dir,
        chunk_artifact_sha256=digest,
        batch_size=batch_size,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
