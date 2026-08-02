"""Benchmark document-embedding throughput and peak VRAM by batch size."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from viettheory.schema import Chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    chunks = [
        Chunk.model_validate_json(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
    ]
    texts = [chunk.text for chunk in chunks[: args.sample_size]]
    model = SentenceTransformer(args.model_path, device="cuda")
    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            vectors = model.encode_document(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "documents": len(texts),
                    "dimension": int(vectors.shape[1]),
                    "elapsed_seconds": round(elapsed, 3),
                    "documents_per_second": round(len(texts) / elapsed, 3),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                }
            )
        except torch.cuda.OutOfMemoryError:
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "documents": len(texts),
                }
            )
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "total_vram_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
