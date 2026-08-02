# GTX 1660 Ti Runtime Optimization

## Objective

Maximize useful retrieval throughput on the 6 GB GPU while preserving enough
headroom to keep both the embedding model and reranker resident. High utilization
alone is not the target; paging and oversized batches reduce throughput.

## Embedding benchmark

Qwen3-Embedding-0.6B was benchmarked on 128 real MLN111 chunks:

| Batch | Documents/s | Peak allocated VRAM |
|------:|------------:|--------------------:|
| 8 | 3.945 | 2,279.8 MB |
| 16 | 3.900 | 3,410.8 MB |
| 24 | 1.468 | 4,544.5 MB |
| 32 | 0.988 | 5,675.9 MB |
| 40 | 0.468 | 6,811.2 MB* |

`*` The final value includes memory pressure beyond dedicated VRAM and corresponds
to severe slowdown.

Batch 8 is the selected indexing default for GPUs with at most 8 GB VRAM. Query
embedding uses batch 1.

## Resident model test

- Embedding model after query: approximately 1,144.5 MB allocated.
- Embedding plus FP16 Qwen3-Reranker-0.6B: approximately 2,280.9 MB resident.
- Relevant/irrelevant Vietnamese smoke pair logits: `6.4297` and `-11.0781`.

Both models can remain on the GTX 1660 Ti without request-time load/unload.

## Reranker latency benchmark

Twenty candidates at max length 1024 took about 18.9–19.9 seconds for batch sizes
1–8. Increasing batch size did not materially improve latency.

The selected runtime policy is:

- retrieve/fuse up to 12 candidates;
- reranker batch size 4;
- maximum sequence length 512;
- return top 5;
- FP16 reranker on CUDA;
- embedding and reranker remain resident;
- FAISS and BM25 remain on CPU.

This configuration reranked 12 real candidates in approximately 9.9 seconds with
about 2.75 GB peak allocated VRAM and retained the expected source pages.

## Implementation

- Index CLI defaults to hardware-aware `--device auto`.
- GTX 1660 Ti resolves to CUDA, document batch 8, query batch 1.
- Explicit `--device cpu` and `--device cuda` remain available.
- Reranker preserves chunk source spans and is pipeline-compatible through
  `RerankedRetriever`.

## Gate result

**PASS for hardware-aware GPU operation and resident dual-model feasibility.**
Final latency/quality decisions remain subject to reviewed dev-set evaluation.
