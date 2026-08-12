# MLN111 Benchmark v1.0

This frozen release contains 70 human-reviewed development questions and a private
30-question human-verified hidden test. The corpus and all evidence IDs validate against artifact
`mln111_corpus_2026_07_1` without staleness.

The retrieval configuration was frozen before the hidden evaluation. Hidden metrics were
not used to tune the system. Hidden questions, gold answers, review records, and detailed
hidden reports stay under the Git-ignored `benchmark_private/` tree. Any gold or schema
change now requires a new benchmark version.

## Public development result

- Recall@1: 0.8235
- Recall@3: 0.9706
- Recall@5: 0.9706
- MRR: 0.8922
- nDCG@5: 0.8770
- Full Evidence Success@5: 0.9265
- Latency p50: 10.32 seconds

See `release_manifest.json` for integrity hashes and release status.
