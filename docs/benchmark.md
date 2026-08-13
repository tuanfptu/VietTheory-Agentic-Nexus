# MLN111 Retrieval Benchmark v1.0

## Release contract

Version `1.0.0` was frozen on 2026-08-12 against corpus artifact
`mln111_corpus_2026_07_1`. Public development data lives in
`benchmark/v1.0/mln111_development.jsonl`; hidden questions, answers, review records and detailed
reports live in the Git-ignored `benchmark_private/v1.0/` tree.

Any change to gold evidence or schema requires a new benchmark version. The release manifest
records split counts, review state, corpus identity, hashes, validation status and aggregate
metrics.

## Composition

| Dimension | Distribution |
|---|---|
| Split | 70 development, 30 hidden test |
| Difficulty (development) | 21 easy, 31 medium, 18 hard |
| Reasoning scope | 59 single-chunk, 11 multi-chunk |
| Answerability | 66 answerable, 2 false-premise, 2 out-of-scope |
| Chapter scope | 70 single-chapter |

Question-type tags are non-exclusive, so their counts can exceed 70: definition 8, explanation
10, paraphrase 9, synthesis 15, misconception 2, out-of-scope 2, temporal 3, entity 4,
cause–effect 8, enumeration 4, comparison 11 and application 1.

## Evidence model

A normal question may define primary and acceptable gold chunks. Multi-evidence questions use
required evidence groups; retrieval is fully successful only when at least one acceptable child
is returned from every required group. This avoids incorrectly declaring success after retrieving
only one half of a comparison or synthesis question.

## Evaluation configuration

- Lexical: BM25
- Dense: Qwen/Qwen3-Embedding-0.6B
- Fusion: Reciprocal Rank Fusion
- Reranker: Qwen/Qwen3-Reranker-0.6B
- Rerank candidate K: 12
- Evaluation top K: 10
- Configuration frozen before hidden evaluation

## Public development metrics

The development file has 70 questions. Retrieval metrics evaluate 68 retrieval-answerable items;
the two explicit out-of-scope questions are validated but excluded from the retrieval denominator.

| Metric | Value |
|---|---:|
| Recall@1 | 0.8382 |
| Recall@3 | 0.9853 |
| Recall@5 | 0.9853 |
| Recall@10 | 0.9853 |
| MRR | 0.9069 |
| nDCG@5 | 0.8878 |
| Evidence Group Recall@5 | 0.9367 |
| Partial Evidence Coverage@5 | 0.9559 |
| Full Evidence Success@5 | 0.9265 |
| Full Evidence Success@10 | 0.9412 |
| Latency p50 | 11.41 s |
| Latency p95 | 11.91 s |

## Development ablations

The controlled development ablation uses all 68 retrieval-answerable questions. Fixed-size gold
evidence is mapped from verified structured children through exact shared source spans. The full
machine-readable results and runner are available at
`benchmark/reports/mln111_v1_ablations.json` and `scripts/evaluate_mln111_ablations.py`.

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 | Full evidence@5 |
|---|---:|---:|---:|---:|---:|
| Fixed BM25 | 0.5294 | 0.8676 | 0.6693 | 0.5304 | 0.8676 |
| Fixed dense | 0.6618 | 0.8971 | 0.7645 | 0.5969 | 0.8676 |
| Fixed hybrid RRF | 0.7500 | 0.9559 | 0.8286 | 0.6372 | 0.9559 |
| Structured BM25 | 0.5147 | 0.8529 | 0.6633 | 0.6760 | 0.7794 |
| Structured dense | 0.5882 | 0.9265 | 0.7415 | 0.7534 | 0.8529 |
| Structured hybrid RRF | 0.6176 | 0.9559 | 0.7648 | 0.7819 | 0.8971 |
| Structured hybrid + reranker | 0.8382 | 0.9853 | 0.9069 | 0.8878 | 0.9265 |
| Structured planner + hybrid + reranker | 0.8382 | 0.9853 | 0.9069 | 0.8878 | 0.9265 |

These are quality-only ablations computed from cached query embeddings and rankings; their wall
times are not online latency measurements. The separately frozen canonical full-system report
above remains the release result used for development/hidden comparison.

## Hidden aggregate metrics

The hidden split has 30 verified questions. Its 28 retrieval-answerable items were evaluated once
after configuration freeze; two explicit out-of-scope items are excluded from the denominator.

| Metric | Value |
|---|---:|
| Recall@1 | 0.6429 |
| Recall@3 | 0.8214 |
| Recall@5 | 0.9286 |
| Recall@10 | 0.9286 |
| MRR | 0.7512 |
| nDCG@5 | 0.7804 |
| Evidence Group Recall@1 | 0.6000 |
| Evidence Group Recall@3 | 0.8000 |
| Evidence Group Recall@5 | 0.9333 |
| Evidence Group Recall@10 | 0.9333 |
| Partial Evidence Coverage@5 | 0.9286 |
| Full Evidence Success@5 | 0.9286 |
| Full Evidence Success@10 | 0.9286 |
| Latency p50 | 10.33 s |
| Latency p95 | 10.86 s |

The private hidden report SHA-256 is pinned in `release_manifest.json` and was verified before
publishing these aggregate values. Hidden questions, gold evidence and per-question results remain
private so the test set cannot become a tuning target.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\prepare_mln111_v1.py
.\.venv\Scripts\python.exe scripts\evaluate_mln111_v1.py `
  benchmark\v1.0\mln111_development.jsonl `
  --report benchmark\reports\mln111_v1_development_retrieval.json
```

Validation requires the local corpus artifacts and evaluation requires the local CUDA models.
The canonical public reports are `benchmark/reports/mln111_v1_development_validation.json` and
`benchmark/reports/mln111_v1_development_retrieval.json`.
