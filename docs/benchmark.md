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
| Recall@1 | 0.8235 |
| Recall@3 | 0.9706 |
| Recall@5 | 0.9706 |
| Recall@10 | 0.9706 |
| MRR | 0.8922 |
| nDCG@5 | 0.8770 |
| Evidence Group Recall@5 | 0.9367 |
| Partial Evidence Coverage@5 | 0.9485 |
| Full Evidence Success@5 | 0.9265 |
| Full Evidence Success@10 | 0.9412 |
| Latency p50 | 10.32 s |
| Latency p95 | 11.15 s |

Hidden aggregate Recall@5 and Full Evidence Success@5 are both `0.9286`. Hidden per-question
results remain private so the test set cannot become a tuning target.

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
