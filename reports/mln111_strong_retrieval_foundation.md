# MLN111 Strong Retrieval Foundation Gate

## Scope

This gate covers the local, model-independent components required around dense
retrieval. It does not claim reranker quality or benchmark accuracy.

## Implemented

- Unicode-aware Vietnamese tokenization and Okapi BM25.
- Optional subject filtering.
- Reciprocal Rank Fusion over BM25 and dense rankings.
- Stable chunk-level deduplication independent of raw score scales.
- Rule-based pre-routing for subject, question type, cross-course intent and only
  obvious out-of-domain queries.
- Evidence decisions with configurable thresholds and at most one rewrite.
- Deterministic threshold selection from labeled development examples.

## Verification

- Ruff lint: pass.
- Ruff format check: pass.
- mypy strict: pass.
- pytest: 31 passed.
- Tests cover Vietnamese diacritics, lexical ranking, subject filtering, RRF
  consensus/deduplication, routing, out-of-domain behavior, one-retry enforcement
  and dev calibration.

## Gate result

**PASS for implementation correctness.** Retrieval quality remains unproven until
the benchmark is human-reviewed and the real dense index is complete. Thresholds
must not be selected on the test split.
