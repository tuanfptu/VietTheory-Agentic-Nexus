# VietTheory-RAG Five-subject Protocol

## Objective

VietTheory-RAG is one shared framework for MLN111, MLN122, MLN131, HCM202, and VNR202. Subject
corpora, indexes, metadata, and benchmarks vary; retrieval, generation, agentic policy, and
evaluation contracts remain shared unless an experiment explicitly tests a subject-specific
choice.

## Research boundary

- MLN111 remains the development and deep-ablation subject.
- B0 remains frozen: structured children + BM25 + Qwen3 dense + RRF + Qwen3 reranker.
- Method choices are developed on MLN111, then transferred unchanged to the other four subjects.
- Subject-specific tuning is prohibited in transfer claims unless reported as a separate ablation.
- Hidden tests remain isolated by subject and are run only after code, prompts, thresholds, corpus,
  indexes, and manifests are frozen.

## Shared layers

### Data

Every subject must provide versioned extraction, page geometry, chapter/section structure,
parent/child chunks, stable IDs, source checksums, and index manifests. Native PDF extraction is
used for MLN111 and MLN122. Tesseract Vietnamese OCR is used for HCM202, MLN131, and VNR202.

### Retrieval

The shared path is subject routing → BM25 and dense retrieval → RRF → cross-encoder reranking →
parent expansion. Parent-expanded context is the production evidence boundary; child IDs are the
retrieval evaluation boundary.

### Generation

Retrieval success, answer correctness, groundedness, citation correctness, citation completeness,
and abstention are reported separately. No combined score may obscure which layer failed.

### Agentic policy

Agentic behavior is not implemented until the Evidence Sufficiency pilot is audited. The eventual
bounded policy operates on required/covered/missing aspects after parent expansion, with at most
two rounds and two targeted recovery queries.

### Evaluation

- MLN111 v1 remains frozen and reproducible.
- Natural QA v2 is a new versioned line; it does not mutate v1.
- Deep component ablations run on MLN111.
- Transfer evaluation on the other subjects compares only research-relevant configurations.
- Controlled Judge and Recovery benchmarks remain distinct from natural QA evaluation.
- Cross-subject evaluation distinguishes unrestricted retrieval, wrong-subject rejection, and
  genuine multi-subject synthesis; it is not merely five isolated chatbot evaluations.
- Per-query win/loss/mixed/tie analysis is sliced by subject, chapter, difficulty, question type,
  and reasoning scope when sample sizes permit.

## Target benchmark portfolio

| Benchmark | Target | Role |
|---|---:|---|
| MLN111 Retrieval v1 | 68 answerable dev + 30 hidden records | Frozen B0 foundation |
| Natural QA v2 | 1,100–1,400 across five subjects | Cross-subject realism |
| Evidence Sufficiency | 40–60 cases per subject | Judge diagnostics |
| Recovery | Missing/partial controlled cases | Corrective action |
| Cross-subject hidden | Separate versioned set | Routing and confusion |

Natural QA v2 targets 250 questions for each of the five subjects. Candidate generation and review
proceed in 50-question batches; quality and human verification take priority over reaching a
headline count. The frozen MLN111 v1 records remain historical evaluation artifacts and are not
silently counted as reviewed Natural QA v2 cases.

Natural QA v2 now uses a concrete final target of 250 questions per subject. The pilot quota,
four-gate review contract, deterministic portfolio plan, and validation commands are documented in
[`benchmark/v2/README.md`](../benchmark/v2/README.md).

## Freeze and leakage rules

- Split natural questions by evidence parent, semantic family, and source section—not randomly by
  question ID.
- Split controlled cases by `source_question_id`.
- Do not tune prompts or thresholds on held-out or hidden cases.
- Every release pins canonical JSON serialization, stable IDs, model revisions, artifact hashes,
  benchmark hashes, prompts, thresholds, and code commit.
- Natural and controlled results are reported separately.

## Current status

All five subjects have extraction artifacts, structured parent/child chunks, and Qwen3 FAISS
indexes. Quantitative human-reviewed retrieval benchmarks exist only for MLN111. Therefore the
current bottleneck is benchmark construction and review, not OCR, indexing, or model availability.
