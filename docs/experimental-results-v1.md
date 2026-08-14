# Experimental Results v1

All selections below were made on public development data. Natural QA hidden and Evidence
Sufficiency held-out remained untouched until after `configs/final_candidate_v1.json` was written.

## Retrieval B0

| Mode | Recall@5 | Full Evidence@5 |
|---|---:|---:|
| Within-subject parent-aware B0 | 98.49% | 93.96% |
| Global parent-aware B0 | 97.89% | 92.15% |

B0 is Structured Hybrid (BM25 + Qwen dense + RRF), Qwen cross-encoder reranking, and parent
expansion. The frozen Natural QA development evaluation contains 331 answerable questions.

### One-shot held-out result

After all candidate decisions were written to `configs/final_candidate_v1.json`, frozen B0 was
run once on the private 150-question held-out split (147 retrieval-answerable questions).

| Mode | Recall@1 | Recall@5 | MRR | nDCG@5 | Evidence Group Recall@5 | Full Evidence@5 |
|---|---:|---:|---:|---:|---:|---:|
| Within-subject | 87.76% | 98.64% | 92.60% | 92.44% | 95.38% | 94.56% |
| Global | 85.03% | 97.28% | 90.74% | 90.47% | 93.64% | 93.20% |

Question-level rankings remain private. The public aggregate includes hashes of both private input
and report. Results were not used for post-hoc model, prompt, index, or threshold changes.

## Evidence sufficiency

| Candidate | Accuracy | Macro-F1 | Decision |
|---|---:|---:|---|
| L0 lexical shortcut | 50.00% | 41.61% | sanity baseline |
| L1 TF-IDF shortcut | 50.00% | 41.61% | sanity baseline |
| J1 Gemini structured judge | 96.88% | 96.86% | freeze for held-out |

J1 made one disagreement in 32 cases. Manual evidence inspection found that the remaining parent
actually supported both answer requirements, so the frozen `PARTIAL` label is likely redundant.
Raw metrics are retained; no post-error prompt tuning occurred.

The frozen J1 was then evaluated exactly once on the private 16-case held-out split and achieved
**100% accuracy and 100% macro-F1**. The held-out set contains four source-question groups and four
active labels (`SUFFICIENT`, `PARTIAL`, `MISSING`, `WRONG_ASPECT`); contradiction remains explicitly
exploratory because no natural reviewed contradiction cases were available. Per-case data remains
private and only aggregate confusion plus private-artifact hashes are published.

## Targeted recovery

J1 activated on 15/32 controlled development cases. Up to two missing-aspect queries were sent
through frozen B0. Only 2/15 deficient activated cases recovered all annotated parent groups
(13.33%). The candidate is rejected from the default path. This result also exposes narrow
acceptable-parent annotations and redundant evidence groups that should be addressed in a future
Recovery benchmark rather than by tuning against this pilot.

## Graph and coordination ablation

| Slice | B0 Full Evidence@5 | B0 + graph | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|
| All 331 answerable dev | 93.96% | 89.73% | 2 | 16 | 313 |
| Relationship/multi-hop (101) | 82.18% | 71.29% | 2 | 13 | 86 |

The graph candidate uses only corpus parent adjacency, never benchmark gold, and returns source-page
provenance for every expansion. Despite only ~3.0 ms median local expansion overhead, its evidence
quality regresses. Graph-only Full Evidence@5 is effectively zero. The candidate and its
role-separated multi-agent form are rejected; the single bounded controller remains the default.

## Memory and tools

- Account isolation, foreign deletion denial, owner deletion, effective deletion, and rejection of
  memory-as-textbook-evidence: **5/5 safety checks passed**.
- Frozen rule router on the 12-case typed-tool sanity set: **12/12 acceptable-tool selections**.

These are engineering contract checks, not claims of general model reasoning quality. Memory and
tools remain isolated from the grounded answer path until larger reviewed benchmarks exist.

## Validation notes

The post-review Gold profile validates 500/500 records with no blocking issues. Five non-blocking
warnings remain public: two section-concentration warnings and three high question/answer lexical
overlap warnings. They are retained as benchmark limitations instead of rewriting verified items
after evaluation. All five corpus readiness audits pass, including checksums, parent links, subject
purity, vector mapping, and pinned model identity.
