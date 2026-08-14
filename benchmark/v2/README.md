# Natural QA v2

Natural QA v2 is the new five-subject benchmark line. It does not mutate or replace the frozen
MLN111 v1 benchmark.

## Portfolio

- Pilot: 50 reviewed questions per subject.
- V1: 150 reviewed questions per subject.
- Final target: 250 reviewed questions per subject (1,250 total).
- Cross-subject challenge: 100–150 separate cases.
- Evidence Sufficiency: 150–250 controlled cases in a separate benchmark.

The five physical textbooks remain separate. Their chunks are exposed through a unified logical
catalog with subject, document, chapter, section, page, parent, and child provenance.

## Pilot quota per subject

| Primary category | Cases |
|---|---:|
| Direct / single evidence | 13 |
| Explanation / cause-effect | 10 |
| Comparison / relationship | 8 |
| Multi-chunk | 7 |
| Synthesis | 5 |
| Multi-hop / cross-chapter | 3 |
| Negative | 4 |
| **Total** | **50** |

Categories are primary labels for quota accounting. Additional reasoning metadata may overlap.

## Human review gates

Every question must pass:

1. **Question validity:** natural, intentionally scoped, unambiguous, and without answer leakage.
2. **Gold-answer validity:** supported by the textbook without unsupported external claims.
3. **Evidence validity:** child evidence and parent-expanded context genuinely support the answer.
4. **Difficulty validity:** the assigned difficulty reflects the evidence/reasoning burden.

The record may become `verified` only when all four booleans in `review_gates` are true. LLM output
always begins as `draft`; generation never grants review status.

## Workflow

```text
corpus → candidates → schema validation → human review 1 → correction
       → second review / 10–20% spot audit → version freeze
```

Generate the deterministic plan:

```powershell
python scripts/create_benchmark_v2_plan.py
```

Validate a candidate JSONL file:

```powershell
python scripts/validate_benchmark_v2.py benchmark/v2/candidates/MLN111_batch_01.jsonl
```

The report includes quota progress by subject, difficulty and chapter coverage, plus diversity
warnings for near-duplicate questions, overused gold parents, overrepresented sections, anomalous
answer lengths, and unusually high question/answer lexical overlap. Warnings require review but do
not automatically invalidate otherwise well-formed records.

Split policy must group semantic families and shared evidence parents. Random splitting by question
ID is prohibited.

## Human review import (2026-08-14)

Import the 500-row human review without promoting revised or rejected records:

```powershell
python scripts/import_natural_qa_v2_review.py `
  benchmark/v2/review/natural_qa_v2_500_draft.jsonl `
  benchmark/v2/reviews/human_2026-08-14/natural_qa_v2_500_verified.csv `
  benchmark/v2/review/natural_qa_v2_500_reviewed.jsonl `
  --manifest benchmark/v2/reviews/human_2026-08-14/review_manifest.json `
  --verified-output benchmark/v2/review/natural_qa_v2_452_verified.jsonl `
  --needs-action-output benchmark/v2/review/natural_qa_v2_48_needs_action.jsonl
```

This review contains 452 approved, 26 revise, and 22 rejected records. Only approved records are
promoted to `verified`. Revised records remain `draft`; rejected records remain in the audit trail
but are excluded from a Gold release. Corrected and replacement records require a new human check
because their content hashes differ from the reviewed drafts.
