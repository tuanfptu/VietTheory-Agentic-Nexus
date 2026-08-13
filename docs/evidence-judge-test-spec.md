# Evidence Judge Test Specification v1

## Boundary

This document specifies tests; it does not implement an Evidence Judge or agent. The judge receives
a question and expanded parent evidence. It never receives benchmark gold groups, gold answers, or
expected recovery queries.

## Core contract

The future judge must:

1. infer independently supportable required aspects from the question;
2. assign each aspect one frozen coverage label;
3. cite evidence IDs for every `covered` or `contradicted` label;
4. return `missing_evidence` only after evaluating expanded parent context;
5. emit missing-aspect text suitable for a targeted recovery query;
6. produce deterministic schema-valid output at the application boundary.

## Positive recovery case

### EJ-001 — genuine definition gap

- Input question: `mln111_0002`.
- Input context: B0 top-five expanded parents.
- Expected required aspects: definition as reflection by the human brain; highest form of material
  reflection.
- Expected decision: `missing_evidence`.
- Expected missing aspects: both explicit definitional propositions, or one combined definition
  aspect containing both.
- Forbidden behavior: marking the context sufficient merely because it discusses the brain as the
  organ of consciousness.

## Mandatory non-activation cases

### EJ-002 — natural and social origin

- Input question: `mln111_0021`.
- Expected decision after parent expansion: `sufficient`.
- Regression guarded: strict child g1 is rank 7, but rank-1 parent contains both gold groups.

### EJ-003 — role of material production

- Input question: `mln111_0057`.
- Expected decision: `sufficient`.
- Regression guarded: missing sibling child must not cause a retry when its parent is present.

### EJ-004 — socio-economic formation theory

- Input question: `mln111_0082`.
- Expected decision: `sufficient`.
- Regression guarded: adjacent theory-component child is restored by parent expansion.

### EJ-005 — idealism, Engels, and Lenin on motion

- Input question: `mln111_0095`.
- Expected decision: `sufficient`.
- Regression guarded: the rank-2 parent contains both required author-specific aspects.

## Dataset-wide tests

- Schema validity: 68/68 retrieval-answerable development questions produce valid output.
- Evidence-link validity: every supporting evidence ID exists in the supplied parent context.
- Gold isolation: no test helper passes gold groups or gold answers into judge inputs.
- Determinism: repeated evaluation with frozen prompt/model/config returns the same decision.
- False activation: the four parent-resolved cases above must remain inactive.
- False sufficient: `EJ-001` must not be accepted without recovery.

## Metrics for B1

- missing-evidence precision, recall, and F1;
- false-sufficient rate;
- false-activation rate;
- aspect-level coverage accuracy;
- schema failure rate;
- judge latency and LLM calls per query.

No B1 implementation may proceed until these cases and metric definitions are reviewed and frozen.
