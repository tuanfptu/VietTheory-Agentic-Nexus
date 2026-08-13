# Advanced Research Program

## Scope

VietTheory-RAG is intended to grow from a five-subject citation-grounded RAG system into an AI
tutoring and research system. Graph retrieval, tools, memory, and multi-agent coordination are
research candidates—not automatically enabled product features.

Every candidate follows the same lifecycle:

`motivation → hypothesis → isolated implementation → benchmark → baseline/ablation → failure
analysis → keep, revise, or discard`

No feature enters the default answer path merely because it has been implemented.

## Target control flow

```text
Question
  → intent and subject routing
  → conversational context resolution
  → bounded controller
      ├─ structured hybrid retrieval
      ├─ graph retrieval
      ├─ source inspection tools
      └─ cross-subject retrieval
  → evidence aggregation
  → evidence sufficiency decision
      ├─ sufficient: answer
      ├─ partial/missing: targeted recovery
      └─ contradicted: conflict handling or abstention
  → citation verification
  → grounded answer
```

The structured Hybrid + Reranker pipeline remains B0. Advanced paths must demonstrate an
improvement over B0 on their intended query slice and report their additional cost.

## RQ6 — GraphRAG

### Hypothesis

A graph path improves evidence coverage for relationship, multi-hop, cross-chapter, and
cross-subject questions that are poorly represented by lexical or embedding similarity alone.

### Graph contract

Initial node types are `Concept`, `Person`, `Work`, `Event`, `Organization`, `HistoricalPeriod`,
and `Theory`. Initial edge types are `defines`, `influences`, `contradicts`, `develops_from`,
`causes`, `is_condition_of`, `applied_in`, and `associated_with`.

Every node and edge must retain provenance to one or more source spans. Uncited graph facts cannot
be used as answer evidence.

### Evaluation

Compare B0, Graph-only, B0 + Graph, and an adaptive Hybrid/Graph router. Report Recall@k, Full
Evidence@k, evidence-group recall, answer groundedness, latency, and graph traversal cost on
relationship and multi-hop slices. Graph construction quality is evaluated separately from graph
retrieval quality.

### Gate

Do not build the graph until the five-subject runtime and reviewed multi-hop/cross-subject gold
questions exist. Keep GraphRAG only if it adds evidence coverage without unacceptable provenance
or latency regressions.

## RQ7/RQ8 — Controller and tool use

The bounded controller may select from a small typed tool set:

- `retrieve_subject`
- `retrieve_cross_subject`
- `graph_search`
- `lookup_source_page`
- `compare_concepts`
- `generate_quiz`
- `grade_answer`
- `inspect_citation`

Tools must have validated input/output schemas, deterministic error contracts, trace IDs, and
explicit subject/provenance metadata. Retrieval tools return evidence, not final prose answers.

Tool selection is evaluated with exact/acceptable-tool accuracy, unnecessary-call rate, recovery
success, latency, LLM calls, tokens, and end-to-end groundedness. A rule-based or single-call
controller is the baseline. Tool count stays small until a benchmark proves that a new tool has a
distinct function.

## RQ9 — Memory

Memory is divided into three stores:

1. **Conversation state:** recent turns and resolved references needed for the current dialogue.
2. **Learning state:** reviewed mastery estimates, misconceptions, completed topics, and quiz
   outcomes.
3. **System trace:** retrieval/tool decisions used for debugging and evaluation; never presented as
   user memory.

User memories are account-isolated, inspectable, deletable, and excluded from retrieval evidence.
The system must not turn an old assistant claim into textbook evidence.

Memory evaluation measures reference resolution, conversational consistency, false-memory
injection, cross-account isolation, mastery calibration, and usefulness for quiz adaptation. The
baseline is a bounded recent-turn window without persistent learning state.

## RQ10 — Multi-agent coordination

Multi-agent coordination is the final optional phase. Candidate roles are Controller, Retrieval,
Graph, Evidence Aggregator, Critic/Judge, and Citation Verifier. Roles must have non-overlapping
contracts; free-form agent discussion is not an architecture requirement.

Compare a single bounded controller with the multi-agent system on answer quality, Full
Evidence@k, citation validity, failure rate, latency, LLM calls, and token/cost overhead. If
coordination increases cost without a material quality or reliability gain, the single-controller
design remains the production choice and the negative result is reported.

## Evaluation portfolio

| Capability | Primary benchmark | Required baseline |
|---|---|---|
| Five-subject retrieval | Natural QA v2 | B0 by subject |
| Evidence decision | Controlled sufficiency | L0 lexical, L1 TF-IDF |
| Recovery | Missing/partial cases | B0 without retry |
| Graph retrieval | Multi-hop/relationship subset | B0 and Graph-only |
| Routing/tool use | Tool-selection set | Rule-based/single-call controller |
| Memory | Conversational and tutoring set | Recent-turn window |
| Multi-agent | End-to-end hard subset | Single bounded controller |

All reports include per-query deltas and cost/latency alongside aggregate quality. Hidden sets are
used only after the relevant prompts, thresholds, models, indexes, and policies are frozen.

## Delivery order

1. Five-subject runtime, VNR202 audit, and smoke checks.
2. Natural QA benchmarks and Evidence Sufficiency pilot.
3. Evidence Judge, targeted recovery, and bounded adaptive RAG.
4. Provenance-backed GraphRAG and graph-specific benchmark.
5. Typed tool controller and tool-selection evaluation.
6. Conversation and learning memory with privacy tests.
7. Multi-agent ablation only if a measured single-controller bottleneck remains.
