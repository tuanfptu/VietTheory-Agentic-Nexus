# Evidence Coverage Taxonomy v1

## Purpose

This taxonomy defines the state consumed by a future Evidence Judge. It is frozen before agent
implementation so that implementation cannot redefine success after seeing results.

## Unit of judgment

The judge evaluates **expanded production context**, not raw child retrieval IDs. Gold benchmark
groups are evaluation labels only and must never be passed to the judge at inference time.

## Required-aspect state

```json
{
  "required_aspects": [
    {
      "aspect_id": "a1",
      "description": "one independently supportable requirement inferred from the question",
      "importance": "required"
    }
  ],
  "coverage": [
    {
      "aspect_id": "a1",
      "status": "covered",
      "supporting_evidence_ids": ["parent_id"]
    }
  ],
  "missing_aspects": [],
  "decision": "sufficient"
}
```

## Coverage labels

- `covered`: at least one expanded passage directly supports the aspect;
- `partial`: evidence is related but omits a necessary proposition;
- `missing`: no retrieved passage supports the aspect;
- `contradicted`: retrieved evidence directly conflicts with the proposed aspect;
- `not_applicable`: the aspect was incorrectly inferred and should not be required.

## Decisions

- `sufficient`: every required aspect is `covered`;
- `missing_evidence`: at least one required aspect is `partial` or `missing`;
- `conflicting_evidence`: at least one required aspect is `contradicted`;
- `abstain`: requirements cannot be reliably inferred or recovery budget is exhausted.

## Failure taxonomy

- `true_context_gap`: required support is absent after parent expansion;
- `semantic_neighbour`: retrieved text is topically close but does not state the required claim;
- `lexical_mismatch`: terminology mismatch prevents retrieval;
- `multi_aspect_gap`: only a subset of required aspects is covered;
- `sibling_child_gap`: gold child is absent but its shared parent is retrieved;
- `child_id_gap`: strict child metric misses semantically sufficient production context;
- `fusion_regression`: fusion ranks a gold candidate below a component retriever;
- `reranker_regression`: reranking degrades a previously successful result;
- `annotation_gap`: retrieved evidence appears acceptable but is absent from benchmark gold.

Only `true_context_gap`, `semantic_neighbour`, `lexical_mismatch`, and `multi_aspect_gap` may trigger
targeted recovery. Parent-resolved and annotation gaps must not activate the agent.
