# Research Candidate v0.1

This release is an **isolated research candidate**, not the production answer path. Structured
Hybrid + Reranker remains the frozen B0 baseline.

## Frozen contracts

- J1 produces `SUFFICIENT`, `PARTIAL`, `MISSING`, `WRONG_ASPECT`, or `CONTRADICTED`, plus required,
  covered, and missing aspects.
- Recovery activates only for `PARTIAL` and `MISSING`, permits at most two rounds and two queries
  per round, and stops on sufficiency or contradiction.
- Graph paths are selective, bounded to at most three hops by contract, and every node, edge, and
  returned path must retain source spans.
- Memory is account-scoped, inspectable, deletable, and structurally forbidden from becoming
  textbook evidence.
- Tool calls have typed inputs/results, trace IDs, bounded budgets, deterministic error classes,
  and explicit latency.
- Multi-agent coordination is role-separated rather than free-form. It is compared per query with
  a single bounded controller using evidence hits, latency, and LLM-call counts.

## Current evaluation state

| Capability | State | Evidence |
|---|---|---|
| B0 five-subject retrieval | Development evaluated | `benchmark/v2/reports/` |
| Evidence pilot | Frozen | 32 public development + 16 private held-out cases |
| L0/L1 shortcuts | Development evaluated | `shortcut_baselines.json` |
| J1 | Implementation validated; external development run pending explicit data-export approval | `run_evidence_judge.py` |
| Recovery | Contract and deterministic tests complete; empirical evaluation pending J1 freeze | `recovery.py` |
| GraphRAG | Provenance-preserving search primitive complete; corpus graph and benchmark evaluation pending | `graph.py` |
| Memory | Isolation/evidence-safety primitive complete; conversational benchmark pending | `memory.py` |
| Tools | Typed bounded controller complete; tool-selection benchmark pending | `tools.py` |
| Multi-agent | Fair per-query ablation harness complete; end-to-end run pending candidate inputs | `coordination.py`, `ablation.py` |

No Natural QA hidden set or Evidence Sufficiency held-out set has been evaluated. They remain
locked until the respective model, prompt, threshold, graph, and controller candidates are frozen.

## Reproduction

```powershell
python scripts/build_research_release_manifest.py outputs/releases/research_candidate_v0.1.json
ruff check .
ruff format --check .
mypy --strict src
pytest -q
git diff --check
```

The manifest contains canonical sorted paths and SHA-256 digests and intentionally excludes all
hidden and held-out artifacts.
