# Five-subject Roadmap

## Gate 0 — preserved foundation

- MLN111 B0 and benchmark v1 remain frozen.
- Existing PDFs, audit data, reports, and private hidden material remain untouched.
- Shared changes must reproduce MLN111 B0 before transfer evaluation.

## Gate 1 — corpus readiness

- Validate manifests, checksums, parent links, subject purity, vector mappings, page counts, and
  chapter counts for all subjects.
- Audit 20–30 representative VNR202 pages, including chapter transitions and heading-heavy pages,
  because only seven headings were detected across 230 pages.
- Determine whether the sparse structure reflects the textbook or heading-detector under-segmentation
  before freezing VNR202 parent boundaries.
- Record any OCR correction as a new corpus artifact version.

Status: completed. The 25-page audit confirmed OCR under-segmentation; the general detector was
corrected and VNR202 was rebuilt with 28 headings. See
[`reports/vnr202_structure_audit.md`](../reports/vnr202_structure_audit.md).

## Gate 2 — shared retrieval runtime

- Replace MLN111 constants with the frozen subject registry.
- Load one or multiple subject indexes through one assembly path.
- Add subject routing, wrong-subject handling, and cross-subject confusion tests.
- Reproduce MLN111 B0 unchanged.

## Gate 3 — five-subject smoke evaluation

- Run B0 through the same code path on all five subjects before constructing large benchmarks.
- Verify within-subject retrieval, unrestricted cross-subject retrieval, subject filtering, parent
  expansion, and citation provenance.
- Treat these as smoke checks, not benchmark results, until human-reviewed gold exists.

## Gate 4 — natural benchmark foundations

- Define Natural QA v2 lineage and grouped split policy.
- Generate candidates in batches of 50 per subject, validate automatically, and human-review before
  scaling to 100 and beyond.
- Evaluate four distinct categories: within-subject retrieval, cross-subject retrieval,
  wrong-subject rejection, and cross-subject synthesis.

## Gate 5 — Evidence Sufficiency foundation

- Complete the MLN111 pilot before implementing J1 or an agent: 12 source questions and about 40
  controlled cases.
- Audit label clarity, semantic validity after parent expansion, and lexical shortcuts.
- Split controlled development and held-out cases by `source_question_id`.

## Gate 6 — transfer retrieval evaluation

- Apply the frozen B0 method unchanged to MLN122, MLN131, HCM202, and VNR202.
- Report metrics and deltas by subject and evidence scope.
- Do not run the complete MLN111 ablation grid on every subject without a research question.

## Gate 7 — generation evaluation

- Freeze grounded generation and citation contracts.
- Evaluate answer correctness, groundedness, citation correctness/completeness, and abstention
  independently.

## Gate 8 — bounded Agentic RAG

- Implement J1 only after the Judge pilot audit.
- Freeze J1 before recovery experiments.
- Evaluate targeted missing-aspect recovery before combining it with adaptive activation.
- Limit recovery to two rounds and two queries.

## Gate 9 — scale and final freeze

- Scale natural QA only after each 50-question batch passes review and distribution checks. The
  final target is approximately 1,100–1,400 reviewed questions across five subjects.
- Freeze code, prompts, models, thresholds, indexes, benchmark versions, manifests, and checksums.
- Run subject-specific hidden tests and the cross-subject hidden set once.
- Publish aggregate hidden results without leaking question-level gold.

## Post-v1 research phases

After the five-subject core and bounded evidence-recovery path are proven, advanced capabilities
follow the gated [Advanced Research Program](advanced-research-program.md):

1. provenance-backed GraphRAG and graph benchmark;
2. typed tool use and tool-selection evaluation;
3. conversation/learning memory and privacy evaluation;
4. multi-agent coordination only if a single-controller bottleneck is measured.

Each capability requires an isolated baseline, ablation, cost analysis, and failure analysis. A
negative result may remove a feature from the production path without invalidating the research.

## Human bottlenecks

The remaining high-effort work is human verification of natural questions, controlled sufficiency
cases, contradiction cases, acceptable evidence, and hidden tests. Generated JSON volume is not a
substitute for reviewed gold quality.
