# Five-subject Selective Gemini Structure Audit

## Scope

- 100 pages total: 20 pages per subject.
- 20 contiguous five-page windows selected by the local anomaly detector.
- Model: `gemini-3.5-flash-lite`.
- Execution: structure-only, image plus existing extracted blocks.
- Requests: 22, including one rejected MLN131 batch that was split into 3+2 pages.
- Result: 100/100 pages validated; zero failed singleton pages.
- Corpus/index v1, benchmark artifacts, hidden tests, and B0 metrics were not modified.

## Validated output summary

| Subject | Pages | Elements | Validator-warning pages | Main page roles |
|---|---:|---:|---:|---|
| MLN111 | 20 | 27 | 1 | front matter, body, review, references, contents |
| MLN122 | 20 | 42 | 5 | front matter, body, review, chapter opening, references |
| MLN131 | 20 | 49 | 13 | body, review, chapter opening, references, contents |
| HCM202 | 20 | 57 | 1 | body, review, chapter opening, references, contents |
| VNR202 | 20 | 39 | 2 | front matter, contents, chapter opening, body |

Validator warnings mean one or more Gemini elements cited a cross-page or unknown
anchor. Those elements were dropped individually; valid elements on the page were
retained. OCR corrections were disabled for this pass, so the audit did not rewrite
source text.

## Important VNR202 result

Zero-based PDF page 225 (human PDF page 226) was classified as `body`. Gemini
identified the real level-3 section:

> 4. Kết hợp sức mạnh dân tộc với sức mạnh thời đại, sức mạnh trong nước với sức
> mạnh quốc tế

The section is anchored to two existing OCR blocks with confidence 1.0. This
confirms that v1 parent exclusion around the late-book review boundary is too
broad for this page.

## Interpretation and next gate

This audit identifies candidate repairs; it does not automatically promote them
to corpus truth. Before building `structured_v2`, the accepted Gemini elements
must be merged with deterministic v1 structure through a cross-page resolver.
Pages with validator warnings and all parent-coverage changes require human
spot-checking. OCR correction should be a separate sparse pass over confirmed
problem blocks rather than a whole-page rewrite.

