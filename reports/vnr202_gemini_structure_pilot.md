# VNR202 Gemini Structure Pilot

## Scope and safety

- Model: `gemini-3.5-flash-lite`
- Sample: five zero-based PDF pages: 21, 68, 124, 180, 225
- Inputs per request: rendered page image plus existing OCR blocks and block IDs
- Validated page records: five
- API calls during the pilot: six; one first response for page 180 was rejected
  because its elements were not in source reading order, then the deterministic
  canonical-order guard was added before retrying
- Output: schema-constrained JSON validated against source page/block IDs
- API key: loaded in memory only; not logged or persisted
- `structured_v1`, indexes, benchmark data, hidden tests, and B0 metrics: unchanged

This is a pilot artifact pending human review. It is not a full-corpus structure
release and is not used by production retrieval.

## Observed results

| PDF page (zero-based) | Role | Rule headings | Gemini heading-like elements | Observation |
|---:|---|---:|---:|---|
| 21 | chapter opening | 3 | 3 | Gemini recovered the complete Chapter 1 title and corrected `Đáng` to `Đảng`. |
| 68 | chapter opening | 3 | 3 | Gemini recovered the complete Chapter 2 title; numbering invariants normalized `I.`/`1.` to levels 2/3. |
| 124 | chapter opening | 3 | 3 | Gemini recovered the complete multi-line Chapter 3 title. |
| 180 | body | 0 | 1 | Gemini detected an unnumbered italic subheading missed by the rule parser and separated the footnote/page number. |
| 225 | body | 1 | 1 | Gemini confirmed a real numbered section and separated the footnote/page number. |

Across the sample, the rule parser emitted 10 headings; the validated Gemini
result emitted 11 heading-like elements, plus four page numbers and two footnotes.

## Important finding

The v1 parent artifact has no parent touching page 225 even though both the source
image and Gemini output show substantive body content and a valid numbered
section. This suggests that the v1 review-section state may exclude too much late
book content. The finding must be audited around the preceding transition pages
before scaling or replacing v1.

## Decision

The pilot supports continuing the Gemini-assisted approach, but it does not yet
justify processing all 230 pages. Human adjudication and a cross-page resolver are
still required. The five page records remain `pending_human_review`.
