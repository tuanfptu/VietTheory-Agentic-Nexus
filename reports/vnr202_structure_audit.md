# VNR202 Structure Audit

## Finding

The previous seven-heading result was caused by genuine structure under-segmentation, not by an
unusually flat textbook. A stratified visual review of 25 PDF pages found visible Roman-numbered
divisions and numbered sections that existed in the PDF and OCR text but were rejected by the old
detector.

The sample covered front matter, all three chapter transitions, heading-heavy pages, end-of-chapter
review pages, and normal prose controls. The exact zero-based PDF page sample is recorded in the
machine-readable [audit report](vnr202_structure_audit.json).

## Root cause

- Tesseract line blocks do not provide reliable bold font flags.
- Roman-dot headings were accepted only when the complete text was uppercase.
- Numbered headings required bold typography.
- Asterisk-prefixed OCR footnotes could be mistaken for level-five headings.
- A small number of heading prefixes contained scoped OCR errors (`I ` and `1J.`).

The fix is general to OCR textbooks. No VNR202 heading allowlist or page-specific heading table was
added. Conservative OCR cues reject terminal prose/list punctuation, inline enumerations, review
questions, footnotes, large numeric years, and implausibly long candidates. Wrapped OCR heading
lines are joined only when geometry and continuation cues agree.

## Artifact changes

| Artifact | Before | After |
|---|---:|---:|
| Headings | 7 | 28 |
| Chapter headings | 3 | 3 |
| Division headings | 0 | 6 |
| Section headings | 0 | 19 |
| Parent chunks | 100 | 106 |
| Child chunks / vectors | 490 | 459 |

The structured artifacts and Qwen3 FAISS index were rebuilt. Current checksums are pinned in the
JSON audit report and validated by the five-subject readiness audit. Parent-child links, subject
purity, source provenance, vector mapping coverage, and all manifest checks pass.

## Interpretation

The lower child count is expected: correct section boundaries change parent grouping and overlap,
while review-question blocks are no longer indexed as textbook answer evidence. This correction
must be frozen before VNR202 Natural QA v2 gold evidence is created.
