# MLN111 Extraction Validation

## Automated validation

- Source pages: **285**
- Extracted records: **285**
- Pages with more than 100 characters: **284**
- Pages flagged for OCR review: **1**
- Total extracted characters: **701,796**
- Text blocks: **9,167**
- Text lines: **9,374**
- Page-number blocks: **284**
- Repeated header/footer blocks at the 80% threshold: **0**
- Mean native-text quality score: **0.9929**
- Minimum native-text quality score: **0.1820**
- Schema: **1.0**
- Stable IDs: verified across repeated extraction
- Bounding boxes within page bounds: verified for all blocks
- Role tagging is non-destructive: text and source geometry are retained
- Source and artifact SHA-256 checks: verified against companion manifest
- Extractor version: **PyMuPDF 1.28.0**
- Postprocessor chain: **marginal_roles_v1**

Full output is generated locally at `data/processed/MLN111/pages.jsonl`. It is excluded
from Git because processed corpora are reproducible artifacts.

## Visual sample

The deterministic 20-page sample uses zero-based PDF indices:

`0, 1, 7, 14, 30, 42, 58, 73, 88, 104, 120, 135, 150, 166, 181, 198, 215, 233, 250, 284`

The overlay artifact is generated locally at
`output/pdf/mln111_bbox_validation.pdf`. Red rectangles represent text blocks; blue
rectangles represent text lines.

## Manual review checklist

For each sampled page, record `pass`, `warning`, or `fail` for:

- text agrees with the visible PDF;
- block rectangles enclose the intended text;
- line rectangles align with individual lines;
- reading order follows the page;
- printed-page mapping is correct;
- repeated headers and footers are identified appropriately.

All 20 sampled pages were rendered and visually inspected. Each page passed text/bbox
alignment, basic one-column reading order, printed-page localization, and footnote
localization checks. No sampled page showed clipping or inverted reading order.

## Preliminary rendered inspection

Overlay pages 1, 10, and 20 of the validation artifact (source PDF indices 0, 104,
and 284) were rasterized at 120 DPI and inspected:

- cover title lines align with their visible text;
- dense body text retains top-to-bottom reading order and line-level boxes;
- footnotes and the printed page number are localized separately;
- the final publication-information page retains its multi-block layout;
- no clipped or out-of-page rectangles were observed.

Poppler emitted missing-display-font warnings for `Symbol` and `ArialUnicode` during
rasterization. The inspected pages remained legible; this is recorded as a rendering
environment warning rather than an extraction failure.

**Day 05 visual gate: PASS.**
