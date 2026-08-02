# MLN111 Fixed-Size Chunking Baseline

## Configuration

- Chunker version: `fixed_lines_v1`
- Target size: **400 deterministic Unicode word/punctuation tokens**
- Overlap target: **50 tokens**
- Split boundary: complete extracted text lines
- Excluded roles: page number, repeated header, repeated footer
- Provenance granularity: line-level page ID, PDF page, printed page, bbox, and text

## Results

- Chunks: **527**
- Minimum tokens: **215**
- Median tokens: **391**
- Mean tokens: **390.1**
- Maximum tokens: **400**
- Unique stable chunk IDs: **527/527**
- Source body lines: **9,090**
- Source body lines covered by chunks: **9,090**
- Last source PDF page represented: **284**
- Empty chunks: **0**
- Output: `data/processed/MLN111/chunks.jsonl` (generated, Git-ignored)

## Gate result

Every body line is represented in at least one chunk, every source span resolves to an
existing page and valid bbox, page-number metadata is excluded from chunk text, and the
final source page is retained.

**Day 06 chunking gate: PASS.**
