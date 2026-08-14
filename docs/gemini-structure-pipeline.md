# Gemini-assisted document structure

## Status

`structured_v1` remains the frozen rule-based corpus. The new
`structured_v2_gemini` namespace is experimental and does not feed retrieval,
benchmarks, or reported B0 metrics yet.

## Design

```text
PDF page image + existing OCR blocks
                 |
                 v
Gemini semantic structure extraction
  - schema-constrained JSON
  - page role
  - chapter/division/section/subsection
  - footnote/page number/review-region labels
                 |
                 v
Deterministic validation
  - exact pdf_page match
  - existing block_id anchors only
  - source blocks in reading order
  - no reused heading anchors
  - textbook numbering invariants
                 |
                 v
Human review -> cross-page resolver -> v2 chunks/index
```

Gemini is not allowed to invent source IDs. It may correct obvious OCR errors in
heading text, but every proposed element must cite one or more OCR block IDs that
already exist on the page. The API credential is loaded from `GEMINI_API_KEY` in
the process environment or `.env`; it is never logged or persisted.

## Five-page VNR202 pilot

Run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_gemini_structure_pilot.py `
  --pdf "Tài liệu\Giáo trình VNR202.pdf" `
  --pages-jsonl data\processed\VNR202\pages.jsonl `
  --output-dir data\processed\VNR202\structured_v2_gemini\pilot_5_pages `
  --pages 21 68 124 180 225 `
  --model gemini-3.5-flash-lite `
  --requests-per-minute 4 `
  --max-requests 5
```

The command checkpoints one canonical JSON record per page. Re-running without
`--refresh` validates the cached records and performs no API calls. The request
budget and rate limit are explicit CLI arguments.

## Promotion gate

Do not run the complete 230-page conversion or rebuild retrieval artifacts until:

1. A human reviews all five pilot pages against the source images.
2. Page-role and hierarchy disagreements are adjudicated.
3. The cross-page hierarchy resolver is implemented and tested.
4. A 20-page stratified pilot passes, including review/back-matter transitions.
5. The full output receives a structural audit before becoming `structured_v2`.

