# MLN122 Native Pipeline Gate

## Native extraction

- Source: `Tài liệu/Giáo trình MLN122.pdf`.
- Pages extracted: 262 / 262.
- Extractor: PyMuPDF 1.28.0.
- Text characters: 391,078.
- Empty pages: 0.
- Pages requiring OCR: 0.
- Text blocks: 5,698.
- Text lines: 7,222.
- Page artifact SHA-256:
  `7eb6a9a028a0f127b7bce14ecbf589f998e4720ba6ae07782e45fafb8223b58f`.

The marginal-role postprocessor identified 262 page-number blocks and retained
5,436 body blocks. The source document checksum and extraction range are recorded in
`data/processed/MLN122/pages.manifest.json`.

## Structure and parent-child chunks

- Headings: 183.
- Heading levels 1–5: 6 / 17 / 46 / 71 / 43.
- Detected chapter starts: PDF pages 7, 28, 74, 111, 151 and 200.
- Parents: 148.
- Children: 348.
- Parent target: at most 1,500 tokens.
- Child target: at most 400 tokens with 50-token overlap.
- Child artifact SHA-256:
  `cae0a7794de9750775fc3aaccfc9647075d51aafb6df661572876668cc143cd3`.

The parser rejects prose that merely starts with “Chương N”, preserves the section
path on every chunk and never creates a child that crosses its parent boundary.

## Structured dense index

- Model: Qwen/Qwen3-Embedding-0.6B.
- Revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Index: normalized FAISS `IndexFlatIP`.
- Vectors: 348 x 1024.
- GPU batch size: 8.
- Index SHA-256:
  `a658b63c071798ad09583ebd94fea9419b3f60a8aeb70aeb0772b3db684e329a`.
- Mapping SHA-256:
  `786f4d6db214110ac7c2c7ef2f517b95c18840a35e924e82341a0d71e8e7b43a`.

For “Giá trị thặng dư là gì?”, rank 1 expands to the parent under Chapter 3,
section `2. Bản chất của giá trị thặng dư`, on PDF pages 87–89.

## Visual validation

The bbox overlay artifact is `output/pdf/mln122_bbox_validation.pdf`. All 12 sampled
source pages were rendered and inspected: 7, 28, 74, 87, 111, 151, 200, 204, 230,
250, 260 and 261. The sample includes all chapter transitions, regular prose, a
table, the table of contents and publication metadata.

- Bounding boxes align with visible text: PASS.
- Reading order is coherent: PASS.
- Page numbers are separated from body content: PASS.
- Table and contents layouts remain traceable to their source regions: PASS.

## Gate result

**PASS for native extraction completeness, provenance, structural chunks, persisted
dense-index integrity, qualitative retrieval and visual bbox validation.**
Quantitative retrieval evaluation remains dependent on human-reviewed relevance
labels.
