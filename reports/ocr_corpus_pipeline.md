# Scanned Corpus OCR Pipeline Gate

## Engine selection

The representative benchmark compared PaddleOCR and Tesseract on Vietnamese scan
pages:

- PaddleOCR 3.7 / PP-OCRv6 incorrectly selected a model without Vietnamese support.
- PP-OCRv3 accepted `vi`, but retained few Vietnamese marks and took about
  12.5 seconds per page at 144 DPI.
- Tesseract 5.4 with `tessdata_best/vie` retained Vietnamese diacritics and averaged
  3.472 seconds per page over 15 distributed samples.

The production OCR configuration is therefore Tesseract, 144 DPI, PSM 6. PaddleOCR
remains isolated in `.venv-ocr` and is not a runtime dependency of the application.

## Full extraction

| Subject | Pages | Characters | Mean quality | Review pages | Page SHA-256 |
|---|---:|---:|---:|---|---|
| HCM202 | 271 | 356,046 | 0.9011 | 0, 1, 7 | `cece299a…f155` |
| MLN131 | 273 | 368,220 | 0.9488 | 0, 1, 46 | `01c067f4…51ff` |
| VNR202 | 230 | 615,663 | 0.9379 | 0, 1, 3, 7 | `7d9801f5…9a63` |

All 774 pages are present and all three artifact checksums match their manifests.
The ten review pages are covers, sparse front matter or blank scan pages; they remain
in the page artifacts and are not silently discarded.

Tesseract TSV is parsed with CSV quoting disabled. This prevents OCR-recognized
quotation marks from swallowing subsequent TSV rows. Thirteen pages affected during
the first pass were re-OCRed and their manifests refreshed. No TSV marker or
character-count outlier remains.

OCR lines are promoted to individual blocks before marginal-role tagging. This
preserves line IDs, text and PDF-coordinate bboxes while allowing page-number lines
to be separated from body text.

## Structure and dense indexes

| Subject | Chapter headings | All headings | Parents | Children/vectors | Index SHA-256 |
|---|---:|---:|---:|---:|---|
| HCM202 | 6 | 86 | 112 | 331 | `543d01e3…5e8c2` |
| MLN131 | 7 | 122 | 137 | 334 | `0c44a19d…fa7bb` |
| VNR202 | 3 | 7 | 100 | 490 | `b5ced14a…07ff1` |

Every parent is at most 1,500 tokens, every child at most 400 tokens and every child
resolves to an existing parent. Contents and clustered tail chapter lists are
retained in page artifacts but excluded from retrieval chunks.

All indexes use normalized FAISS `IndexFlatIP`, Qwen3-Embedding-0.6B revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, dimension 1024 and CUDA batch 8.
Manifest child, index and mapping checksums are current.

## Visual and retrieval validation

Twelve bbox overlay pages were rendered and inspected:

- HCM202: PDF pages 8, 70, 166 and 243;
- MLN131: PDF pages 7, 82, 161 and 235;
- VNR202: PDF pages 21, 68, 124 and 206.

Line boxes align with visible text on normal and chapter-transition pages. VNR202
requires its source rotation metadata to be applied by the overlay renderer; this is
validated against regenerated upright previews. Edge-scan noise can create small
marginal boxes but does not shift body geometry.

Dense retrieval plus parent expansion returned:

- HCM202 “đại đoàn kết dân tộc”: top 3 in Chapter 5, pages 167–202;
- MLN131 “sứ mệnh lịch sử của giai cấp công nhân”: top results in Chapter 2;
- VNR202 “ý nghĩa lịch sử của việc thành lập Đảng”: top 3 in Chapter 1,
  pages 21–38.

No contents chunk appears in these post-fix results.

## Gate result

**PASS for full OCR coverage, checksum integrity, Vietnamese text quality,
line-level provenance, visual bbox alignment, parent-child constraints, current GPU
indexes and qualitative retrieval.** The sparse review pages remain explicitly
flagged, and quantitative retrieval evaluation still requires human-reviewed labels.
