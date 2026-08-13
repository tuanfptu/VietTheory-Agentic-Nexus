# Corpus Processing Status

## Eligibility audit

| Subject | Pages | Native-text pages | OCR pages required | Current gate |
|---|---:|---:|---:|---|
| MLN111 | 285 | 284 | 1 | Native pipeline PASS |
| MLN122 | 262 | 262 | 0 | Native pipeline PASS |
| HCM202 | 271 | 0 | 271 | OCR pipeline PASS |
| MLN131 | 273 | 0 | 273 | OCR pipeline PASS |
| VNR202 | 230 | 0 | 230 | OCR pipeline PASS |

All five subjects now have extraction artifacts, structural parent-child chunks and
persisted Qwen dense indexes. MLN111 and MLN122 use native extraction. HCM202,
MLN131 and VNR202 use Tesseract `tessdata_best/vie` OCR because their PDFs contain
no usable text layer.

The deterministic readiness audit is available at `reports/five_subject_readiness.json` and can
be reproduced with:

```powershell
.\.venv\Scripts\python.exe scripts\audit_subject_readiness.py
```

It validates 15 integrity and identity checks per subject, including source/chunk/index checksums,
parent links, vector mappings, subject purity, extraction mode, page counts, and chapter counts.

## Next gate

The next corpus-wide gate is human review of benchmark relevance labels followed by
quantitative hybrid retrieval and reranker evaluation across all five subjects.
