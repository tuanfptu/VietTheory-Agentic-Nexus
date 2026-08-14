# Structured v2 Audit

The frozen `structured_v1` artifacts were not modified. `structured_v2` is a
candidate namespace and is not used by production retrieval yet.

| Subject | v1 children | v2 children | v1 pages | v2 pages | Gemini exact level |
|---|---:|---:|---:|---:|---:|
| VNR202 | 459 | 526 | 198 | 224 | 93.75% |
| MLN131 | 334 | 381 | 269 | 266 | 70.83% |

## Interpretation

A Gemini disagreement is an audit candidate, not an automatic correction. Any
unmatched anchor or hierarchy mismatch must be reviewed before v2 promotion.
Dense indexes must only be rebuilt after this promotion gate.

### VNR202

- Gemini headings: 16
- Exact anchor and level matches: 15
- Level mismatches: 0
- Unmatched anchors: 1
- Review candidates: 1

### MLN131

- Gemini headings: 24
- Exact anchor and level matches: 17
- Level mismatches: 1
- Unmatched anchors: 6
- Review candidates: 7
