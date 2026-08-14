# Structured v2 Visual Adjudication

## Method

The remaining Gemini-versus-parser disagreements were checked against rendered
source PDF pages. Decisions below apply only to candidate `structured_v2`; frozen
`structured_v1` and production indexes remain unchanged.

## VNR202

| PDF page (zero-based) | Candidate | Decision | Reason |
|---:|---|---|---|
| 15 | `Phương pháp lịch sử` | reject Gemini anchor | The cited anchor is the opening quotation/body sentence, not the italic heading printed on the preceding page. |
| 15 | `Phương pháp logic` | pending override | This is a real italic subheading, but it was not retained as a validated Gemini element. Add only through a provenance-tracked override. |
| 223 | `Những bài học lớn về sự lãnh đạo của Đảng` | accept | Visually bold standalone heading introducing the numbered lessons. |

The recovered `KẾT LUẬN` boundary and numbered headings on pages 223-226 are
visually valid. The reference section beginning on page 229 is correctly excluded.

## MLN131

| PDF page (zero-based) | Candidate | Decision | Reason |
|---:|---|---|---|
| 158-159 | `Một là` ... `Bốn là` | reject Gemini level 3 | These are italic enumerated subpoints under subsection `b)`, not level-3 sections. |
| 159 | Two numbered review questions | reject | They belong to `C. CÂU HỎI ÔN TẬP` and must remain excluded from retrieval. |
| 235 | `I- KHÁI NIỆM, VỊ TRÍ VÀ CHỨC NĂNG CỦA GIA ĐÌNH` | keep deterministic level 2 | The visual hierarchy places it under `B. NỘI DUNG`; Gemini level 3 is inconsistent with the book layout. |

## Promotion decision

`structured_v2` remains a candidate. Do not build its dense indexes until the two
accepted/pending VNR202 overrides are represented by a deterministic override file
whose checksum is recorded in the structured manifest. MLN131 requires no Gemini
override from this reviewed disagreement set.
