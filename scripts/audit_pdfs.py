"""
Phase 0: PDF Quality Audit
Kiểm tra chất lượng 5 giáo trình PDF trước khi xây pipeline RAG.

Output: reports/pdf_audit_report.md + data/audit/audit_results.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

# ── Config ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "Tài liệu"
REPORT_DIR = PROJECT_ROOT / "reports"
AUDIT_DATA_DIR = PROJECT_ROOT / "data" / "audit"

SUBJECT_MAP = {
    "Giáo trình MLN111.pdf": {"code": "MLN111", "name": "Triết học Mác - Lênin"},
    "Giáo trình MLN122.pdf": {"code": "MLN122", "name": "Kinh tế Chính trị Mác - Lênin"},
    "Giáo trình MLN131.pdf": {"code": "MLN131", "name": "Chủ nghĩa Xã hội Khoa học"},
    "Giáo trình HCM202.pdf": {"code": "HCM202", "name": "Tư tưởng Hồ Chí Minh"},
    "Giáo trình VNR202.pdf": {"code": "VNR202", "name": "Lịch sử Đảng Cộng sản Việt Nam"},
}

# Ký tự thay thế Unicode phổ biến khi font lỗi
REPLACEMENT_CHARS = {"\ufffd", "\x00", "\ufffe", "\uffff"}

# Vietnamese diacritics pattern
VIETNAMESE_PATTERN = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)

# Heading patterns
HEADING_PATTERNS = [
    re.compile(r"^(Chương|CHƯƠNG)\s+[IVXLCDM\d]+", re.MULTILINE),
    re.compile(r"^(Phần|PHẦN)\s+[IVXLCDM\d]+", re.MULTILINE),
    re.compile(r"^(Mục|MỤC)\s+\d+", re.MULTILINE),
    re.compile(r"^\d+\.\d+\.?\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸ]", re.MULTILINE),
    re.compile(r"^[IVXLCDM]+\.\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸ]", re.MULTILINE),
]

# Printed page number patterns
PRINTED_PAGE_PATTERN = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)
ROMAN_PAGE_PATTERN = re.compile(r"^\s*([ivxlcdm]+)\s*$", re.MULTILINE | re.IGNORECASE)

# ── Helper Functions ────────────────────────────────────────────────────────


def compute_quality_score(text: str, page_area: float) -> dict:
    """Compute quality score for a page's extracted text."""
    char_count = len(text)

    if char_count == 0:
        return {
            "char_count": 0,
            "unicode_error_ratio": 0.0,
            "vietnamese_word_ratio": 0.0,
            "replacement_char_count": 0,
            "quality_score": 0.0,
            "needs_ocr": True,
        }

    # Count replacement/error characters
    replacement_count = sum(1 for c in text if c in REPLACEMENT_CHARS)
    unicode_error_ratio = replacement_count / char_count

    # Count Vietnamese characters
    viet_chars = len(VIETNAMESE_PATTERN.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    vietnamese_ratio = viet_chars / max(alpha_chars, 1)

    # Simple word-level check: split and count "readable" words
    words = text.split()
    word_count = len(words)

    # Compute composite quality score (0-1)
    score = 1.0
    if char_count < 50:
        score *= 0.1  # Almost empty page
    elif char_count < 200:
        score *= 0.5  # Very little text

    if unicode_error_ratio > 0.1:
        score *= 0.2
    elif unicode_error_ratio > 0.05:
        score *= 0.5

    # Pages with text content should have some Vietnamese chars
    # (unless it's a table of contents or index with mostly numbers)
    if alpha_chars > 50 and vietnamese_ratio < 0.02:
        score *= 0.3  # Likely garbled text

    needs_ocr = (
        char_count < 100
        or unicode_error_ratio > 0.05
        or (alpha_chars > 50 and vietnamese_ratio < 0.02)
        or score < 0.3
    )

    return {
        "char_count": char_count,
        "word_count": word_count,
        "unicode_error_ratio": round(unicode_error_ratio, 4),
        "vietnamese_char_count": viet_chars,
        "vietnamese_ratio": round(vietnamese_ratio, 4),
        "replacement_char_count": replacement_count,
        "quality_score": round(score, 3),
        "needs_ocr": needs_ocr,
    }


def detect_headings(text: str) -> list[dict]:
    """Detect heading patterns in text."""
    headings = []
    for pattern in HEADING_PATTERNS:
        for m in pattern.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            heading_text = text[line_start:line_end].strip()
            if len(heading_text) < 200:  # Sanity check
                headings.append(
                    {
                        "text": heading_text[:150],
                        "pattern": pattern.pattern[:50],
                        "position": m.start(),
                    }
                )
    return headings


def detect_printed_page(text: str) -> str | None:
    """Try to detect printed page number from page text."""
    lines = text.strip().split("\n")

    # Check first and last 3 lines for page numbers
    candidates = lines[:3] + lines[-3:]
    for line in candidates:
        line = line.strip()
        # Simple numeric
        if re.match(r"^\d{1,4}$", line):
            return line
        # Roman numeral
        if re.match(r"^[ivxlcdm]+$", line, re.IGNORECASE) and len(line) < 8:
            return line
    return None


def detect_font_sizes(page) -> dict:
    """Extract font size distribution from a page."""
    font_sizes = Counter()
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = round(span["size"], 1)
                font_sizes[size] += len(span["text"])
    return dict(font_sizes.most_common(10))


def detect_repeated_lines(pages_text: list[str], threshold: float = 0.7) -> list[str]:
    """Find lines that appear on >threshold fraction of pages."""
    line_counts = Counter()
    total_pages = len(pages_text)

    for text in pages_text:
        lines = set()
        for line in text.split("\n"):
            cleaned = line.strip()
            if len(cleaned) > 3 and len(cleaned) < 200:
                lines.add(cleaned)
        for line in lines:
            line_counts[line] += 1

    repeated = []
    for line, count in line_counts.most_common(20):
        if count / total_pages >= threshold:
            repeated.append(f"{line}  ({count}/{total_pages} pages)")
    return repeated


def detect_layout_issues(page) -> dict:
    """Check for multi-column layout and reading order issues."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    text_blocks = [b for b in blocks if b.get("type") == 0 and "lines" in b]

    if len(text_blocks) < 2:
        return {"multi_column": False, "block_count": len(text_blocks)}

    # Check if blocks are side-by-side (multi-column indicator)
    page_width = page.rect.width
    left_blocks = [b for b in text_blocks if b["bbox"][2] < page_width * 0.55]
    right_blocks = [b for b in text_blocks if b["bbox"][0] > page_width * 0.45]

    has_both = len(left_blocks) > 0 and len(right_blocks) > 0
    # Check for significant overlap in Y coordinates (side-by-side)
    multi_column = False
    if has_both:
        for lb in left_blocks:
            for rb in right_blocks:
                y_overlap = min(lb["bbox"][3], rb["bbox"][3]) - max(lb["bbox"][1], rb["bbox"][1])
                if y_overlap > 50:
                    multi_column = True
                    break

    return {
        "multi_column": multi_column,
        "block_count": len(text_blocks),
        "left_blocks": len(left_blocks),
        "right_blocks": len(right_blocks),
    }


def count_tables_and_images(page) -> dict:
    """Count tables and images on a page."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    image_blocks = [b for b in blocks if b.get("type") == 1]

    # Simple table detection: look for lines with multiple tab/space-separated columns
    text = page.get_text()
    table_indicators = 0
    for line in text.split("\n"):
        # Lines with multiple segments separated by 3+ spaces suggest tables
        parts = re.split(r"\s{3,}", line.strip())
        if len(parts) >= 3 and all(len(p) > 0 for p in parts[:3]):
            table_indicators += 1

    return {
        "image_count": len(image_blocks),
        "table_indicator_lines": table_indicators,
    }


# ── Main Audit ──────────────────────────────────────────────────────────────


def audit_pdf(filepath: Path) -> dict:
    """Perform comprehensive audit on a single PDF file."""
    filename = filepath.name
    subject_info = SUBJECT_MAP.get(filename, {"code": "UNKNOWN", "name": "Unknown"})

    print(f"\n{'=' * 60}")
    print(f"Auditing: {filename}")
    print(f"Subject: {subject_info['code']} — {subject_info['name']}")
    print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")

    doc = fitz.open(str(filepath))
    total_pages = len(doc)

    pages_data = []
    all_pages_text = []
    all_headings = []
    font_size_global = Counter()
    ocr_needed_pages = []
    multi_column_pages = []
    total_images = 0
    total_table_lines = 0
    sample_texts = []

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        all_pages_text.append(text)

        # Quality score
        page_area = page.rect.width * page.rect.height
        quality = compute_quality_score(text, page_area)

        # Printed page detection
        printed_page = detect_printed_page(text)

        # Font sizes (sample every 10th page + first 5)
        font_sizes = {}
        if page_num < 5 or page_num % 10 == 0:
            font_sizes = detect_font_sizes(page)
            for size, count in font_sizes.items():
                font_size_global[size] += count

        # Headings
        headings = detect_headings(text)
        for h in headings:
            h["pdf_page"] = page_num
        all_headings.extend(headings)

        # Layout
        layout = detect_layout_issues(page)
        if layout["multi_column"]:
            multi_column_pages.append(page_num)

        # Tables and images
        ti = count_tables_and_images(page)
        total_images += ti["image_count"]
        total_table_lines += ti["table_indicator_lines"]

        # Track OCR needs
        if quality["needs_ocr"]:
            ocr_needed_pages.append(page_num)

        # Sample text (pages 5, 15, 30, 50, 80 or proportional)
        sample_indices = [int(total_pages * f) for f in [0.05, 0.15, 0.3, 0.5, 0.8]]
        if page_num in sample_indices and len(sample_texts) < 5:
            snippet = text[:500].strip()
            if snippet:
                sample_texts.append(
                    {
                        "pdf_page": page_num,
                        "printed_page": printed_page,
                        "snippet": snippet,
                        "quality_score": quality["quality_score"],
                    }
                )

        page_data = {
            "pdf_page": page_num,
            "printed_page": printed_page,
            "quality": quality,
            "heading_count": len(headings),
            "layout": layout,
        }
        pages_data.append(page_data)

        # Progress
        if (page_num + 1) % 50 == 0 or page_num == total_pages - 1:
            print(f"  Processed {page_num + 1}/{total_pages} pages...")

    # Repeated headers/footers
    repeated_lines = detect_repeated_lines(all_pages_text)

    # Page stats
    pages_with_text = sum(1 for p in pages_data if p["quality"]["char_count"] > 100)
    avg_chars = sum(p["quality"]["char_count"] for p in pages_data) / max(total_pages, 1)
    avg_quality = sum(p["quality"]["quality_score"] for p in pages_data) / max(total_pages, 1)

    # Page number mapping analysis
    printed_pages_found = [p for p in pages_data if p["printed_page"] is not None]
    page_offset_samples = []
    for p in printed_pages_found[:20]:
        try:
            printed_num = int(p["printed_page"])
            offset = p["pdf_page"] - printed_num
            page_offset_samples.append(
                {"pdf_page": p["pdf_page"], "printed": printed_num, "offset": offset}
            )
        except (ValueError, TypeError):
            page_offset_samples.append(
                {"pdf_page": p["pdf_page"], "printed": p["printed_page"], "offset": "non-numeric"}
            )

    doc.close()

    result = {
        "file": filename,
        "subject_code": subject_info["code"],
        "subject_name": subject_info["name"],
        "file_size_mb": round(filepath.stat().st_size / 1024 / 1024, 1),
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "pages_needing_ocr": len(ocr_needed_pages),
        "ocr_page_indices": ocr_needed_pages[:50],  # First 50 for brevity
        "avg_chars_per_page": round(avg_chars),
        "avg_quality_score": round(avg_quality, 3),
        "total_headings_detected": len(all_headings),
        "heading_samples": all_headings[:15],
        "font_size_distribution": dict(font_size_global.most_common(10)),
        "multi_column_pages": multi_column_pages[:20],
        "multi_column_count": len(multi_column_pages),
        "repeated_headers_footers": repeated_lines[:10],
        "total_images": total_images,
        "total_table_indicator_lines": total_table_lines,
        "page_offset_samples": page_offset_samples[:15],
        "sample_texts": sample_texts,
        "pages_detail": pages_data,  # Full per-page data
    }

    # Print summary
    print("\n  Summary:")
    print(f"    Total pages: {total_pages}")
    print(f"    Pages with text: {pages_with_text} ({pages_with_text / total_pages * 100:.0f}%)")
    print(
        f"    Pages needing OCR: {len(ocr_needed_pages)} "
        f"({len(ocr_needed_pages) / total_pages * 100:.0f}%)"
    )
    print(f"    Avg chars/page: {avg_chars:.0f}")
    print(f"    Avg quality: {avg_quality:.3f}")
    print(f"    Headings found: {len(all_headings)}")
    print(f"    Multi-column pages: {len(multi_column_pages)}")
    print(f"    Images: {total_images}")
    print(f"    Repeated lines: {len(repeated_lines)}")

    return result


def generate_report(results: list[dict]) -> str:
    """Generate markdown report from audit results."""
    lines = []
    lines.append("# PDF Quality Audit Report")
    lines.append("")
    lines.append("Generated by `scripts/audit_pdfs.py`")
    lines.append("")

    # Summary table
    lines.append("## Tổng quan")
    lines.append("")
    lines.append(
        "| File | Mã môn | Trang | Có text | Cần OCR | Chars/trang | "
        "Quality | Headings | Multi-col | Images |"
    )
    lines.append(
        "|------|--------|-------|---------|---------|-------------|---------|----------|-----------|--------|"
    )

    for r in results:
        ocr_pct = f"{r['pages_needing_ocr']}/{r['total_pages']}"
        text_pct = f"{r['pages_with_text']}/{r['total_pages']}"
        lines.append(
            f"| {r['file']} | {r['subject_code']} | {r['total_pages']} | {text_pct} "
            f"| {ocr_pct} | {r['avg_chars_per_page']} | {r['avg_quality_score']:.3f} "
            f"| {r['total_headings_detected']} | {r['multi_column_count']} | {r['total_images']} |"
        )

    lines.append("")

    # Per-file details
    for r in results:
        lines.append("---")
        lines.append("")
        lines.append(f"## {r['subject_code']} — {r['subject_name']}")
        lines.append(f"**File**: `{r['file']}` ({r['file_size_mb']} MB)")
        lines.append("")

        # Basic stats
        lines.append("### Thống kê cơ bản")
        lines.append(f"- Tổng trang: **{r['total_pages']}**")
        lines.append(
            f"- Trang có text (>100 chars): **{r['pages_with_text']}** "
            f"({r['pages_with_text'] / r['total_pages'] * 100:.0f}%)"
        )
        lines.append(
            f"- Trang cần OCR: **{r['pages_needing_ocr']}** "
            f"({r['pages_needing_ocr'] / r['total_pages'] * 100:.0f}%)"
        )
        lines.append(f"- Trung bình chars/trang: **{r['avg_chars_per_page']}**")
        lines.append(f"- Quality score trung bình: **{r['avg_quality_score']:.3f}**")
        lines.append("")

        # OCR pages
        if r["pages_needing_ocr"] > 0:
            lines.append("### Trang cần OCR")
            if r["pages_needing_ocr"] <= 20:
                lines.append(f"Pages: {r['ocr_page_indices']}")
            else:
                lines.append(
                    f"Tổng: {r['pages_needing_ocr']} trang. "
                    f"Đầu tiên: {r['ocr_page_indices'][:20]}..."
                )
            lines.append("")

        # Headings
        lines.append("### Headings phát hiện")
        lines.append(f"Tổng: {r['total_headings_detected']} headings")
        lines.append("")
        if r["heading_samples"]:
            lines.append("Mẫu:")
            for h in r["heading_samples"][:10]:
                lines.append(f"- Page {h['pdf_page']}: `{h['text']}`")
            lines.append("")

        # Font sizes
        if r["font_size_distribution"]:
            lines.append("### Font size distribution")
            lines.append("| Size | Char count |")
            lines.append("|------|-----------|")
            for size, count in sorted(r["font_size_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| {size} | {count} |")
            lines.append("")

        # Page mapping
        if r["page_offset_samples"]:
            lines.append("### Page number mapping (PDF page → Printed page)")
            lines.append("| PDF page | Printed page | Offset |")
            lines.append("|----------|-------------|--------|")
            for p in r["page_offset_samples"][:15]:
                lines.append(f"| {p['pdf_page']} | {p['printed']} | {p['offset']} |")
            lines.append("")

        # Multi-column
        if r["multi_column_count"] > 0:
            lines.append("### Layout 2 cột")
            lines.append(f"Phát hiện **{r['multi_column_count']}** trang có thể là 2 cột.")
            if r["multi_column_pages"]:
                lines.append(f"Pages: {r['multi_column_pages'][:20]}")
            lines.append("")

        # Repeated lines
        if r["repeated_headers_footers"]:
            lines.append("### Header/Footer lặp")
            for line in r["repeated_headers_footers"]:
                lines.append(f"- `{line}`")
            lines.append("")

        # Images and tables
        lines.append("### Bảng và hình ảnh")
        lines.append(f"- Hình ảnh: **{r['total_images']}**")
        lines.append(f"- Dòng có dấu hiệu bảng: **{r['total_table_indicator_lines']}**")
        lines.append("")

        # Sample texts
        if r["sample_texts"]:
            lines.append("### Sample text")
            for s in r["sample_texts"]:
                lines.append(
                    f"#### Page {s['pdf_page']} (printed: {s['printed_page']}, "
                    f"quality: {s['quality_score']})"
                )
                lines.append("```")
                lines.append(s["snippet"])
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def main():
    """Run audit on all PDFs."""
    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {PDF_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

    # Create output directories
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for pdf_path in pdf_files:
        result = audit_pdf(pdf_path)
        results.append(result)

    # Save JSON (without per-page detail for readability)
    results_summary = []
    for r in results:
        summary = {k: v for k, v in r.items() if k != "pages_detail"}
        results_summary.append(summary)

    json_path = AUDIT_DATA_DIR / "audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save full per-page data separately
    for r in results:
        detail_path = AUDIT_DATA_DIR / f"{r['subject_code']}_pages.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(r["pages_detail"], f, ensure_ascii=False, indent=2)

    # Generate and save markdown report
    report = generate_report(results)
    report_path = REPORT_DIR / "pdf_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Markdown report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
