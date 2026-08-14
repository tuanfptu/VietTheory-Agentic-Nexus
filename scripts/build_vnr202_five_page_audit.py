"""Build a compact five-page visual audit pack for the VNR202 corpus."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = ROOT / "Tài liệu" / "Giáo trình VNR202.pdf"
PAGES_PATH = ROOT / "data" / "processed" / "VNR202" / "pages.jsonl"
HEADINGS_PATH = ROOT / "data" / "processed" / "VNR202" / "structured_v1" / "headings.jsonl"
PARENTS_PATH = ROOT / "data" / "processed" / "VNR202" / "structured_v1" / "parents.jsonl"
GEMINI_COMPARISON_PATH = (
    ROOT
    / "data"
    / "processed"
    / "VNR202"
    / "structured_v2_gemini"
    / "pilot_5_pages"
    / "comparison.json"
)
OUTPUT_PDF = ROOT / "output" / "pdf" / "VNR202_five_page_human_audit.pdf"
OUTPUT_JSON = ROOT / "reports" / "vnr202_five_page_human_review.json"
TMP_DIR = ROOT / "tmp" / "pdfs" / "vnr202_five_page_audit"
PDFTOPPM = Path(
    r"C:\Users\TUAN\.cache\codex-runtimes\codex-primary-runtime\dependencies"
    r"\native\poppler\Library\bin\pdftoppm.exe"
)

# Zero-based PDF page numbers. These cover all three chapter openings, a normal
# content page, and a late heading-heavy page.
SAMPLES = (
    (21, "Chapter 1 opening"),
    (68, "Chapter 2 opening"),
    (124, "Chapter 3 opening"),
    (180, "Normal content control"),
    (225, "Late review-section exclusion control"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    pdfmetrics.registerFont(TTFont("AuditSans", str(regular)))
    pdfmetrics.registerFont(TTFont("AuditSans-Bold", str(bold)))


def render_source_page(pdf_page: int) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_DIR / f"page_{pdf_page + 1:03d}"
    expected = prefix.with_suffix(".png")
    subprocess.run(
        [
            str(PDFTOPPM),
            "-f",
            str(pdf_page + 1),
            "-l",
            str(pdf_page + 1),
            "-singlefile",
            "-r",
            "120",
            "-png",
            str(SOURCE_PDF),
            str(prefix),
        ],
        check=True,
    )
    return expected


def parents_for_page(parents: list[dict[str, Any]], pdf_page: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for parent in parents:
        if any(span.get("pdf_page") == pdf_page for span in parent.get("source_spans", [])):
            matches.append(parent)
    return matches


def draw_wrapped(
    canvas: Canvas,
    text: str,
    x: float,
    y_top: float,
    width: float,
    height: float,
    *,
    size: float = 10,
    color: str = "#172033",
) -> None:
    style = ParagraphStyle(
        "audit",
        fontName="AuditSans",
        fontSize=size,
        leading=size * 1.35,
        textColor=HexColor(color),
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    raw = text

    def escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    safe = escape(raw)
    paragraph = Paragraph(safe, style)
    _, rendered_height = paragraph.wrap(width, height)
    while rendered_height > height and len(raw) > 160:
        raw = raw[: int(len(raw) * 0.88)].rstrip() + " ..."
        safe = escape(raw)
        paragraph = Paragraph(safe, style)
        _, rendered_height = paragraph.wrap(width, height)
    paragraph.drawOn(canvas, x, y_top - rendered_height)


def build() -> None:
    register_fonts()
    pages = {item["pdf_page"]: item for item in read_jsonl(PAGES_PATH)}
    headings = read_jsonl(HEADINGS_PATH)
    parents = read_jsonl(PARENTS_PATH)
    gemini_pages: dict[int, dict[str, Any]] = {}
    if GEMINI_COMPARISON_PATH.exists():
        comparison = json.loads(GEMINI_COMPARISON_PATH.read_text(encoding="utf-8"))
        gemini_pages = {item["pdf_page"]: item for item in comparison["pages"]}
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = landscape(A3)
    canvas = Canvas(str(OUTPUT_PDF), pagesize=(page_width, page_height))
    review_cases: list[dict[str, Any]] = []

    for index, (pdf_page, category) in enumerate(SAMPLES, start=1):
        page = pages[pdf_page]
        page_headings = [item for item in headings if item["pdf_page"] == pdf_page]
        page_parents = parents_for_page(parents, pdf_page)
        image_path = render_source_page(pdf_page)

        canvas.setFillColor(HexColor("#F5F7FB"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#172033"))
        canvas.setFont("AuditSans-Bold", 20)
        canvas.drawString(28, page_height - 34, f"VNR202 human audit {index}/5")
        canvas.setFont("AuditSans", 11)
        canvas.setFillColor(HexColor("#526078"))
        canvas.drawString(
            28,
            page_height - 53,
            f"PDF page {pdf_page + 1} (stored pdf_page={pdf_page}) - {category}",
        )

        image_x, image_y = 28, 52
        image_w, image_h = 515, page_height - 124
        image = ImageReader(str(image_path))
        source_w, source_h = image.getSize()
        scale = min(image_w / source_w, image_h / source_h)
        draw_w, draw_h = source_w * scale, source_h * scale
        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.roundRect(image_x - 8, image_y - 8, image_w + 16, image_h + 16, 8, fill=1, stroke=0)
        canvas.drawImage(
            image,
            image_x + (image_w - draw_w) / 2,
            image_y + (image_h - draw_h) / 2,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
        )

        right_x = 575
        right_w = page_width - right_x - 30
        top = page_height - 32
        canvas.setFont("AuditSans-Bold", 12)
        canvas.setFillColor(HexColor("#172033"))
        canvas.drawString(right_x, top, "Machine extraction")
        canvas.setFont("AuditSans", 9)
        canvas.setFillColor(HexColor("#526078"))
        canvas.drawString(
            right_x,
            top - 18,
            f"method={page.get('extraction_method')} | rotation={page.get('rotation')} | "
            f"chars={page.get('char_count')} | quality={page.get('quality_score')}",
        )

        heading_text = (
            "\n".join(f"L{item['level']}: {item['text']}" for item in page_headings)
            or "(No heading detected on this page)"
        )
        gemini_elements = gemini_pages.get(pdf_page, {}).get("gemini_elements", [])
        gemini_heading_text = (
            "\n".join(
                f"L{item['level']}: {item['text']}"
                for item in gemini_elements
                if item.get("level") is not None
            )
            or "(No Gemini heading detected on this page)"
        )
        canvas.setFont("AuditSans-Bold", 10)
        canvas.setFillColor(HexColor("#2F6FED"))
        canvas.drawString(right_x, top - 43, "Rule-based v1 heading(s)")
        draw_wrapped(canvas, heading_text, right_x, top - 51, right_w, 52, size=8)
        canvas.setFillColor(HexColor("#7C3AED"))
        canvas.drawString(right_x, top - 108, "Gemini v2 heading(s) - review this output")
        draw_wrapped(canvas, gemini_heading_text, right_x, top - 116, right_w, 72, size=8)

        boundary_lines = []
        for parent in page_parents:
            section = parent.get("section") or "-"
            if len(section) > 72:
                section = section[:69] + "..."
            boundary_lines.append(f"{parent['chunk_id']} | section={section}")
        canvas.setFont("AuditSans-Bold", 10)
        canvas.setFillColor(HexColor("#2F6FED"))
        canvas.drawString(right_x, top - 204, "Parent context touching this page (v1)")
        draw_wrapped(
            canvas,
            "\n".join(boundary_lines) or "(No parent found: verify that exclusion is intentional)",
            right_x,
            top - 212,
            right_w,
            70,
            size=8,
        )

        canvas.setFont("AuditSans-Bold", 10)
        canvas.setFillColor(HexColor("#2F6FED"))
        canvas.drawString(right_x, top - 298, "OCR text - compare with the source image")
        draw_wrapped(
            canvas,
            page.get("text", ""),
            right_x,
            top - 320,
            right_w,
            278,
            size=8,
        )

        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.roundRect(right_x, 26, right_w, 76, 7, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#172033"))
        canvas.setFont("AuditSans-Bold", 9)
        canvas.drawString(right_x + 12, 82, "Manual decision")
        canvas.setFont("AuditSans", 9)
        canvas.drawString(right_x + 12, 63, "OCR: [ ] pass  [ ] minor  [ ] fail")
        canvas.drawString(right_x + 205, 63, "Heading: [ ] pass  [ ] missing  [ ] false positive")
        canvas.drawString(right_x + 12, 44, "Parent/chapter: [ ] pass  [ ] fail")
        canvas.drawString(right_x + 205, 44, "Notes: __________________________________")

        canvas.setFont("AuditSans", 7)
        canvas.setFillColor(HexColor("#6B7280"))
        canvas.drawRightString(
            page_width - 28, 12, "Quick audit only - not a full-corpus quality certificate"
        )
        canvas.showPage()

        review_cases.append(
            {
                "case_id": f"vnr202_page_{pdf_page + 1:03d}",
                "pdf_page_zero_based": pdf_page,
                "pdf_page_human": pdf_page + 1,
                "category": category,
                "detected_headings": page_headings,
                "parent_chunk_ids": [parent["chunk_id"] for parent in page_parents],
                "human_review": {
                    "ocr": "pending",
                    "heading": "pending",
                    "parent_and_chapter": "pending",
                    "notes": "",
                },
            }
        )

    canvas.save()
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "subject": "VNR202",
                "scope": "five_page_quick_human_audit",
                "status": "pending_human_review",
                "cases": review_cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
