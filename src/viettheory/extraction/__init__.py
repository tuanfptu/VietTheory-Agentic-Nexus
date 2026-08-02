"""PDF extraction primitives with citation-preserving metadata."""

from viettheory.extraction.pdf_extractor import extract_pdf, iter_pdf_pages, read_document
from viettheory.schema import BoundingBox, Page, TextBlock, TextLine

__all__ = [
    "BoundingBox",
    "Page",
    "TextBlock",
    "TextLine",
    "extract_pdf",
    "iter_pdf_pages",
    "read_document",
]
