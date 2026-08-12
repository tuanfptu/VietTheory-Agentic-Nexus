"""Versioned, validated contracts shared by the MLN111 pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SchemaVersion = Literal["1.0"]
SCHEMA_VERSION: SchemaVersion = "1.0"
BoundingBox = tuple[float, float, float, float]

NonEmptyText = Annotated[str, Field(min_length=1)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]


class VietTheoryModel(BaseModel):
    """Strict and immutable base class for persisted pipeline records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: SchemaVersion = SCHEMA_VERSION


class ExtractionMethod(StrEnum):
    """Method that produced page text."""

    PYMUPDF = "pymupdf"
    OCR = "ocr"
    NONE = "none"


class BlockRole(StrEnum):
    """Semantic role assigned without discarding source geometry."""

    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class Document(VietTheoryModel):
    """A source document identified by its content digest."""

    document_id: NonEmptyText
    file_name: NonEmptyText
    subject_code: NonEmptyText
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_count: Annotated[int, Field(gt=0)]


class ExtractionManifest(VietTheoryModel):
    """Reproducibility metadata for a page JSONL artifact."""

    document: Document
    extractor: Literal["pymupdf", "tesseract"] = "pymupdf"
    extractor_version: NonEmptyText
    output_format: Literal["jsonl"] = "jsonl"
    postprocessors: tuple[str, ...] = ()
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    start_page: Annotated[int, Field(ge=0)]
    end_page: Annotated[int, Field(gt=0)]
    extracted_page_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_range(self) -> ExtractionManifest:
        """Keep the recorded half-open range consistent with artifact size."""
        if self.end_page <= self.start_page:
            raise ValueError("end_page must be greater than start_page")
        if self.end_page - self.start_page != self.extracted_page_count:
            raise ValueError("page range must match extracted_page_count")
        if self.end_page > self.document.page_count:
            raise ValueError("end_page cannot exceed document page_count")
        return self


class TextLine(VietTheoryModel):
    """One line of source text in PDF point coordinates."""

    line_id: NonEmptyText
    bbox: BoundingBox
    text: NonEmptyText
    font_size: Annotated[float, Field(gt=0.0)] | None = None
    font_flags: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_bbox(self) -> TextLine:
        """Reject inverted source rectangles."""
        x0, y0, x1, y1 = self.bbox
        if x0 > x1 or y0 > y1:
            raise ValueError("bbox must satisfy x0 <= x1 and y0 <= y1")
        return self


class TextBlock(VietTheoryModel):
    """An ordered text block made from source lines."""

    block_id: NonEmptyText
    bbox: BoundingBox
    text: NonEmptyText
    lines: tuple[TextLine, ...]
    role: BlockRole = BlockRole.BODY

    @model_validator(mode="after")
    def validate_content(self) -> TextBlock:
        """Require lines and a valid block rectangle."""
        if not self.lines:
            raise ValueError("text blocks must contain at least one line")
        x0, y0, x1, y1 = self.bbox
        if x0 > x1 or y0 > y1:
            raise ValueError("bbox must satisfy x0 <= x1 and y0 <= y1")
        return self


class Page(VietTheoryModel):
    """Citation-ready representation of a zero-based PDF page."""

    page_id: NonEmptyText
    document_id: NonEmptyText
    pdf_file: NonEmptyText
    subject_code: NonEmptyText
    pdf_page: Annotated[int, Field(ge=0)]
    printed_page: str | None = None
    width: Annotated[float, Field(gt=0.0)]
    height: Annotated[float, Field(gt=0.0)]
    rotation: Literal[0, 90, 180, 270] = 0
    text: str
    extraction_method: ExtractionMethod
    char_count: Annotated[int, Field(ge=0)]
    quality_score: UnitScore
    needs_ocr: bool
    image_count: Annotated[int, Field(ge=0)] = 0
    blocks: tuple[TextBlock, ...]

    @model_validator(mode="after")
    def validate_page(self) -> Page:
        """Keep derived fields and all source rectangles internally consistent."""
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal len(text)")
        if self.extraction_method is ExtractionMethod.NONE and self.text:
            raise ValueError("extraction_method='none' requires empty text")
        for block in self.blocks:
            x0, y0, x1, y1 = block.bbox
            if x0 < 0 or y0 < 0 or x1 > self.width or y1 > self.height:
                raise ValueError(f"block {block.block_id} bbox is outside page bounds")
        return self


class SourceSpan(VietTheoryModel):
    """Exact page region supporting a chunk or citation."""

    page_id: NonEmptyText
    pdf_page: Annotated[int, Field(ge=0)]
    printed_page: str | None = None
    bbox: BoundingBox
    text: NonEmptyText

    @model_validator(mode="after")
    def validate_bbox(self) -> SourceSpan:
        """Reject inverted citation rectangles."""
        x0, y0, x1, y1 = self.bbox
        if x0 > x1 or y0 > y1:
            raise ValueError("bbox must satisfy x0 <= x1 and y0 <= y1")
        return self


class Chunk(VietTheoryModel):
    """Retrieval unit that retains complete source provenance."""

    chunk_id: NonEmptyText
    document_id: NonEmptyText
    subject_code: NonEmptyText
    text: NonEmptyText
    token_count: Annotated[int, Field(gt=0)]
    source_spans: tuple[SourceSpan, ...]
    chunk_kind: Literal["baseline", "parent", "child"] = "baseline"
    parent_chunk_id: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None

    @model_validator(mode="after")
    def require_provenance(self) -> Chunk:
        """Forbid chunks that cannot be traced to source pages."""
        if not self.source_spans:
            raise ValueError("chunks must contain at least one source span")
        if self.chunk_kind == "child" and not self.parent_chunk_id:
            raise ValueError("child chunks require parent_chunk_id")
        if self.chunk_kind != "child" and self.parent_chunk_id:
            raise ValueError("only child chunks may reference a parent")
        return self


class RetrievedEvidence(VietTheoryModel):
    """A ranked chunk returned by retrieval."""

    evidence_id: NonEmptyText
    chunk: Chunk
    score: float
    rank: Annotated[int, Field(gt=0)]
    retrieval_method: NonEmptyText


class Citation(VietTheoryModel):
    """A claim-to-evidence link constrained to retrieved evidence."""

    citation_id: NonEmptyText
    evidence_id: NonEmptyText
    source_span: SourceSpan
    context_text: str | None = None


class Claim(VietTheoryModel):
    """One independently verifiable statement in an answer."""

    claim_id: NonEmptyText
    text: NonEmptyText
    citation_ids: tuple[str, ...]


class Answer(VietTheoryModel):
    """Structured generator result with claim-level citations."""

    answer_id: NonEmptyText
    question: NonEmptyText
    direct_answer: NonEmptyText
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    refused: bool = False
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def validate_citation_links(self) -> Answer:
        """Ensure every claim citation resolves within the answer."""
        citation_ids = {citation.citation_id for citation in self.citations}
        if len(citation_ids) != len(self.citations):
            raise ValueError("citation IDs must be unique within an answer")
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim IDs must be unique within an answer")
        missing = {
            citation_id
            for claim in self.claims
            for citation_id in claim.citation_ids
            if citation_id not in citation_ids
        }
        if missing:
            raise ValueError(f"claims reference unknown citations: {sorted(missing)}")
        if self.refused and not self.refusal_reason:
            raise ValueError("refused answers require refusal_reason")
        return self
