"""Versioned subject registry shared by data, retrieval, and evaluation layers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ExtractionMode(StrEnum):
    NATIVE = "native"
    OCR = "ocr"


class SubjectSpec(BaseModel):
    """Stable metadata for one supported textbook corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: str
    name_vi: str
    name_en: str
    extraction_mode: ExtractionMode
    expected_pages: int
    expected_chapters: int


SUBJECTS: tuple[SubjectSpec, ...] = (
    SubjectSpec(
        code="MLN111",
        name_vi="Triết học Mác - Lênin",
        name_en="Marxist-Leninist Philosophy",
        extraction_mode=ExtractionMode.NATIVE,
        expected_pages=285,
        expected_chapters=3,
    ),
    SubjectSpec(
        code="MLN122",
        name_vi="Kinh tế Chính trị Mác - Lênin",
        name_en="Marxist-Leninist Political Economy",
        extraction_mode=ExtractionMode.NATIVE,
        expected_pages=262,
        expected_chapters=6,
    ),
    SubjectSpec(
        code="MLN131",
        name_vi="Chủ nghĩa Xã hội Khoa học",
        name_en="Scientific Socialism",
        extraction_mode=ExtractionMode.OCR,
        expected_pages=273,
        expected_chapters=7,
    ),
    SubjectSpec(
        code="HCM202",
        name_vi="Tư tưởng Hồ Chí Minh",
        name_en="Ho Chi Minh Ideology",
        extraction_mode=ExtractionMode.OCR,
        expected_pages=271,
        expected_chapters=6,
    ),
    SubjectSpec(
        code="VNR202",
        name_vi="Lịch sử Đảng Cộng sản Việt Nam",
        name_en="History of the Communist Party of Vietnam",
        extraction_mode=ExtractionMode.OCR,
        expected_pages=230,
        expected_chapters=3,
    ),
)

SUBJECT_BY_CODE = {subject.code: subject for subject in SUBJECTS}


def get_subject(code: str) -> SubjectSpec:
    """Return one registered subject or fail before touching corpus artifacts."""
    try:
        return SUBJECT_BY_CODE[code.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported subject code: {code}") from exc
