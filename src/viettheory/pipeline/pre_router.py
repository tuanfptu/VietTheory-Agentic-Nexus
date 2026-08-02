"""Transparent rule-based routing before retrieval."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class QuestionType(StrEnum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    MCQ = "mcq"
    ESSAY = "essay"
    EXACT_QUOTE = "exact_quote"
    GENERAL = "general"


class RouteDecision(BaseModel):
    """Deterministic routing metadata passed to retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_codes: frozenset[str]
    question_type: QuestionType
    cross_course: bool
    obvious_out_of_domain: bool
    matched_terms: tuple[str, ...]


_SUBJECT_TERMS: dict[str, tuple[str, ...]] = {
    "MLN111": ("triết học", "duy vật", "biện chứng", "mác-lênin"),
    "MLN122": ("kinh tế chính trị", "giá trị thặng dư", "hàng hóa", "tư bản"),
    "MLN131": ("chủ nghĩa xã hội khoa học", "cnxh", "giai cấp công nhân"),
    "HCM202": ("hồ chí minh", "tư tưởng hồ chí minh", "bác hồ"),
    "VNR202": ("lịch sử đảng", "đảng cộng sản việt nam", "đcsvn"),
}
_OUT_OF_DOMAIN = (
    "thời tiết",
    "tỷ giá",
    "giá bitcoin",
    "bóng đá",
    "nấu ăn",
    "lập trình",
)


def route_question(question: str) -> RouteDecision:
    """Classify obvious routing signals without making semantic claims."""
    normalized = " ".join(question.casefold().split())
    if not normalized:
        raise ValueError("question must not be blank")
    matches: list[str] = []
    subjects: set[str] = set()
    for subject, terms in _SUBJECT_TERMS.items():
        for term in terms:
            if term in normalized:
                subjects.add(subject)
                matches.append(term)
    out_terms = [term for term in _OUT_OF_DOMAIN if term in normalized]
    matches.extend(out_terms)
    if re.search(r"\b[abcd]\s*[.)]", normalized) or "đáp án" in normalized:
        question_type = QuestionType.MCQ
    elif any(term in normalized for term in ("so sánh", "khác nhau", "giống nhau")):
        question_type = QuestionType.COMPARISON
    elif any(term in normalized for term in ("trích nguyên văn", "nguyên văn", "trích dẫn")):
        question_type = QuestionType.EXACT_QUOTE
    elif any(term in normalized for term in ("là gì", "khái niệm", "định nghĩa")):
        question_type = QuestionType.DEFINITION
    elif any(term in normalized for term in ("phân tích", "trình bày", "chứng minh")):
        question_type = QuestionType.ESSAY
    else:
        question_type = QuestionType.GENERAL
    return RouteDecision(
        subject_codes=frozenset(subjects),
        question_type=question_type,
        cross_course=len(subjects) > 1 or question_type is QuestionType.COMPARISON,
        obvious_out_of_domain=bool(out_terms) and not subjects,
        matched_terms=tuple(dict.fromkeys(matches)),
    )
