"""Transparent MLN111 scope and question-type routing."""

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
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_type: QuestionType
    obvious_out_of_scope: bool
    matched_terms: tuple[str, ...]


_OUT_OF_SCOPE = (
    "thời tiết",
    "tỷ giá",
    "giá bitcoin",
    "bóng đá",
    "nấu ăn",
    "lập trình",
    "đạo hàm",
    "adn",
)


def route_question(question: str) -> RouteDecision:
    """Reject only explicit non-MLN111 topics; retrieval handles academic scope."""
    normalized = " ".join(question.casefold().split())
    if not normalized:
        raise ValueError("question must not be blank")
    out_terms = tuple(term for term in _OUT_OF_SCOPE if term in normalized)
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
        question_type=question_type,
        obvious_out_of_scope=bool(out_terms),
        matched_terms=out_terms,
    )
