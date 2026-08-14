"""Transparent textbook scope and question-type routing."""

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

_SUBJECT_TERMS: dict[str, tuple[str, ...]] = {
    "MLN111": (
        "triết học",
        "vật chất",
        "ý thức",
        "phép biện chứng",
        "lý luận nhận thức",
        "vai trò của thực tiễn",
    ),
    "MLN122": (
        "kinh tế chính trị",
        "giá trị thặng dư",
        "hàng hóa",
        "kinh tế thị trường",
        "tư bản bất biến",
        "tư bản khả biến",
        "quan hệ lợi ích",
    ),
    "MLN131": (
        "chủ nghĩa xã hội khoa học",
        "sứ mệnh lịch sử của giai cấp công nhân",
        "cơ cấu xã hội - giai cấp",
        "dân tộc tộc người",
        "vấn đề gia đình",
    ),
    "HCM202": (
        "tư tưởng hồ chí minh",
        "quan điểm của hồ chí minh",
        "hồ chí minh quan niệm",
        "đại đoàn kết dân tộc",
        "đạo đức cách mạng",
    ),
    "VNR202": (
        "lịch sử đảng",
        "đảng cộng sản việt nam",
        "đại hội đại biểu toàn quốc",
        "đường lối đổi mới",
        "cách mạng tháng tám",
        "ba giai đoạn",
    ),
}


def route_subject(question: str, allowed_subjects: frozenset[str]) -> str | None:
    """Route only on strong course-specific phrases; ties stay global."""
    normalized = " ".join(question.casefold().split())
    scores = {
        subject: sum(term in normalized for term in terms)
        for subject, terms in _SUBJECT_TERMS.items()
        if subject in allowed_subjects
    }
    if not scores or max(scores.values()) == 0:
        return None
    best_score = max(scores.values())
    winners = [subject for subject, score in scores.items() if score == best_score]
    return winners[0] if len(winners) == 1 else None


def route_question(question: str) -> RouteDecision:
    """Reject only explicit non-textbook topics; retrieval handles academic scope."""
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
