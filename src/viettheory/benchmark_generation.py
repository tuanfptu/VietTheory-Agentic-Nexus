"""Contracts and guards for LLM-assisted benchmark candidate generation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viettheory.benchmark import (
    Answerability,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    QuestionType,
    ReasoningScope,
)
from viettheory.schema import Chunk


def load_gemini_key(dotenv_path: Path) -> str | None:
    """Load the Gemini credential without exposing or exporting other secrets."""
    import os

    for name in ("GEMINI_API_KEY", "LLM_API_KEY"):
        existing = os.getenv(name)
        if existing:
            return existing
    if not dotenv_path.is_file():
        return None
    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in {"GEMINI_API_KEY", "LLM_API_KEY"}:
            continue
        key = value.strip()
        if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
            key = key[1:-1]
        if key:
            values[name] = key
    return values.get("GEMINI_API_KEY") or values.get("LLM_API_KEY")


def is_substantive_chunk(chunk: Chunk) -> bool:
    """Exclude front matter and chapter learning objectives."""
    if not chunk.chapter:
        return False
    labels = " ".join(label for label in (chunk.section, chunk.subsection) if label).casefold()
    objective_patterns = (
        r"\bmục tiêu\b",
        r"^\s*\d+\.\s*về\s+(kiến thức|kỹ năng|tư tưởng)",
    )
    return not any(re.search(pattern, labels) for pattern in objective_patterns)


class CandidateEvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    required: bool = True
    child_ids: tuple[str, ...] = Field(min_length=1)


class BenchmarkCandidate(BaseModel):
    """Provider response before corpus metadata and release fields are attached."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=8)
    answerability: Answerability
    unanswerable_reason: str | None = None
    expected_behavior: ExpectedBehavior
    question_types: tuple[QuestionType, ...] = Field(min_length=1)
    reasoning_scope: ReasoningScope
    chapter_scope: ChapterScope
    difficulty: Difficulty
    gold_answer: str | None = None
    required_concepts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    evidence_groups: tuple[CandidateEvidenceGroup, ...] = ()

    @model_validator(mode="after")
    def validate_candidate(self) -> BenchmarkCandidate:
        if self.answerability is Answerability.ANSWERABLE:
            if not self.gold_answer or not self.evidence_groups:
                raise ValueError("answerable candidates require an answer and evidence")
            if self.expected_behavior is not ExpectedBehavior.ANSWER:
                raise ValueError("answerable candidates must expect an answer")
        elif not self.unanswerable_reason:
            raise ValueError("unanswerable candidates require a reason")
        return self


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: tuple[BenchmarkCandidate, ...]


def normalized_question(text: str) -> str:
    """Normalize text for deterministic exact/near-exact duplicate screening."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", without_marks).strip()


def reject_unknown_evidence(
    candidates: Iterable[BenchmarkCandidate],
    allowed_child_ids: frozenset[str],
) -> tuple[BenchmarkCandidate, ...]:
    """Reject a whole provider response if it invents corpus IDs."""
    checked: list[BenchmarkCandidate] = []
    for candidate in candidates:
        used = {child_id for group in candidate.evidence_groups for child_id in group.child_ids}
        unknown = used.difference(allowed_child_ids)
        if unknown:
            raise ValueError(f"candidate cites unknown child IDs: {sorted(unknown)}")
        checked.append(candidate)
    return tuple(checked)


def deduplicate_candidates(
    candidates: Iterable[BenchmarkCandidate],
) -> tuple[BenchmarkCandidate, ...]:
    """Remove normalized exact duplicates while preserving provider order."""
    seen: set[str] = set()
    unique: list[BenchmarkCandidate] = []
    for candidate in candidates:
        key = normalized_question(candidate.question)
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)
