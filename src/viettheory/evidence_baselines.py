"""Frozen shortcut baselines for evidence-sufficiency diagnostics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from viettheory.evidence_sufficiency import EvidenceSufficiencyCase, SufficiencyLabel

VIETNAMESE_STOPWORDS = frozenset(
    {
        "ai",
        "bao",
        "các",
        "có",
        "của",
        "đã",
        "được",
        "gì",
        "khi",
        "là",
        "nào",
        "những",
        "như",
        "theo",
        "thế",
        "trong",
        "và",
        "về",
        "vì",
    }
)


def tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(
        token
        for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
        if len(token) > 1 and token not in VIETNAMESE_STOPWORDS
    )


def context_text(case: EvidenceSufficiencyCase) -> str:
    return "\n".join(context.text for context in case.provided_contexts)


@dataclass(frozen=True)
class LexicalCoverageBaseline:
    sufficient_threshold: float = 0.55
    partial_threshold: float = 0.25

    def score(self, case: EvidenceSufficiencyCase) -> float:
        question = frozenset(tokens(case.question))
        context = frozenset(tokens(context_text(case)))
        return len(question & context) / len(question) if question else 0.0

    def predict(self, case: EvidenceSufficiencyCase) -> SufficiencyLabel:
        if not case.provided_contexts:
            return SufficiencyLabel.MISSING
        score = self.score(case)
        if score >= self.sufficient_threshold:
            return SufficiencyLabel.SUFFICIENT
        if score >= self.partial_threshold:
            return SufficiencyLabel.PARTIAL
        return SufficiencyLabel.WRONG_ASPECT


class TfidfSimilarityBaseline:
    """Small dependency-free TF-IDF cosine baseline with frozen thresholds."""

    def __init__(
        self,
        cases: Iterable[EvidenceSufficiencyCase],
        *,
        sufficient_threshold: float = 0.30,
        partial_threshold: float = 0.12,
    ) -> None:
        documents = [frozenset(tokens(context_text(case))) for case in cases]
        self._document_count = len(documents)
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(document)
        self._idf = {
            token: math.log((1 + self._document_count) / (1 + frequency)) + 1.0
            for token, frequency in frequencies.items()
        }
        self.sufficient_threshold = sufficient_threshold
        self.partial_threshold = partial_threshold

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokens(text))
        total = sum(counts.values())
        if total == 0:
            return {}
        return {token: count / total * self._idf.get(token, 1.0) for token, count in counts.items()}

    def score(self, case: EvidenceSufficiencyCase) -> float:
        left = self._vector(case.question)
        right = self._vector(context_text(case))
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def predict(self, case: EvidenceSufficiencyCase) -> SufficiencyLabel:
        if not case.provided_contexts:
            return SufficiencyLabel.MISSING
        score = self.score(case)
        if score >= self.sufficient_threshold:
            return SufficiencyLabel.SUFFICIENT
        if score >= self.partial_threshold:
            return SufficiencyLabel.PARTIAL
        return SufficiencyLabel.WRONG_ASPECT
