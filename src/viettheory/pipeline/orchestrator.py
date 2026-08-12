"""End-to-end orchestration with bounded corrective retrieval."""

from __future__ import annotations

import re
from typing import Protocol

from viettheory.ids import stable_id
from viettheory.pipeline.citation_verifier import verify_citations
from viettheory.pipeline.evidence_gate import GateAction, GateThresholds, decide_evidence
from viettheory.pipeline.generator import GenerationError, GeneratorAdapter
from viettheory.pipeline.pre_router import route_question
from viettheory.schema import Answer, RetrievedEvidence


class RoutedRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]: ...


class QueryRewriter(Protocol):
    def rewrite(self, question: str) -> str: ...


class IdentityRewriter:
    """Safe baseline, replaceable after comparative evaluation."""

    def rewrite(self, question: str) -> str:
        return " ".join(question.split())


class RagPipeline:
    """Route, retrieve, gate, retry at most once, generate, and verify."""

    def __init__(
        self,
        retriever: RoutedRetriever,
        generator: GeneratorAdapter,
        thresholds: GateThresholds,
        *,
        rewriter: QueryRewriter | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._retriever = retriever
        self._generator = generator
        self._thresholds = thresholds
        self._rewriter = rewriter or IdentityRewriter()
        self._top_k = top_k

    def ask(self, question: str, context: tuple[str, ...] = ()) -> Answer:
        contextual_question = _contextual_question(question, context)
        route = route_question(contextual_question)
        if route.obvious_out_of_scope:
            return _refusal(question, "Câu hỏi nằm ngoài phạm vi giáo trình MLN111.")
        subjects = frozenset({"MLN111"})
        evidence = self._retriever.search(
            contextual_question, top_k=self._top_k, subject_codes=subjects
        )
        decision = decide_evidence(evidence, self._thresholds)
        if decision.action is GateAction.REWRITE:
            rewritten = self._rewriter.rewrite(contextual_question)
            evidence = self._retriever.search(rewritten, top_k=self._top_k, subject_codes=subjects)
            decision = decide_evidence(evidence, self._thresholds, already_retried=True)
        if decision.action is GateAction.REFUSE_OUT_OF_DOMAIN:
            return _refusal(question, "Không tìm thấy bằng chứng phù hợp trong giáo trình.")
        if decision.action is GateAction.REFUSE_INSUFFICIENT:
            return _refusal(question, "Chưa tìm đủ căn cứ trong giáo trình để trả lời.")
        answer = self._generator.generate(contextual_question, evidence)
        if context:
            answer = answer.model_copy(update={"question": question})
        verification = verify_citations(answer, evidence)
        if not verification.valid:
            raise GenerationError("generated answer failed deterministic citation verification")
        return answer


def _refusal(question: str, reason: str) -> Answer:
    return Answer(
        answer_id=stable_id("answer", question, reason),
        question=question,
        direct_answer=reason,
        claims=(),
        citations=(),
        refused=True,
        refusal_reason=reason,
    )


def _contextual_question(question: str, context: tuple[str, ...]) -> str:
    if not context or not _needs_conversation_context(question):
        return question
    # Only the immediately preceding user/assistant turn is relevant to an
    # elliptical follow-up. Older turns can contain different MLN111 concepts
    # and badly contaminate retrieval (for example, "chúng" resolving to a
    # topic discussed two questions earlier).
    history = "\n".join(context[-2:])
    return (
        f"Ngữ cảnh hội thoại trước:\n{history}\nCâu hỏi hiện tại (chỉ trả lời câu này): {question}"
    )


def _needs_conversation_context(question: str) -> bool:
    """Use history only for genuinely elliptical follow-ups, not every new topic."""
    normalized = " ".join(question.casefold().split())
    references = (
        "chúng",
        "chúng nó",
        "hai khái niệm",
        "hai yếu tố",
        "hai cái",
        "điều đó",
        "ý đó",
        "cái đó",
        "định nghĩa đó",
        "khái niệm đó",
        "quan điểm đó",
        "vấn đề đó",
        "nó ",
        "họ ",
        "ông ấy",
        "tại sao vậy",
        "vì sao vậy",
        "giải thích thêm",
        "nói rõ hơn",
        "so sánh thêm",
    )
    return any(term in normalized for term in references) or bool(
        re.fullmatch(r"(?:tại sao|vì sao|như thế nào|cụ thể là gì)[?.!]*", normalized)
    )
