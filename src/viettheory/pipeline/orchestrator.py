"""End-to-end orchestration with bounded corrective retrieval."""

from __future__ import annotations

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

    def ask(self, question: str) -> Answer:
        route = route_question(question)
        if route.obvious_out_of_domain:
            return _refusal(question, "Câu hỏi nằm ngoài phạm vi năm giáo trình.")
        subjects = route.subject_codes or None
        evidence = self._retriever.search(question, top_k=self._top_k, subject_codes=subjects)
        decision = decide_evidence(evidence, self._thresholds)
        if decision.action is GateAction.REWRITE:
            rewritten = self._rewriter.rewrite(question)
            evidence = self._retriever.search(rewritten, top_k=self._top_k, subject_codes=subjects)
            decision = decide_evidence(evidence, self._thresholds, already_retried=True)
        if decision.action is GateAction.REFUSE_OUT_OF_DOMAIN:
            return _refusal(question, "Không tìm thấy bằng chứng phù hợp trong giáo trình.")
        if decision.action is GateAction.REFUSE_INSUFFICIENT:
            return _refusal(question, "Chưa tìm đủ căn cứ trong giáo trình để trả lời.")
        answer = self._generator.generate(question, evidence)
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
