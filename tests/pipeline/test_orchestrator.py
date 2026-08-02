from viettheory.pipeline.evidence_gate import GateThresholds
from viettheory.pipeline.generator import GeneratorAdapter
from viettheory.pipeline.orchestrator import RagPipeline
from viettheory.schema import Answer, Chunk, Citation, Claim, RetrievedEvidence, SourceSpan


def _evidence(score: float = 0.9) -> RetrievedEvidence:
    text = "Vật chất là một phạm trù triết học."
    span = SourceSpan(page_id="p1", pdf_page=1, bbox=(0.0, 0.0, 1.0, 1.0), text=text)
    chunk = Chunk(
        chunk_id="c1",
        document_id="doc",
        subject_code="MLN111",
        text=text,
        token_count=7,
        source_spans=(span,),
    )
    return RetrievedEvidence(
        evidence_id="S1", chunk=chunk, score=score, rank=1, retrieval_method="rrf"
    )


class StubRetriever:
    def __init__(self, results: tuple[RetrievedEvidence, ...]) -> None:
        self.results = results
        self.calls = 0

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        self.calls += 1
        return self.results[:top_k]


class StubGenerator(GeneratorAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, evidence: tuple[RetrievedEvidence, ...]) -> Answer:
        self.calls += 1
        citation = Citation(
            citation_id="CIT1",
            evidence_id=evidence[0].evidence_id,
            source_span=evidence[0].chunk.source_spans[0],
        )
        return Answer(
            answer_id="A1",
            question=question,
            direct_answer="Một câu trả lời.",
            claims=(Claim(claim_id="C1", text="Nhận định", citation_ids=("CIT1",)),),
            citations=(citation,),
        )


def _pipeline(retriever: StubRetriever, generator: StubGenerator) -> RagPipeline:
    return RagPipeline(
        retriever,
        generator,
        GateThresholds(sufficient_score=0.7, related_score=0.3),
    )


def test_obvious_ood_refuses_without_retrieval_or_generation() -> None:
    retriever = StubRetriever((_evidence(),))
    generator = StubGenerator()
    answer = _pipeline(retriever, generator).ask("Thời tiết hôm nay thế nào?")
    assert answer.refused
    assert retriever.calls == 0
    assert generator.calls == 0


def test_grounded_answer_passes_end_to_end() -> None:
    retriever = StubRetriever((_evidence(),))
    generator = StubGenerator()
    answer = _pipeline(retriever, generator).ask("Vật chất là gì?")
    assert not answer.refused
    assert retriever.calls == 1
    assert generator.calls == 1


def test_related_evidence_retries_once_then_refuses() -> None:
    retriever = StubRetriever((_evidence(0.5),))
    generator = StubGenerator()
    answer = _pipeline(retriever, generator).ask("Vật chất là gì?")
    assert answer.refused
    assert retriever.calls == 2
    assert generator.calls == 0
