from viettheory.pipeline.citation_verifier import verify_citations
from viettheory.schema import Answer, Chunk, Citation, Claim, RetrievedEvidence, SourceSpan


def _evidence() -> RetrievedEvidence:
    span = SourceSpan(
        page_id="page-1", pdf_page=1, printed_page="1", bbox=(1.0, 2.0, 3.0, 4.0), text="Nguồn"
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        subject_code="MLN111",
        text="Nguồn",
        token_count=1,
        source_spans=(span,),
    )
    return RetrievedEvidence(
        evidence_id="S1", chunk=chunk, score=0.9, rank=1, retrieval_method="dense"
    )


def test_accepts_claim_grounded_in_retrieved_span() -> None:
    evidence = _evidence()
    answer = Answer(
        answer_id="answer-1",
        question="Câu hỏi?",
        direct_answer="Trả lời.",
        claims=(Claim(claim_id="C1", text="Một nhận định", citation_ids=("CIT1",)),),
        citations=(
            Citation(
                citation_id="CIT1",
                evidence_id=evidence.evidence_id,
                source_span=evidence.chunk.source_spans[0],
            ),
        ),
    )
    result = verify_citations(answer, (evidence,))
    assert result.valid
    assert result.unsupported_claim_ids == ()


def test_rejects_claim_without_citation() -> None:
    evidence = _evidence()
    answer = Answer(
        answer_id="answer-1",
        question="Câu hỏi?",
        direct_answer="Trả lời.",
        claims=(Claim(claim_id="C1", text="Một nhận định", citation_ids=()),),
        citations=(),
    )
    result = verify_citations(answer, (evidence,))
    assert not result.valid
    assert result.unsupported_claim_ids == ("C1",)
