from viettheory.pipeline.generator import (
    GeneratedAnswer,
    GeneratedClaim,
    _canonicalize_citations,
    _deduplicate_citations,
    _materialize_answer,
    _normalize_schema_versions,
)
from viettheory.schema import Answer, Chunk, Citation, Claim, RetrievedEvidence, SourceSpan


def test_deduplicate_citations_rewrites_claim_links() -> None:
    span = SourceSpan(
        page_id="p1",
        pdf_page=88,
        printed_page="89",
        bbox=(0.0, 0.0, 10.0, 10.0),
        text="Nguồn gốc của ý thức.",
    )
    answer = Answer(
        answer_id="a1",
        question="Ý thức có nguồn gốc như thế nào?",
        direct_answer="Ý thức có nguồn gốc tự nhiên và xã hội.",
        claims=(
            Claim(claim_id="c1", text="Nguồn gốc tự nhiên.", citation_ids=("cit1",)),
            Claim(claim_id="c2", text="Nguồn gốc xã hội.", citation_ids=("cit2", "cit3")),
        ),
        citations=(
            Citation(citation_id="cit1", evidence_id="e1", source_span=span),
            Citation(citation_id="cit2", evidence_id="e1", source_span=span),
            Citation(citation_id="cit3", evidence_id="e1", source_span=span),
        ),
    )

    result = _deduplicate_citations(answer)

    assert tuple(item.citation_id for item in result.citations) == ("cit1",)
    assert result.claims[0].citation_ids == ("cit1",)
    assert result.claims[1].citation_ids == ("cit1",)


def test_canonical_citation_includes_full_retrieved_passage() -> None:
    span = SourceSpan(
        page_id="p1",
        pdf_page=99,
        printed_page="100",
        bbox=(0.0, 0.0, 10.0, 10.0),
        text="Vật chất quyết định ý thức.",
    )
    chunk = Chunk(
        chunk_id="parent_1",
        document_id="doc_1",
        subject_code="MLN111",
        text="Đây là toàn bộ đoạn dài. Vật chất quyết định ý thức và ý thức tác động trở lại.",
        token_count=18,
        source_spans=(span,),
    )
    evidence = RetrievedEvidence(
        evidence_id="e1",
        chunk=chunk,
        score=0.9,
        rank=1,
        retrieval_method="parent_expansion",
    )
    citation = Citation(citation_id="cit1", evidence_id="e1", source_span=span)
    answer = Answer(
        answer_id="a1",
        question="Mối quan hệ là gì?",
        direct_answer="Vật chất quyết định ý thức.",
        claims=(Claim(claim_id="c1", text="Vật chất quyết định ý thức.", citation_ids=("cit1",)),),
        citations=(citation,),
    )

    result = _canonicalize_citations(answer, (evidence,))

    assert result.citations[0].context_text == chunk.text


def test_provider_schema_version_aliases_are_normalized_recursively() -> None:
    raw = {
        "schema_version": "v1",
        "citations": [{"schema_version": "v1"}],
        "claims": [{"schema_version": "1"}],
    }
    _normalize_schema_versions(raw)
    assert raw["schema_version"] == "1.0"
    assert raw["citations"][0]["schema_version"] == "1.0"
    assert raw["claims"][0]["schema_version"] == "1.0"


def test_minimal_provider_answer_materializes_canonical_provenance() -> None:
    span = SourceSpan(
        page_id="VNR202_p1",
        pdf_page=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        text="Giai đoạn 1930-1945.",
    )
    chunk = Chunk(
        chunk_id="parent_vnr",
        document_id="doc",
        subject_code="VNR202",
        text="Giai đoạn 1930-1945 là giai đoạn đấu tranh giành chính quyền.",
        token_count=10,
        source_spans=(span,),
    )
    evidence = RetrievedEvidence(
        evidence_id="ev1",
        chunk=chunk,
        score=1.0,
        rank=1,
        retrieval_method="parent_expansion",
    )
    generated = GeneratedAnswer(
        direct_answer="Giai đoạn đầu là 1930-1945.",
        claims=(GeneratedClaim(text="Giai đoạn đầu là 1930-1945.", evidence_ids=("ev1",)),),
    )
    answer = _materialize_answer("Tóm tắt các giai đoạn.", generated, (evidence,))
    assert answer.citations[0].source_span == span
    assert answer.citations[0].context_text == chunk.text
    assert answer.claims[0].citation_ids == (answer.citations[0].citation_id,)
