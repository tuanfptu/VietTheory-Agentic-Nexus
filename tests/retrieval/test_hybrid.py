from viettheory.retrieval.bm25 import BM25Retriever, tokenize
from viettheory.retrieval.hybrid import reciprocal_rank_fusion
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _chunk(chunk_id: str, subject: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        subject_code=subject,
        text=text,
        token_count=len(text.split()),
        source_spans=(
            SourceSpan(page_id=f"p-{chunk_id}", pdf_page=0, bbox=(0.0, 0.0, 1.0, 1.0), text=text),
        ),
    )


def _result(chunk: Chunk, rank: int, method: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=f"{method}-{rank}",
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retrieval_method=method,
    )


def test_tokenizer_preserves_vietnamese_diacritics() -> None:
    assert tokenize("Giá trị thặng dư!") == ("giá", "trị", "thặng", "dư")


def test_bm25_ranks_exact_vietnamese_terms_and_filters_subject() -> None:
    relevant = _chunk("c1", "MLN111", "Định nghĩa vật chất của V.I. Lênin")
    other = _chunk("c2", "MLN111", "Phép biện chứng duy vật")
    retriever = BM25Retriever((other, relevant))
    result = retriever.search("định nghĩa vật chất", top_k=5, subject_codes=frozenset({"MLN111"}))
    assert result[0].chunk.chunk_id == "c1"
    assert {item.chunk.subject_code for item in result} == {"MLN111"}


def test_rrf_deduplicates_and_rewards_consensus() -> None:
    consensus = _chunk("consensus", "MLN111", "A")
    lexical_only = _chunk("lexical", "MLN111", "B")
    dense_only = _chunk("dense", "MLN111", "C")
    fused = reciprocal_rank_fusion(
        (
            (_result(lexical_only, 1, "bm25"), _result(consensus, 2, "bm25")),
            (_result(dense_only, 1, "dense"), _result(consensus, 2, "dense")),
        ),
        top_k=3,
    )
    assert fused[0].chunk.chunk_id == "consensus"
    assert len({item.chunk.chunk_id for item in fused}) == 3
