import numpy as np
from numpy.typing import NDArray

from viettheory.retrieval.reranker import RerankedRetriever, Reranker
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _evidence(chunk_id: str, text: str, rank: int) -> RetrievedEvidence:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        subject_code="MLN111",
        text=text,
        token_count=len(text.split()),
        source_spans=(
            SourceSpan(
                page_id=f"p-{chunk_id}",
                pdf_page=rank,
                bbox=(0.0, 0.0, 1.0, 1.0),
                text=text,
            ),
        ),
    )
    return RetrievedEvidence(
        evidence_id=f"S{rank}",
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retrieval_method="rrf",
    )


class StubScorer:
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> NDArray[np.float32]:
        assert batch_size == 4
        return np.asarray([0.1, 0.9], dtype=np.float32)


def test_reranker_reorders_and_preserves_source_spans() -> None:
    first = _evidence("c1", "less relevant", 1)
    second = _evidence("c2", "more relevant", 2)
    reranked = Reranker(StubScorer()).rerank("query", (first, second), top_k=2)
    assert [item.chunk.chunk_id for item in reranked] == ["c2", "c1"]
    assert reranked[0].chunk.source_spans == second.chunk.source_spans
    assert reranked[0].retrieval_method == "qwen_reranker"


def test_reranker_handles_no_candidates_without_model_call() -> None:
    assert Reranker(StubScorer()).rerank("query", ()) == ()


class StubCandidateRetriever:
    def __init__(self, candidates: tuple[RetrievedEvidence, ...]) -> None:
        self.candidates = candidates

    def search(
        self,
        query: str,
        *,
        top_k: int,
        subject_codes: frozenset[str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        return self.candidates[:top_k]


def test_reranked_retriever_is_pipeline_compatible() -> None:
    candidates = (
        _evidence("c1", "less relevant", 1),
        _evidence("c2", "more relevant", 2),
    )
    retriever = RerankedRetriever(StubCandidateRetriever(candidates), Reranker(StubScorer()))
    result = retriever.search("query", top_k=1, subject_codes=frozenset({"MLN111"}))
    assert len(result) == 1
    assert result[0].chunk.chunk_id == "c2"
