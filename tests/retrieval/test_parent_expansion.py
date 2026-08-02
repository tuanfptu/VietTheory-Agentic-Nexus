from viettheory.retrieval.parent import ParentChunkStore, expand_to_parents
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _span(text: str) -> SourceSpan:
    return SourceSpan(
        page_id="page-1",
        pdf_page=1,
        bbox=(0.0, 0.0, 1.0, 1.0),
        text=text,
    )


def _parent(parent_id: str) -> Chunk:
    return Chunk(
        chunk_id=parent_id,
        document_id="doc",
        subject_code="MLN111",
        text="parent context",
        token_count=2,
        source_spans=(_span("parent context"),),
        chunk_kind="parent",
    )


def _child(child_id: str, parent_id: str, rank: int) -> RetrievedEvidence:
    chunk = Chunk(
        chunk_id=child_id,
        document_id="doc",
        subject_code="MLN111",
        text="child context",
        token_count=2,
        source_spans=(_span("child context"),),
        chunk_kind="child",
        parent_chunk_id=parent_id,
    )
    return RetrievedEvidence(
        evidence_id=f"S{rank}",
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retrieval_method="qwen_reranker",
    )


def test_parent_expansion_deduplicates_siblings() -> None:
    first_parent = _parent("p1")
    second_parent = _parent("p2")
    store = ParentChunkStore((first_parent, second_parent))
    evidence = (
        _child("c1", "p1", 1),
        _child("c2", "p1", 2),
        _child("c3", "p2", 3),
    )
    expanded = expand_to_parents(evidence, store)
    assert [item.chunk.chunk_id for item in expanded] == ["p1", "p2"]
    assert expanded[0].score == 1.0
    assert expanded[0].retrieval_method == "parent_expansion"
