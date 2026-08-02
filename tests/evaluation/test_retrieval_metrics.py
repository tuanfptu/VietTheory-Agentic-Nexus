"""Tests for child-ID and multi-group retrieval metrics."""

import pytest

from viettheory.benchmark import (
    Answerability,
    BenchmarkQuestion,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
    ReviewStatus,
)
from viettheory.evaluation import evaluate_retrieval
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _span(page: int) -> SourceSpan:
    return SourceSpan(
        page_id=f"page_{page}",
        pdf_page=page,
        bbox=(1.0, 1.0, 2.0, 2.0),
        text=f"evidence {page}",
    )


def _group(number: int, child_id: str) -> GoldEvidenceGroup:
    return GoldEvidenceGroup(
        group_id=f"g{number}",
        subject_code="TEST",
        role="answer",
        primary_child_ids=(child_id,),
        gold_parent_ids=(f"parent_{number}",),
        gold_pdf_pages=(number,),
    )


def _question(status: ReviewStatus) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        benchmark_version="1.0.0",
        id="test_0001",
        subject_code="TEST",
        question="Gold question?",
        answerability=Answerability.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        question_types=(QuestionType.SYNTHESIS,),
        reasoning_scope=ReasoningScope.MULTI_CHUNK,
        chapter_scope=ChapterScope.SINGLE_CHAPTER,
        difficulty=Difficulty.MEDIUM,
        split=BenchmarkSplit.DEVELOPMENT,
        gold_evidence_groups=(_group(1, "child_3"), _group(2, "child_5")),
        gold_answer="Gold answer",
        generation=GenerationMetadata(method="human"),
        artifact_manifest_id="test_manifest",
        review_status=status,
    )


def _retrieve(query: str, top_k: int) -> tuple[RetrievedEvidence, ...]:
    del query, top_k
    return tuple(
        RetrievedEvidence(
            evidence_id=f"evidence_{rank}",
            chunk=Chunk(
                chunk_id=f"child_{rank}",
                document_id="doc_1",
                subject_code="TEST",
                text="chunk text",
                token_count=2,
                source_spans=(_span(rank),),
                chunk_kind="child",
                parent_chunk_id=f"parent_{rank}",
            ),
            score=1.0 / rank,
            rank=rank,
            retrieval_method="dense",
        )
        for rank in range(1, 7)
    )


def test_evaluate_retrieval_computes_group_metrics() -> None:
    metrics = evaluate_retrieval((_question(ReviewStatus.VERIFIED),), _retrieve)

    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr == pytest.approx(1 / 3)
    assert metrics.group_recall_at_3 == 0.5
    assert metrics.group_recall_at_5 == 1.0
    assert metrics.partial_evidence_coverage_at_5 == 1.0
    assert metrics.full_evidence_success_at_5 == 1.0
    assert metrics.full_evidence_success_at_10 == 1.0
    assert 0.0 < metrics.ndcg_at_5 <= 1.0


def test_evaluate_retrieval_refuses_draft_gold_set() -> None:
    with pytest.raises(ValueError, match="human-verified"):
        evaluate_retrieval((_question(ReviewStatus.DRAFT),), _retrieve)
