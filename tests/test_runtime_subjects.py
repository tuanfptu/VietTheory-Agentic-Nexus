from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viettheory.benchmark import BenchmarkQuestion
from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hybrid import HybridRetriever
from viettheory.retrieval.parent import ParentChunkStore, ParentExpandedRetriever
from viettheory.retrieval.planned import PlannedRerankedRetriever
from viettheory.retrieval.reranker import Reranker
from viettheory.retrieval.retriever import DenseRetriever
from viettheory.runtime import EMBEDDING_MODEL_ID, EMBEDDING_REVISION, build_retrieval
from viettheory.schema import Chunk
from viettheory.subjects import SUBJECTS


class SmokeEmbedder:
    model_id = EMBEDDING_MODEL_ID
    model_revision = EMBEDDING_REVISION

    def encode_queries(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        assert batch_size == 1
        return np.ones((len(texts), 1024), dtype=np.float32)


class SmokeScorer:
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> NDArray[np.float32]:
        assert batch_size > 0
        return np.asarray([len(text) / 10_000 for _, text in pairs], dtype=np.float32)


def _query_for_subject(root: Path, subject_code: str) -> str:
    chunk = UnifiedCorpusCatalog(root).load_children(SearchMode.WITHIN_SUBJECT, subject_code)[0]
    return " ".join(chunk.text.split()[:24])


def test_same_b0_path_retrieves_parent_evidence_for_all_subjects() -> None:
    root = Path(".").resolve()
    for subject in SUBJECTS:
        retrieval = build_retrieval(
            root,
            search_mode=SearchMode.WITHIN_SUBJECT,
            subject_code=subject.code,
            embedder=SmokeEmbedder(),
            scorer=SmokeScorer(),
        )
        query = _query_for_subject(root, subject.code)
        children = retrieval.search_children(query, top_k=5)
        parents = retrieval.search(query, top_k=3)

        assert children
        assert parents
        assert all(item.chunk.subject_code == subject.code for item in children)
        assert all(item.chunk.subject_code == subject.code for item in parents)
        assert all(item.retrieval_method == "qwen_reranker" for item in children)
        assert all(item.retrieval_method == "parent_expansion" for item in parents)


def test_global_b0_path_preserves_multi_subject_provenance() -> None:
    root = Path(".").resolve()
    retrieval = build_retrieval(
        root,
        search_mode=SearchMode.GLOBAL,
        subject_code=None,
        embedder=SmokeEmbedder(),
        scorer=SmokeScorer(),
    )
    results = retrieval.search_children("Đảng cách mạng xã hội chủ nghĩa", top_k=20)

    assert results
    assert {item.chunk.subject_code for item in results}.issubset(
        {subject.code for subject in SUBJECTS}
    )


def test_mln111_shared_assembly_matches_frozen_legacy_b0_path() -> None:
    root = Path(".").resolve()
    embedder = SmokeEmbedder()
    scorer = SmokeScorer()
    shared = build_retrieval(
        root,
        search_mode=SearchMode.WITHIN_SUBJECT,
        subject_code="MLN111",
        embedder=embedder,
        scorer=scorer,
    )

    structured = root / "data/processed/MLN111/structured_v1"
    children_path = structured / "children.jsonl"
    chunks = tuple(
        chunk
        for line in children_path.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line)).chapter is not None
    )
    legacy = ParentExpandedRetriever(
        PlannedRerankedRetriever(
            HybridRetriever(
                BM25Retriever(chunks),
                DenseRetriever(structured / "dense_index", children_path, embedder),
                candidate_k=30,
            ),
            Reranker(scorer, batch_size=4),
            candidate_k=12,
        ),
        ParentChunkStore.from_jsonl(structured / "parents.jsonl"),
    )
    questions = tuple(
        BenchmarkQuestion.model_validate_json(line).question
        for line in (root / "benchmark/v1.0/mln111_development.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )

    for question in questions:
        shared_ids = tuple(
            item.chunk.chunk_id
            for item in shared.search_children(
                question, top_k=10, subject_codes=frozenset({"MLN111"})
            )
        )
        legacy_ids = tuple(
            item.chunk.chunk_id
            for item in legacy.search_children(
                question, top_k=10, subject_codes=frozenset({"MLN111"})
            )
        )
        assert shared_ids == legacy_ids
