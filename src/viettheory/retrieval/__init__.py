"""Dense, lexical, and hybrid retrieval primitives."""

from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.retrieval.hardware import RetrievalRuntimeProfile, detect_runtime
from viettheory.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from viettheory.retrieval.indexer import Embedder, build_faiss_index
from viettheory.retrieval.parent import ParentChunkStore, ParentExpandedRetriever, expand_to_parents
from viettheory.retrieval.planned import PlannedRerankedRetriever, comparison_query_variants
from viettheory.retrieval.reranker import (
    QwenCrossEncoderScorer,
    RerankedRetriever,
    Reranker,
)

__all__ = [
    "BM25Retriever",
    "Embedder",
    "HybridRetriever",
    "ParentChunkStore",
    "ParentExpandedRetriever",
    "PlannedRerankedRetriever",
    "QwenCrossEncoderScorer",
    "RerankedRetriever",
    "Reranker",
    "RetrievalRuntimeProfile",
    "build_faiss_index",
    "comparison_query_variants",
    "detect_runtime",
    "expand_to_parents",
    "reciprocal_rank_fusion",
]
