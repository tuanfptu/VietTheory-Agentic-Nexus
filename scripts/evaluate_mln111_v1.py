"""Evaluate the frozen MLN111 retrieval configuration on one isolated split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from viettheory.benchmark import BenchmarkQuestion
from viettheory.evaluation import evaluate_retrieval
from viettheory.retrieval.parent import ParentExpandedRetriever
from viettheory.runtime import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-checked", action="store_true")
    args = parser.parse_args()
    questions = tuple(
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    pipeline = build_pipeline()
    retriever = cast(ParentExpandedRetriever, pipeline._retriever)
    per_query: list[dict[str, object]] = []
    question_by_text = {question.question: question for question in questions}

    def retrieve(question: str, top_k: int):  # type: ignore[no-untyped-def]
        results = retriever.search_children(
            question,
            top_k=top_k,
            subject_codes=frozenset({"MLN111"}),
        )
        item = question_by_text[question]
        gold_ids = frozenset(
            child_id
            for group in item.gold_evidence_groups
            if group.required
            for child_id in group.all_child_ids
        )
        retrieved_ids = [result.chunk.chunk_id for result in results]
        per_query.append(
            {
                "question_id": item.id,
                "retrieved_ids": retrieved_ids,
                "first_gold_rank": next(
                    (
                        rank
                        for rank, child_id in enumerate(retrieved_ids, start=1)
                        if child_id in gold_ids
                    ),
                    None,
                ),
            }
        )
        return results

    metrics = evaluate_retrieval(
        questions,
        retrieve,
        require_verified=not args.allow_checked,
    )
    report = {
        "benchmark_version": questions[0].benchmark_version,
        "split": questions[0].split.value,
        "retrieval_configuration": {
            "lexical": "BM25",
            "dense": "Qwen/Qwen3-Embedding-0.6B",
            "fusion": "RRF",
            "reranker": "Qwen/Qwen3-Reranker-0.6B",
            "candidate_k": 12,
            "top_k": 10,
        },
        "metrics": asdict(metrics),
        "per_query": per_query,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
