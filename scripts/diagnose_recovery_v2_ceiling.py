"""Estimate recovery headroom on B0 dev failures using gold concepts (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from viettheory.corpus import SearchMode
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.runtime import build_retrieval


def _load_questions(path: Path) -> dict[str, NaturalQuestionV2]:
    return {
        question.id: question
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for question in (NaturalQuestionV2.model_validate_json(line),)
    }


def _covered_groups(question: NaturalQuestionV2, parents: frozenset[str]) -> int:
    return sum(
        bool(parents & frozenset(group.gold_parent_ids))
        for group in question.required_evidence_groups
        if group.required
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    b0 = json.loads(args.b0_report.read_text(encoding="utf-8"))
    rows = b0["variants"]["within_subject_parent_aware_b0"]["per_query"]
    failures = [row for row in rows if not row["full_evidence_at_5"]]
    retriever = build_retrieval(Path.cwd(), search_mode=SearchMode.GLOBAL, subject_code=None)
    output: list[dict[str, Any]] = []
    for index, baseline in enumerate(failures, 1):
        question = questions[baseline["question_id"]]
        before = frozenset(baseline["retrieved_ids"][: args.top_k])
        started = time.perf_counter()
        recovered: list[str] = []
        queries: list[str] = []
        for concept in question.required_concepts[:2]:
            query = f"{question.question} {concept}"
            queries.append(query)
            for item in retriever.search(
                query,
                top_k=args.top_k,
                subject_codes=frozenset({question.subject_code}),
            ):
                if item.chunk.chunk_id not in recovered:
                    recovered.append(item.chunk.chunk_id)
        merged = before | frozenset(recovered)
        required_count = sum(group.required for group in question.required_evidence_groups)
        before_count = _covered_groups(question, before)
        after_count = _covered_groups(question, merged)
        output.append(
            {
                "question_id": question.id,
                "queries": queries,
                "groups_before": before_count,
                "groups_after_union": after_count,
                "required_groups": required_count,
                "recoverable_in_union": after_count == required_count,
                "recovered_parent_ids": recovered,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
        print(f"{index}/{len(failures)} {question.id} {before_count}->{after_count}", flush=True)
    report = {
        "schema_version": "1.0",
        "split": "development",
        "diagnostic_only": True,
        "uses_gold_concepts": True,
        "must_not_be_reported_as_agentic_performance": True,
        "failure_count": len(output),
        "recoverable_count": sum(row["recoverable_in_union"] for row in output),
        "per_query": output,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"recoverable": report["recoverable_count"], "total": len(output)}))


if __name__ == "__main__":
    main()
