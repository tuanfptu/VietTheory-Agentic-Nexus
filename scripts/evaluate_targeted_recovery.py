"""Evaluate deterministic J1 missing-aspect recovery on controlled development cases."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.evidence_judge import JudgeDecision
from viettheory.evidence_sufficiency import EvidenceSufficiencyCase, SufficiencyLabel
from viettheory.retrieval.bm25 import BM25Retriever
from viettheory.runtime import build_retrieval


def _load_cases(path: Path) -> tuple[EvidenceSufficiencyCase, ...]:
    return tuple(
        EvidenceSufficiencyCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _load_decisions(path: Path) -> dict[str, JudgeDecision]:
    return {
        decision.case_id: decision
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for decision in (JudgeDecision.model_validate_json(line),)
    }


def _parent_ranking(items: tuple[Any, ...], top_k: int) -> tuple[str, ...]:
    parents: list[str] = []
    for item in items:
        parent_id = (
            item.chunk.chunk_id if item.chunk.chunk_kind == "parent" else item.chunk.parent_chunk_id
        )
        if parent_id is not None and parent_id not in parents:
            parents.append(parent_id)
        if len(parents) == top_k:
            break
    return tuple(parents)


def _full_coverage(case: EvidenceSufficiencyCase, parent_ids: frozenset[str]) -> bool:
    return all(
        bool(parent_ids & frozenset(aspect.acceptable_parent_ids))
        for aspect in case.required_aspects
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("j1_checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--parent-k", type=int, default=5)
    parser.add_argument("--retriever", choices=("bm25", "b0"), default="bm25")
    args = parser.parse_args()
    if args.candidate_k < args.parent_k or args.parent_k <= 0:
        raise ValueError("candidate-k must be >= positive parent-k")

    cases = _load_cases(args.cases)
    decisions = _load_decisions(args.j1_checkpoint)
    catalog = UnifiedCorpusCatalog(Path.cwd())
    retriever = (
        BM25Retriever(catalog.load_children(SearchMode.GLOBAL))
        if args.retriever == "bm25"
        else build_retrieval(Path.cwd(), search_mode=SearchMode.GLOBAL, subject_code=None)
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        decision = decisions[case.case_id]
        baseline_parents = frozenset(context.parent_id for context in case.provided_contexts)
        baseline_complete = _full_coverage(case, baseline_parents)
        activated = decision.label in {SufficiencyLabel.PARTIAL, SufficiencyLabel.MISSING}
        started = time.perf_counter()
        queries = (
            tuple(f"{case.question} {aspect}" for aspect in decision.missing_aspects[:2])
            if activated
            else ()
        )
        recovered_parents: tuple[str, ...] = ()
        if activated:
            retrieved = tuple(
                item
                for query in queries
                for item in retriever.search(
                    query,
                    top_k=args.candidate_k if args.retriever == "bm25" else args.parent_k,
                    subject_codes=frozenset({case.subject_code}),
                )
            )
            recovered_parents = _parent_ranking(retrieved, args.parent_k * len(queries))
        merged = baseline_parents | frozenset(recovered_parents)
        after_complete = _full_coverage(case, merged)
        rows.append(
            {
                "case_id": case.case_id,
                "source_question_id": case.source_question_id,
                "gold_label": case.expected_label.value,
                "j1_label": decision.label.value,
                "activated": activated,
                "queries": list(queries),
                "baseline_parent_ids": sorted(baseline_parents),
                "recovered_parent_ids": list(recovered_parents),
                "baseline_full_coverage": baseline_complete,
                "after_full_coverage": after_complete,
                "recovery_success": activated and not baseline_complete and after_complete,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )

    deficient = [row for row in rows if not row["baseline_full_coverage"]]
    activated_rows = [row for row in rows if row["activated"]]
    recoverable = [row for row in deficient if row["activated"]]
    successful = [row for row in recoverable if row["recovery_success"]]
    report = {
        "schema_version": "1.0",
        "benchmark_version": cases[0].benchmark_version,
        "split": "development",
        "policy": "J1 partial/missing -> one deterministic BM25 targeted query",
        "retriever": args.retriever,
        "candidate_k": args.candidate_k,
        "parent_k": args.parent_k,
        "held_out_accessed": False,
        "case_count": len(rows),
        "baseline_deficient_cases": len(deficient),
        "activated_deficient_cases": len(recoverable),
        "agent_activation_count": len(activated_rows),
        "agent_activation_rate": len(activated_rows) / len(rows),
        "recovery_success_count": len(successful),
        "recovery_success_rate": (len(successful) / len(recoverable) if recoverable else 0.0),
        "full_coverage_before": sum(bool(row["baseline_full_coverage"]) for row in rows)
        / len(rows),
        "full_coverage_after": sum(bool(row["after_full_coverage"]) for row in rows) / len(rows),
        "latency_p50_ms": statistics.median(float(row["latency_ms"]) for row in rows),
        "per_case": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "activation_rate": report["agent_activation_rate"],
                "recovery_success_rate": report["recovery_success_rate"],
                "full_coverage_after": report["full_coverage_after"],
            }
        )
    )


if __name__ == "__main__":
    main()
