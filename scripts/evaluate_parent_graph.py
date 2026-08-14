"""Evaluate a provenance-backed adjacent-parent graph without benchmark-derived edges."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.retrieval.bm25 import tokenize
from viettheory.schema import Chunk


def _load_questions(path: Path) -> dict[str, NaturalQuestionV2]:
    return {
        question.id: question
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for question in (NaturalQuestionV2.model_validate_json(line),)
        if question.required_evidence_groups
    }


def _gold_groups(question: NaturalQuestionV2) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(group.gold_parent_ids)
        for group in question.required_evidence_groups
        if group.required
    )


def _complete(ids: tuple[str, ...], groups: tuple[frozenset[str], ...], k: int = 5) -> bool:
    top = frozenset(ids[:k])
    return all(bool(top & group) for group in groups)


def _graph_candidates(
    question: str,
    baseline: tuple[str, ...],
    ordered_ids: tuple[str, ...],
    parents: dict[str, Chunk],
) -> tuple[tuple[str, str, float], ...]:
    positions = {parent_id: index for index, parent_id in enumerate(ordered_ids)}
    query_terms = frozenset(tokenize(question))
    scores: dict[str, tuple[str, float]] = {}
    for seed_rank, seed_id in enumerate(baseline[:5], 1):
        position = positions.get(seed_id)
        if position is None:
            continue
        for neighbor_position in (position - 1, position + 1):
            if not 0 <= neighbor_position < len(ordered_ids):
                continue
            candidate_id = ordered_ids[neighbor_position]
            if candidate_id in baseline[:5]:
                continue
            candidate = parents[candidate_id]
            overlap = len(query_terms & frozenset(tokenize(candidate.text))) / max(
                len(query_terms), 1
            )
            score = 1.0 / (60 + seed_rank) + overlap
            previous = scores.get(candidate_id)
            if previous is None or score > previous[1]:
                scores[candidate_id] = (seed_id, score)
    return tuple(
        (candidate_id, seed_id, score)
        for candidate_id, (seed_id, score) in sorted(
            scores.items(), key=lambda item: (-item[1][1], item[0])
        )
    )


def _rrf_merge(
    baseline: tuple[str, ...], graph: tuple[tuple[str, str, float], ...], top_k: int = 10
) -> tuple[str, ...]:
    scores: dict[str, float] = {}
    for rank, parent_id in enumerate(baseline, 1):
        scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (60 + rank)
    for rank, (parent_id, _, _) in enumerate(graph, 1):
        scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (60 + rank)
    return tuple(sorted(scores, key=lambda parent_id: (-scores[parent_id], parent_id))[:top_k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("b0_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    questions = _load_questions(args.questions)
    b0 = json.loads(args.b0_report.read_text(encoding="utf-8"))
    source_rows = b0["variants"]["within_subject_parent_aware_b0"]["per_query"]

    catalog = UnifiedCorpusCatalog(Path.cwd())
    parents: dict[str, Chunk] = {}
    ordered_by_subject: dict[str, tuple[str, ...]] = {}
    edge_count = 0
    for corpus in catalog.resolve(SearchMode.GLOBAL):
        ordered = tuple(
            Chunk.model_validate_json(line)
            for line in corpus.parents_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        parents.update({parent.chunk_id: parent for parent in ordered})
        ordered_by_subject[corpus.subject_code] = tuple(parent.chunk_id for parent in ordered)
        edge_count += max(0, 2 * (len(ordered) - 1))

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        question = questions[source["question_id"]]
        baseline = tuple(source["retrieved_ids"])
        graph_started = time.perf_counter()
        graph = _graph_candidates(
            question.question,
            baseline,
            ordered_by_subject[question.subject_code],
            parents,
        )
        graph_only = tuple(item[0] for item in graph[:10])
        combined = _rrf_merge(baseline, graph)
        graph_latency_ms = (time.perf_counter() - graph_started) * 1000
        groups = _gold_groups(question)
        graph_provenance = [
            {
                "parent_id": parent_id,
                "expanded_from": seed_id,
                "relation": "adjacent_to",
                "subject_code": parents[parent_id].subject_code,
                "source_pages": sorted({span.pdf_page for span in parents[parent_id].source_spans}),
            }
            for parent_id, seed_id, _ in graph[:10]
        ]
        rows.append(
            {
                "question_id": question.id,
                "subject_code": question.subject_code,
                "category": question.primary_category.value,
                "reasoning_scope": question.reasoning_scope.value,
                "baseline_ids": list(baseline),
                "graph_ids": list(graph_only),
                "combined_ids": list(combined),
                "baseline_full_evidence_at_5": _complete(baseline, groups),
                "graph_full_evidence_at_5": _complete(graph_only, groups),
                "combined_full_evidence_at_5": _complete(combined, groups),
                "graph_provenance": graph_provenance,
                "b0_latency_ms": source["latency_ms"],
                "graph_overhead_ms": graph_latency_ms,
            }
        )

    target = [
        row
        for row in rows
        if row["category"] in {"comparison", "synthesis", "multi_hop"}
        or row["reasoning_scope"] != "single_chunk"
    ]

    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        return {
            "count": count,
            "b0_full_evidence_at_5": sum(
                bool(row["baseline_full_evidence_at_5"]) for row in selected
            )
            / count,
            "graph_only_full_evidence_at_5": sum(
                bool(row["graph_full_evidence_at_5"]) for row in selected
            )
            / count,
            "b0_plus_graph_full_evidence_at_5": sum(
                bool(row["combined_full_evidence_at_5"]) for row in selected
            )
            / count,
            "wins": sum(
                not row["baseline_full_evidence_at_5"] and row["combined_full_evidence_at_5"]
                for row in selected
            ),
            "losses": sum(
                row["baseline_full_evidence_at_5"] and not row["combined_full_evidence_at_5"]
                for row in selected
            ),
            "ties": sum(
                row["baseline_full_evidence_at_5"] == row["combined_full_evidence_at_5"]
                for row in selected
            ),
            "graph_overhead_p50_ms": statistics.median(
                float(row["graph_overhead_ms"]) for row in selected
            ),
            "mean_graph_candidates": statistics.fmean(len(row["graph_ids"]) for row in selected),
        }

    report = {
        "schema_version": "1.0",
        "split": "development",
        "candidate": "provenance_adjacent_parent_graph_v0",
        "construction_uses_benchmark_gold": False,
        "hidden_accessed": False,
        "node_count": len(parents),
        "directed_edge_count": edge_count,
        "all": metrics(rows),
        "relationship_multihop_slice": metrics(target),
        "per_query": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"all": report["all"], "target": report["relationship_multihop_slice"]}))


if __name__ == "__main__":
    main()
