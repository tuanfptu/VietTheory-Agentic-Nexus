"""Select 70 MLN111 candidates and assemble a 100-question review draft."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from viettheory.benchmark import (
    BenchmarkQuestion,
    BenchmarkSplit,
    GenerationMetadata,
    GoldEvidenceGroup,
    ReviewStatus,
)
from viettheory.benchmark_generation import BenchmarkCandidate, normalized_question
from viettheory.schema import Chunk

NEW_QUOTAS = {"easy": 16, "medium": 33, "hard": 21}
NEW_DEVELOPMENT_QUOTAS = {"easy": 11, "medium": 25, "hard": 16}


def _read_questions(path: Path) -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_candidates(path: Path) -> list[BenchmarkCandidate]:
    return [
        BenchmarkCandidate.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate_chapters(
    candidate: BenchmarkCandidate,
    children: dict[str, Chunk],
) -> tuple[str, ...]:
    chapters = {
        children[child_id].chapter or "unassigned"
        for group in candidate.evidence_groups
        for child_id in group.child_ids
    }
    return tuple(sorted(chapters))


def _select(
    candidates: list[BenchmarkCandidate],
    *,
    children: dict[str, Chunk],
    existing_questions: set[str],
) -> list[BenchmarkCandidate]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.reasoning_scope.value != "cross_subject"
        and normalized_question(candidate.question) not in existing_questions
        and all(
            child_id in children
            for group in candidate.evidence_groups
            for child_id in group.child_ids
        )
    ]
    selected: list[BenchmarkCandidate] = []
    chapter_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for difficulty, quota in NEW_QUOTAS.items():
        pool = [item for item in eligible if item.difficulty.value == difficulty]
        for _ in range(quota):
            if not pool:
                raise ValueError(f"not enough eligible {difficulty} candidates")

            def score(item: BenchmarkCandidate) -> tuple[int, int, str]:
                chapters = _candidate_chapters(item, children)
                chapter_load = sum(chapter_counts[chapter] for chapter in chapters)
                type_load = sum(type_counts[kind.value] for kind in item.question_types)
                return chapter_load, type_load, normalized_question(item.question)

            chosen = min(pool, key=score)
            pool.remove(chosen)
            selected.append(chosen)
            chapter_counts.update(_candidate_chapters(chosen, children))
            type_counts.update(kind.value for kind in chosen.question_types)
    return selected


def _convert(
    selected: list[BenchmarkCandidate],
    *,
    children: dict[str, Chunk],
    artifact_manifest_id: str,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkQuestion]]:
    development: list[BenchmarkQuestion] = []
    held_out: list[BenchmarkQuestion] = []
    development_used: Counter[str] = Counter()
    for offset, candidate in enumerate(selected, start=31):
        difficulty = candidate.difficulty.value
        is_development = development_used[difficulty] < NEW_DEVELOPMENT_QUOTAS[difficulty]
        split = BenchmarkSplit.DEVELOPMENT if is_development else BenchmarkSplit.HELD_OUT_TEST
        if is_development:
            development_used[difficulty] += 1
        groups: list[GoldEvidenceGroup] = []
        for number, candidate_group in enumerate(candidate.evidence_groups, start=1):
            group_children = [children[child_id] for child_id in candidate_group.child_ids]
            groups.append(
                GoldEvidenceGroup(
                    group_id=f"g{number}",
                    subject_code="MLN111",
                    role=candidate_group.role,
                    required=candidate_group.required,
                    primary_child_ids=candidate_group.child_ids,
                    gold_parent_ids=tuple(
                        sorted(
                            {
                                child.parent_chunk_id
                                for child in group_children
                                if child.parent_chunk_id
                            }
                        )
                    ),
                    gold_pdf_pages=tuple(
                        sorted(
                            {
                                span.pdf_page
                                for child in group_children
                                for span in child.source_spans
                            }
                        )
                    ),
                    gold_printed_pages=tuple(
                        sorted(
                            {
                                span.printed_page
                                for child in group_children
                                for span in child.source_spans
                                if span.printed_page
                            }
                        )
                    ),
                )
            )
        chapter_labels = tuple(
            chapter
            for chapter in _candidate_chapters(candidate, children)
            if chapter != "unassigned"
        )
        question = BenchmarkQuestion(
            benchmark_version="1.0.0-draft",
            id=f"mln111_{offset:04d}",
            subject_code="MLN111",
            question=candidate.question,
            answerability=candidate.answerability,
            unanswerable_reason=candidate.unanswerable_reason,
            expected_behavior=candidate.expected_behavior,
            question_types=candidate.question_types,
            reasoning_scope=candidate.reasoning_scope,
            chapter_scope=candidate.chapter_scope,
            difficulty=candidate.difficulty,
            chapter_labels=chapter_labels,
            split=split,
            gold_evidence_groups=tuple(groups),
            gold_answer=candidate.gold_answer,
            required_concepts=candidate.required_concepts,
            forbidden_claims=candidate.forbidden_claims,
            generation=GenerationMetadata(
                method="llm_assisted",
                model="gemini-3.5-flash-lite",
                prompt_version="mln111-candidate-v2",
            ),
            artifact_manifest_id=artifact_manifest_id,
            review_status=ReviewStatus.DRAFT,
        )
        (development if is_development else held_out).append(question)
    return development, held_out


def _write(path: Path, questions: list[BenchmarkQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(question.model_dump_json() + "\n" for question in questions),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--existing-development", type=Path, required=True)
    parser.add_argument("--existing-held-out", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--held-out-output", type=Path, required=True)
    args = parser.parse_args()

    existing_development = _read_questions(args.existing_development)
    existing_held_out = _read_questions(args.existing_held_out)
    existing = existing_development + existing_held_out
    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    selected = _select(
        _read_candidates(args.candidates),
        children=children,
        existing_questions={normalized_question(item.question) for item in existing},
    )
    development, held_out = _convert(
        selected,
        children=children,
        artifact_manifest_id=existing[0].artifact_manifest_id,
    )
    merged_development = existing_development + development
    merged_held_out = existing_held_out + held_out
    _write(args.development_output, merged_development)
    _write(args.held_out_output, merged_held_out)
    print(
        f"selected={len(selected)} development={len(merged_development)} "
        f"held_out={len(merged_held_out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
