"""Migrate the 30-question MLN111 page-span benchmark to evidence groups."""

from __future__ import annotations

import json
import re
from pathlib import Path

from viettheory.benchmark import (
    Answerability,
    ArtifactManifest,
    BenchmarkQuestion,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    EvidenceRequirement,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
)
from viettheory.chunking.manifest import StructuredArtifactManifest
from viettheory.schema import Chunk, SourceSpan

_SPACE = re.compile(r"\s+")


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _candidate_score(chunk: Chunk, evidence: SourceSpan) -> float:
    evidence_text = _normalized(evidence.text)
    score = 0.0
    for span in chunk.source_spans:
        if span.pdf_page != evidence.pdf_page:
            continue
        area = max(
            1.0, (evidence.bbox[2] - evidence.bbox[0]) * (evidence.bbox[3] - evidence.bbox[1])
        )
        score = max(score, _intersection(span.bbox, evidence.bbox) / area)
        span_text = _normalized(span.text)
        if evidence_text in span_text or span_text in evidence_text:
            score = max(score, 1.0)
    return score


def _group_for_span(
    evidence: SourceSpan,
    group_number: int,
    children: tuple[Chunk, ...],
) -> GoldEvidenceGroup:
    scored = sorted(
        (
            (_candidate_score(chunk, evidence), chunk)
            for chunk in children
            if any(span.pdf_page == evidence.pdf_page for span in chunk.source_spans)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored or scored[0][0] <= 0:
        raise ValueError(f"No child matches evidence on PDF page {evidence.pdf_page}")
    best_score, primary = scored[0]
    acceptable = tuple(
        chunk.chunk_id
        for score, chunk in scored[1:]
        if score >= max(0.5, best_score * 0.75) and chunk.parent_chunk_id is not None
    )
    parent_ids = tuple(
        dict.fromkeys(
            child.parent_chunk_id
            for child in (
                primary,
                *(chunk for score, chunk in scored[1:] if chunk.chunk_id in acceptable),
            )
            if child.parent_chunk_id is not None
        )
    )
    return GoldEvidenceGroup(
        group_id=f"g{group_number}",
        subject_code="MLN111",
        role="direct_answer" if group_number == 1 else "supporting_evidence",
        primary_child_ids=(primary.chunk_id,),
        acceptable_child_ids=acceptable,
        gold_parent_ids=parent_ids,
        gold_pdf_pages=(evidence.pdf_page,),
        gold_printed_pages=((evidence.printed_page,) if evidence.printed_page else ()),
    )


def main() -> int:
    root = Path.cwd()
    legacy_path = root / "benchmark" / "mln111_questions.jsonl"
    structured_dir = root / "data" / "processed" / "MLN111" / "structured_v1"
    pages_path = root / "data" / "processed" / "MLN111" / "pages.jsonl"
    structured = StructuredArtifactManifest.model_validate_json(
        (structured_dir / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = ArtifactManifest(
        artifact_manifest_id="mln111_corpus_2026_07_1",
        subject_code="MLN111",
        source_artifact_sha256=structured.source_pages_sha256,
        chunk_artifact_sha256=structured.children_sha256,
        chunking_config_sha256=structured.config_sha256,
        chunk_schema_version=structured.chunk_schema_version,
        retrieval_corpus_version="2026.07.1",
    )
    children = tuple(
        Chunk.model_validate_json(line)
        for line in (structured_dir / "children.jsonl").read_text(encoding="utf-8").splitlines()
    )
    legacy = [json.loads(line) for line in legacy_path.read_text(encoding="utf-8").splitlines()]
    migrated: list[BenchmarkQuestion] = []
    for index, record in enumerate(legacy, start=1):
        old_type = record["question_type"]
        evidences = tuple(
            SourceSpan.model_validate_json(json.dumps(item, ensure_ascii=False))
            for item in record["gold_evidence"]
        )
        groups = tuple(
            _group_for_span(evidence, number, children)
            for number, evidence in enumerate(evidences, start=1)
        )
        answerability = (
            Answerability.ANSWERABLE
            if record["answerable"]
            else (
                Answerability.FALSE_PREMISE
                if old_type == "false_premise"
                else Answerability.OUT_OF_SCOPE
            )
        )
        migrated_types = {
            "multi_hop": QuestionType.SYNTHESIS,
            "false_premise": QuestionType.MISCONCEPTION,
            "out_of_domain": QuestionType.MISCONCEPTION,
        }
        question_type = migrated_types.get(old_type)
        if question_type is None:
            question_type = QuestionType(old_type)
        chapters = tuple(
            dict.fromkeys(
                child.chapter
                for group in groups
                for child in children
                if child.chunk_id in group.all_child_ids and child.chapter
            )
        )
        migrated.append(
            BenchmarkQuestion(
                benchmark_version="1.0.0-draft",
                id=f"mln111_{index:04d}",
                subject_code="MLN111",
                question=record["question"],
                answerability=answerability,
                unanswerable_reason=(
                    None if answerability is Answerability.ANSWERABLE else record["gold_answer"]
                ),
                expected_behavior=(
                    ExpectedBehavior.ANSWER
                    if answerability is Answerability.ANSWERABLE
                    else (
                        ExpectedBehavior.CORRECT_PREMISE
                        if answerability is Answerability.FALSE_PREMISE
                        else ExpectedBehavior.REFUSE
                    )
                ),
                question_types=(question_type,),
                reasoning_scope=(
                    ReasoningScope.MULTI_HOP
                    if old_type == "multi_hop"
                    else ReasoningScope.SINGLE_CHUNK
                ),
                chapter_scope=(
                    ChapterScope.MULTI_CHAPTER if len(chapters) > 1 else ChapterScope.SINGLE_CHAPTER
                ),
                difficulty=Difficulty(record["difficulty"]),
                chapter_labels=chapters,
                split=(
                    BenchmarkSplit.DEVELOPMENT
                    if record["split"] == "development"
                    else BenchmarkSplit.HELD_OUT_TEST
                ),
                gold_evidence_groups=groups,
                evidence_requirement=EvidenceRequirement(),
                gold_answer=record["gold_answer"],
                generation=GenerationMetadata(method="migration"),
                artifact_manifest_id=artifact.artifact_manifest_id,
            )
        )

    development = root / "benchmark" / "development" / "mln111_questions.jsonl"
    held_out = root / "benchmark_private" / "held_out_test_gold" / "mln111_test_gold.jsonl"
    manifest_path = root / "benchmark" / "manifests" / "mln111_artifacts.json"
    development.parent.mkdir(parents=True, exist_ok=True)
    held_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    development.write_text(
        "".join(
            question.model_dump_json() + "\n"
            for question in migrated
            if question.split is BenchmarkSplit.DEVELOPMENT
        ),
        encoding="utf-8",
    )
    held_out.write_text(
        "".join(
            question.model_dump_json() + "\n"
            for question in migrated
            if question.split is BenchmarkSplit.HELD_OUT_TEST
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Migrated {len(migrated)} questions: "
        f"{sum(q.split is BenchmarkSplit.DEVELOPMENT for q in migrated)} development, "
        f"{sum(q.split is BenchmarkSplit.HELD_OUT_TEST for q in migrated)} held-out"
    )
    print(f"Source pages: {pages_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
