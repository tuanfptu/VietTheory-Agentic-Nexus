"""Attach corpus provenance and stable IDs to raw Natural QA v2 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viettheory.benchmark import (
    BenchmarkSplit,
    GenerationMetadata,
    GoldEvidenceGroup,
    ReasoningScope,
)
from viettheory.benchmark_generation import BenchmarkCandidate
from viettheory.natural_benchmark import BenchmarkCategory, NaturalQuestionV2
from viettheory.schema import Chunk


def normalized_reasoning_scope(candidate: BenchmarkCandidate) -> ReasoningScope:
    """Keep per-subject drafts from claiming unsupported cross-subject evidence."""
    if candidate.reasoning_scope is not ReasoningScope.CROSS_SUBJECT:
        return candidate.reasoning_scope
    if candidate.primary_category is BenchmarkCategory.MULTI_HOP_CROSS_CHAPTER:
        return ReasoningScope.MULTI_HOP
    if len(candidate.evidence_groups) > 1:
        return ReasoningScope.MULTI_CHUNK
    return ReasoningScope.SINGLE_CHUNK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--benchmark-version", default="natural_qa_v2_500_draft")
    parser.add_argument("--prompt-version", default="natural_qa_v2_one_pass_500_v1")
    args = parser.parse_args()

    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    candidates = [
        BenchmarkCandidate.model_validate_json(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artifact_id = f"structured_v1:{manifest['children_sha256']}"
    records: list[NaturalQuestionV2] = []
    for index, candidate in enumerate(candidates, start=1):
        groups: list[GoldEvidenceGroup] = []
        cited: list[Chunk] = []
        for group_index, group in enumerate(candidate.evidence_groups, start=1):
            group_chunks = [children[child_id] for child_id in group.child_ids]
            cited.extend(group_chunks)
            groups.append(
                GoldEvidenceGroup(
                    group_id=f"g{group_index}",
                    subject_code=args.subject,
                    role=group.role,
                    required=group.required,
                    primary_child_ids=group.child_ids,
                    gold_parent_ids=tuple(
                        dict.fromkeys(
                            chunk.parent_chunk_id
                            for chunk in group_chunks
                            if chunk.parent_chunk_id is not None
                        )
                    ),
                    gold_pdf_pages=tuple(
                        sorted(
                            {span.pdf_page for chunk in group_chunks for span in chunk.source_spans}
                        )
                    ),
                )
            )
        records.append(
            NaturalQuestionV2(
                benchmark_version=args.benchmark_version,
                id=f"{args.subject.lower()}_{index:04d}",
                subject_code=args.subject,
                chapter_labels=tuple(
                    dict.fromkeys(chunk.chapter for chunk in cited if chunk.chapter)
                ),
                section_labels=tuple(
                    dict.fromkeys(chunk.section for chunk in cited if chunk.section)
                ),
                question=candidate.question,
                question_types=candidate.question_types,
                primary_category=candidate.primary_category,
                difficulty=candidate.difficulty,
                reasoning_scope=normalized_reasoning_scope(candidate),
                chapter_scope=candidate.chapter_scope,
                answerability=candidate.answerability,
                unanswerable_reason=candidate.unanswerable_reason,
                negative_type=candidate.negative_type,
                expected_behavior=candidate.expected_behavior,
                gold_answer=candidate.gold_answer,
                required_evidence_groups=tuple(groups),
                required_concepts=candidate.required_concepts,
                forbidden_claims=candidate.forbidden_claims,
                split=BenchmarkSplit.DEVELOPMENT,
                generation=GenerationMetadata(
                    method="llm_assisted",
                    model=args.model,
                    prompt_version=args.prompt_version,
                ),
                artifact_manifest_ids=(artifact_id,),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    print(f"materialized={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
