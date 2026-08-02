"""Read-only integrity and staleness validation for benchmark releases."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from viettheory.benchmark import ArtifactManifest, BenchmarkQuestion
from viettheory.chunking.manifest import StructuredArtifactManifest, sha256_file
from viettheory.schema import Chunk


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    severity: str
    code: str
    question_id: str | None = None
    message: str


class BenchmarkValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    valid: bool
    stale: bool
    question_count: int
    issues: tuple[ValidationIssue, ...]
    distribution: dict[str, dict[str, int]]


def _issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    *,
    question_id: str | None = None,
    severity: str = "error",
) -> None:
    issues.append(
        ValidationIssue(
            severity=severity,
            code=code,
            question_id=question_id,
            message=message,
        )
    )


def validate_benchmark(
    questions: tuple[BenchmarkQuestion, ...],
    artifact: ArtifactManifest,
    *,
    pages_path: Path,
    structured_dir: Path,
) -> BenchmarkValidationReport:
    """Validate without modifying questions or their review state."""
    issues: list[ValidationIssue] = []
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        _issue(issues, "duplicate_question_id", "Question IDs must be unique")

    structured = StructuredArtifactManifest.model_validate_json(
        (structured_dir / "manifest.json").read_text(encoding="utf-8")
    )
    stale_checks = {
        "source_artifact_sha256": (
            artifact.source_artifact_sha256,
            sha256_file(pages_path),
        ),
        "chunk_artifact_sha256": (
            artifact.chunk_artifact_sha256,
            sha256_file(structured_dir / "children.jsonl"),
        ),
        "chunking_config_sha256": (
            artifact.chunking_config_sha256,
            structured.config_sha256,
        ),
    }
    stale = False
    for field, (expected, actual) in stale_checks.items():
        if expected != actual:
            stale = True
            _issue(issues, f"{field}_mismatch", f"{field} does not match current corpus")

    parents = {
        chunk.chunk_id: chunk
        for line in (structured_dir / "parents.jsonl").read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    children = {
        chunk.chunk_id: chunk
        for line in (structured_dir / "children.jsonl").read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }

    for question in questions:
        if question.artifact_manifest_id != artifact.artifact_manifest_id:
            _issue(
                issues,
                "artifact_manifest_id_mismatch",
                "Question references a different artifact manifest",
                question_id=question.id,
            )
        for group in question.gold_evidence_groups:
            if group.subject_code != artifact.subject_code:
                _issue(
                    issues,
                    "evidence_subject_mismatch",
                    "Evidence subject differs from artifact subject",
                    question_id=question.id,
                )
            declared_parents = set(group.gold_parent_ids)
            declared_pages = set(group.gold_pdf_pages)
            for parent_id in declared_parents:
                if parent_id not in parents:
                    _issue(
                        issues,
                        "missing_parent",
                        f"Unknown parent ID: {parent_id}",
                        question_id=question.id,
                    )
            for child_id in group.all_child_ids:
                child = children.get(child_id)
                if child is None:
                    _issue(
                        issues,
                        "missing_child",
                        f"Unknown child ID: {child_id}",
                        question_id=question.id,
                    )
                    continue
                if child.subject_code != group.subject_code:
                    _issue(
                        issues,
                        "child_subject_mismatch",
                        f"Child {child_id} has the wrong subject",
                        question_id=question.id,
                    )
                if child.parent_chunk_id not in declared_parents:
                    _issue(
                        issues,
                        "child_parent_mismatch",
                        f"Child {child_id} is not under a declared parent",
                        question_id=question.id,
                    )
                child_pages = {span.pdf_page for span in child.source_spans}
                if not child_pages.intersection(declared_pages):
                    _issue(
                        issues,
                        "child_page_mismatch",
                        f"Child {child_id} does not overlap declared PDF pages",
                        question_id=question.id,
                    )

    distribution = {
        "split": dict(Counter(question.split.value for question in questions)),
        "difficulty": dict(Counter(question.difficulty.value for question in questions)),
        "question_type": dict(
            Counter(kind.value for question in questions for kind in question.question_types)
        ),
        "reasoning_scope": dict(Counter(question.reasoning_scope.value for question in questions)),
        "chapter_scope": dict(Counter(question.chapter_scope.value for question in questions)),
        "answerability": dict(Counter(question.answerability.value for question in questions)),
    }
    error_count = sum(issue.severity == "error" for issue in issues)
    return BenchmarkValidationReport(
        valid=error_count == 0,
        stale=stale,
        question_count=len(questions),
        issues=tuple(issues),
        distribution=distribution,
    )
