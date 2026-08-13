"""Subject-preserving logical corpus catalog for unified retrieval."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from viettheory.schema import Chunk
from viettheory.subjects import SUBJECTS, get_subject


class SearchMode(StrEnum):
    WITHIN_SUBJECT = "within_subject"
    GLOBAL = "global"


class SubjectCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_code: str
    structured_dir: Path
    children_path: Path
    parents_path: Path
    dense_index_dir: Path


class UnifiedCorpusCatalog:
    """Expose five physical corpora through one subject-aware logical boundary."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._corpora = {subject.code: self._subject_corpus(subject.code) for subject in SUBJECTS}

    def _subject_corpus(self, subject_code: str) -> SubjectCorpus:
        get_subject(subject_code)
        structured = self.project_root / "data" / "processed" / subject_code / "structured_v1"
        return SubjectCorpus(
            subject_code=subject_code,
            structured_dir=structured,
            children_path=structured / "children.jsonl",
            parents_path=structured / "parents.jsonl",
            dense_index_dir=structured / "dense_index",
        )

    @property
    def subject_codes(self) -> tuple[str, ...]:
        return tuple(self._corpora)

    def resolve(
        self, mode: SearchMode, subject_code: str | None = None
    ) -> tuple[SubjectCorpus, ...]:
        if mode is SearchMode.WITHIN_SUBJECT:
            if subject_code is None:
                raise ValueError("within_subject search requires subject_code")
            get_subject(subject_code)
            return (self._corpora[subject_code],)
        if subject_code is not None:
            raise ValueError("global search must not pre-filter subject_code")
        return tuple(self._corpora.values())

    def load_children(self, mode: SearchMode, subject_code: str | None = None) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for corpus in self.resolve(mode, subject_code):
            for line in corpus.children_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = Chunk.model_validate_json(line)
                if chunk.subject_code != corpus.subject_code:
                    raise ValueError(
                        f"chunk {chunk.chunk_id} belongs to {chunk.subject_code}, "
                        f"not {corpus.subject_code}"
                    )
                if chunk.chunk_id in seen:
                    raise ValueError(f"duplicate global chunk ID: {chunk.chunk_id}")
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
        return tuple(chunks)
