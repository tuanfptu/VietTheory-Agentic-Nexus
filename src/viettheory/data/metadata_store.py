"""SQLite metadata store for pages, chunks, spans, and subjects."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from viettheory.schema import Chunk, Page


class MetadataStore:
    """Transactional store retaining validated JSON payloads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS subjects (subject_code TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                subject_code TEXT NOT NULL REFERENCES subjects(subject_code),
                pdf_page INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                subject_code TEXT NOT NULL REFERENCES subjects(subject_code),
                parent_chunk_id TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_spans (
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                span_index INTEGER NOT NULL,
                page_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (chunk_id, span_index)
            );
            CREATE INDEX IF NOT EXISTS idx_pages_subject_page
                ON pages(subject_code, pdf_page);
            CREATE INDEX IF NOT EXISTS idx_chunks_subject
                ON chunks(subject_code);
            """
        )

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def upsert_pages(self, pages: Iterable[Page]) -> None:
        with self._connection:
            for page in pages:
                self._connection.execute(
                    "INSERT OR IGNORE INTO subjects(subject_code) VALUES (?)",
                    (page.subject_code,),
                )
                self._connection.execute(
                    """INSERT OR REPLACE INTO pages
                       (page_id, subject_code, pdf_page, payload_json) VALUES (?, ?, ?, ?)""",
                    (page.page_id, page.subject_code, page.pdf_page, page.model_dump_json()),
                )

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self._connection:
            for chunk in chunks:
                self._connection.execute(
                    "INSERT OR IGNORE INTO subjects(subject_code) VALUES (?)",
                    (chunk.subject_code,),
                )
                self._connection.execute(
                    """INSERT OR REPLACE INTO chunks
                       (chunk_id, subject_code, parent_chunk_id, payload_json)
                       VALUES (?, ?, ?, ?)""",
                    (
                        chunk.chunk_id,
                        chunk.subject_code,
                        chunk.parent_chunk_id,
                        chunk.model_dump_json(),
                    ),
                )
                self._connection.execute(
                    "DELETE FROM source_spans WHERE chunk_id = ?", (chunk.chunk_id,)
                )
                self._connection.executemany(
                    """INSERT INTO source_spans
                       (chunk_id, span_index, page_id, payload_json) VALUES (?, ?, ?, ?)""",
                    [
                        (chunk.chunk_id, index, span.page_id, span.model_dump_json())
                        for index, span in enumerate(chunk.source_spans)
                    ],
                )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._connection.execute(
            "SELECT payload_json FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return None if row is None else Chunk.model_validate_json(row[0])
