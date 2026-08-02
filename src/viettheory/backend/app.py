"""FastAPI application factory with dependency-injected RAG runtime."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from viettheory.ids import stable_id
from viettheory.schema import Answer


class AnsweringPipeline(Protocol):
    def ask(self, question: str) -> Answer: ...


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1, max_length=128)
    helpful: bool
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    accepted: bool = True


class FeedbackStore:
    """Thread-safe, minimal SQLite feedback persistence."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = Lock()
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                answer_id TEXT NOT NULL,
                helpful INTEGER NOT NULL,
                comment TEXT
            )"""
        )
        self._connection.commit()

    def add(self, request: FeedbackRequest) -> str:
        feedback_id = stable_id(
            "feedback", request.answer_id, request.helpful, request.comment or ""
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR REPLACE INTO feedback
                   (feedback_id, answer_id, helpful, comment) VALUES (?, ?, ?, ?)""",
                (feedback_id, request.answer_id, int(request.helpful), request.comment),
            )
        return feedback_id


def create_app(
    pipeline: AnsweringPipeline | None = None,
    *,
    feedback_path: Path = Path("data/local/feedback.sqlite3"),
) -> FastAPI:
    """Build an app without loading model artifacts at import time."""
    app = FastAPI(title="VietTheory-RAG API", version="0.1.0")
    feedback_store = FeedbackStore(feedback_path)

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "pipeline_ready": pipeline is not None}

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline is not configured")
        try:
            return pipeline.ask(request.question.strip())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        return FeedbackResponse(feedback_id=feedback_store.add(request))

    return app


app = create_app()
