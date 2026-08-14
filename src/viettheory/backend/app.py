"""FastAPI application factory with dependency-injected RAG runtime."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, cast

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from viettheory.backend.auth import AuthStore
from viettheory.backend.conversations import ChatMessage, Conversation, ConversationStore
from viettheory.ids import stable_id
from viettheory.pipeline.generator import GenerationError
from viettheory.schema import Answer


class AnsweringPipeline(Protocol):
    def ask(self, question: str) -> Answer: ...


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    subject_code: Literal["MLN111", "MLN122", "MLN131", "HCM202", "VNR202"] | None = None


class ChatRequest(AskRequest):
    conversation_id: str = Field(min_length=1, max_length=128)


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="Cuộc trò chuyện mới", min_length=1, max_length=100)


class CredentialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class SessionResponse(BaseModel):
    username: str
    access_token: str


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
    feedback_path: Path = Path("data/local/mln111_feedback.sqlite3"),
    conversation_path: Path | None = None,
) -> FastAPI:
    """Build an app without loading model artifacts at import time."""
    app = FastAPI(title="VietTheory Agentic Nexus API", version="2.0.0")
    feedback_store = FeedbackStore(feedback_path)
    conversation_store = ConversationStore(
        conversation_path or feedback_path.with_name("mln111_conversations.sqlite3")
    )
    auth_store = AuthStore(feedback_path.with_name("mln111_auth.sqlite3"))

    def current_user(authorization: str | None = Header(default=None)) -> tuple[str, str]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bạn cần đăng nhập")
        try:
            return auth_store.authenticate(authorization.removeprefix("Bearer ").strip())
        except KeyError as exc:
            raise HTTPException(status_code=401, detail="Phiên đăng nhập không hợp lệ") from exc

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "pipeline_ready": pipeline is not None,
            "subject": "MLN111, MLN122, MLN131, HCM202, VNR202",
            "benchmark_version": "2.0.0",
        }

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline is not configured")
        try:
            if request.subject_code is None:
                return pipeline.ask(request.question.strip())
            return cast(
                Answer,
                cast(Any, pipeline).ask(
                    request.question.strip(), subject_code=request.subject_code
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail="Gemini tạm thời không tạo được câu trả lời hợp lệ. Hãy thử lại.",
            ) from exc

    @app.post("/auth/register", response_model=SessionResponse, status_code=201)
    def register(request: CredentialsRequest) -> SessionResponse:
        try:
            session = auth_store.register(request.username, request.password)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionResponse(username=session.username, access_token=session.token)

    @app.post("/auth/login", response_model=SessionResponse)
    def login(request: CredentialsRequest) -> SessionResponse:
        try:
            session = auth_store.login(request.username, request.password)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return SessionResponse(username=session.username, access_token=session.token)

    @app.post("/auth/logout", status_code=204)
    def logout(
        authorization: str | None = Header(default=None),
        _: tuple[str, str] = Depends(current_user),
    ) -> None:
        assert authorization is not None
        auth_store.logout(authorization.removeprefix("Bearer ").strip())

    @app.get("/conversations", response_model=tuple[Conversation, ...])
    def list_conversations(
        user: tuple[str, str] = Depends(current_user),
    ) -> tuple[Conversation, ...]:
        return conversation_store.list(user[0])

    @app.post("/conversations", response_model=Conversation, status_code=201)
    def create_conversation(
        request: CreateConversationRequest, user: tuple[str, str] = Depends(current_user)
    ) -> Conversation:
        return conversation_store.create(user[0], request.title.strip())

    @app.get("/conversations/{conversation_id}/messages", response_model=tuple[ChatMessage, ...])
    def conversation_messages(
        conversation_id: str, user: tuple[str, str] = Depends(current_user)
    ) -> tuple[ChatMessage, ...]:
        try:
            return conversation_store.messages(user[0], conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện") from exc

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: str, user: tuple[str, str] = Depends(current_user)
    ) -> None:
        try:
            conversation_store.delete(user[0], conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện") from exc

    @app.post("/chat", response_model=ChatMessage)
    def chat(request: ChatRequest, user: tuple[str, str] = Depends(current_user)) -> ChatMessage:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline is not configured")
        question = request.question.strip()
        try:
            context = conversation_store.recent_context(user[0], request.conversation_id)
            conversation_store.append_user(user[0], request.conversation_id, question)
            if request.subject_code is None:
                answer = cast(Any, pipeline).ask(question, context)
            else:
                answer = cast(Any, pipeline).ask(
                    question, context, subject_code=request.subject_code
                )
            return conversation_store.append_assistant(user[0], request.conversation_id, answer)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail="Gemini tạm thời không tạo được câu trả lời hợp lệ. Hãy gửi lại câu hỏi.",
            ) from exc

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        return FeedbackResponse(feedback_id=feedback_store.add(request))

    return app


app = create_app()
