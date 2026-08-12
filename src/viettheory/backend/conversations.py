"""Persistent local conversation history for the chat application."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, ConfigDict

from viettheory.ids import stable_id
from viettheory.schema import Answer


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    conversation_id: str
    role: str
    content: str
    answer: Answer | None = None
    created_at: str


class ConversationStore:
    """Thread-safe SQLite storage for conversations and structured answers."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'legacy',
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    answer_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );
                """
            )
            columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(conversations)")
            }
            if "user_id" not in columns:
                self._connection.execute(
                    "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'"
                )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)"
            )

    def create(self, user_id: str, title: str = "Cuộc trò chuyện mới") -> Conversation:
        conversation_id = stable_id("conversation", uuid.uuid4().hex)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO conversations(conversation_id, user_id, title) VALUES (?, ?, ?)",
                (conversation_id, user_id, title),
            )
        return self.get(user_id, conversation_id)

    def get(self, user_id: str, conversation_id: str) -> Conversation:
        row = self._connection.execute(
            """SELECT conversation_id, title, created_at, updated_at
               FROM conversations WHERE conversation_id = ? AND user_id = ?""",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return Conversation.model_validate(dict(row))

    def list(self, user_id: str) -> tuple[Conversation, ...]:
        rows = self._connection.execute(
            """SELECT conversation_id, title, created_at, updated_at
               FROM conversations WHERE user_id = ?
               ORDER BY updated_at DESC, rowid DESC""",
            (user_id,),
        ).fetchall()
        return tuple(Conversation.model_validate(dict(row)) for row in rows)

    def messages(self, user_id: str, conversation_id: str) -> tuple[ChatMessage, ...]:
        self.get(user_id, conversation_id)
        rows = self._connection.execute(
            "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY rowid",
            (conversation_id,),
        ).fetchall()
        return tuple(self._message(row) for row in rows)

    def append_user(self, user_id: str, conversation_id: str, content: str) -> ChatMessage:
        return self._append(user_id, conversation_id, "user", content, None)

    def append_assistant(self, user_id: str, conversation_id: str, answer: Answer) -> ChatMessage:
        return self._append(user_id, conversation_id, "assistant", answer.direct_answer, answer)

    def recent_context(
        self, user_id: str, conversation_id: str, *, limit: int = 6
    ) -> tuple[str, ...]:
        messages = self.messages(user_id, conversation_id)[-limit:]
        return tuple(f"{message.role}: {message.content}" for message in messages)

    def delete(self, user_id: str, conversation_id: str) -> None:
        self.get(user_id, conversation_id)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,)
            )
            self._connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )

    def _append(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        answer: Answer | None,
    ) -> ChatMessage:
        self.get(user_id, conversation_id)
        message_id = stable_id("message", conversation_id, uuid.uuid4().hex)
        answer_json = answer.model_dump_json() if answer else None
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO chat_messages
                   (message_id, conversation_id, role, content, answer_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (message_id, conversation_id, role, content, answer_json),
            )
            count = self._connection.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if role == "user" and count == 1:
                title = content[:60] + ("…" if len(content) > 60 else "")
                self._connection.execute(
                    """UPDATE conversations
                       SET title = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE conversation_id = ?""",
                    (title, conversation_id),
                )
            else:
                self._connection.execute(
                    """UPDATE conversations
                       SET updated_at = CURRENT_TIMESTAMP
                       WHERE conversation_id = ?""",
                    (conversation_id,),
                )
        row = self._connection.execute(
            "SELECT * FROM chat_messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        assert row is not None
        return self._message(row)

    @staticmethod
    def _message(row: sqlite3.Row) -> ChatMessage:
        raw = dict(row)
        answer_json = raw.pop("answer_json")
        raw["answer"] = Answer.model_validate_json(answer_json) if answer_json else None
        return ChatMessage.model_validate(raw)
