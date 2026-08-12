"""Local account authentication with scrypt password hashing."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

from viettheory.ids import stable_id


@dataclass(frozen=True)
class AuthSession:
    user_id: str
    username: str
    token: str


class AuthStore:
    """Persist users and revocable opaque sessions in SQLite."""

    def __init__(self, path: Path, *, session_days: int = 7) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._session_days = session_days
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash BLOB PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                """
            )

    def register(self, username: str, password: str) -> AuthSession:
        username = username.strip()
        salt = secrets.token_bytes(16)
        password_hash = self._password_hash(password, salt)
        user_id = stable_id("user", username.casefold(), uuid.uuid4().hex)
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    """INSERT INTO users(user_id, username, password_salt, password_hash)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, username, salt, password_hash),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Tên đăng nhập đã tồn tại") from exc
        return self._create_session(user_id, username)

    def login(self, username: str, password: str) -> AuthSession:
        row = self._connection.execute(
            """SELECT user_id, username, password_salt, password_hash
               FROM users WHERE username = ? COLLATE NOCASE""",
            (username.strip(),),
        ).fetchone()
        if row is None or not hmac.compare_digest(
            self._password_hash(password, row["password_salt"]), row["password_hash"]
        ):
            raise ValueError("Tên đăng nhập hoặc mật khẩu không đúng")
        return self._create_session(row["user_id"], row["username"])

    def authenticate(self, token: str) -> tuple[str, str]:
        now = datetime.now(UTC).isoformat()
        row = self._connection.execute(
            """SELECT users.user_id, users.username
               FROM auth_sessions JOIN users USING(user_id)
               WHERE token_hash = ? AND expires_at > ?""",
            (self._token_hash(token), now),
        ).fetchone()
        if row is None:
            raise KeyError("invalid session")
        return row["user_id"], row["username"]

    def logout(self, token: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (self._token_hash(token),)
            )

    def _create_session(self, user_id: str, username: str) -> AuthSession:
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(UTC) + timedelta(days=self._session_days)).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO auth_sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (self._token_hash(token), user_id, expires_at),
            )
        return AuthSession(user_id=user_id, username=username, token=token)

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)

    @staticmethod
    def _token_hash(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()
