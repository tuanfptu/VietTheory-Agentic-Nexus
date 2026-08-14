"""Account-isolated memory records that can never become textbook evidence."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import model_validator

from viettheory.schema import NonEmptyText, VietTheoryModel


class MemoryKind(StrEnum):
    CONVERSATION = "conversation"
    LEARNING = "learning"
    SYSTEM_TRACE = "system_trace"


class MemoryRecord(VietTheoryModel):
    memory_id: NonEmptyText
    account_id: NonEmptyText
    kind: MemoryKind
    content: NonEmptyText
    subject_code: str | None = None
    created_at: datetime
    evidence_eligible: bool = False

    @model_validator(mode="after")
    def forbid_memory_as_evidence(self) -> MemoryRecord:
        if self.evidence_eligible:
            raise ValueError("memory records must never be eligible as textbook evidence")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class InMemoryAccountStore:
    """Reference store used by tests and local runtime; access is always account-scoped."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, MemoryRecord]] = defaultdict(dict)

    def put(self, record: MemoryRecord) -> None:
        self._records[record.account_id][record.memory_id] = record

    def list_for_account(
        self, account_id: str, *, kind: MemoryKind | None = None
    ) -> tuple[MemoryRecord, ...]:
        records = self._records.get(account_id, {}).values()
        return tuple(
            sorted(
                (record for record in records if kind is None or record.kind is kind),
                key=lambda record: (record.created_at, record.memory_id),
            )
        )

    def delete(self, account_id: str, memory_id: str) -> bool:
        return self._records.get(account_id, {}).pop(memory_id, None) is not None

    def clear_account(self, account_id: str) -> int:
        count = len(self._records.get(account_id, {}))
        self._records.pop(account_id, None)
        return count


def utc_now() -> datetime:
    return datetime.now(UTC)
