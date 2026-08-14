from __future__ import annotations

from datetime import UTC, datetime

import pytest

from viettheory.memory import InMemoryAccountStore, MemoryKind, MemoryRecord


def _record(memory_id: str, account_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        account_id=account_id,
        kind=MemoryKind.LEARNING,
        content="User is reviewing dialectical materialism.",
        subject_code="MLN111",
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def test_memory_is_strictly_account_isolated_and_deletable() -> None:
    store = InMemoryAccountStore()
    store.put(_record("m1", "alice"))
    store.put(_record("m2", "bob"))

    assert [item.memory_id for item in store.list_for_account("alice")] == ["m1"]
    assert store.delete("bob", "m1") is False
    assert store.delete("alice", "m1") is True
    assert store.list_for_account("alice") == ()


def test_memory_can_never_be_evidence() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="m1",
            account_id="alice",
            kind=MemoryKind.CONVERSATION,
            content="An assistant once claimed something.",
            created_at=datetime(2026, 8, 14, tzinfo=UTC),
            evidence_eligible=True,
        )
