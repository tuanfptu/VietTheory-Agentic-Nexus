"""Run deterministic account-isolation, deletion, and false-evidence memory checks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from viettheory.memory import InMemoryAccountStore, MemoryKind, MemoryRecord


def _record(memory_id: str, account_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        account_id=account_id,
        kind=MemoryKind.LEARNING,
        content=content,
        subject_code="MLN111",
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    store = InMemoryAccountStore()
    store.put(_record("alice_1", "alice", "Needs review of materialism."))
    store.put(_record("bob_1", "bob", "Understands materialism."))
    checks = {
        "alice_cannot_read_bob": all(
            record.account_id == "alice" for record in store.list_for_account("alice")
        ),
        "bob_cannot_delete_alice": not store.delete("bob", "alice_1"),
        "owner_can_delete": store.delete("alice", "alice_1"),
        "deletion_is_effective": not store.list_for_account("alice"),
    }
    false_evidence_rejected = False
    try:
        MemoryRecord(
            memory_id="unsafe",
            account_id="alice",
            kind=MemoryKind.CONVERSATION,
            content="Assistant claim presented as evidence.",
            created_at=datetime(2026, 8, 14, tzinfo=UTC),
            evidence_eligible=True,
        )
    except ValidationError:
        false_evidence_rejected = True
    checks["false_memory_evidence_rejected"] = false_evidence_rejected
    report = {
        "schema_version": "1.0",
        "evaluation": "memory_safety_contract_v1",
        "check_count": len(checks),
        "passed": sum(checks.values()),
        "all_passed": all(checks.values()),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "total": report["check_count"]}))


if __name__ == "__main__":
    main()
