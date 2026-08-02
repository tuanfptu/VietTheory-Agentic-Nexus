"""Deterministic identifiers for versioned pipeline artifacts."""

from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: object) -> str:
    """Build a compact deterministic ID from canonical string components."""
    if not prefix or not parts:
        raise ValueError("stable_id requires a prefix and at least one component")
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"{prefix}_{digest}"
