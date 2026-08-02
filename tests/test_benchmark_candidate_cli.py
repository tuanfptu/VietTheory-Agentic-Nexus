"""Tests for credential-safe candidate CLI helpers."""

from pathlib import Path

import pytest

from viettheory.benchmark_generation import load_gemini_key


def test_load_gemini_key_reads_only_named_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OTHER_SECRET=do-not-use\nGEMINI_API_KEY='test-key'\n",
        encoding="utf-8",
    )

    assert load_gemini_key(dotenv) == "test-key"


def test_environment_key_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")

    assert load_gemini_key(dotenv) == "environment-key"


def test_legacy_llm_api_key_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_API_KEY=legacy-key\n", encoding="utf-8")

    assert load_gemini_key(dotenv) == "legacy-key"
