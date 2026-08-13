from pathlib import Path

from fastapi.testclient import TestClient

from viettheory.backend.app import create_app
from viettheory.schema import Answer


class StubPipeline:
    def ask(self, question: str) -> Answer:
        return Answer(
            answer_id="A1",
            question=question,
            direct_answer="Không đủ căn cứ.",
            claims=(),
            citations=(),
            refused=True,
            refusal_reason="Không đủ căn cứ.",
        )


class ContextStubPipeline(StubPipeline):
    def __init__(self) -> None:
        self.contexts: list[tuple[str, ...]] = []

    def ask(self, question: str, context: tuple[str, ...] = ()) -> Answer:
        self.contexts.append(context)
        return super().ask(question)


def test_health_reports_pipeline_readiness(tmp_path: Path) -> None:
    client = TestClient(create_app(StubPipeline(), feedback_path=tmp_path / "feedback.db"))
    assert client.get("/health").json() == {
        "status": "ok",
        "pipeline_ready": True,
        "subject": "MLN111",
        "benchmark_version": "1.0.0",
    }


def test_ask_returns_validated_answer(tmp_path: Path) -> None:
    client = TestClient(create_app(StubPipeline(), feedback_path=tmp_path / "feedback.db"))
    response = client.post("/ask", json={"question": "Vật chất là gì?"})
    assert response.status_code == 200
    assert response.json()["question"] == "Vật chất là gì?"


def test_unconfigured_ask_returns_service_unavailable(tmp_path: Path) -> None:
    client = TestClient(create_app(feedback_path=tmp_path / "feedback.db"))
    assert client.post("/ask", json={"question": "Câu hỏi"}).status_code == 503


def test_feedback_is_accepted_without_storing_secret_data(tmp_path: Path) -> None:
    client = TestClient(create_app(feedback_path=tmp_path / "feedback.db"))
    response = client.post("/feedback", json={"answer_id": "A1", "helpful": True, "comment": "Tốt"})
    assert response.status_code == 201
    assert response.json()["accepted"] is True


def test_chat_persists_messages_and_passes_prior_context(tmp_path: Path) -> None:
    pipeline = ContextStubPipeline()
    client = TestClient(create_app(pipeline, feedback_path=tmp_path / "feedback.db"))
    token = _register(client, "tuan_user")
    headers = {"Authorization": f"Bearer {token}"}
    conversation = client.post(
        "/conversations", json={"title": "Cuộc trò chuyện mới"}, headers=headers
    ).json()
    conversation_id = conversation["conversation_id"]

    first = client.post(
        "/chat",
        json={"conversation_id": conversation_id, "question": "Vật chất là gì?"},
        headers=headers,
    )
    second = client.post(
        "/chat",
        json={"conversation_id": conversation_id, "question": "Ý đó có nghĩa gì?"},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    messages = client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert pipeline.contexts[0] == ()
    assert any("Vật chất là gì?" in item for item in pipeline.contexts[1])


def _register(client: TestClient, username: str) -> str:
    response = client.post(
        "/auth/register", json={"username": username, "password": "strong-password-123"}
    )
    assert response.status_code == 201
    access_token = response.json()["access_token"]
    assert isinstance(access_token, str)
    return access_token


def test_accounts_have_isolated_conversation_histories(tmp_path: Path) -> None:
    client = TestClient(create_app(StubPipeline(), feedback_path=tmp_path / "feedback.db"))
    token_a = _register(client, "student_a")
    token_b = _register(client, "student_b")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    conversation = client.post(
        "/conversations", json={"title": "Lịch sử của A"}, headers=headers_a
    ).json()

    assert len(client.get("/conversations", headers=headers_a).json()) == 1
    assert client.get("/conversations", headers=headers_b).json() == []
    assert (
        client.get(
            f"/conversations/{conversation['conversation_id']}/messages", headers=headers_b
        ).status_code
        == 404
    )


def test_conversation_endpoints_require_login(tmp_path: Path) -> None:
    client = TestClient(create_app(StubPipeline(), feedback_path=tmp_path / "feedback.db"))
    assert client.get("/conversations").status_code == 401
