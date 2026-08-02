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


def test_health_reports_pipeline_readiness(tmp_path: Path) -> None:
    client = TestClient(create_app(StubPipeline(), feedback_path=tmp_path / "feedback.db"))
    assert client.get("/health").json() == {"status": "ok", "pipeline_ready": True}


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
