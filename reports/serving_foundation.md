# API and UI Foundation Gate

## Implemented

- FastAPI application factory with no model loading at import time.
- `GET /health`, `POST /ask`, and `POST /feedback`.
- Explicit HTTP 503 while the production RAG runtime is not configured.
- Strict request validation and schema-backed `Answer` responses.
- Thread-safe local SQLite feedback persistence.
- Streamlit client rendering direct answers, claims, citation page metadata, source
  text and bounding boxes without unsafe HTML.
- App dependencies isolated in the `app` optional dependency group.

## Verification

- Ruff lint and format: pass.
- mypy strict: pass (49 source/test files).
- pytest: 38 passed.
- API tests cover readiness, configured and unconfigured `/ask`, and feedback.

One Starlette deprecation warning originates from FastAPI's current `TestClient`
compatibility layer. It does not change endpoint behavior.

## Gate result

**PASS for serving foundation.** Production readiness is not claimed: the API must
still construct the real retriever/generator runtime, and the citation-to-PDF
highlight route requires implementation and browser verification.
