"""Streamlit client for the VietTheory-RAG API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError("Không kết nối được VietTheory-RAG API") from exc


def render_answer(answer: dict[str, Any]) -> None:
    """Render structured claims and source geometry without unsafe HTML."""
    st.subheader("Trả lời")
    st.write(answer["direct_answer"])
    if answer.get("refused"):
        st.warning(answer.get("refusal_reason", "Không đủ căn cứ để trả lời."))
        return
    citations = {item["citation_id"]: item for item in answer.get("citations", [])}
    for claim in answer.get("claims", []):
        st.markdown(f"- {claim['text']}")
        for citation_id in claim.get("citation_ids", []):
            citation = citations.get(citation_id)
            if citation is None:
                continue
            span = citation["source_span"]
            printed = span.get("printed_page") or "—"
            with st.expander(
                f"{citation_id} · PDF trang {span['pdf_page'] + 1} · trang in {printed}"
            ):
                st.write(span["text"])
                st.caption(f"bbox: {span['bbox']}")


def main() -> None:
    st.set_page_config(page_title="VietTheory-RAG", page_icon="📚", layout="wide")
    st.title("VietTheory-RAG")
    st.caption("Hỏi đáp năm giáo trình lý luận chính trị với dẫn nguồn theo trang PDF.")
    api_base = os.getenv("VIETTHEORY_API_URL", "http://127.0.0.1:8000").rstrip("/")
    with st.form("ask-form"):
        question = st.text_area("Câu hỏi", max_chars=4000)
        submitted = st.form_submit_button("Tìm và trả lời", type="primary")
    if submitted:
        if not question.strip():
            st.warning("Hãy nhập câu hỏi.")
        else:
            try:
                with st.spinner("Đang truy xuất giáo trình..."):
                    answer = _post_json(f"{api_base}/ask", {"question": question})
                render_answer(answer)
            except RuntimeError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
