"""ChatGPT-style Streamlit client for the MLN111 assistant."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import streamlit as st

ASSET_DIR = Path(__file__).with_name("assets")
CREATOR_LINE = "Trợ lí được tạo bởi Tuân, một gymer thích học Triết."


def _render_login_hero() -> None:
    """Introduce the product with a compact, playful proof point."""
    st.title("📚 MLN111 Assistant")
    st.markdown(
        '### Muốn học MLN111 đỡ "triết lý" và dễ hiểu hơn? '
        "Thử ngay trợ lí Triết học Mác-Lênin này nhé."
    )
    image_column, text_column = st.columns((1, 1.5), vertical_alignment="center")
    image_column.image(
        str(ASSET_DIR / "academic-result-mln111.png"),
        use_container_width=True,
    )
    with text_column:
        st.markdown("**Tác giả đã thử trước và kết quả cũng khá có sức thuyết phục :))**")
        st.caption(CREATOR_LINE)


def _render_creator_details() -> None:
    """Keep a short creator note after sign-in without repeating the score image."""
    st.markdown(CREATOR_LINE)
    st.caption("Tác giả đã thử chatbot trước khi rủ mọi người dùng chung :)")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status == 204:
                return None
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = None
        raise RuntimeError(
            _friendly_error(detail) or f"MLN111 Assistant API trả lỗi HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError("Không kết nối được MLN111 Assistant API") from exc


def _friendly_error(detail: Any) -> str | None:
    """Translate FastAPI validation payloads into concise UI messages."""
    if isinstance(detail, str):
        return detail
    if not isinstance(detail, list):
        return None
    fields = {item.get("loc", [None])[-1]: item for item in detail if isinstance(item, dict)}
    if "password" in fields:
        return "Mật khẩu phải có ít nhất 8 ký tự."
    if "username" in fields:
        return (
            "Tên đăng nhập cần 3-40 ký tự và chỉ gồm chữ, số, dấu chấm, gạch ngang hoặc gạch dưới."
        )
    return "Thông tin nhập vào chưa hợp lệ."


def _api_supports_accounts(api_base: str) -> bool:
    try:
        schema = _request_json(f"{api_base}/openapi.json")
    except RuntimeError:
        return False
    return "/auth/register" in schema.get("paths", {})


def _api_base() -> str:
    """Select an explicitly configured API or discover the local account API."""
    configured = os.getenv("VIETTHEORY_API_URL")
    if configured:
        candidate = configured.rstrip("/")
        if _api_supports_accounts(candidate):
            return candidate
    for candidate in ("http://127.0.0.1:8001", "http://127.0.0.1:8000"):
        if _api_supports_accounts(candidate):
            return candidate
    return (configured or "http://127.0.0.1:8001").rstrip("/")


def _render_sources(answer: dict[str, Any]) -> None:
    citations = {item["citation_id"]: item for item in answer.get("citations", [])}
    if not citations:
        return
    st.caption(f"Nguồn tham khảo · {len(citations)} trích dẫn")
    for index, citation in enumerate(citations.values(), start=1):
        span = citation["source_span"]
        printed = span.get("printed_page") or "—"
        label = (
            f"[{index}] MLN111 · PDF trang {span['pdf_page'] + 1} "
            f"· trang in {printed} · Bấm để xem đoạn nguồn"
        )
        with st.expander(label):
            st.markdown(citation.get("context_text") or span["text"])


def _render_message(message: dict[str, Any]) -> None:
    role = message["role"]
    with st.chat_message(role, avatar="📚" if role == "assistant" else None):
        answer = message.get("answer")
        if answer:
            if answer.get("refused"):
                st.warning(answer.get("refusal_reason", answer["direct_answer"]))
            else:
                st.markdown(answer["direct_answer"])
                _render_sources(answer)
        else:
            st.markdown(message["content"])


def _new_conversation(api_base: str, token: str) -> str:
    conversation = _request_json(
        f"{api_base}/conversations",
        method="POST",
        payload={"title": "Cuộc trò chuyện mới"},
        token=token,
    )
    return str(conversation["conversation_id"])


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        [data-testid="stSidebar"] { background: #f7f7f8; }
        [data-testid="stSidebar"] button { text-align: left; border-radius: 10px; }
        [data-testid="stChatMessage"] { max-width: 820px; margin: 0 auto; }
        [data-testid="stChatInput"] { max-width: 820px; margin: 0 auto; }
        .block-container { max-width: 1050px; padding-top: 2rem; padding-bottom: 7rem; }
        h1 { letter-spacing: -0.04em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="MLN111 Assistant", page_icon="📚", layout="wide")
    _apply_style()
    api_base = _api_base()

    if "access_token" not in st.session_state:
        _render_login_hero()
        st.divider()
        login_tab, register_tab = st.tabs(("Đăng nhập", "Tạo tài khoản"))
        for tab, endpoint, submit_label in (
            (login_tab, "login", "Đăng nhập"),
            (register_tab, "register", "Tạo tài khoản"),
        ):
            with tab, st.form(f"auth-{endpoint}"):
                username = st.text_input("Tên đăng nhập", key=f"username-{endpoint}")
                password = st.text_input("Mật khẩu", type="password", key=f"password-{endpoint}")
                st.caption("Tên đăng nhập: 3-40 ký tự · Mật khẩu: ít nhất 8 ký tự")
                if st.form_submit_button(submit_label, type="primary"):
                    if len(username.strip()) < 3:
                        st.error("Tên đăng nhập phải có ít nhất 3 ký tự.")
                    elif len(password) < 8:
                        st.error("Mật khẩu phải có ít nhất 8 ký tự.")
                    else:
                        try:
                            session = _request_json(
                                f"{api_base}/auth/{endpoint}",
                                method="POST",
                                payload={"username": username, "password": password},
                            )
                            st.session_state.access_token = session["access_token"]
                            st.session_state.username = session["username"]
                            st.rerun()
                        except RuntimeError as exc:
                            st.error(str(exc))
        return

    token = st.session_state.access_token

    try:
        conversations: list[dict[str, Any]] = _request_json(
            f"{api_base}/conversations", token=token
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.info("Hãy khởi động API và đợi dòng 'Application startup complete'.")
        return

    with st.sidebar:
        st.markdown("## 📚 MLN111")
        st.caption(f"Đang đăng nhập: {st.session_state.username}")
        if st.button("Đăng xuất", use_container_width=True):
            try:
                _request_json(f"{api_base}/auth/logout", method="POST", token=token)
            except RuntimeError:
                pass
            st.session_state.clear()
            st.rerun()
        st.caption("Trợ lý Triết học Mác-Lênin")
        if st.button("+ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
            st.session_state.conversation_id = _new_conversation(api_base, token)
            st.rerun()
        st.markdown("#### Lịch sử")
        for conversation in conversations:
            active = conversation["conversation_id"] == st.session_state.get("conversation_id")
            label = ("● " if active else "") + conversation["title"]
            if st.button(
                label,
                key=f"conversation-{conversation['conversation_id']}",
                use_container_width=True,
            ):
                st.session_state.conversation_id = conversation["conversation_id"]
                st.rerun()
        st.divider()
        st.caption("Qwen GPU retrieval · Gemini generation · Citation theo PDF")
        with st.expander("Một chút về tác giả"):
            _render_creator_details()

    if "conversation_id" not in st.session_state:
        if conversations:
            st.session_state.conversation_id = conversations[0]["conversation_id"]
        else:
            st.session_state.conversation_id = _new_conversation(api_base, token)
            st.rerun()

    conversation_id = st.session_state.conversation_id
    messages: list[dict[str, Any]] = _request_json(
        f"{api_base}/conversations/{conversation_id}/messages", token=token
    )

    if not messages:
        st.title("Hôm nay bạn muốn học gì?")
        st.caption(
            "Hỏi về giáo trình MLN111. Bạn có thể hỏi tiếp bằng 'ý đó', 'vì sao' "
            "hoặc 'so sánh thêm' - hệ thống sẽ dùng ngữ cảnh cuộc trò chuyện."
        )
        cols = st.columns(2)
        suggestions = (
            "Vật chất theo định nghĩa của Lênin là gì?",
            "Thực tiễn có vai trò gì đối với nhận thức?",
            "Phân biệt duy vật biện chứng và duy vật siêu hình.",
            "Nguồn gốc của ý thức gồm những yếu tố nào?",
        )
        for index, suggestion in enumerate(suggestions):
            if cols[index % 2].button(
                suggestion, key=f"suggestion-{index}", use_container_width=True
            ):
                st.session_state.pending_question = suggestion
                st.rerun()
    else:
        for message in messages:
            _render_message(message)

    question = st.chat_input("Nhắn tin cho trợ lý MLN111…")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant", avatar="📚"):
                with st.spinner("Đang đọc giáo trình và kiểm tra nguồn…"):
                    message = _request_json(
                        f"{api_base}/chat",
                        method="POST",
                        payload={"conversation_id": conversation_id, "question": question},
                        token=token,
                    )
                answer = message["answer"]
                if answer.get("refused"):
                    st.warning(answer.get("refusal_reason", answer["direct_answer"]))
                else:
                    st.markdown(answer["direct_answer"])
                    _render_sources(answer)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
