"""Gradio chat UI — pure HTTP client of the FastAPI backend (zero RAG logic).

Login → JWT kept in gr.State → streaming chat over SSE → sources shown in a
collapsible block under each answer → visible session badge + New Conversation.
"""

import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

Message = dict[str, Any]


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _badge(session_id: str) -> str:
    return f"**Session:** `{session_id}`"


def _sources_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    rows = []
    for s in sources:
        section = f" — *{s['section']}*" if s.get("section") else ""
        snippet = (s.get("snippet") or "").replace("\n", " ")
        rows.append(
            f"- **[{s['tag']}]** {s['filename']}, p.{s['page']}{section}<br>"
            f"<small>{snippet}…</small>"
        )
    body = "\n".join(rows)
    return (
        f"\n\n<details><summary>📄 Sources ({len(sources)})</summary>\n\n{body}\n\n</details>"
    )


def login(username: str, password: str) -> tuple[Any, ...]:
    if not username or not password:
        return None, "", gr.update(visible=True), gr.update(visible=False), "Enter username and password.", ""
    try:
        resp = httpx.post(
            f"{API_URL}/auth/token",
            json={"username": username, "password": password},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        return None, "", gr.update(visible=True), gr.update(visible=False), f"API unreachable: {exc}", ""
    if resp.status_code == 401:
        return None, "", gr.update(visible=True), gr.update(visible=False), "Invalid credentials.", ""
    if resp.status_code != 200:
        return None, "", gr.update(visible=True), gr.update(visible=False), f"Login failed ({resp.status_code}).", ""
    token = resp.json()["access_token"]
    session_id = _new_session_id()
    return token, session_id, gr.update(visible=False), gr.update(visible=True), "", _badge(session_id)


def new_conversation() -> tuple[str, list[Message], str]:
    session_id = _new_session_id()
    return session_id, [], _badge(session_id)


def respond(
    message: str, history: list[Message], token: str | None, session_id: str
) -> Iterator[tuple[list[Message], str]]:
    message = (message or "").strip()
    if not message:
        yield history, ""
        return
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, ""

    def set_answer(text: str) -> None:
        history[-1]["content"] = text

    if not token:
        set_answer("⚠️ Not logged in — please log in first.")
        yield history, ""
        return

    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/chat",
            json={"session_id": session_id, "message": message, "stream": True},
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(10.0, read=180.0),
        ) as resp:
            if resp.status_code == 401:
                set_answer("⚠️ Session expired — please log in again.")
                yield history, ""
                return
            if resp.status_code == 429:
                set_answer("⚠️ Slow down — rate limit reached. Try again in a minute.")
                yield history, ""
                return
            if resp.status_code != 200:
                resp.read()
                set_answer(f"⚠️ API error {resp.status_code}: {resp.text[:200]}")
                yield history, ""
                return

            answer = ""
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if "delta" in event:
                    answer += event["delta"]
                    set_answer(answer)
                    yield history, ""
                elif "sources" in event:
                    set_answer(answer + _sources_block(event["sources"]))
                    yield history, ""
                elif "error" in event:
                    status = event.get("status")
                    hint = (
                        "LLM quota exceeded — retry shortly."
                        if status == 503
                        else event["error"]
                    )
                    set_answer(answer + f"\n\n⚠️ {hint}")
                    yield history, ""
    except httpx.HTTPError as exc:
        set_answer(f"⚠️ Connection error: {exc}")
        yield history, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Opkey Procurement Assistant") as demo:
        token_state = gr.State(None)
        session_state = gr.State("")

        gr.Markdown("# 🛒 Opkey Procurement Assistant")

        with gr.Column(visible=True) as login_panel:
            gr.Markdown("Log in to start chatting (demo credentials are in `.env.example`).")
            username = gr.Textbox(label="Username", value="demo")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Log in", variant="primary")
            login_error = gr.Markdown("")

        with gr.Column(visible=False) as chat_panel:
            with gr.Row():
                session_badge = gr.Markdown("")
                new_btn = gr.Button("🔄 New Conversation", scale=0)
            chatbot = gr.Chatbot(height=480)  # Gradio 6: messages format is the default
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about purchase orders, requisitions, approval limits…",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        login_btn.click(
            login,
            inputs=[username, password],
            outputs=[token_state, session_state, login_panel, chat_panel, login_error, session_badge],
        )
        password.submit(
            login,
            inputs=[username, password],
            outputs=[token_state, session_state, login_panel, chat_panel, login_error, session_badge],
        )
        new_btn.click(new_conversation, outputs=[session_state, chatbot, session_badge])
        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, token_state, session_state],
            outputs=[chatbot, msg_box],
        )
        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, token_state, session_state],
            outputs=[chatbot, msg_box],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", "7860")))
