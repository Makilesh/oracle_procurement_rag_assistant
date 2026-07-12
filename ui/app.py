"""Gradio chat UI — pure HTTP client of the FastAPI backend (zero RAG logic).

Login → JWT kept in gr.State → streaming chat over SSE with a live
"thinking… (Ns)" indicator → sources in a styled collapsible block →
response-time footer → session badge + New Conversation.
"""

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

Message = dict[str, Any]

EXAMPLE_QUESTIONS = [
    "What are the competitive bidding thresholds at the University of Richmond?",
    "What is the difference between a purchase order, a blanket purchase agreement, and a contract agreement?",
    "What is a purchase requisition in Oracle Procurement?",
    "Can a university purchase card be used to pay an invoice?",
]

WELCOME = """
<div style="text-align:center; opacity:0.75; padding: 28px 12px;">
<h3 style="margin-bottom:6px;">👋 Welcome to the Opkey Procurement Assistant</h3>
<p style="max-width:560px; margin:0 auto;">
I answer questions grounded in two documents — the <b>Oracle Fusion Cloud
Procurement guide</b> (requisitions, purchase orders, agreements, approvals)
and the <b>University of Richmond Procurement Policy</b> (bidding thresholds,
purchase methods, signature authority) — always with page-level citations.
</p>
<p style="margin-top:10px; font-size:0.9em;">Try one of the examples below, or ask your own question.</p>
</div>
"""

CSS = """
footer { display: none !important; }
#header-block { text-align: center; margin-bottom: 4px; }
#header-block p { opacity: 0.7; margin-top: 2px; }
#login-panel { max-width: 460px; margin: 8vh auto 0 auto; }
#session-row { align-items: center; }
.message details {
    background: rgba(128,128,128,0.07);
    border-left: 3px solid #f97316;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 12px;
}
.message details summary { cursor: pointer; font-weight: 600; opacity: 0.85; }
.message details li { margin: 6px 0; }
.thinking { opacity: 0.65; font-style: italic; }
.latency-note { opacity: 0.5; font-size: 0.8em; }
#chat-window { height: 62vh !important; min-height: 420px; }
"""


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _clean_section(path: str) -> str:
    """Cosmetic: merge PDF drop-cap artifacts ('R > ICHMOND' -> 'RICHMOND')."""
    merged: list[str] = []
    for part in (p.strip() for p in path.split(">")):
        if not part:
            continue
        if merged and len(merged[-1]) == 1 and part.isupper():
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return " › ".join(merged)


def _sources_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    rows = []
    for s in sources:
        section = f" — <i>{_clean_section(s['section'])}</i>" if s.get("section") else ""
        snippet = (s.get("snippet") or "").replace("\n", " ")
        rows.append(
            f"<li><b>[{s['tag']}]</b> {s['filename']}, p.{s['page']}{section}<br>"
            f"<small>{snippet}…</small></li>"
        )
    body = "\n".join(rows)
    return (
        f"\n\n<details><summary>📄 Sources ({len(sources)})</summary>\n"
        f"<ul>\n{body}\n</ul>\n</details>"
    )


ALL_DOCUMENTS = "All documents"


def _document_choices(token: str) -> list[str]:
    """Populate the 'Search in' dropdown from the backend's document list."""
    try:
        resp = httpx.get(
            f"{API_URL}/documents",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code == 200:
            return [ALL_DOCUMENTS] + [d["filename"] for d in resp.json()["documents"]]
    except httpx.HTTPError:
        pass
    return [ALL_DOCUMENTS]


def login(username: str, password: str) -> tuple[Any, ...]:
    def fail(msg: str) -> tuple[Any, ...]:
        return (
            None, "", [],
            gr.update(visible=True), gr.update(visible=False), f"⚠️ {msg}",
            gr.update(), gr.update(),
        )

    if not username or not password:
        return fail("Enter username and password.")
    try:
        resp = httpx.post(
            f"{API_URL}/auth/token",
            json={"username": username, "password": password},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        return fail(f"API unreachable: {exc}")
    if resp.status_code == 401:
        return fail("Invalid credentials.")
    if resp.status_code != 200:
        return fail(f"Login failed ({resp.status_code}).")
    token = resp.json()["access_token"]
    session_id = _new_session_id()
    return (
        token,
        session_id,
        [session_id],
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        gr.update(choices=[session_id], value=session_id),
        gr.update(choices=_document_choices(token), value=ALL_DOCUMENTS),
    )


def new_conversation(sessions: list[str]) -> tuple[str, list[str], list[Message], Any]:
    session_id = _new_session_id()
    sessions = [session_id] + [s for s in (sessions or []) if s != session_id]
    return session_id, sessions, [], gr.update(choices=sessions, value=session_id)


def switch_session(session_id: str, token: str | None, sessions: list[str]) -> tuple[str, list[str], list[Message]]:
    """Load a previous conversation's full history from the backend
    (sessions live in Redis, so anything ever chatted is recoverable —
    you can even paste an old session id from a previous login)."""
    session_id = (session_id or "").strip()
    if not token or not session_id:
        return session_id, sessions or [], []
    if session_id not in (sessions or []):
        sessions = [session_id] + (sessions or [])
    messages: list[Message] = []
    try:
        resp = httpx.get(
            f"{API_URL}/sessions/{session_id}/history",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code == 200:
            for turn in resp.json()["turns"]:
                content = turn.get("content", "")
                if turn.get("role") == "assistant" and turn.get("sources"):
                    content += _sources_block(turn["sources"])
                messages.append({"role": turn.get("role", "assistant"), "content": content})
        # 404 = a fresh session with no turns yet — an empty chat is correct
    except httpx.HTTPError:
        messages = [{"role": "assistant", "content": "⚠️ Could not load this session's history."}]
    return session_id, sessions, messages


def _stream_worker(payload: dict[str, Any], token: str, events: "queue.Queue") -> None:
    """Reads the SSE stream on a thread so the UI can tick a timer meanwhile."""
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/chat",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(10.0, read=180.0),
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                events.put(("http_error", resp.status_code, resp.text[:200]))
                return
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                events.put(("event", json.loads(data), None))
    except httpx.HTTPError as exc:
        events.put(("conn_error", str(exc), None))
    finally:
        events.put(("done", None, None))


def respond(
    message: str,
    history: list[Message],
    token: str | None,
    session_id: str,
    doc_filter: str | None = None,
) -> Iterator[tuple[list[Message], str]]:
    message = (message or "").strip()
    if not message:
        yield history, ""
        return
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    def set_answer(text: str) -> None:
        history[-1]["content"] = text

    if not token:
        set_answer("⚠️ Not logged in — please log in first.")
        yield history, ""
        return

    started = time.monotonic()
    set_answer('<span class="thinking">🔍 Searching the knowledge base…</span>')
    yield history, ""

    payload: dict[str, Any] = {"session_id": session_id, "message": message, "stream": True}
    if doc_filter and doc_filter != ALL_DOCUMENTS:
        payload["doc_filter"] = doc_filter

    events: queue.Queue = queue.Queue()
    threading.Thread(
        target=_stream_worker,
        args=(payload, token, events),
        daemon=True,
    ).start()

    answer = ""
    streaming = False
    while True:
        try:
            kind, payload, extra = events.get(timeout=1.0)
        except queue.Empty:
            if not streaming:  # tick the elapsed-time indicator while waiting
                elapsed = int(time.monotonic() - started)
                set_answer(
                    f'<span class="thinking">🔍 Searching documents & generating… '
                    f"({elapsed}s)</span>"
                )
                yield history, ""
            continue

        if kind == "done":
            break
        if kind == "http_error":
            if payload == 401:
                set_answer("⚠️ Session expired — please log in again.")
            elif payload == 429:
                set_answer("⚠️ Slow down — rate limit reached. Try again in a minute.")
            else:
                set_answer(f"⚠️ API error {payload}: {extra}")
            yield history, ""
            continue
        if kind == "conn_error":
            set_answer(f"⚠️ Connection error: {payload}")
            yield history, ""
            continue

        event = payload
        if "delta" in event:
            streaming = True
            answer += event["delta"]
            set_answer(answer)
            yield history, ""
        elif "sources" in event:
            elapsed = time.monotonic() - started
            set_answer(
                answer
                + _sources_block(event["sources"])
                + f'\n\n<span class="latency-note">⏱ answered in {elapsed:.1f}s</span>'
            )
            yield history, ""
        elif "error" in event:
            hint = (
                "LLM quota exceeded — retry shortly."
                if event.get("status") == 503
                else event["error"]
            )
            set_answer((answer + "\n\n" if answer else "") + f"⚠️ {hint}")
            yield history, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Opkey Procurement Assistant") as demo:
        token_state = gr.State(None)
        session_state = gr.State("")

        with gr.Column(elem_id="header-block"):
            gr.Markdown(
                "# 🛒 Opkey Procurement Assistant\n"
                "Grounded answers from the Oracle Fusion Procurement guide and the "
                "University of Richmond procurement policy — with page-level citations."
            )

        with gr.Column(visible=True, elem_id="login-panel") as login_panel:
            gr.Markdown("### Sign in")
            username = gr.Textbox(label="Username", value="demo")
            password = gr.Textbox(label="Password", type="password", placeholder="demo123")
            login_btn = gr.Button("Log in", variant="primary")
            login_error = gr.Markdown("")

        sessions_state = gr.State([])

        with gr.Column(visible=False) as chat_panel:
            with gr.Row(elem_id="session-row"):
                session_picker = gr.Dropdown(
                    choices=[],
                    label="🪪 Session",
                    scale=3,
                    allow_custom_value=True,  # paste an old session id to restore it
                    info="Switch back to a previous conversation — history is kept server-side",
                )
                doc_filter_dd = gr.Dropdown(
                    choices=[ALL_DOCUMENTS],
                    value=ALL_DOCUMENTS,
                    label="📚 Search in",
                    scale=2,
                    info="Restrict answers to one document",
                )
                new_btn = gr.Button("🔄 New Conversation", scale=0, size="sm")
            chatbot = gr.Chatbot(
                height="62vh",
                show_label=False,
                placeholder=WELCOME,
                elem_id="chat-window",
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about purchase orders, requisitions, approval limits…",
                    show_label=False,
                    scale=9,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[msg_box], label="Try asking")

        login_outputs = [
            token_state, session_state, sessions_state,
            login_panel, chat_panel, login_error, session_picker, doc_filter_dd,
        ]
        login_btn.click(login, inputs=[username, password], outputs=login_outputs)
        password.submit(login, inputs=[username, password], outputs=login_outputs)

        new_btn.click(
            new_conversation,
            inputs=[sessions_state],
            outputs=[session_state, sessions_state, chatbot, session_picker],
        )
        # .input fires only on USER interaction, not on programmatic updates —
        # so New Conversation doesn't trigger a spurious history reload.
        session_picker.input(
            switch_session,
            inputs=[session_picker, token_state, sessions_state],
            outputs=[session_state, sessions_state, chatbot],
        )

        chat_inputs = [msg_box, chatbot, token_state, session_state, doc_filter_dd]
        send_btn.click(respond, inputs=chat_inputs, outputs=[chatbot, msg_box])
        msg_box.submit(respond, inputs=chat_inputs, outputs=[chatbot, msg_box])
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("UI_PORT", "7860")),
        css=CSS,  # Gradio 6: styling params live on launch()
    )
