"""Route-level security guards: per-user session scoping and the upload cap."""

import asyncio

from fastapi.testclient import TestClient

from api.main import app
from core.config import settings
from core.sessions import SessionStore, scoped_session_id
from tests.test_sessions import FakeRedis

class FakeIndex:
    """Just enough index for routes that only read the registry
    (also satisfies /health, since `app` is shared across test modules)."""

    def list_docs(self) -> list:
        return []

    def doc_count(self) -> int:
        return 0

    def chunk_count(self) -> int:
        return 0


store = SessionStore(FakeRedis())  # type: ignore[arg-type]
app.state.sessions = store
app.state.index = FakeIndex()
client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    resp = client.post(
        "/auth/token",
        json={"username": settings.demo_username, "password": settings.demo_password},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _seed(session_key: str) -> None:
    asyncio.run(
        store.append_turn(session_key, {"role": "user", "content": "hello", "sources": []})
    )


def test_scoped_session_id_namespaces_by_user() -> None:
    assert scoped_session_id("demo", "abc") == "demo:abc"
    assert scoped_session_id("alice", "abc") != scoped_session_id("bob", "abc")


def test_history_reads_only_own_namespace() -> None:
    headers = _auth_headers()
    _seed(scoped_session_id(settings.demo_username, "mine"))
    _seed("other-user:mine")  # someone else's session with the same raw id

    resp = client.get("/sessions/mine/history", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["turns"]) == 1  # only the demo-scoped turn, not both


def test_foreign_session_is_invisible(monkeypatch) -> None:
    # Run as a NON-admin: demo is an admin by default and admins may read any
    # session by key (the deliberate fallthrough tested further down).
    monkeypatch.setattr(settings, "admin_usernames", "someone-else")
    headers = _auth_headers()
    _seed("victim:secret-chat")  # exists in Redis, but under another user

    assert client.get("/sessions/secret-chat/history", headers=headers).status_code == 404
    assert client.delete("/sessions/secret-chat", headers=headers).status_code == 404
    # ...and the raw storage key can't be reached by pasting it either:
    # "demo:victim:secret-chat" is a different key entirely.
    assert client.get("/sessions/victim:secret-chat/history", headers=headers).status_code == 404


def test_delete_scopes_to_own_session() -> None:
    headers = _auth_headers()
    _seed(scoped_session_id(settings.demo_username, "todelete"))

    assert client.delete("/sessions/todelete", headers=headers).json() == {"deleted": True}
    assert client.delete("/sessions/todelete", headers=headers).status_code == 404


def test_admin_lists_all_sessions() -> None:
    headers = _auth_headers()
    _seed(scoped_session_id(settings.demo_username, "list-me"))
    _seed("ghost:hidden-chat")  # another user's session

    resp = client.get("/sessions", headers=headers)
    assert resp.status_code == 200
    by_key = {s["key"]: s for s in resp.json()["sessions"]}
    assert f"{settings.demo_username}:list-me" in by_key
    assert by_key["ghost:hidden-chat"]["owner"] == "ghost"
    assert by_key["ghost:hidden-chat"]["turns"] == 1


def test_non_admin_gets_403_on_sessions_list(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_usernames", "someone-else")
    resp = client.get("/sessions", headers=_auth_headers())
    assert resp.status_code == 403


def test_admin_fallthrough_reads_any_session_by_key() -> None:
    headers = _auth_headers()
    _seed("ghost:their-chat")
    _seed("legacy-unscoped")  # pre-scoping session with no owner prefix

    assert client.get("/sessions/ghost:their-chat/history", headers=headers).status_code == 200
    assert client.get("/sessions/legacy-unscoped/history", headers=headers).status_code == 200


def test_non_admin_has_no_fallthrough(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_usernames", "someone-else")
    headers = _auth_headers()
    _seed("ghost:private-chat")
    assert client.get("/sessions/ghost:private-chat/history", headers=headers).status_code == 404


def test_non_admin_cannot_ingest_or_delete_documents(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_usernames", "someone-else")
    headers = _auth_headers()

    resp = client.post(
        "/ingest", headers=headers, files={"file": ("doc.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 403
    assert client.delete("/documents/any-doc-id", headers=headers).status_code == 403
    # read access to the shared knowledge base is NOT admin-gated
    assert client.get("/documents", headers=headers).status_code == 200


def test_oversized_upload_is_413(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    payload = b"x" * (1024 * 1024 + 1)
    resp = client.post(
        "/ingest",
        headers=_auth_headers(),
        files={"file": ("big.txt", payload, "text/plain")},
    )
    assert resp.status_code == 413
    assert "upload limit" in resp.json()["detail"]
