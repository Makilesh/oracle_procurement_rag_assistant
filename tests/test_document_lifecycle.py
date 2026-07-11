"""Integration test (slow): full document lifecycle against a LIVE api service.

Verifies the prebuilt index is only a bootstrap optimization: delete every
document via the API -> /documents is empty and chat refuses -> re-ingest a
PDF via /ingest -> retrieval works again, all with no restart.

Run: pytest -m slow tests/test_document_lifecycle.py
Requires docker-compose up (api on localhost:8000) and data/ PDFs.
"""

from pathlib import Path

import httpx
import pytest

API = "http://localhost:8000"
PDF = Path("data/richmond_procurement_policy.pdf")

pytestmark = pytest.mark.slow


def _api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    if not _api_up():
        pytest.skip("live api not reachable on localhost:8000")
    with httpx.Client(base_url=API, timeout=600) as c:
        token = c.post(
            "/auth/token", json={"username": "demo", "password": "demo123"}
        ).json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


def test_delete_all_then_reingest_cycle(client: httpx.Client) -> None:
    # 1. delete every indexed document
    docs = client.get("/documents").json()["documents"]
    for doc in docs:
        assert client.delete(f"/documents/{doc['doc_id']}").json() == {"deleted": True}

    assert client.get("/documents").json() == {"documents": []}
    health = client.get("/health").json()
    assert health["docs_indexed"] == 0 and health["chunks"] == 0

    # 2. chat must refuse with an empty index (no LLM hallucination path)
    resp = client.post(
        "/chat",
        json={"session_id": "lifecycle-test", "message": "What are the bidding thresholds?"},
    ).json()
    assert resp["sources"] == []
    assert "couldn't find" in resp["answer"].lower()

    # 3. re-ingest through the API — no restart, no manual steps
    with PDF.open("rb") as fh:
        ingested = client.post("/ingest", files={"file": (PDF.name, fh, "application/pdf")}).json()
    assert ingested["chunks_created"] > 0

    # 4. retrieval works again
    resp = client.post(
        "/chat",
        json={
            "session_id": "lifecycle-test",
            "message": "What are the competitive bidding thresholds at the University of Richmond?",
        },
    ).json()
    assert resp["sources"], f"expected sources after re-ingest, got: {resp}"
    assert any(s["filename"] == PDF.name for s in resp["sources"])

    client.delete("/sessions/lifecycle-test")
