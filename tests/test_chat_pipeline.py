"""Chat pipeline tests with mocked LLM and mocked retrieval."""

from typing import Any

import pytest

import core.chat_pipeline as pipeline
from core import prompts
from core.chat_pipeline import chat_once, prepare_turn, route_small_talk
from core.retrieval import RetrievedChunk
from core.sessions import SessionStore
from tests.test_sessions import FakeRedis


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


def make_store() -> SessionStore:
    return SessionStore(FakeRedis())  # type: ignore[arg-type]


def make_chunk(text: str = "The PO approval workflow has three steps.") -> RetrievedChunk:
    return RetrievedChunk(
        id="doc1:0",
        text=text,
        metadata={
            "source_filename": "oracle.pdf",
            "page_start": 212,
            "page_end": 213,
            "section_path": "Purchase Orders > Approvals",
            "doc_id": "doc1",
            "chunk_index": 0,
        },
        score=0.9,
    )


def test_small_talk_router_hits() -> None:
    assert route_small_talk("hi") == prompts.GREETING_RESPONSE
    assert route_small_talk("Hello!") == prompts.GREETING_RESPONSE
    assert route_small_talk("thanks") == prompts.THANKS_RESPONSE
    assert route_small_talk("what can you do?") == prompts.CAPABILITY_BLURB
    assert route_small_talk("who are you") == prompts.CAPABILITY_BLURB


def test_small_talk_router_passes_domain_questions() -> None:
    assert route_small_talk("what is the PO approval workflow?") is None
    assert route_small_talk("hello, what is a purchase order?") is None


async def test_first_turn_skips_condensation(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    condense_calls: list[str] = []

    async def fake_condense(history: Any, message: str) -> str:
        condense_calls.append(message)
        return message

    async def fake_retrieve(index: Any, query: str) -> list[RetrievedChunk]:
        return [make_chunk()]

    monkeypatch.setattr(pipeline, "condense_query", fake_condense)
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    prepared = await prepare_turn(None, store, "s1", "what is a purchase order?")
    assert prepared.kind == "rag"
    assert condense_calls == []  # empty history -> no condensation call


async def test_followup_invokes_condensation(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    await store.append_turn("s1", {"role": "user", "content": "what is a purchase requisition?"})
    await store.append_turn("s1", {"role": "assistant", "content": "A PR is...", "sources": []})

    condense_calls: list[str] = []
    retrieved_queries: list[str] = []

    async def fake_condense(history: Any, message: str) -> str:
        condense_calls.append(message)
        return "purchase requisition approval limit"

    async def fake_retrieve(index: Any, query: str) -> list[RetrievedChunk]:
        retrieved_queries.append(query)
        return [make_chunk()]

    monkeypatch.setattr(pipeline, "condense_query", fake_condense)
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    prepared = await prepare_turn(None, store, "s1", "what about its approval limit?")
    assert condense_calls == ["what about its approval limit?"]
    assert retrieved_queries == ["purchase requisition approval limit"]
    assert prepared.condensed_query == "purchase requisition approval limit"


async def test_refusal_when_nothing_passes_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()

    async def fake_retrieve(index: Any, query: str) -> list[RetrievedChunk]:
        return []  # confidence gate rejected everything

    llm_calls: list[Any] = []

    async def fake_complete(*args: Any, **kwargs: Any) -> Any:
        llm_calls.append(args)
        return FakeResponse("should never be called")

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "complete", fake_complete)

    answer, sources = await chat_once(None, store, "s1", "what is the meaning of life?")
    assert answer == prompts.REFUSAL_RESPONSE
    assert sources == []
    assert llm_calls == []  # refusal must not call the answer LLM

    history = await store.history("s1")
    assert history is not None and len(history) == 2  # refusal turn still persisted


async def test_rag_answer_persists_turn_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()

    async def fake_retrieve(index: Any, query: str) -> list[RetrievedChunk]:
        return [make_chunk()]

    async def fake_complete(role: str, messages: Any, **kwargs: Any) -> Any:
        assert role == "main"
        assert "[S1]" in messages[1]["content"]
        return FakeResponse("The workflow has three steps [S1].")

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "complete", fake_complete)

    answer, sources = await chat_once(None, store, "s1", "what is the PO approval workflow?")
    assert "[S1]" in answer
    assert sources[0]["filename"] == "oracle.pdf"
    assert sources[0]["page"] == 212

    history = await store.history("s1")
    assert history is not None
    assert history[-1]["role"] == "assistant"
    assert history[-1]["sources"][0]["tag"] == "S1"


async def test_small_talk_persisted_and_no_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    retrieve_calls: list[str] = []

    async def fake_retrieve(index: Any, query: str) -> list[RetrievedChunk]:
        retrieve_calls.append(query)
        return []

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    answer, sources = await chat_once(None, store, "s1", "hi")
    assert answer == prompts.GREETING_RESPONSE
    assert retrieve_calls == []
    history = await store.history("s1")
    assert history is not None and len(history) == 2
