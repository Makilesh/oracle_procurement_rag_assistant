"""Chat orchestration: small-talk router → history → condensation → retrieval
→ grounded answer (streamed or not) → turn persistence."""

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from core import metrics, prompts
from core.index import IndexStore
from core.llm import QuotaExceededError, complete, response_text, stream_deltas
from core.logging import log_stage
from core.models import embed_texts
from core.retrieval import RetrievedChunk, retrieve
from core.semcache import SemanticCache
from core.sessions import SessionStore, Turn

logger = logging.getLogger("chat")

# ---- rule-based small-talk / capability router (no LLM, no retrieval) ----

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))[\s!.,]*$", re.IGNORECASE
)
_THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx|ty)[\s!.,]*$", re.IGNORECASE)
_CAPABILITY_RE = re.compile(
    r"^\s*(who\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do|help|capabilities)\??[\s!.,]*$",
    re.IGNORECASE,
)


def route_small_talk(message: str) -> str | None:
    if _GREETING_RE.match(message):
        return prompts.GREETING_RESPONSE
    if _THANKS_RE.match(message):
        return prompts.THANKS_RESPONSE
    if _CAPABILITY_RE.match(message):
        return prompts.CAPABILITY_BLURB
    return None


# ---- helpers ----


def _format_history(turns: list[Turn]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def _sources_payload(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        body = re.sub(r"^\[[^\]]*\]\n", "", chunk.text)  # drop section header line
        sources.append(
            {
                "tag": f"S{i}",
                "filename": str(meta.get("source_filename", "")),
                "page": int(meta.get("page_start", 0)),
                "section": str(meta.get("section_path", "")),
                "snippet": body[:240],
            }
        )
    return sources


def _answer_messages(
    chunks: list[RetrievedChunk], history: list[Turn], message: str
) -> list[dict[str, str]]:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        body = re.sub(r"^\[[^\]]*\]\n", "", chunk.text)
        blocks.append(
            prompts.CONTEXT_BLOCK_TEMPLATE.format(
                tag=f"S{i}",
                filename=meta.get("source_filename", ""),
                page=meta.get("page_start", 0),
                section=meta.get("section_path", ""),
                text=body,
            )
        )
    user = prompts.ANSWER_USER_TEMPLATE.format(
        context_blocks="\n\n".join(blocks),
        history=_format_history(history),
        message=message,
    )
    return [
        {"role": "system", "content": prompts.ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def condense_query(history: list[Turn], message: str) -> str:
    """Rewrite a follow-up into a standalone query via the cheap model.
    3s budget; any failure falls back to the raw message — never blocks chat."""
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            complete(
                "cheap",
                [
                    {"role": "system", "content": prompts.CONDENSE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompts.CONDENSE_USER_TEMPLATE.format(
                            history=_format_history(history), message=message
                        ),
                    },
                ],
                timeout=3.0,
            ),
            timeout=3.5,
        )
        condensed = response_text(response).strip().strip('"').splitlines()[0].strip()
        if not condensed or len(condensed) > 300:
            return message
        metrics.STAGE_LATENCY.labels("condense").observe(time.perf_counter() - started)
        log_stage(
            logger,
            "condensed query",
            original=message,
            condensed=condensed,
            condense_ms=round((time.perf_counter() - started) * 1000),
        )
        return condensed
    except Exception:
        logger.warning("condensation failed, using raw message", exc_info=True)
        return message


# ---- pipeline ----


@dataclass
class PreparedTurn:
    kind: Literal["canned", "refusal", "rag", "cached"]
    answer: str | None = None  # set for canned/refusal/cached
    messages: list[dict[str, str]] = field(default_factory=list)  # set for rag
    sources: list[dict[str, Any]] = field(default_factory=list)
    condensed_query: str | None = None
    # carried so a successful rag answer can be stored in the semantic cache
    query_embedding: list[float] | None = None
    doc_filter: str | None = None


async def prepare_turn(
    index: IndexStore,
    store: SessionStore,
    session_id: str,
    message: str,
    doc_filter: str | None = None,
    cache: SemanticCache | None = None,
) -> PreparedTurn:
    canned = route_small_talk(message)
    if canned is not None:
        log_stage(logger, "small-talk routed", session_id=session_id)
        return PreparedTurn(kind="canned", answer=canned)

    history = await store.window(session_id)
    condensed = await condense_query(history, message) if history else message

    # Semantic cache: the condensed query is standalone, so its embedding is a
    # stable key. Embed once here and reuse the vector for dense retrieval.
    query_embedding: list[float] | None = None
    if cache is not None:
        embed_started = time.perf_counter()
        query_embedding = (await embed_texts([condensed]))[0]
        metrics.STAGE_LATENCY.labels("embed").observe(time.perf_counter() - embed_started)
        hit = await cache.lookup(query_embedding, doc_filter)
        if hit is not None:
            return PreparedTurn(
                kind="cached",
                answer=hit["answer"],
                sources=hit["sources"],
                condensed_query=condensed,
                doc_filter=doc_filter,
            )

    chunks = await retrieve(
        index,
        condensed,
        filenames=[doc_filter] if doc_filter else None,
        query_embedding=query_embedding,
    )
    if not chunks:
        return PreparedTurn(kind="refusal", answer=prompts.REFUSAL_RESPONSE, condensed_query=condensed)
    return PreparedTurn(
        kind="rag",
        messages=_answer_messages(chunks, history, message),
        sources=_sources_payload(chunks),
        condensed_query=condensed,
        query_embedding=query_embedding,
        doc_filter=doc_filter,
    )


# When a client disconnects mid-stream the request scope is already cancelled,
# so any await inside the generator's cleanup can be cancelled too. Salvage
# persistence therefore runs on detached tasks; the set keeps live references
# so they aren't garbage-collected before completing.
_salvage_tasks: set[asyncio.Task[None]] = set()


def _persist_detached(coro: Coroutine[Any, Any, None]) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _salvage_tasks.add(task)
    task.add_done_callback(_salvage_tasks.discard)


async def persist_turn(
    store: SessionStore,
    session_id: str,
    message: str,
    answer: str,
    sources: list[dict[str, Any]],
    condensed_query: str | None,
) -> None:
    await store.append_turn(
        session_id, {"role": "user", "content": message, "sources": [], "condensed_query": condensed_query}
    )
    await store.append_turn(
        session_id, {"role": "assistant", "content": answer, "sources": sources}
    )


async def _store_in_cache(cache: SemanticCache | None, prepared: PreparedTurn, answer: str) -> None:
    """Cache a successful rag answer (never canned/refusal — they cost no LLM)."""
    if cache is None or prepared.kind != "rag" or prepared.query_embedding is None:
        return
    await cache.store(
        prepared.query_embedding,
        prepared.condensed_query or "",
        prepared.doc_filter,
        answer,
        prepared.sources,
    )


async def chat_once(
    index: IndexStore,
    store: SessionStore,
    session_id: str,
    message: str,
    doc_filter: str | None = None,
    cache: SemanticCache | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Non-streaming chat: returns (answer, sources) and persists the turn."""
    started = time.perf_counter()
    prepared = await prepare_turn(index, store, session_id, message, doc_filter, cache)
    if prepared.kind in ("canned", "refusal", "cached"):
        answer = prepared.answer or ""
        await persist_turn(
            store, session_id, message, answer, prepared.sources, prepared.condensed_query
        )
        return answer, prepared.sources

    generate_started = time.perf_counter()
    response = await complete("main", prepared.messages, timeout=45.0)
    metrics.STAGE_LATENCY.labels("generate").observe(time.perf_counter() - generate_started)
    answer = response_text(response)
    await _store_in_cache(cache, prepared, answer)
    await persist_turn(store, session_id, message, answer, prepared.sources, prepared.condensed_query)
    log_stage(
        logger,
        "chat complete",
        session_id=session_id,
        total_ms=round((time.perf_counter() - started) * 1000),
        sources=len(prepared.sources),
    )
    return answer, prepared.sources


async def chat_stream(
    index: IndexStore,
    store: SessionStore,
    session_id: str,
    message: str,
    doc_filter: str | None = None,
    cache: SemanticCache | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming chat: yields {"delta": ...} events then a final
    {"sources": [...], "session_id": ...} event; persists the turn at the end."""
    started = time.perf_counter()
    prepared = await prepare_turn(index, store, session_id, message, doc_filter, cache)

    if prepared.kind in ("canned", "refusal", "cached"):
        answer = prepared.answer or ""
        yield {"delta": answer}
        yield {"sources": prepared.sources, "session_id": session_id}
        await persist_turn(
            store, session_id, message, answer, prepared.sources, prepared.condensed_query
        )
        return

    stream = await complete("main", prepared.messages, stream=True, timeout=45.0)
    collected: list[str] = []
    first_token_ms: int | None = None
    salvaged = False
    try:
        async for delta in stream_deltas(stream):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started) * 1000)
                metrics.STAGE_LATENCY.labels("first_token").observe(first_token_ms / 1000)
            collected.append(delta)
            yield {"delta": delta}
    except QuotaExceededError:
        raise
    except (GeneratorExit, asyncio.CancelledError):
        # Client disconnected mid-answer. Without this, the whole turn (the
        # user's message included) would vanish from history and the next
        # follow-up would condense against a hole in the conversation.
        logger.warning("client disconnected mid-stream, salvaging partial turn")
        _persist_detached(
            persist_turn(
                store,
                session_id,
                message,
                "".join(collected),
                prepared.sources if collected else [],
                prepared.condensed_query,
            )
        )
        raise
    except Exception:
        # Gemini intermittently drops streaming connections mid-body. Salvage:
        # keep what we have, or retry once non-streaming if nothing arrived.
        logger.warning("mid-stream failure, salvaging", exc_info=True)
        if not collected:
            response = await complete("main", prepared.messages, timeout=45.0)
            fallback_answer = response_text(response)
            collected.append(fallback_answer)
            yield {"delta": fallback_answer}
        else:
            salvaged = True  # possibly truncated — never cache it
            yield {"delta": "\n\n_(stream interrupted — answer may be truncated)_"}
    answer = "".join(collected)
    if not salvaged:
        await _store_in_cache(cache, prepared, answer)
    yield {"sources": prepared.sources, "session_id": session_id}
    await persist_turn(store, session_id, message, answer, prepared.sources, prepared.condensed_query)
    log_stage(
        logger,
        "chat stream complete",
        session_id=session_id,
        first_token_ms=first_token_ms,
        total_ms=round((time.perf_counter() - started) * 1000),
        sources=len(prepared.sources),
    )
