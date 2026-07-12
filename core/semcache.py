"""Semantic answer cache: condensed-query embedding → answer + sources.

Keyed on the CONDENSED query (already rewritten into standalone form, so
session context is baked in) plus the active document filter — a filtered
answer never serves an unfiltered question. Entries live in Redis under the
current kb_version, which ingest/delete bumps, so any knowledge-base change
instantly orphans the whole cache (TTL reclaims the old key). Lookup is a
linear cosine scan — trivial at <= SEMCACHE_MAX_ENTRIES, and embeddings are
already L2-normalized so dot product == cosine similarity.
"""

import json
import logging
import time
import uuid
from typing import Any

from redis.asyncio import Redis, from_url

from core import metrics
from core.config import settings
from core.logging import log_stage

logger = logging.getLogger("semcache")

KB_VERSION_KEY = "kb_version"


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class SemanticCache:
    """Redis failures degrade to cache-off behavior — never block a chat."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @classmethod
    def from_settings(cls) -> "SemanticCache":
        return cls(from_url(settings.redis_url, decode_responses=True))

    async def _key(self) -> str:
        version = await self._redis.get(KB_VERSION_KEY) or "0"
        return f"semcache:{version}"

    async def bump_kb_version(self) -> None:
        """Called after any ingest/delete: orphans every cached answer."""
        try:
            await self._redis.incr(KB_VERSION_KEY)
        except Exception:
            logger.warning("could not bump kb_version; semcache may serve stale answers")

    async def lookup(
        self, embedding: list[float], doc_filter: str | None
    ) -> dict[str, Any] | None:
        if not settings.semcache_enabled:
            return None
        try:
            entries = await self._redis.hgetall(await self._key())
        except Exception:
            logger.warning("semcache lookup skipped (redis unavailable)")
            return None
        best: dict[str, Any] | None = None
        best_sim = 0.0
        for raw in entries.values():
            entry = json.loads(raw)
            if entry.get("doc_filter") != doc_filter:
                continue
            sim = _dot(embedding, entry["embedding"])
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= settings.semcache_threshold:
            metrics.SEMCACHE_LOOKUPS.labels("hit").inc()
            log_stage(
                logger,
                "semcache hit",
                similarity=round(best_sim, 4),
                cached_query=best.get("query", ""),
            )
            return {"answer": best["answer"], "sources": best["sources"]}
        metrics.SEMCACHE_LOOKUPS.labels("miss").inc()
        return None

    async def store(
        self,
        embedding: list[float],
        query: str,
        doc_filter: str | None,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> None:
        if not settings.semcache_enabled:
            return
        try:
            key = await self._key()
            entry = json.dumps(
                {
                    "embedding": embedding,
                    "query": query,
                    "doc_filter": doc_filter,
                    "answer": answer,
                    "sources": sources,
                    "ts": time.time(),
                },
                ensure_ascii=False,
            )
            await self._redis.hset(key, uuid.uuid4().hex[:12], entry)
            await self._redis.expire(key, settings.semcache_ttl_hours * 3600)
            if await self._redis.hlen(key) > settings.semcache_max_entries:
                entries = await self._redis.hgetall(key)
                oldest = min(entries, key=lambda f: json.loads(entries[f]).get("ts", 0.0))
                await self._redis.hdel(key, oldest)
        except Exception:
            logger.warning("semcache store skipped (redis unavailable)", exc_info=True)
