"""Hybrid retrieval: dense + BM25 → RRF fusion → cross-encoder rerank → gate.

RRF is rank-based only — raw dense/sparse scores are never mixed (scale
mismatch). The confidence gate returns an empty list when nothing clears
MIN_RERANK_SCORE; callers must take the refusal path instead of calling the
answer LLM with empty context.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from core.config import settings
from core.index import IndexStore
from core.logging import log_stage
from core.models import embed_texts, rerank_pairs, run_in_embed_pool

logger = logging.getLogger("retrieval")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # sigmoid-normalized rerank score in [0, 1]


def rrf_fuse(rankings: list[list[str]], k: int) -> list[str]:
    """Reciprocal-rank fusion over id rankings; deduplicates across lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


async def retrieve(index: IndexStore, query: str) -> list[RetrievedChunk]:
    """Return the top chunks passing the confidence gate (may be empty)."""
    started = time.perf_counter()
    query_embedding = (await embed_texts([query]))[0]
    embed_ms = round((time.perf_counter() - started) * 1000)

    search_started = time.perf_counter()
    dense = await run_in_embed_pool(index.dense_search_sync, query_embedding, settings.dense_top_k)
    sparse = await run_in_embed_pool(index.sparse_search_sync, query, settings.sparse_top_k)
    search_ms = round((time.perf_counter() - search_started) * 1000)

    by_id: dict[str, dict[str, Any]] = {}
    for candidate in dense + sparse:
        by_id.setdefault(candidate["id"], candidate)
    fused_ids = rrf_fuse(
        [[c["id"] for c in dense], [c["id"] for c in sparse]], settings.rrf_k
    )
    candidates = [by_id[chunk_id] for chunk_id in fused_ids[: settings.rerank_candidates]]
    if not candidates:
        log_stage(logger, "retrieval empty", query=query, embed_ms=embed_ms, search_ms=search_ms)
        return []

    rerank_started = time.perf_counter()
    scores = await rerank_pairs(query, [c["text"] for c in candidates])
    rerank_ms = round((time.perf_counter() - rerank_started) * 1000)

    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    kept = [
        RetrievedChunk(id=c["id"], text=c["text"], metadata=c["metadata"], score=s)
        for c, s in scored[: settings.final_top_k]
        if s >= settings.min_rerank_score
    ]
    log_stage(
        logger,
        "retrieval complete",
        query=query,
        dense=len(dense),
        sparse=len(sparse),
        candidates=len(candidates),
        kept=len(kept),
        top_score=round(scored[0][1], 3) if scored else None,
        embed_ms=embed_ms,
        search_ms=search_ms,
        rerank_ms=rerank_ms,
    )
    return kept
