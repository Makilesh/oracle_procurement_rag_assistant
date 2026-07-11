"""Lazy singletons for the embedding/reranking models plus the thread pools
that keep their synchronous inference off the event loop.

Non-blocking rule: SentenceTransformer.encode(), CrossEncoder.predict() and
BM25 scoring are synchronous — they are only ever called through
run_in_executor on these dedicated pools (embed and rerank are separate so one
request cannot starve the other; BM25 shares the embed pool).
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("models")

EMBED_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
RERANK_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank")


@lru_cache(maxsize=1)
def get_embedder() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info(
        "loaded embedder %s on %s in %.1fs",
        settings.embedding_model,
        settings.embedding_device,
        time.perf_counter() - started,
    )
    return model


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":
    import torch
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    # bge rerankers emit raw unbounded logits by default; sigmoid normalizes
    # scores to [0,1] so MIN_RERANK_SCORE thresholds are meaningful.
    # (sentence-transformers >=4 renamed activation_fct -> activation_fn)
    model = CrossEncoder(
        settings.reranker_model,
        device=settings.embedding_device,
        activation_fn=torch.nn.Sigmoid(),
    )
    logger.info(
        "loaded reranker %s on %s in %.1fs",
        settings.reranker_model,
        settings.embedding_device,
        time.perf_counter() - started,
    )
    return model


@lru_cache(maxsize=1)
def get_tokenizer() -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def count_tokens(text: str) -> int:
    """Real embedder-tokenizer count — never len(text.split())."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    embeddings = get_embedder().encode(
        texts,
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in embeddings]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EMBED_POOL, embed_texts_sync, texts)


async def rerank_pairs(query: str, passages: list[str]) -> list[float]:
    loop = asyncio.get_running_loop()

    def _predict() -> list[float]:
        scores: Any = get_reranker().predict([(query, passage) for passage in passages])
        return [float(score) for score in scores]

    return await loop.run_in_executor(RERANK_POOL, _predict)


async def run_in_embed_pool(fn: Any, *args: Any) -> Any:
    """Run any sync CPU-bound callable (parsing, BM25) on the embed pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EMBED_POOL, fn, *args)
