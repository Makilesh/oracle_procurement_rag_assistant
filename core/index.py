"""Persistent knowledge-base index: Chroma (dense) + BM25 (sparse) + doc registry.

Chroma is the single source of truth. It runs either as a dedicated service
(CHROMA_HOST set — the docker-compose/production shape) or embedded in-process
(local dev, unit tests). Everything the api keeps locally is a derived cache
that can be rebuilt from Chroma alone: the BM25 index (re-pickled on every
ingest/delete) and the docs registry (docs.json; recovered by scanning chunk
metadata if the cache is missing) — so a fresh api instance needs nothing but
Chroma to come up.

Single-instance constraint: these caches are refreshed only by the process
that served the ingest/delete, and the ingest lock is per-process. Running
multiple api replicas would serve stale sparse results after an ingest until
restart — add cross-replica invalidation (e.g. Redis pub/sub) before scaling
out.
"""

import asyncio
import json
import logging
import pickle
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from core.config import settings
from core.ingestion import parse_and_chunk
from core.models import embed_texts, run_in_embed_pool

logger = logging.getLogger("index")

_WORD_RE = re.compile(r"\w+")


def _bm25_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _make_client() -> Any:
    """Server mode when CHROMA_HOST is set (with connection retries so the api
    tolerates the chroma service still booting), embedded mode otherwise."""
    if settings.chroma_host:
        last_exc: Exception | None = None
        for attempt in range(20):
            try:
                client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
                client.heartbeat()
                logger.info(
                    "connected to chroma service at %s:%d",
                    settings.chroma_host,
                    settings.chroma_port,
                )
                return client
            except Exception as exc:
                last_exc = exc
                logger.info("waiting for chroma service (attempt %d/20)...", attempt + 1)
                time.sleep(3)
        raise RuntimeError(f"chroma service unreachable: {last_exc}")
    chroma_dir = Path(settings.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


class IndexStore:
    def __init__(self) -> None:
        state_dir = Path(settings.chroma_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self._client = _make_client()
        self._collection = self._client.get_or_create_collection(
            "knowledge_base", metadata={"hnsw:space": "cosine"}
        )
        self._registry_path = state_dir / "docs.json"
        self._registry: dict[str, dict[str, Any]] = self._load_registry()
        if not self._registry and self._collection.count() > 0:
            self._recover_registry()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._lock = asyncio.Lock()
        self._load_bm25()
        logger.info(
            "index ready (%s): %d docs, %d chunks, bm25=%s",
            "server" if settings.chroma_host else "embedded",
            self.doc_count(),
            self.chunk_count(),
            self._bm25 is not None,
        )

    # ---------- registry ----------

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        return {}

    def _recover_registry(self) -> None:
        """Rebuild the docs registry by scanning chunk metadata in Chroma —
        the registry is only a cache; Chroma stays the source of truth."""
        data = self._collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in data["metadatas"] or []:
            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue
            entry = docs.setdefault(
                doc_id,
                {
                    "filename": str(meta.get("source_filename", "unknown")),
                    "pages": 0,
                    "chunks": 0,
                    "ingested_at": datetime.now(UTC).isoformat() + " (recovered)",
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], int(meta.get("page_end", 0)))
        self._registry = docs
        self._save_registry()
        logger.info("registry recovered from chroma metadata: %d docs", len(docs))

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def doc_count(self) -> int:
        return len(self._registry)

    def chunk_count(self) -> int:
        return self._collection.count()

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self._registry

    def list_docs(self) -> list[dict[str, Any]]:
        return [
            {"doc_id": doc_id, **meta}
            for doc_id, meta in sorted(
                self._registry.items(), key=lambda item: item[1]["ingested_at"]
            )
        ]

    # ---------- BM25 ----------

    def _load_bm25(self) -> None:
        path = Path(settings.bm25_path)
        if not path.exists():
            self._rebuild_bm25_sync()
            return
        try:
            payload = pickle.loads(path.read_bytes())
            self._bm25_ids = payload["ids"]
            self._bm25 = BM25Okapi(payload["tokenized"]) if payload["tokenized"] else None
        except Exception:
            logger.warning("bm25 pickle unreadable, rebuilding", exc_info=True)
            self._rebuild_bm25_sync()

    def _rebuild_bm25_sync(self) -> None:
        started = time.perf_counter()
        data = self._collection.get(include=["documents"])
        ids, documents = data["ids"], data["documents"] or []
        tokenized = [_bm25_tokenize(text) for text in documents]
        self._bm25_ids = ids
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        path = Path(settings.bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"ids": ids, "tokenized": tokenized}))
        logger.info(
            "bm25 rebuilt over %d chunks in %.2fs", len(ids), time.perf_counter() - started
        )

    # ---------- ingest / delete ----------

    async def ingest(self, filename: str, data: bytes) -> tuple[str, int, int]:
        """Parse, chunk, embed and index one uploaded file.
        Duplicate filename replaces the previous document's chunks."""
        async with self._lock:
            chunks, pages = await run_in_embed_pool(parse_and_chunk, filename, data)
            if not chunks:
                raise ValueError(f"No extractable text in {filename}")

            for doc_id, meta in list(self._registry.items()):
                if meta["filename"] == filename:
                    logger.info("replacing existing doc %s (%s)", doc_id, filename)
                    await self._delete_unlocked(doc_id, rebuild=False)

            doc_id = uuid.uuid4().hex[:12]
            embeddings = await embed_texts([chunk.text for chunk in chunks])
            self._collection.upsert(
                ids=[f"{doc_id}:{chunk.chunk_index}" for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "source_filename": filename,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_index": chunk.chunk_index,
                        "section_path": chunk.section_path,
                    }
                    for chunk in chunks
                ],
            )
            self._registry[doc_id] = {
                "filename": filename,
                "pages": pages,
                "chunks": len(chunks),
                "ingested_at": datetime.now(UTC).isoformat(),
            }
            self._save_registry()
            await run_in_embed_pool(self._rebuild_bm25_sync)
            return doc_id, len(chunks), pages

    async def _delete_unlocked(self, doc_id: str, rebuild: bool = True) -> bool:
        if doc_id not in self._registry:
            return False
        self._collection.delete(where={"doc_id": doc_id})
        del self._registry[doc_id]
        self._save_registry()
        if rebuild:
            await run_in_embed_pool(self._rebuild_bm25_sync)
        return True

    async def delete_doc(self, doc_id: str) -> bool:
        async with self._lock:
            return await self._delete_unlocked(doc_id)

    # ---------- search primitives (used by core.retrieval) ----------

    def dense_search_sync(self, query_embedding: list[float], k: int) -> list[dict[str, Any]]:
        if self.chunk_count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.chunk_count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]

    def sparse_search_sync(self, query: str, k: int) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._bm25_ids:
            return []
        scores = self._bm25.get_scores(_bm25_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_ids = [self._bm25_ids[i] for i in ranked if scores[i] > 0]
        if not top_ids:
            return []
        data = self._collection.get(ids=top_ids, include=["documents", "metadatas"])
        by_id = {
            data["ids"][i]: {
                "id": data["ids"][i],
                "text": (data["documents"] or [])[i],
                "metadata": (data["metadatas"] or [])[i],
            }
            for i in range(len(data["ids"]))
        }
        return [by_id[chunk_id] for chunk_id in top_ids if chunk_id in by_id]
