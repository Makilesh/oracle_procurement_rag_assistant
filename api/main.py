"""FastAPI application entrypoint. Models and stores are loaded once in lifespan."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler

from core import metrics

from api.deps import limiter
from api.routes import auth, chat, documents, evaluate, health, ingest, sessions
from core.bootstrap import full_ingest_data_dir, prepare_index_dir
from core.config import settings
from core.index import IndexStore
from core.logging import setup_logging
from core.models import get_embedder, run_in_embed_pool
from core.sessions import SessionStore

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info("starting api service")
    app.state.sessions = SessionStore.from_settings()
    bootstrap_path = prepare_index_dir()
    app.state.index = IndexStore()
    ingest_task: asyncio.Task[None] | None = None
    if settings.eager_load_models:
        # Load the embedder once at startup (first boot downloads into hf_cache).
        await run_in_embed_pool(get_embedder)
        # Last-resort path: nothing seeded the index (no volume, no prebuilt).
        if bootstrap_path == "full-ingest" or (
            bootstrap_path == "server" and app.state.index.chunk_count() == 0
        ):
            ingest_task = asyncio.create_task(full_ingest_data_dir(app.state.index))
    logger.info("startup complete (bootstrap=%s)", bootstrap_path)
    yield
    if ingest_task is not None and not ingest_task.done():
        ingest_task.cancel()
    logger.info("api service shutting down")


app = FastAPI(
    title="Opkey Procurement RAG Chatbot",
    description="Session-aware RAG chatbot over Oracle Fusion Procurement and "
    "University of Richmond procurement policy documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    # label by route template (e.g. /sessions/{session_id}) to keep cardinality bounded
    route_path = getattr(route, "path", "unmatched")
    metrics.HTTP_REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
    metrics.HTTP_LATENCY.labels(request.method, route_path).observe(time.perf_counter() - started)
    return response


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)

app.include_router(health.router, tags=["health"])
app.include_router(auth.router, tags=["auth"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(chat.router, tags=["chat"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(documents.router, tags=["documents"])
app.include_router(evaluate.router, tags=["evaluate"])
