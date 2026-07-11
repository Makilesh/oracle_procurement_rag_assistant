"""FastAPI application entrypoint. Models and stores are loaded once in lifespan."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler

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
        if bootstrap_path == "full-ingest":
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

app.include_router(health.router, tags=["health"])
app.include_router(auth.router, tags=["auth"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(chat.router, tags=["chat"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(documents.router, tags=["documents"])
app.include_router(evaluate.router, tags=["evaluate"])
