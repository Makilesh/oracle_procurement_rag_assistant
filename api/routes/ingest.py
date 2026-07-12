import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from api.deps import get_admin_user, limiter
from api.schemas import IngestResponse
from core.config import settings
from core.index import IndexStore
from core.logging import log_stage

logger = logging.getLogger("ingest")

router = APIRouter()

ALLOWED_SUFFIXES = (".pdf", ".txt")


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_document(
    request: Request,
    file: UploadFile,
    user: str = Depends(get_admin_user),
) -> IngestResponse:
    """Admin-only: the knowledge base is shared by every user, and ingesting a
    duplicate filename REPLACES the existing document — a write this global
    shouldn't be open to any chat user (non-admins get 403)."""
    filename = file.filename or "upload"
    if not filename.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(status_code=422, detail="Only PDF and TXT files are supported")
    # Read in 1 MB slices so the size cap rejects an oversized upload without
    # ever buffering it whole. The cap is an abuse guard (multi-GB uploads),
    # NOT a document-size policy — the default 200 MB fits PDFs of tens of
    # thousands of pages (the 670-page Oracle guide is ~3 MB).
    max_bytes = settings.max_upload_mb * 1024 * 1024
    buffer = bytearray()
    while chunk := await file.read(1 << 20):
        buffer += chunk
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_mb} MB upload limit "
                "(configurable via MAX_UPLOAD_MB)",
            )
    data = bytes(buffer)
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    index: IndexStore = request.app.state.index
    started = time.perf_counter()
    try:
        doc_id, chunks_created, pages = await index.ingest(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log_stage(
        logger,
        "ingest complete",
        filename=filename,
        doc_id=doc_id,
        chunks=chunks_created,
        pages=pages,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return IngestResponse(doc_id=doc_id, chunks_created=chunks_created, pages=pages)
