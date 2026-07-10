from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness check plus index counts. No auth required."""
    index = getattr(request.app.state, "index", None)
    if index is None:
        return HealthResponse(status="ok", docs_indexed=0, chunks=0)
    return HealthResponse(status="ok", docs_indexed=index.doc_count(), chunks=index.chunk_count())
