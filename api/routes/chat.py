import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.deps import get_current_user, limiter
from api.schemas import ChatRequest, ChatResponse, Source
from core.chat_pipeline import chat_once, chat_stream
from core.config import settings
from core.llm import QuotaExceededError

logger = logging.getLogger("chat.route")

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    user: str = Depends(get_current_user),
):
    index = request.app.state.index
    store = request.app.state.sessions

    if body.stream:
        async def event_stream() -> AsyncIterator[str]:
            try:
                async for event in chat_stream(index, store, body.session_id, body.message):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except QuotaExceededError as exc:
                yield f"data: {json.dumps({'error': str(exc), 'status': 503})}\n\n"
            except Exception:
                logger.exception("stream failed")
                yield f"data: {json.dumps({'error': 'internal error', 'status': 500})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        answer, sources = await chat_once(index, store, body.session_id, body.message)
    except QuotaExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return ChatResponse(
        answer=answer,
        sources=[Source(**source) for source in sources],
        session_id=body.session_id,
    )
