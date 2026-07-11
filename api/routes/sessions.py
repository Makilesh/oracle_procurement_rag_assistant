from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_current_user
from api.schemas import DeletedResponse, HistoryResponse
from core.sessions import SessionStore, scoped_session_id

router = APIRouter()


def _store(request: Request) -> SessionStore:
    return request.app.state.sessions


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def session_history(
    session_id: str,
    request: Request,
    user: str = Depends(get_current_user),
) -> HistoryResponse:
    turns = await _store(request).history(scoped_session_id(user, session_id))
    if turns is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return HistoryResponse(session_id=session_id, turns=turns)


@router.delete("/sessions/{session_id}", response_model=DeletedResponse)
async def delete_session(
    session_id: str,
    request: Request,
    user: str = Depends(get_current_user),
) -> DeletedResponse:
    deleted = await _store(request).delete(scoped_session_id(user, session_id))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return DeletedResponse()
