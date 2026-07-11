from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_admin_user, get_current_user, is_admin
from api.schemas import DeletedResponse, HistoryResponse, SessionInfo, SessionsListResponse
from core.sessions import SessionStore, scoped_session_id

router = APIRouter()


def _store(request: Request) -> SessionStore:
    return request.app.state.sessions


@router.get("/sessions", response_model=SessionsListResponse)
async def list_sessions(
    request: Request, admin: str = Depends(get_admin_user)
) -> SessionsListResponse:
    """Admin-only: every stored conversation across all users (plus legacy
    unscoped sessions), newest activity first. Non-admins get 403."""
    sessions = await _store(request).list_sessions()
    return SessionsListResponse(sessions=[SessionInfo(**s) for s in sessions])


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def session_history(
    session_id: str,
    request: Request,
    user: str = Depends(get_current_user),
) -> HistoryResponse:
    store = _store(request)
    turns = await store.history(scoped_session_id(user, session_id))
    if turns is None and is_admin(user):
        # Admin fallthrough: read any session by its exact storage key
        # (the `key` field returned by GET /sessions), e.g. "alice:abc" or a
        # legacy unscoped id.
        turns = await store.history(session_id)
    if turns is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return HistoryResponse(session_id=session_id, turns=turns)


@router.delete("/sessions/{session_id}", response_model=DeletedResponse)
async def delete_session(
    session_id: str,
    request: Request,
    user: str = Depends(get_current_user),
) -> DeletedResponse:
    store = _store(request)
    deleted = await store.delete(scoped_session_id(user, session_id))
    if not deleted and is_admin(user):
        deleted = await store.delete(session_id)  # admin fallthrough, as above
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return DeletedResponse()
