from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_admin_user, get_current_user
from api.schemas import DeletedResponse, DocumentInfo, DocumentsResponse
from core.index import IndexStore

router = APIRouter()


def _index(request: Request) -> IndexStore:
    return request.app.state.index


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    request: Request, user: str = Depends(get_current_user)
) -> DocumentsResponse:
    docs = _index(request).list_docs()
    return DocumentsResponse(documents=[DocumentInfo(**doc) for doc in docs])


@router.delete("/documents/{doc_id}", response_model=DeletedResponse)
async def delete_document(
    doc_id: str, request: Request, user: str = Depends(get_admin_user)
) -> DeletedResponse:
    """Admin-only: deleting a document destroys shared retrieval context for
    every user at once (non-admins get 403). Listing stays open to all users."""
    deleted = await _index(request).delete_doc(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    return DeletedResponse()
