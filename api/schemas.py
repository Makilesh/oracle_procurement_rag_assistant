"""Pydantic request/response models for every endpoint."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    docs_indexed: int = 0
    chunks: int = 0


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    pages: int


class Source(BaseModel):
    tag: str
    filename: str
    page: int
    section: str = ""
    snippet: str = ""


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


class Turn(BaseModel):
    role: str
    content: str
    sources: list[Source] = []
    condensed_query: str | None = None
    ts: float


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[Turn]


class SessionInfo(BaseModel):
    owner: str | None = Field(None, description="JWT subject; None for legacy unscoped sessions")
    session_id: str
    key: str = Field(description="Exact storage id — admins can pass it to /sessions/{key}/history")
    turns: int
    created_at: float | None = None
    updated_at: float | None = None


class SessionsListResponse(BaseModel):
    sessions: list[SessionInfo]


class DeletedResponse(BaseModel):
    deleted: bool = True


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    ingested_at: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class EvalQuestionResult(BaseModel):
    id: str
    question: str
    hit: bool | None = None
    answer_relevance: float | None = None
    faithfulness: float | None = None
    keyword_coverage: float | None = None
    notes: str = ""


class EvalResponse(BaseModel):
    hit_rate: float
    answer_relevance: float
    faithfulness: float
    keyword_coverage: float
    llm_calls: int
    per_question: list[EvalQuestionResult]
    extra: dict[str, Any] = {}
