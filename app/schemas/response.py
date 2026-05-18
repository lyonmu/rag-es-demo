"""Response body and SSE event models."""

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    doc_id: str
    filename: str
    heading_path: str
    content: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


class KbInfo(BaseModel):
    kb_id: str
    name: str
    description: str
    index_name: str
    doc_count: int = 0
    created_at: str = ""


class KbCreateResponse(BaseModel):
    kb_id: str
    index_name: str
    created_at: str = ""


class KbListResponse(BaseModel):
    knowledge_bases: list[KbInfo]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int


class SourceDoc(BaseModel):
    """引用文档，用于 SSE events 和问答响应。"""
    doc_id: str
    filename: str
    heading_path: str
    content: str
    score: float


class ChatDoneData(BaseModel):
    answer: str
    sources: list[SourceDoc]


class ChatErrorData(BaseModel):
    message: str
