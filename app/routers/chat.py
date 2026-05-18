"""SSE chat endpoint for RAG-based Q&A."""

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core import get_kb
from app.core.error_codes import KB_NOT_FOUND
from app.core.response import ApiResponse
from app.schemas import ChatRequest
from app.services.chat_service import chat_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/chat", tags=["chat"])


@router.post("")
async def chat(kb_id: str, body: ChatRequest):
    """SSE streaming intelligent Q&A."""
    # Verify KB exists
    kb = get_kb(kb_id)
    if kb is None:
        return ApiResponse.error(code=KB_NOT_FOUND, message="知识库不存在")

    return StreamingResponse(
        chat_stream(kb_id, body.query, top_k=body.top_k),
        media_type="text/event-stream",
    )
