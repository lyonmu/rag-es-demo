"""SSE chat endpoint for RAG-based Q&A."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core import get_kb_info
from app.schemas import ChatRequest
from app.services.chat_service import chat_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/chat", tags=["chat"])


@router.post("", response_class=StreamingResponse)
async def chat(kb_id: str, body: ChatRequest):
    """SSE streaming intelligent Q&A."""
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")

    return StreamingResponse(
        chat_stream(kb_id, body.query, top_k=body.top_k),
        media_type="text/event-stream",
    )
