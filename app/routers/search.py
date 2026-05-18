"""Search router for hybrid retrieval."""

import logging

from fastapi import APIRouter, HTTPException

from app.core import get_kb_info
from app.retrievers import hybrid_search
from app.schemas import SearchRequest
from app.schemas.response import SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(kb_id: str, body: SearchRequest):
    """Hybrid search in a knowledge base."""
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")

    results = await hybrid_search(kb_id, body.query, top_k=body.top_k)
    return SearchResponse(results=results, total=len(results))
