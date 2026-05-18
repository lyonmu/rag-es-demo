"""Search router for hybrid retrieval."""

import logging

from fastapi import APIRouter

from app.core import get_kb, get_error_message
from app.core.error_codes import KB_NOT_FOUND, SEARCH_ERROR
from app.core.response import ApiResponse
from app.retrievers import hybrid_search
from app.schemas import SearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/search", tags=["search"])


@router.post("")
async def search(kb_id: str, body: SearchRequest):
    """Hybrid search in a knowledge base."""
    # Verify KB exists
    kb = get_kb(kb_id)
    if kb is None:
        return ApiResponse.error(code=KB_NOT_FOUND, message="知识库不存在")

    try:
        results = await hybrid_search(kb_id, body.query, top_k=body.top_k)
        return ApiResponse.success(
            data={
                "results": [item.model_dump() for item in results],
                "total": len(results),
            }
        )
    except Exception as e:
        logger.exception("Search failed for kb=%s query=%s", kb_id, body.query)
        return ApiResponse.error(code=SEARCH_ERROR, message=f"检索失败: {e}")
