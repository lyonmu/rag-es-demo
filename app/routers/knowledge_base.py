"""Knowledge base management router."""

import logging

from fastapi import APIRouter, HTTPException

from app.core import create_kb_index, list_kb_indices
from app.schemas import CreateKbRequest
from app.schemas.response import KbCreateResponse, KbInfo, KbListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["kb"])


@router.post("", response_model=KbCreateResponse, status_code=201)
async def create_kb(body: CreateKbRequest):
    """Create a new knowledge base."""
    result = await create_kb_index(body.name, description=body.description)
    return KbCreateResponse(
        kb_id=result.kb_id,
        index_name=result.index_name,
        created_at=getattr(result, "created_at", ""),
    )


@router.get("", response_model=KbListResponse)
async def list_kbs():
    """List all knowledge bases."""
    items = await list_kb_indices()
    return KbListResponse(
        knowledge_bases=[
            KbInfo(
                kb_id=item.kb_id,
                index_name=item.index_name,
                name=item.name,
                description=item.description,
                doc_count=item.doc_count,
            )
            for item in items
        ]
    )
