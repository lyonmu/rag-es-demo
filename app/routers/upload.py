"""File upload router."""

import logging
import os

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/upload", tags=["upload"])


@router.post("")
async def upload_file(kb_id: str, file: UploadFile):
    """Upload a markdown file to a knowledge base."""
    if not file.filename.endswith(".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Only .md files are supported, got '{os.path.splitext(file.filename)[1]}'",
        )
    # Ingestion handled in Task 6 (ingest_service)
    logger.info("Upload received for kb=%s file=%s", kb_id, file.filename)
    return {"filename": file.filename, "status": "received"}
