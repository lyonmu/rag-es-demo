"""File upload router — now wired to ingest_markdown service."""

import logging
import os

from fastapi import APIRouter, UploadFile

from app.config import settings
from app.core import get_error_message
from app.core.error_codes import KB_NOT_FOUND, UNSUPPORTED_FILE_TYPE
from app.core.kb_store import get_kb, update_doc_count
from app.core.response import ApiResponse
from app.services.ingest_service import ingest_markdown

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/upload", tags=["upload"])


@router.post("")
async def upload_file(kb_id: str, file: UploadFile):
    """Upload a markdown/txt file to a knowledge base and ingest it."""
    # 1. Verify KB exists
    kb = get_kb(kb_id)
    if kb is None:
        return ApiResponse.error(code=KB_NOT_FOUND, message="知识库不存在")

    # 2. Validate file type
    if file.filename is None:
        return ApiResponse.error(code=UNSUPPORTED_FILE_TYPE, message="文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".md", ".txt"):
        return ApiResponse.error(
            code=UNSUPPORTED_FILE_TYPE,
            message=get_error_message(UNSUPPORTED_FILE_TYPE),
        )

    # 3. Read file content
    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="replace")

    if not content.strip():
        return ApiResponse.error(code=UNSUPPORTED_FILE_TYPE, message="文件内容为空")

    # 4. Ingest into ES
    try:
        result = await ingest_markdown(kb_id, file.filename, content)
    except Exception as e:
        logger.exception("Ingest failed for kb=%s file=%s", kb_id, file.filename)
        from app.core.error_codes import ES_OPERATION_ERROR

        return ApiResponse.error(code=ES_OPERATION_ERROR, message=f"文档入库失败: {e}")

    # 5. Update doc_count in SQLite
    if result.get("indexed_chunks_count", result.get("chunks_count", 0)) > 0:
        update_doc_count(kb_id, increment=1)

    return ApiResponse.success(data=result)
