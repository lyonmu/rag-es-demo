"""Knowledge base management router."""

import logging

from fastapi import APIRouter

from app.core import (
    create_kb,
    create_kb_index,
    delete_kb,
    delete_kb_index,
    get_kb,
    get_kb_info,
    list_kbs,
)
from app.core.error_codes import (
    ES_CREATE_INDEX_ERROR,
    KB_CREATE_ERROR,
    KB_DELETE_ERROR,
    KB_NOT_FOUND,
)
from app.core.response import ApiResponse
from app.schemas import CreateKbRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["kb"])


@router.post("")
async def create_kb_endpoint(body: CreateKbRequest):
    """Create a new knowledge base. Writes to SQLite then ES."""
    try:
        # 1. Create ES index first
        es_info = await create_kb_index(body.name, description=body.description)

        # 2. Store metadata in SQLite
        kb_record = create_kb(
            kb_id=es_info.kb_id,
            name=body.name,
            description=body.description,
            index_name=es_info.index_name,
        )

        return ApiResponse.success(
            data={
                "kb_id": kb_record["kb_id"],
                "name": kb_record["name"],
                "description": kb_record["description"],
                "index_name": kb_record["index_name"],
                "doc_count": kb_record["doc_count"],
                "created_at": kb_record["created_at"],
            },
        )
    except ConnectionError as e:
        logger.error("Failed to create ES index: %s", e)
        return ApiResponse.error(code=ES_CREATE_INDEX_ERROR, message=f"ES 创建索引失败: {e}")
    except Exception as e:
        logger.exception("Failed to create KB: %s", e)
        return ApiResponse.error(code=KB_CREATE_ERROR, message=f"知识库创建失败: {e}")


@router.get("")
async def list_kbs_endpoint():
    """List all knowledge bases. Reads name/description from SQLite, doc_count from ES."""
    try:
        sqlite_kbs = list_kbs()

        result = []
        for kb in sqlite_kbs:
            # Refresh doc_count from ES
            es_info = await get_kb_info(kb["kb_id"])
            if es_info is not None:
                if es_info.doc_count != kb["doc_count"]:
                    from app.core.kb_store import set_doc_count

                    set_doc_count(kb["kb_id"], es_info.doc_count)
                    kb["doc_count"] = es_info.doc_count

            result.append(
                {
                    "kb_id": kb["kb_id"],
                    "name": kb["name"],
                    "description": kb["description"],
                    "index_name": kb["index_name"],
                    "doc_count": kb["doc_count"],
                    "created_at": kb["created_at"],
                }
            )

        return ApiResponse.success(data={"knowledge_bases": result})
    except Exception as e:
        logger.exception("Failed to list KBs: %s", e)
        return ApiResponse.error(code=KB_NOT_FOUND, message=f"知识库列表获取失败: {e}")


@router.get("/{kb_id}")
async def get_kb_endpoint(kb_id: str):
    """Get a single knowledge base by ID."""
    kb = get_kb(kb_id)
    if kb is None:
        return ApiResponse.error(code=KB_NOT_FOUND, message="知识库不存在")

    # Refresh doc_count from ES
    es_info = await get_kb_info(kb_id)
    if es_info is not None and es_info.doc_count != kb["doc_count"]:
        from app.core.kb_store import set_doc_count

        set_doc_count(kb_id, es_info.doc_count)
        kb["doc_count"] = es_info.doc_count

    return ApiResponse.success(data=kb)


@router.delete("/{kb_id}")
async def delete_kb_endpoint(kb_id: str):
    """Delete a knowledge base. Removes from SQLite and ES."""
    kb = get_kb(kb_id)
    if kb is None:
        return ApiResponse.error(code=KB_NOT_FOUND, message="知识库不存在")

    try:
        # 1. Delete ES index
        deleted = await delete_kb_index(kb_id)
        # 2. Delete SQLite record
        delete_kb(kb_id)
        return ApiResponse.success(data={"kb_id": kb_id, "deleted": True})
    except Exception as e:
        logger.exception("Failed to delete KB: %s", e)
        return ApiResponse.error(code=KB_DELETE_ERROR, message=f"知识库删除失败: {e}")
