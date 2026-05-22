"""Elasticsearch async client singleton with lifecycle management."""

import logging
from uuid import uuid4

from elasticsearch import AsyncElasticsearch

from app.config import settings
from app.schemas.response import KbInfo

logger = logging.getLogger(__name__)

# Global ES client instance
_es_client: AsyncElasticsearch | None = None

# ES index mapping template
KB_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "heading_path": {"type": "text", "analyzer": "ik_max_word"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text", "analyzer": "ik_max_word"},
            "content_with_heading": {"type": "text", "analyzer": "ik_max_word"},
            "content_length": {"type": "integer"},
            "chunk_text_hash": {"type": "keyword"},
            "mapping_version": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 512},
            "created_at": {"type": "date"},
        }
    }
}

INDEX_PREFIX = "rag_kb_"


def _index_name(kb_id: str) -> str:
    return f"{INDEX_PREFIX}{kb_id}"


def _kb_id_from_index(index_name: str) -> str:
    return index_name.removeprefix(INDEX_PREFIX)


async def init_es_client() -> AsyncElasticsearch:
    """Initialize ES connection and perform health check. Called on startup."""
    global _es_client
    if _es_client is not None:
        return _es_client

    _es_client = AsyncElasticsearch(
        hosts=[settings.es_url],
        http_auth=(settings.es_user, settings.es_password),
        timeout=10,
    )

    # Health check: use info() instead of ping() for ES 7.10 compatibility
    # ping() in elasticsearch-py v8 can return False despite HEAD 200 against ES 7.x
    try:
        info = await _es_client.info()
        es_version = info.get("version", {}).get("number", "unknown")
        logger.info("Connected to Elasticsearch %s at %s", es_version, settings.es_url)
    except Exception as e:
        logger.error("Cannot connect to Elasticsearch at %s: %s", settings.es_url, e)
        await _es_client.close()
        _es_client = None
        raise ConnectionError(f"Cannot connect to Elasticsearch at {settings.es_url}: {e}")

    logger.info("Connected to Elasticsearch at %s", settings.es_url)
    return _es_client


async def get_es_client() -> AsyncElasticsearch:
    """Get the ES client instance."""
    if _es_client is None:
        raise ConnectionError("Elasticsearch client not initialized. Call init_es_client() first.")
    return _es_client


async def close_es_client() -> None:
    """Close ES connection."""
    global _es_client
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
        logger.info("Elasticsearch connection closed")


async def create_kb_index(name: str, description: str) -> KbInfo:
    """Create a knowledge base index. Returns KbInfo."""
    client = await get_es_client()
    kb_id = uuid4().hex
    index = _index_name(kb_id)

    await client.indices.create(
        index=index,
        body=KB_INDEX_MAPPING,
    )

    return KbInfo(
        kb_id=kb_id,
        name=name,
        description=description,
        index_name=index,
        doc_count=0,
    )


async def list_kb_indices() -> list[KbInfo]:
    """List all rag_kb_* indices."""
    client = await get_es_client()
    indices = await client.cat.indices(
        index=f"{INDEX_PREFIX}*",
        format="json",
        h="index,docs.count",
    )

    result = []
    for idx in indices:
        idx_name = idx.get("index", "")
        doc_count = int(idx.get("docs.count", 0) or 0)
        result.append(
            KbInfo(
                kb_id=_kb_id_from_index(idx_name),
                name="",
                description="",
                index_name=idx_name,
                doc_count=doc_count,
            )
        )
    return result


async def get_kb_info(kb_id: str) -> KbInfo | None:
    """Get info for a single knowledge base. Returns None if not found."""
    client = await get_es_client()
    index = _index_name(kb_id)

    exists = await client.indices.exists(index=index)
    if not exists:
        return None

    stats = await client.cat.indices(index=index, format="json", h="index,docs.count")
    idx = stats[0] if stats else {}
    doc_count = int(idx.get("docs.count", 0) or 0)

    return KbInfo(
        kb_id=kb_id,
        name="",
        description="",
        index_name=index,
        doc_count=doc_count,
    )


async def delete_kb_index(kb_id: str) -> bool:
    """Delete a knowledge base index. Returns True on success."""
    client = await get_es_client()
    index = _index_name(kb_id)

    exists = await client.indices.exists(index=index)
    if not exists:
        return False

    await client.indices.delete(index=index)
    return True
