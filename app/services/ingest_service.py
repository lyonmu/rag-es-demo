"""Service for ingesting Markdown files into Elasticsearch."""

import hashlib
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.chunkers import chunk_markdown
from app.config import settings
from app.core import encode_texts, get_es_client

logger = logging.getLogger(__name__)
MAPPING_VERSION = "rag-es-optimization-v1"


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_{chunk_index}"


def _content_with_heading(heading_path: str, content: str) -> str:
    return f"{heading_path}\n{content}" if heading_path else content


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bulk_failures(resp: dict) -> list[str]:
    failures: list[str] = []
    for item in resp.get("items", []):
        index_result = item.get("index", {})
        if "error" in index_result:
            failures.append(index_result.get("_id", ""))
    return [item for item in failures if item]


async def ingest_markdown(kb_id: str, filename: str, content: str) -> dict:
    """Split Markdown into chunks, encode, write to ES.

    Args:
        kb_id: Knowledge base ID
        filename: Uploaded filename
        content: Markdown text content

    Returns:
        {doc_id, filename, chunks_count}
    """
    # 1. Chunk
    chunks = chunk_markdown(content)
    if not chunks:
        return {
            "doc_id": "",
            "filename": filename,
            "chunks_count": 0,
            "indexed_chunks_count": 0,
            "failed_chunks_count": 0,
            "failed_chunk_ids": [],
        }

    # 2. Encode
    texts = [c.content for c in chunks]
    embeddings = encode_texts(texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding/chunk count mismatch: embeddings={len(embeddings)} chunks={len(chunks)}"
        )

    # 3. Build bulk operations
    doc_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    index = _index_name(kb_id)

    actions = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = _chunk_id(doc_id, chunk.chunk_index)
        actions.append({"index": {"_index": index, "_id": chunk_id}})
        actions.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "filename": filename,
                "heading_path": chunk.heading_path,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_with_heading": _content_with_heading(chunk.heading_path, chunk.content),
                "content_length": len(chunk.content),
                "chunk_text_hash": _sha256_text(chunk.content),
                "mapping_version": MAPPING_VERSION,
                "embedding": embedding,
                "created_at": now,
            }
        )

    # 4. Bulk write
    client = await get_es_client()
    resp = await client.bulk(body=actions, refresh=settings.ingest_refresh_policy)

    failed_chunk_ids = _bulk_failures(resp)
    failed_chunks_count = len(failed_chunk_ids)
    indexed_chunks_count = len(chunks) - failed_chunks_count

    if failed_chunks_count:
        logger.error("Bulk indexing failed for chunks: %s", failed_chunk_ids)

    if indexed_chunks_count == 0 and chunks:
        raise RuntimeError("All chunks failed during bulk indexing")

    logger.info("Ingested %d chunks for doc=%s file=%s kb=%s", len(chunks), doc_id, filename, kb_id)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
        "indexed_chunks_count": indexed_chunks_count,
        "failed_chunks_count": failed_chunks_count,
        "failed_chunk_ids": failed_chunk_ids,
    }
