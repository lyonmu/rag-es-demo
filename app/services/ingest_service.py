"""Service for ingesting Markdown files into Elasticsearch."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.chunkers import chunk_markdown
from app.core import encode_texts, get_es_client

logger = logging.getLogger(__name__)


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


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
        return {"doc_id": "", "filename": filename, "chunks_count": 0}

    # 2. Encode
    texts = [c.content for c in chunks]
    embeddings = encode_texts(texts)

    # 3. Build bulk operations
    doc_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    index = _index_name(kb_id)

    actions = []
    for chunk, embedding in zip(chunks, embeddings):
        actions.append({"index": {"_index": index, "_id": f"{doc_id}_{chunk.chunk_index}"}})
        actions.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "heading_path": chunk.heading_path,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": embedding,
                "created_at": now,
            }
        )

    # 4. Bulk write
    client = await get_es_client()
    resp = await client.bulk(body=actions, refresh="wait_for")

    if resp.get("errors"):
        failed = [item for item in resp.get("items", []) if "error" in item.get("index", {})]
        logger.error("Bulk indexing errors: %s", failed)

    logger.info("Ingested %d chunks for doc=%s file=%s kb=%s", len(chunks), doc_id, filename, kb_id)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
    }
