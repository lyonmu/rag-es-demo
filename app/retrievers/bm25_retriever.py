"""BM25 text retriever for Elasticsearch."""

import logging

from app.core import get_es_client

logger = logging.getLogger(__name__)


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


async def bm25_search(
    kb_id: str,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Execute BM25 text search.

    Returns raw ES hits list, each hit contains _source and _score.
    """
    client = await get_es_client()
    index = _index_name(kb_id)

    body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or",
                }
            }
        },
        "size": top_k,
    }

    resp = await client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    logger.debug("BM25 search for kb=%s query=%s returned %d results", kb_id, query, len(hits))
    return hits
