"""Vector retriever using ES 7.10 script_score for cosine similarity."""

import logging

from app.core import encode_texts, get_es_client

logger = logging.getLogger(__name__)

# script_score template: cosineSimilarity + 1.0 ensures positive scores
SCRIPT_SOURCE = "cosineSimilarity(params.query_vector, 'embedding') + 1.0"


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


async def vector_search(
    kb_id: str,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Execute vector search (script_score cosine similarity).

    1. Encode query text to vector
    2. Use script_score brute-force scan in ES
    3. Return hits list

    Note: ES 7.10 does not support native knn, must use script_score.
    """
    # Encode query text
    query_vector = encode_texts([query])[0]

    client = await get_es_client()
    index = _index_name(kb_id)

    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": SCRIPT_SOURCE,
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "size": top_k,
    }

    resp = await client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    logger.debug(
        "Vector search for kb=%s query=%s returned %d results", kb_id, query, len(hits)
    )
    return hits
