"""Hybrid retriever with RRF (Reciprocal Rank Fusion) ranking."""

import logging
from dataclasses import dataclass

from app.config import settings
from app.schemas.response import SearchResultItem

from .bm25_retriever import bm25_search
from .vector_retriever import vector_search

logger = logging.getLogger(__name__)


@dataclass
class RankedDoc:
    chunk_id: str
    doc_id: str
    filename: str
    heading_path: str
    content: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0


async def hybrid_search(
    kb_id: str,
    query: str,
    top_k: int | None = None,
) -> list[SearchResultItem]:
    """Two-stage retrieval + RRF ranking fusion.

    1. BM25 and vector search in parallel, each takes pre_k results
    2. Deduplicate by doc_id
    3. RRF score fusion: 1/(k+bm25_rank) + 1/(k+vector_rank)
    4. Take top_k by RRF score descending
    """
    if top_k is None:
        top_k = settings.retriever_top_k
    pre_k = top_k * settings.retriever_pre_multiplier

    # Run both retrievals in parallel
    import asyncio

    bm25_result, vector_result = await asyncio.gather(
        bm25_search(kb_id, query, top_k=pre_k),
        vector_search(kb_id, query, top_k=pre_k),
        return_exceptions=True,
    )

    bm25_hits: list[dict] = []
    vector_hits: list[dict] = []
    failures: list[str] = []

    if isinstance(bm25_result, Exception):
        failures.append(f"bm25: {bm25_result}")
        logger.exception("BM25 retrieval failed for kb=%s query=%s", kb_id, query)
    else:
        bm25_hits = bm25_result

    if isinstance(vector_result, Exception):
        failures.append(f"vector: {vector_result}")
        logger.exception("Vector retrieval failed for kb=%s query=%s", kb_id, query)
    else:
        vector_hits = vector_result

    if failures and not bm25_hits and not vector_hits:
        raise RuntimeError(f"Both retrieval branches failed: {'; '.join(failures)}")

    return _rrf_merge(bm25_hits, vector_hits, k=settings.rrf_k, top_k=top_k)


def _hit_identity(hit: dict) -> str:
    source = hit.get("_source", {})
    return source.get("chunk_id") or hit.get("_id", "")


def _rrf_merge(
    bm25_hits: list[dict],
    vector_hits: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResultItem]:
    """RRF merge two retrieval result lists."""
    docs: dict[str, RankedDoc] = {}

    # Process BM25 results
    for rank, hit in enumerate(bm25_hits, start=1):
        identity = _hit_identity(hit)
        source = hit.get("_source", {})
        if identity not in docs:
            docs[identity] = RankedDoc(
                chunk_id=identity,
                doc_id=source.get("doc_id", identity),
                filename=source.get("filename", ""),
                heading_path=source.get("heading_path", ""),
                content=source.get("content", ""),
                bm25_score=hit.get("_score", 0.0),
                bm25_rank=rank,
            )
        else:
            docs[identity].bm25_rank = rank
            docs[identity].bm25_score = hit.get("_score", 0.0)

    # Process Vector results
    for rank, hit in enumerate(vector_hits, start=1):
        identity = _hit_identity(hit)
        source = hit.get("_source", {})
        if identity not in docs:
            docs[identity] = RankedDoc(
                chunk_id=identity,
                doc_id=source.get("doc_id", identity),
                filename=source.get("filename", ""),
                heading_path=source.get("heading_path", ""),
                content=source.get("content", ""),
                vector_score=hit.get("_score", 0.0) if hit.get("_score") else 0.0,
                vector_rank=rank,
            )
        else:
            docs[identity].vector_rank = rank
            docs[identity].vector_score = hit.get("_score", 0.0) if hit.get("_score") else 0.0

    # Calculate RRF scores
    for doc in docs.values():
        bm25_part = 1.0 / (k + doc.bm25_rank) if doc.bm25_rank else 0.0
        vector_part = 1.0 / (k + doc.vector_rank) if doc.vector_rank else 0.0
        doc.rrf_score = bm25_part + vector_part

    # Sort by RRF score descending, take top_k
    # Secondary tiebreakers: prefer lower vector_rank, then lower bm25_rank
    ranked = sorted(
        docs.values(),
        key=lambda d: (
            -d.rrf_score,
            d.vector_rank if d.vector_rank is not None else float('inf'),
            d.bm25_rank if d.bm25_rank is not None else float('inf'),
        ),
    )[:top_k]

    return [
        SearchResultItem(
            chunk_id=d.chunk_id,
            doc_id=d.doc_id,
            filename=d.filename,
            heading_path=d.heading_path,
            content=d.content,
            bm25_score=d.bm25_score,
            vector_score=d.vector_score,
            rrf_score=d.rrf_score,
        )
        for d in ranked
    ]
