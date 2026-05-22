"""Tests for RRF hybrid retriever (pure logic, no ES required)."""

import pytest

from app.retrievers.hybrid_retriever import hybrid_search
from app.retrievers.hybrid_retriever import _rrf_merge


def _make_hit(doc_id: str, content: str, score: float = 1.0) -> dict:
    """Helper: construct ES hit object."""
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "content": content,
            "filename": f"{doc_id}.md",
            "heading_path": f"Heading for {doc_id}",
            "chunk_index": 0,
        },
    }


def test_rrf_basic_merge():
    """Basic RRF merge: two retrievals with overlapping docs."""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0), _make_hit("doc_c", "C", 3.0)]
    vector = [_make_hit("doc_b", "B", 0.9), _make_hit("doc_a", "A", 0.8), _make_hit("doc_d", "D", 0.7)]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 4
    # doc_b ranks 1st or 2nd in both, should have highest RRF
    assert results[0].doc_id == "doc_b"
    # doc_a also in both, should be second
    assert results[1].doc_id == "doc_a"


def test_rrf_non_overlapping():
    """Two retrievals with completely non-overlapping docs."""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0)]
    vector = [_make_hit("doc_c", "C", 0.9), _make_hit("doc_d", "D", 0.8)]

    results = _rrf_merge(bm25, vector, k=60, top_k=4)

    assert len(results) == 4
    # doc_a (bm25 rank 1) vs doc_c (vector rank 1): same RRF score
    assert results[0].rrf_score == results[1].rrf_score


def test_rrf_only_one_source():
    """Only one retrieval source has results."""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0)]
    vector: list[dict] = []

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 2
    assert results[0].doc_id == "doc_a"
    assert results[0].bm25_score == 5.0
    assert results[0].vector_score == 0.0


def test_rrf_top_k_limit():
    """top_k limit is enforced."""
    bm25 = [_make_hit(f"doc_{i}", f"C{i}", float(10 - i)) for i in range(10)]
    vector: list[dict] = []

    results = _rrf_merge(bm25, vector, k=60, top_k=3)

    assert len(results) == 3
    assert results[0].doc_id == "doc_0"


def test_rrf_score_formula():
    """Verify RRF score formula: doc rank=1 in both, k=60."""
    bm25 = [_make_hit("doc_x", "X", 5.0)]
    vector = [_make_hit("doc_x", "X", 0.9)]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(results[0].rrf_score - expected) < 1e-10


def test_rrf_prefers_chunk_id_over_es_id():
    bm25 = [
        {
            "_id": "legacy_id_a",
            "_score": 5.0,
            "_source": {
                "chunk_id": "chunk_a",
                "doc_id": "doc_a",
                "content": "A",
                "filename": "a.md",
                "heading_path": "A",
            },
        }
    ]
    vector = [
        {
            "_id": "different_es_id",
            "_score": 0.9,
            "_source": {
                "chunk_id": "chunk_a",
                "doc_id": "doc_a",
                "content": "A",
                "filename": "a.md",
                "heading_path": "A",
            },
        }
    ]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 1
    assert results[0].doc_id == "doc_a"
    assert results[0].bm25_score == 5.0
    assert results[0].vector_score == 0.9


@pytest.mark.asyncio
async def test_hybrid_search_returns_vector_results_when_bm25_fails(monkeypatch):
    async def failing_bm25(*args, **kwargs):
        raise RuntimeError("bm25 down")

    async def working_vector(*args, **kwargs):
        return [_make_hit("doc_v", "V", 0.9)]

    monkeypatch.setattr("app.retrievers.hybrid_retriever.bm25_search", failing_bm25)
    monkeypatch.setattr("app.retrievers.hybrid_retriever.vector_search", working_vector)

    results = await hybrid_search("kb1", "query", top_k=5)

    assert len(results) == 1
    assert results[0].doc_id == "doc_v"
    assert results[0].vector_score == 0.9


@pytest.mark.asyncio
async def test_hybrid_search_raises_when_both_branches_fail(monkeypatch):
    async def failing_bm25(*args, **kwargs):
        raise RuntimeError("bm25 down")

    async def failing_vector(*args, **kwargs):
        raise RuntimeError("vector down")

    monkeypatch.setattr("app.retrievers.hybrid_retriever.bm25_search", failing_bm25)
    monkeypatch.setattr("app.retrievers.hybrid_retriever.vector_search", failing_vector)

    with pytest.raises(RuntimeError, match="Both retrieval branches failed"):
        await hybrid_search("kb1", "query", top_k=5)
