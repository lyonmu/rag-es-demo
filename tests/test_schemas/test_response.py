"""Tests for response schemas."""

from app.schemas.response import SearchResultItem, UploadResponse


def test_search_result_item_accepts_chunk_id():
    item = SearchResultItem(
        chunk_id="chunk1",
        doc_id="doc1",
        filename="a.md",
        heading_path="A",
        content="content",
        bm25_score=1.0,
        vector_score=2.0,
        rrf_score=0.3,
    )

    assert item.chunk_id == "chunk1"


def test_upload_response_accepts_partial_failure_counts():
    response = UploadResponse(
        doc_id="doc1",
        filename="a.md",
        chunks_count=2,
        indexed_chunks_count=1,
        failed_chunks_count=1,
        failed_chunk_ids=["doc1_1"],
    )

    assert response.indexed_chunks_count == 1
    assert response.failed_chunks_count == 1
    assert response.failed_chunk_ids == ["doc1_1"]
