"""Tests for IngestService with mocked ES and embedder."""

import unittest.mock as mock

import pytest

from app.services.ingest_service import ingest_markdown


@pytest.mark.asyncio
async def test_ingest_markdown_chunks_and_embeds():
    """Verify chunk → encode → bulk write pipeline."""
    content = """# 标题一

内容一。

## 标题二

内容二。
"""
    mock_bulk_response = {
        "errors": False,
        "items": [
            {"index": {"_id": "test_0", "status": 201}},
            {"index": {"_id": "test_1", "status": 201}},
        ],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:

        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512, [0.2] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        assert result["doc_id"] != ""
        assert result["filename"] == "test.md"
        assert result["chunks_count"] == 2

        # Verify encode_texts was called
        mock_encode.assert_called_once()
        # Verify bulk was called
        mock_client.bulk.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_markdown_writes_optimized_chunk_fields(monkeypatch):
    content = "# 标题\n\n内容"
    mock_bulk_response = {"errors": False, "items": [{"index": {"_id": "x", "status": 201}}]}

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        body = mock_client.bulk.call_args.kwargs["body"]
        document = body[1]
        assert document["chunk_id"] == body[0]["index"]["_id"]
        assert document["content_with_heading"].startswith("标题\n")
        assert document["content_length"] == len(document["content"])
        assert len(document["chunk_text_hash"]) == 64
        assert document["mapping_version"] == "rag-es-optimization-v1"
        assert result["failed_chunks_count"] == 0


@pytest.mark.asyncio
async def test_ingest_markdown_reports_partial_bulk_failures():
    content = "# A\n\n一\n\n# B\n\n二"
    mock_bulk_response = {
        "errors": True,
        "items": [
            {"index": {"_id": "ok", "status": 201}},
            {"index": {"_id": "bad", "status": 500, "error": {"reason": "boom"}}},
        ],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512, [0.2] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        assert result["chunks_count"] == 2
        assert result["indexed_chunks_count"] == 1
        assert result["failed_chunks_count"] == 1
        assert result["failed_chunk_ids"] == ["bad"]


@pytest.mark.asyncio
async def test_ingest_markdown_raises_when_all_bulk_items_fail():
    content = "# A\n\n一"
    mock_bulk_response = {
        "errors": True,
        "items": [{"index": {"_id": "bad", "status": 500, "error": {"reason": "boom"}}}],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512]

        with pytest.raises(RuntimeError, match="All chunks failed"):
            await ingest_markdown("test-kb-id", "test.md", content)


@pytest.mark.asyncio
async def test_ingest_markdown_raises_on_embedding_chunk_count_mismatch():
    chunks = [
        mock.Mock(content="# A\n\n一", heading_path="A", chunk_index=0),
        mock.Mock(content="# B\n\n二", heading_path="B", chunk_index=1),
    ]

    with mock.patch("app.services.ingest_service.chunk_markdown", return_value=chunks), \
         mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock()
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512]

        with pytest.raises(RuntimeError, match="Embedding/chunk count mismatch"):
            await ingest_markdown("test-kb-id", "test.md", "# A\n\n一\n\n# B\n\n二")

        mock_client.bulk.assert_not_called()
