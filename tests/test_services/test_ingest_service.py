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
