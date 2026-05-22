"""Tests for vector retriever."""

import unittest.mock as mock

import pytest

from app.core import embedder
from app.retrievers.vector_retriever import vector_search


@pytest.mark.asyncio
async def test_vector_search_uses_cached_query_embedding():
    embedder.encode_query.cache_clear()

    with mock.patch("app.core.embedder.encode_texts") as mock_encode, \
         mock.patch("app.retrievers.vector_retriever.get_es_client") as mock_get_es:
        mock_encode.return_value = [[0.1] * 512]
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await vector_search("kb1", "重复问题", top_k=3)
        await vector_search("kb1", "重复问题", top_k=3)

        mock_encode.assert_called_once_with(["重复问题"])
        assert mock_client.search.call_count == 2


@pytest.mark.asyncio
async def test_vector_search_keeps_script_score_query():
    embedder.encode_query.cache_clear()

    with mock.patch("app.core.embedder.encode_texts") as mock_encode, \
         mock.patch("app.retrievers.vector_retriever.get_es_client") as mock_get_es:
        mock_encode.return_value = [[0.2] * 512]
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await vector_search("kb1", "向量检索", top_k=4)

        body = mock_client.search.call_args.kwargs["body"]
        script_score = body["query"]["script_score"]
        assert script_score["query"] == {"match_all": {}}
        assert "cosineSimilarity" in script_score["script"]["source"]
        assert script_score["script"]["params"]["query_vector"] == [0.2] * 512
        assert body["size"] == 4
