"""Tests for BM25 retriever query body."""

import unittest.mock as mock

import pytest

from app.retrievers.bm25_retriever import bm25_search


@pytest.mark.asyncio
async def test_bm25_search_builds_heading_aware_bool_query():
    with mock.patch("app.retrievers.bm25_retriever.get_es_client") as mock_get_es:
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await bm25_search("kb-test", "检索词", top_k=7)

        call_kwargs = mock_client.search.call_args.kwargs
        body = call_kwargs["body"]

        assert body["size"] == 7
        assert body["query"]["bool"]["minimum_should_match"] == 1

        should = body["query"]["bool"]["should"]
        assert len(should) == 2

        multi_match = should[0]["multi_match"]
        assert multi_match["query"] == "检索词"
        assert multi_match["fields"] == [
            "content^1.0",
            "heading_path^2.0",
            "content_with_heading^1.5",
        ]

        match_phrase = should[1]["match_phrase"]
        assert match_phrase["content"]["query"] == "检索词"
        assert match_phrase["content"]["boost"] == 1.5
