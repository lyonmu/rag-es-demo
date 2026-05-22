"""Tests for search endpoint."""

import unittest.mock as mock

import pytest

from tests.conftest import create_test_app


@pytest.mark.asyncio
async def test_search_returns_404_for_missing_kb():
    """Search on non-existent KB returns 10101 (HTTP 200 with error code)."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.search.get_kb") as mock_get:
            mock_get.return_value = None

            resp = await client.post("/rag/api/v1/kb/nonexistent/search", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10101


@pytest.mark.asyncio
async def test_search_success():
    """Successful search returns ApiResponse with results."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.search.get_kb") as mock_get, \
             mock.patch("app.routers.search.hybrid_search") as mock_search:

            mock_get.return_value = {
                "kb_id": "testkb",
                "name": "Test",
                "description": "",
                "index_name": "rag_kb_testkb",
                "doc_count": 5,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            mock_search.return_value = []

            resp = await client.post("/rag/api/v1/kb/testkb/search", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10000
            assert data["data"]["results"] == []
            assert data["data"]["total"] == 0
