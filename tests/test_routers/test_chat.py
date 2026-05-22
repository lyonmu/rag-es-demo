"""Tests for SSE chat endpoint."""

import unittest.mock as mock

import pytest

from tests.conftest import create_test_app


@pytest.mark.asyncio
async def test_chat_returns_404_for_missing_kb():
    """Chat on non-existent KB returns 10101 (HTTP 200 with error code)."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.chat.get_kb") as mock_get:
            mock_get.return_value = None

            resp = await client.post("/rag/api/v1/kb/nonexistent/chat", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10101
