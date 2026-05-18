"""Tests for SSE chat endpoint."""

import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_chat_returns_404_for_missing_kb():
    """Chat on non-existent KB returns 10101 (HTTP 200 with error code)."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.chat.get_kb") as mock_get:
        mock_get.return_value = None

        resp = client.post("/rag/api/v1/kb/nonexistent/chat", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10101
