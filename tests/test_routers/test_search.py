"""Tests for search endpoint."""

import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_search_returns_404_for_missing_kb():
    """Search on non-existent KB returns 10101 (HTTP 200 with error code)."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.search.get_kb") as mock_get:
        mock_get.return_value = None

        resp = client.post("/rag/api/v1/kb/nonexistent/search", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10101


def test_search_success():
    """Successful search returns ApiResponse with results."""
    app = create_test_app()
    client = TestClient(app)

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

        resp = client.post("/rag/api/v1/kb/testkb/search", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10000
        assert data["data"]["results"] == []
        assert data["data"]["total"] == 0
