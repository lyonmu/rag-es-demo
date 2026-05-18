"""Tests for search endpoint."""

import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_search_returns_404_for_missing_kb():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.search.get_kb_info") as mock_get:
        mock_get.return_value = None
        resp = client.post("/kb/nonexistent/search", json={"query": "test"})
        assert resp.status_code == 404
