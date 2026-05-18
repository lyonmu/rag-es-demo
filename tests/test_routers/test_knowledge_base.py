"""Tests for knowledge base CRUD endpoints."""

import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_create_kb_returns_201():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.create_kb_index") as mock_create:
        from types import SimpleNamespace
        mock_create.return_value = SimpleNamespace(
            kb_id="test123",
            index_name="rag_kb_test123",
            name="Test KB",
            description="",
            doc_count=0,
        )

        resp = client.post("/kb", json={"name": "Test KB"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["kb_id"] == "test123"
        assert data["index_name"] == "rag_kb_test123"


def test_list_kbs_returns_list():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.list_kb_indices") as mock_list:
        mock_list.return_value = []
        resp = client.get("/kb")
        assert resp.status_code == 200
        assert "knowledge_bases" in resp.json()
