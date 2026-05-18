"""Tests for knowledge base CRUD endpoints."""

import os
import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_create_kb_success():
    """Create KB writes to SQLite and ES, returns ApiResponse."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.create_kb_index") as mock_es, \
         mock.patch("app.routers.knowledge_base.create_kb") as mock_sqlite:

        from types import SimpleNamespace
        mock_es.return_value = SimpleNamespace(
            kb_id="test123",
            index_name="rag_kb_test123",
            doc_count=0,
        )
        mock_sqlite.return_value = {
            "kb_id": "test123",
            "name": "Test KB",
            "description": "A test",
            "index_name": "rag_kb_test123",
            "doc_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

        resp = client.post("/rag/api/v1/kb", json={"name": "Test KB", "description": "A test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10000
        assert data["data"]["kb_id"] == "test123"
        assert data["data"]["index_name"] == "rag_kb_test123"


def test_create_kb_es_connection_error():
    """ES connection error returns 10003."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.create_kb_index") as mock_es:
        mock_es.side_effect = ConnectionError("ES down")

        resp = client.post("/rag/api/v1/kb", json={"name": "Test KB"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10003


def test_list_kbs_returns_list():
    """List KBs returns ApiResponse with knowledge_bases array."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.list_kbs") as mock_list, \
         mock.patch("app.routers.knowledge_base.get_kb_info") as mock_es_info:

        mock_list.return_value = [
            {
                "kb_id": "abc123",
                "name": "Test KB",
                "description": "desc",
                "index_name": "rag_kb_abc123",
                "doc_count": 5,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
        mock_es_info.return_value = None  # ES not found in test, ok

        resp = client.get("/rag/api/v1/kb")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10000
        assert "knowledge_bases" in data["data"]
        assert len(data["data"]["knowledge_bases"]) == 1


def test_get_kb_not_found():
    """GET /kb/{id} returns 10101 for non-existent KB."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.get_kb") as mock_get:
        mock_get.return_value = None

        resp = client.get("/rag/api/v1/kb/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10101


def test_delete_kb_not_found():
    """DELETE /kb/{id} returns 10101 for non-existent KB."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.get_kb") as mock_get:
        mock_get.return_value = None

        resp = client.delete("/rag/api/v1/kb/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10101
