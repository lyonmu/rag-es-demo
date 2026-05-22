"""Tests for knowledge base CRUD endpoints."""

import os
import unittest.mock as mock

import pytest

from tests.conftest import create_test_app


@pytest.mark.asyncio
async def test_create_kb_success():
    """Create KB writes to SQLite and ES, returns ApiResponse."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

            resp = await client.post("/rag/api/v1/kb", json={"name": "Test KB", "description": "A test"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10000
            assert data["data"]["kb_id"] == "test123"
            assert data["data"]["index_name"] == "rag_kb_test123"


@pytest.mark.asyncio
async def test_create_kb_es_connection_error():
    """ES connection error returns 10003."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.knowledge_base.create_kb_index") as mock_es:
            mock_es.side_effect = ConnectionError("ES down")

            resp = await client.post("/rag/api/v1/kb", json={"name": "Test KB"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10003


@pytest.mark.asyncio
async def test_list_kbs_returns_list():
    """List KBs returns ApiResponse with knowledge_bases array."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

            resp = await client.get("/rag/api/v1/kb")
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10000
            assert "knowledge_bases" in data["data"]
            assert len(data["data"]["knowledge_bases"]) == 1


@pytest.mark.asyncio
async def test_get_kb_not_found():
    """GET /kb/{id} returns 10101 for non-existent KB."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.knowledge_base.get_kb") as mock_get:
            mock_get.return_value = None

            resp = await client.get("/rag/api/v1/kb/nonexistent")
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10101


@pytest.mark.asyncio
async def test_delete_kb_not_found():
    """DELETE /kb/{id} returns 10101 for non-existent KB."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.knowledge_base.get_kb") as mock_get:
            mock_get.return_value = None

            resp = await client.delete("/rag/api/v1/kb/nonexistent")
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10101
