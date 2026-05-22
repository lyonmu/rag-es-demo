"""Tests for file upload endpoint."""

import unittest.mock as mock

import pytest

from tests.conftest import create_test_app


@pytest.mark.asyncio
async def test_upload_non_md_file_returns_error():
    """Uploading .txt file succeeds since .txt is now supported alongside .md."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.upload.get_kb") as mock_get, \
             mock.patch("app.routers.upload.ingest_markdown", new_callable=mock.AsyncMock) as mock_ingest, \
             mock.patch("app.routers.upload.update_doc_count") as mock_update:

            mock_get.return_value = {
                "kb_id": "testkb",
                "name": "Test",
                "description": "",
                "index_name": "rag_kb_testkb",
                "doc_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            mock_ingest.return_value = {
                "doc_id": "abc123",
                "filename": "test.txt",
                "chunks_count": 1,
            }

            resp = await client.post(
                "/rag/api/v1/kb/testkb/upload",
                files={"file": ("test.txt", b"content", "text/plain")},
            )
            # .txt is now supported, so this should succeed
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10000


@pytest.mark.asyncio
async def test_upload_unsupported_file_type():
    """Uploading .pdf file returns 10100."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.upload.get_kb") as mock_get:
            mock_get.return_value = {
                "kb_id": "testkb",
                "name": "Test",
                "description": "",
                "index_name": "rag_kb_testkb",
                "doc_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }

            resp = await client.post(
                "/rag/api/v1/kb/testkb/upload",
                files={"file": ("test.pdf", b"content", "application/pdf")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10100


@pytest.mark.asyncio
async def test_upload_kb_not_found():
    """Upload to non-existent KB returns 10101."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.upload.get_kb") as mock_get:
            mock_get.return_value = None

            resp = await client.post(
                "/rag/api/v1/kb/nonexistent/upload",
                files={"file": ("test.md", b"# Hello", "text/markdown")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10101


@pytest.mark.asyncio
async def test_upload_md_file_success():
    """Uploading .md file calls ingest and returns success."""
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.upload.get_kb") as mock_get, \
             mock.patch("app.routers.upload.ingest_markdown", new_callable=mock.AsyncMock) as mock_ingest, \
             mock.patch("app.routers.upload.update_doc_count") as mock_update:

            mock_get.return_value = {
                "kb_id": "testkb",
                "name": "Test",
                "description": "",
                "index_name": "rag_kb_testkb",
                "doc_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            mock_ingest.return_value = {
                "doc_id": "abc123",
                "filename": "test.md",
                "chunks_count": 2,
            }

            resp = await client.post(
                "/rag/api/v1/kb/testkb/upload",
                files={"file": ("test.md", b"# Hello\n\nWorld", "text/markdown")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 10000
            assert data["data"]["doc_id"] == "abc123"
            assert data["data"]["chunks_count"] == 2
            mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_upload_does_not_increment_doc_count_when_no_chunks_indexed():
    import httpx

    app = create_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with mock.patch("app.routers.upload.get_kb") as mock_get, \
             mock.patch("app.routers.upload.ingest_markdown", new_callable=mock.AsyncMock) as mock_ingest, \
             mock.patch("app.routers.upload.update_doc_count") as mock_update:
            mock_get.return_value = {
                "kb_id": "testkb",
                "name": "Test",
                "description": "",
                "index_name": "rag_kb_testkb",
                "doc_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            mock_ingest.return_value = {
                "doc_id": "abc123",
                "filename": "test.md",
                "chunks_count": 2,
                "indexed_chunks_count": 0,
                "failed_chunks_count": 2,
            }

            resp = await client.post(
                "/rag/api/v1/kb/testkb/upload",
                files={"file": ("test.md", b"# Hello", "text/markdown")},
            )

            assert resp.status_code == 200
            assert resp.json()["code"] == 10000
            mock_update.assert_not_called()
