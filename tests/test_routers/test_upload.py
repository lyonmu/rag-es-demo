"""Tests for file upload endpoint."""

import unittest.mock as mock

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_upload_non_md_file_returns_error():
    """Uploading .txt file succeeds since .txt is now supported alongside .md."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get, \
         mock.patch("app.routers.upload.ingest_markdown") as mock_ingest, \
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

        resp = client.post(
            "/rag/api/v1/kb/testkb/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
        )
        # .txt is now supported, so this should succeed
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10000


def test_upload_unsupported_file_type():
    """Uploading .pdf file returns 10100."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get:
        mock_get.return_value = {
            "kb_id": "testkb",
            "name": "Test",
            "description": "",
            "index_name": "rag_kb_testkb",
            "doc_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

        resp = client.post(
            "/rag/api/v1/kb/testkb/upload",
            files={"file": ("test.pdf", b"content", "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10100


def test_upload_kb_not_found():
    """Upload to non-existent KB returns 10101."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get:
        mock_get.return_value = None

        resp = client.post(
            "/rag/api/v1/kb/nonexistent/upload",
            files={"file": ("test.md", b"# Hello", "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10101


def test_upload_md_file_success():
    """Uploading .md file calls ingest and returns success."""
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get, \
         mock.patch("app.routers.upload.ingest_markdown") as mock_ingest, \
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

        resp = client.post(
            "/rag/api/v1/kb/testkb/upload",
            files={"file": ("test.md", b"# Hello\n\nWorld", "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 10000
        assert data["data"]["doc_id"] == "abc123"
        assert data["data"]["chunks_count"] == 2
        mock_update.assert_called_once()


def test_upload_does_not_increment_doc_count_when_no_chunks_indexed():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get, \
         mock.patch("app.routers.upload.ingest_markdown") as mock_ingest, \
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

        resp = client.post(
            "/rag/api/v1/kb/testkb/upload",
            files={"file": ("test.md", b"# Hello", "text/markdown")},
        )

        assert resp.status_code == 200
        assert resp.json()["code"] == 10000
        mock_update.assert_not_called()
