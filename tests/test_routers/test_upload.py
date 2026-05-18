"""Tests for file upload endpoint."""

from tests.conftest import create_test_app
from fastapi.testclient import TestClient


def test_upload_non_md_file_returns_400():
    app = create_test_app()
    client = TestClient(app)

    resp = client.post(
        "/kb/testkb/upload",
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Only .md files" in resp.json()["detail"]
