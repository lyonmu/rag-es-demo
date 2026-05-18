"""Pytest fixtures and test app configuration."""

import pytest
from fastapi.testclient import TestClient


def create_test_app():
    """Create a test app that skips lifespan to avoid real ES connection."""
    from app.main import create_app as _create_app
    app = _create_app()
    # Replace lifespan with a no-op to avoid connecting to ES during tests
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _null_lifespan(app):
        yield

    app.router.lifespan_context = _null_lifespan
    return app


@pytest.fixture
def client():
    """Provide a TestClient instance."""
    app = create_test_app()
    with TestClient(app) as c:
        yield c
