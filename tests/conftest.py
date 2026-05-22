"""Pytest fixtures and test app configuration."""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient


def create_test_app():
    """Create a test app that skips lifespan to avoid real ES connection."""
    import app.main as app_main

    @asynccontextmanager
    async def _null_lifespan(app):
        yield

    # Override lifespan before app creation so FastAPI binds the no-op lifecycle.
    app_main.lifespan = _null_lifespan
    return app_main.create_app()


@pytest.fixture
def client():
    """Provide a TestClient instance."""
    app = create_test_app()
    with TestClient(app) as c:
        yield c
