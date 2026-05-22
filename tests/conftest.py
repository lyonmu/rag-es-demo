"""Pytest fixtures and test app configuration."""

from contextlib import asynccontextmanager

import httpx
import pytest


def create_test_app():
    """Create a test app that skips lifespan to avoid real ES connection."""
    import app.main as app_main

    @asynccontextmanager
    async def _null_lifespan(app):
        yield

    # Override lifespan before app creation so FastAPI binds the no-op lifecycle.
    app_main.lifespan = _null_lifespan
    return app_main.create_app()


@asynccontextmanager
async def create_async_client():
    """Provide an AsyncClient backed by ASGITransport for route tests."""
    app = create_test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def async_client():
    async with create_async_client() as client:
        yield client
