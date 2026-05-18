"""FastAPI application entry point with lifecycle management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import (
    close_es_client,
    init_db,
    init_es_client,
    get_embedder,
    register_exception_handler,
)
from app.routers import chat_router, kb_router, search_router, upload_router

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init on startup, cleanup on shutdown."""
    # Startup
    logger.info("Starting RAG-ES-Demo...")

    # Initialize SQLite KB store
    init_db()
    logger.info("SQLite KB store initialized")

    # Initialize ES
    await init_es_client()
    logger.info("Elasticsearch connected")

    # Preload embedding model
    try:
        get_embedder()
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG-ES-Demo...")
    await close_es_client()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Factory function for creating the FastAPI app."""
    app = FastAPI(
        title="RAG-ES-Demo",
        description="智能化 RAG 问答系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS (open for frontend debugging)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register global exception handler (unified ApiResponse)
    register_exception_handler(app)

    # Register routers with API versioning prefix
    api_prefix = "/rag/api/v1"
    app.include_router(kb_router, prefix=api_prefix)
    app.include_router(upload_router, prefix=api_prefix)
    app.include_router(search_router, prefix=api_prefix)
    app.include_router(chat_router, prefix=api_prefix)

    from app.core.response import ApiResponse

    # Health check
    @app.get(api_prefix + "/health")
    async def health():
        return ApiResponse.success(data={"status": "ok"})

    return app


# uvicorn entry point
app = create_app()
