"""NexoGraph FastAPI Application Entry Point.

This module initializes the FastAPI application with CORS middleware, API routes,
and lifespan event handlers.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import api_router
from backend.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    This is the modern approach replacing the deprecated @app.on_event decorators.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    # Startup
    logger.info("=" * 60)
    logger.info("NexoGraph API starting...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Debug Mode: {settings.app_debug}")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info("=" * 60)

    # Check Qdrant connectivity (non-blocking)
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=2,
        )
        client.get_collections()
        logger.info("✓ Qdrant connection successful")
    except Exception as e:
        logger.warning(f"✗ Qdrant connection failed: {e}")
        logger.warning("  Qdrant services will be unavailable until connection is restored")

    logger.info("Application startup complete")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("NexoGraph API shutting down...")
    logger.info("Cleanup completed")
    logger.info("=" * 60)


# Initialize FastAPI application
app = FastAPI(
    title="NexoGraph",
    description="Interdisciplinary Scientific Knowledge Graph System",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.app_debug,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://localhost:8000",  # FastAPI dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router with /api/v1 prefix
app.include_router(api_router)


@app.get("/", tags=["root"])
async def root() -> dict[str, Any]:
    """Root endpoint returning API information.

    Returns:
        API metadata including name, version, and available endpoints

    Example Response:
        {
            "name": "NexoGraph API",
            "version": "0.1.0",
            "docs": "/docs",
            "base_url": "/api/v1"
        }
    """
    return {
        "name": "NexoGraph API",
        "version": settings.app_version,
        "docs": "/docs",
        "base_url": "/api/v1",
    }
