"""Health check endpoint for NexoGraph API.

This module provides a health check endpoint that verifies the status of all
critical services including Qdrant, Upstage API configuration, and LightRAG.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_settings
from backend.config import Settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router with health tag
router = APIRouter(tags=["health"])


async def check_qdrant_connection(settings: Settings) -> Literal["connected", "disconnected"]:
    """Check Qdrant vector database connection status.

    Attempts to connect to Qdrant and verify it's responsive. This check is
    non-blocking and will not raise exceptions to prevent the health endpoint
    from failing.

    Args:
        settings: Application settings containing Qdrant configuration

    Returns:
        "connected" if Qdrant is reachable, "disconnected" otherwise
    """
    try:
        from backend.services.qdrant_service import QdrantService

        service = QdrantService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
        )

        if await service.health_check():
            logger.info(
                f"Qdrant connection successful: {settings.qdrant_host}:{settings.qdrant_port}"
            )
            return "connected"
        else:
            logger.warning(
                f"Qdrant health check failed: {settings.qdrant_host}:{settings.qdrant_port}"
            )
            return "disconnected"

    except Exception as e:
        logger.error(
            f"Qdrant connection failed ({settings.qdrant_host}:{settings.qdrant_port}): {e}"
        )
        return "disconnected"


def check_upstage_configuration(settings: Settings) -> Literal["configured", "unconfigured"]:
    """Check if Upstage API is properly configured.

    Verifies that the Upstage API key is set. Does not test actual API connectivity
    to avoid rate limit issues and unnecessary API calls.

    Args:
        settings: Application settings containing Upstage configuration

    Returns:
        "configured" if API key is set, "unconfigured" otherwise
    """
    if settings.upstage_api_key and settings.upstage_api_key.startswith("up_"):
        return "configured"
    return "unconfigured"


def check_lightrag_status(settings: Settings) -> Literal["initialized", "not_initialized"]:
    """Check LightRAG initialization status.

    Checks if LightRAG working directory exists and contains graph data.

    Args:
        settings: Application settings containing LightRAG configuration

    Returns:
        "initialized" if working directory exists, "not_initialized" otherwise
    """
    try:
        from pathlib import Path

        working_dir = Path(settings.lightrag_working_dir)

        # Check if working directory exists
        if not working_dir.exists():
            logger.info(f"LightRAG working directory does not exist: {working_dir}")
            return "not_initialized"

        # Check if directory has any content (graph files)
        has_content = any(working_dir.iterdir())

        if has_content:
            logger.info(f"LightRAG working directory initialized: {working_dir}")
            return "initialized"
        else:
            logger.info(f"LightRAG working directory empty: {working_dir}")
            return "not_initialized"

    except Exception as e:
        logger.warning(f"Error checking LightRAG status: {e}")
        return "not_initialized"


@router.get("")
async def health_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Health check endpoint for NexoGraph API.

    Returns the current status of all services and the application version.
    This endpoint should always return 200 OK, even if some services are degraded.

    Args:
        settings: Application settings (injected dependency)

    Returns:
        Health status response containing:
        - status: Overall system health ("healthy", "degraded", or "unhealthy")
        - version: Application version
        - services: Status of each service component

    Example Response:
        {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "qdrant": "connected",
                "upstage": "configured",
                "lightrag": "initialized"
            }
        }
    """
    # Check all services
    qdrant_status = await check_qdrant_connection(settings)
    upstage_status = check_upstage_configuration(settings)
    lightrag_status = check_lightrag_status(settings)

    # Determine overall status
    # Critical services: Qdrant (required for vector search)
    # Optional services: Upstage (for API calls), LightRAG (for graph operations)
    if qdrant_status == "connected":
        if upstage_status == "configured" and lightrag_status == "initialized":
            overall_status = "healthy"
        else:
            overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "version": settings.app_version,
        "services": {
            "qdrant": qdrant_status,
            "upstage": upstage_status,
            "lightrag": lightrag_status,
        },
    }
