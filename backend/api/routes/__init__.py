"""API routes for NexoGraph.

This module organizes all API endpoints under the /api/v1 prefix.
"""

from fastapi import APIRouter
from backend.api.routes.health import router as health_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.chat import router as chat_router

# Create API v1 router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health_router, prefix="/health")
api_router.include_router(documents_router, prefix="/documents")
api_router.include_router(chat_router, prefix="/chat")

# Export for main.py
__all__ = ["api_router"]
