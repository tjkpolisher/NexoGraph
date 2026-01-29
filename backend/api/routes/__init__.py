"""API routes for NexoGraph.

This module organizes all API endpoints under the /api/v1 prefix.
"""

from fastapi import APIRouter
from backend.api.routes.health import router as health_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.metrics import router as metrics_router
from backend.api.routes.circuit_breaker import router as circuit_breaker_router

# Create API v1 router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health_router, prefix="/health")
api_router.include_router(documents_router, prefix="/documents")
api_router.include_router(chat_router, prefix="/chat")
api_router.include_router(metrics_router, tags=["monitoring"])
api_router.include_router(circuit_breaker_router, tags=["monitoring"])

# Export for main.py
__all__ = ["api_router"]
