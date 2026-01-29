"""Metrics API route for monitoring and observability.

This module provides endpoints for retrieving system metrics including
API performance statistics, error rates, and external service health.
"""

import logging

from fastapi import APIRouter

from backend.models.schemas import MetricsResponse
from backend.services.metrics_service import metrics_collector
from backend.services.circuit_breaker import circuit_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get API Metrics",
    description="Retrieve comprehensive API metrics including uptime, request statistics, "
    "endpoint performance, and Upstage service health.",
    tags=["monitoring"],
)
async def get_metrics() -> MetricsResponse:
    """Get current API metrics and system health.

    Returns detailed metrics about server performance, API usage,
    and external service (Upstage) health including circuit breaker states.

    Returns:
        MetricsResponse: Complete metrics summary with:
            - Server uptime and request statistics
            - Per-endpoint performance metrics
            - Upstage service health (calls, rate limits, retries, circuit state)
            - Current timestamp

    Example Response:
        {
            "uptime_seconds": 3600.5,
            "total_requests": 1500,
            "total_errors": 45,
            "error_rate": 0.03,
            "requests_per_second": 0.417,
            "endpoints": {
                "GET /api/v1/health": {
                    "count": 100,
                    "avg_duration_ms": 5.2,
                    "error_count": 0,
                    "error_rate": 0.0
                },
                "POST /api/v1/chat": {
                    "count": 250,
                    "avg_duration_ms": 2400.8,
                    "error_count": 15,
                    "error_rate": 0.06
                }
            },
            "upstage_services": {
                "embedding": {
                    "calls": 250,
                    "rate_limit_hits": 3,
                    "retry_attempts": 7,
                    "circuit_state": "closed"
                },
                "llm": {
                    "calls": 150,
                    "rate_limit_hits": 1,
                    "retry_attempts": 2,
                    "circuit_state": "closed"
                },
                "parser": {
                    "calls": 50,
                    "rate_limit_hits": 0,
                    "retry_attempts": 0,
                    "circuit_state": "closed"
                }
            },
            "timestamp": "2025-01-29T10:30:00.123456"
        }

    Note:
        This endpoint is excluded from metrics tracking to avoid circular
        references in the metrics collection system.
    """
    # Get base metrics summary
    summary = metrics_collector.get_summary()

    # Enrich with circuit breaker states
    for service_name in ["embedding", "llm", "parser"]:
        summary["upstage_services"][service_name]["circuit_state"] = (
            circuit_manager.get_state(service_name)
        )

    # Convert to Pydantic model for response validation
    return MetricsResponse(**summary)
