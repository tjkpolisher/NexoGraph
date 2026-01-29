"""Metrics collection middleware for FastAPI.

This middleware intercepts all HTTP requests and responses to collect
performance metrics including request count, duration, and error rates.
"""

import time
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.services.metrics_service import metrics_collector

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP middleware for collecting API metrics.

    This middleware tracks:
    - Request count per endpoint
    - Request duration per endpoint
    - Error count and error rates
    - Global request statistics

    Excludes from tracking:
    - Health check endpoints (/api/v1/health)
    - Metrics endpoints (/api/v1/metrics) to avoid circular tracking

    Attributes:
        None (stateless middleware)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and collect metrics.

        Args:
            request: Incoming HTTP request
            call_next: Callable to pass request to next middleware/route

        Returns:
            HTTP response with metrics recorded
        """
        # Record start time
        start_time = time.perf_counter()

        # Call next middleware/route
        response = await call_next(request)

        # Calculate request duration
        duration = time.perf_counter() - start_time

        # Format endpoint name
        endpoint = f"{request.method} {request.url.path}"

        # Record metrics (exclude certain endpoints)
        # Exclude health check and metrics endpoints to avoid noise
        if not request.url.path.startswith("/api/v1/health"):
            if not request.url.path.startswith("/api/v1/metrics"):
                metrics_collector.record_request(
                    endpoint=endpoint,
                    duration=duration,
                    status_code=response.status_code,
                )

        return response
