"""Middleware package for NexoGraph application.

This package contains HTTP middleware for cross-cutting concerns like
metrics collection, logging, and request/response processing.
"""

from .metrics import MetricsMiddleware

__all__ = ["MetricsMiddleware"]
