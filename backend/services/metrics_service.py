"""Metrics collection service for monitoring API performance and external service health.

This module provides in-memory metrics collection for tracking API usage,
error rates, and Upstage service performance. Designed for Phase 1 MVP with
protocol abstraction for future Prometheus integration (Phase 2+).
"""

import threading
from datetime import datetime
from typing import Any, Protocol


class MetricsCollector(Protocol):
    """Metrics collector interface for abstraction.

    This Protocol defines the interface for metrics collection,
    allowing for multiple implementations (in-memory, Prometheus, etc.).
    Designed to support future migration to Prometheus or other monitoring systems.
    """

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            metric: Metric name
            value: Value to increment by (default: 1)
        """
        ...

    def record_request(
        self, endpoint: str, duration: float, status_code: int
    ) -> None:
        """Record an API request.

        Args:
            endpoint: Endpoint name (e.g., "GET /api/v1/chat")
            duration: Request duration in seconds
            status_code: HTTP status code
        """
        ...

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dictionary containing all metrics data
        """
        ...


class InMemoryMetricsCollector:
    """In-memory metrics collector for Phase 1 MVP.

    This implementation stores all metrics in memory using thread-safe counters
    and endpoint statistics. Suitable for single-instance deployments.

    For Phase 2+, this will be replaced with a Prometheus implementation
    following the same protocol.

    Attributes:
        _lock: Thread lock for thread-safe access
        _counters: Counter metrics dictionary
        _endpoint_stats: Per-endpoint statistics
        _start_time: Application start time
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._lock = threading.Lock()

        # Counter metrics
        self._counters = {
            # General API metrics
            "total_requests": 0,
            "total_errors": 0,
            # Upstage Embedding API metrics
            "upstage_embedding_calls": 0,
            "upstage_embedding_rate_limits": 0,
            "upstage_embedding_retries": 0,
            # Upstage LLM API metrics
            "upstage_llm_calls": 0,
            "upstage_llm_rate_limits": 0,
            "upstage_llm_retries": 0,
            # Upstage Document Parser API metrics
            "upstage_parser_calls": 0,
            "upstage_parser_rate_limits": 0,
            "upstage_parser_retries": 0,
        }

        # Per-endpoint statistics
        # Format: {"endpoint_name": {"count": int, "total_duration": float, "errors": int}}
        self._endpoint_stats: dict[str, dict[str, Any]] = {}

        # Application start time
        self._start_time = datetime.now()

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric.

        Thread-safe implementation using lock.

        Args:
            metric: Metric name to increment
            value: Value to increment by (default: 1)

        Example:
            >>> collector.increment("upstage_embedding_calls")
            >>> collector.increment("upstage_embedding_rate_limits", 2)
        """
        with self._lock:
            if metric in self._counters:
                self._counters[metric] += value

    def record_request(
        self, endpoint: str, duration: float, status_code: int
    ) -> None:
        """Record an API request with duration and status.

        Thread-safe implementation that updates both global and per-endpoint statistics.

        Args:
            endpoint: Endpoint name (e.g., "GET /api/v1/chat")
            duration: Request duration in seconds
            status_code: HTTP status code

        Example:
            >>> collector.record_request("POST /api/v1/chat", 1.25, 200)
            >>> collector.record_request("GET /api/v1/documents", 0.05, 500)
        """
        with self._lock:
            # Update global counters
            self._counters["total_requests"] += 1
            if status_code >= 400:
                self._counters["total_errors"] += 1

            # Initialize endpoint stats if not exists
            if endpoint not in self._endpoint_stats:
                self._endpoint_stats[endpoint] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "errors": 0,
                }

            # Update endpoint stats
            stats = self._endpoint_stats[endpoint]
            stats["count"] += 1
            stats["total_duration"] += duration
            if status_code >= 400:
                stats["errors"] += 1

    def get_summary(self) -> dict[str, Any]:
        """Get a complete metrics summary.

        Computes derived metrics (error rates, averages) from raw counters.
        Thread-safe using lock.

        Returns:
            Dictionary containing:
                - uptime_seconds: Application uptime
                - total_requests: Total request count
                - total_errors: Total error count
                - error_rate: Overall error rate (0.0-1.0)
                - requests_per_second: Average RPS
                - endpoints: Per-endpoint statistics with computed averages
                - upstage_services: Service-specific metrics
                - timestamp: Current timestamp

        Example:
            >>> summary = collector.get_summary()
            >>> print(summary["total_requests"])
            >>> print(summary["upstage_services"]["embedding"])
        """
        with self._lock:
            # Calculate uptime
            uptime = (datetime.now() - self._start_time).total_seconds()

            # Calculate total requests (avoid division by zero)
            total = max(self._counters["total_requests"], 1)

            # Build endpoint statistics
            endpoints_data = {}
            for endpoint, stats in self._endpoint_stats.items():
                count = stats["count"]
                endpoints_data[endpoint] = {
                    "count": count,
                    "avg_duration_ms": (stats["total_duration"] / count) * 1000
                    if count > 0
                    else 0.0,
                    "error_count": stats["errors"],
                    "error_rate": stats["errors"] / count if count > 0 else 0.0,
                }

            # Build service statistics
            upstage_services_data = {
                "embedding": {
                    "calls": self._counters["upstage_embedding_calls"],
                    "rate_limit_hits": self._counters[
                        "upstage_embedding_rate_limits"
                    ],
                    "retry_attempts": self._counters["upstage_embedding_retries"],
                },
                "llm": {
                    "calls": self._counters["upstage_llm_calls"],
                    "rate_limit_hits": self._counters["upstage_llm_rate_limits"],
                    "retry_attempts": self._counters["upstage_llm_retries"],
                },
                "parser": {
                    "calls": self._counters["upstage_parser_calls"],
                    "rate_limit_hits": self._counters["upstage_parser_rate_limits"],
                    "retry_attempts": self._counters["upstage_parser_retries"],
                },
            }

            return {
                "uptime_seconds": uptime,
                "total_requests": self._counters["total_requests"],
                "total_errors": self._counters["total_errors"],
                "error_rate": self._counters["total_errors"] / total,
                "requests_per_second": self._counters["total_requests"]
                / max(uptime, 1),
                "endpoints": endpoints_data,
                "upstage_services": upstage_services_data,
                "timestamp": datetime.now().isoformat(),
            }


# Global singleton instance
metrics_collector = InMemoryMetricsCollector()
