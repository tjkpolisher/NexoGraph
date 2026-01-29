"""Common exceptions for NexoGraph application.

This module defines base exceptions and service-specific exceptions
used throughout the application.
"""


class NexoGraphError(Exception):
    """Base exception for NexoGraph."""

    pass


class CircuitOpenError(NexoGraphError):
    """Raised when circuit breaker is open.

    Attributes:
        service: Name of the service with open circuit
        retry_after: Recommended retry time in seconds
    """

    def __init__(self, service: str, retry_after: int = 60):
        """Initialize CircuitOpenError.

        Args:
            service: Name of the service (e.g., "embedding", "llm", "parser")
            retry_after: Recommended retry time in seconds (default: 60)
        """
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker is open for {service}")
