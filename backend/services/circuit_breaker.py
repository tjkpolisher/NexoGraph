"""Circuit Breaker management for Upstage API services.

This module provides circuit breaker protection for Upstage API calls
to gracefully handle and recover from service failures.
"""

import logging
from typing import Dict

from circuitbreaker import CircuitBreaker

logger = logging.getLogger(__name__)


class CircuitBreakerManager:
    """Manager for service-specific circuit breakers.

    Maintains separate circuit breakers for embedding, LLM, and document
    parsing services with independent failure thresholds and recovery timeouts.
    """

    def __init__(self):
        """Initialize circuit breaker manager with service-specific breakers."""
        self._breakers: Dict[str, CircuitBreaker] = {}

        # Service-specific configurations
        self._configs = {
            "embedding": {"failure_threshold": 5, "recovery_timeout": 60},
            "llm": {"failure_threshold": 5, "recovery_timeout": 60},
            "parser": {"failure_threshold": 3, "recovery_timeout": 120},
        }

        self._initialize_breakers()

    def _initialize_breakers(self):
        """Initialize circuit breakers for each service."""
        for name, config in self._configs.items():
            self._breakers[name] = CircuitBreaker(
                failure_threshold=config["failure_threshold"],
                recovery_timeout=config["recovery_timeout"],
                name=f"upstage_{name}",
            )
            logger.info(
                f"Circuit breaker initialized: {name} "
                f"(threshold={config['failure_threshold']}, "
                f"timeout={config['recovery_timeout']}s)"
            )

    def get_breaker(self, service: str) -> CircuitBreaker:
        """Get circuit breaker for a service.

        Args:
            service: Service name (embedding, llm, parser)

        Returns:
            CircuitBreaker instance for the service

        Raises:
            ValueError: If service name is unknown
        """
        if service not in self._breakers:
            raise ValueError(f"Unknown service: {service}")
        return self._breakers[service]

    def get_state(self, service: str) -> str:
        """Get current state of a service's circuit breaker.

        Args:
            service: Service name

        Returns:
            Circuit state: "closed", "open", or "half_open"
        """
        breaker = self._breakers.get(service)
        if breaker is None:
            return "unknown"

        # circuitbreaker 2.1.3+ returns state as string directly
        state = breaker.state
        if isinstance(state, str):
            return state.lower()
        # Fallback for older versions that return enum
        return state.name.lower()

    def get_all_states(self) -> dict:
        """Get states of all circuit breakers.

        Returns:
            Dictionary mapping service names to circuit states
        """
        return {name: self.get_state(name) for name in self._breakers}


# Global singleton instance
circuit_manager = CircuitBreakerManager()
