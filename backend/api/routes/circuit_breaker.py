"""Circuit Breaker management endpoints.

This module provides REST API endpoints for managing and monitoring
circuit breaker states for external services.
"""

from fastapi import APIRouter, HTTPException

from backend.services.circuit_breaker import circuit_manager

router = APIRouter()


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(service: str | None = None) -> dict:
    """Reset circuit breaker state for a service.

    Args:
        service: Service name to reset (embedding, llm, parser).
                If not provided, resets all services.

    Returns:
        Dictionary with reset status and current circuit states

    Raises:
        HTTPException: If service name is invalid

    Example:
        POST /api/v1/circuit-breaker/reset?service=embedding
    """
    try:
        if service:
            # Reset specific service
            breaker = circuit_manager.get_breaker(service)
            breaker.reset()
            return {
                "status": "success",
                "message": f"Circuit breaker reset for {service}",
                "circuit_states": circuit_manager.get_all_states(),
            }
        else:
            # Reset all services
            for svc_name in ["embedding", "llm", "parser"]:
                breaker = circuit_manager.get_breaker(svc_name)
                breaker.reset()

            return {
                "status": "success",
                "message": "All circuit breakers reset",
                "circuit_states": circuit_manager.get_all_states(),
            }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_service",
                "message": str(e),
                "valid_services": ["embedding", "llm", "parser"],
            },
        )
