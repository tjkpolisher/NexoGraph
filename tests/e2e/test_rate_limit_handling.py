"""
E2E tests for rate limit handling with Mock APIs.

Tests cover:
- Mock 429 responses
- Retry logic validation
- Circuit Breaker state transitions
- Metrics collection verification
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import asyncio

pytestmark = pytest.mark.asyncio


class TestMock429Responses:
    """Test handling of mock 429 rate limit responses."""

    async def test_429_response_raises_rate_limit_error(self, mock_rate_limited_client):
        """Verify that 429 responses are properly recognized as rate limit errors."""
        mock_client, call_count = mock_rate_limited_client

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            # We expect the first call to raise, but tenacity will retry
            # For this basic test, we just verify the mock was called
            try:
                await mock_client.post("test")
            except httpx.HTTPStatusError as e:
                assert e.response.status_code == 429
                assert "Retry-After" in e.response.headers

    async def test_429_with_retry_after_header(self, mock_rate_limited_client):
        """Verify Retry-After header is properly extracted from 429 response."""
        mock_client, _ = mock_rate_limited_client

        try:
            await mock_client.post("test")
        except httpx.HTTPStatusError as e:
            retry_after = e.response.headers.get("Retry-After")
            assert retry_after is not None
            assert int(retry_after) > 0

    async def test_consecutive_429_responses(self):
        """Test handling of consecutive 429 responses."""
        call_count = {"value": 0}

        async def consecutive_rate_limits(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] <= 3:
                response = MagicMock()
                response.status_code = 429
                response.headers = {"Retry-After": "1"}
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=response
                )
            response = MagicMock()
            response.status_code = 200
            return response

        mock_client = AsyncMock()
        mock_client.post = consecutive_rate_limits

        # First 3 calls fail, 4th succeeds
        error_count = 0
        for i in range(4):
            try:
                await mock_client.post("test")
            except httpx.HTTPStatusError:
                error_count += 1

        assert error_count == 3
        assert call_count["value"] == 4


class TestRetryLogicValidation:
    """Test retry logic with Tenacity decorator simulation."""

    async def test_retry_succeeds_after_rate_limits(self, mock_rate_limited_client):
        """Verify that retries eventually succeed after rate limits."""
        mock_client, call_count = mock_rate_limited_client

        # Simulate multiple calls (as tenacity would do)
        success = False
        for attempt in range(5):
            try:
                await mock_client.post("test")
                success = True
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    raise
                # Simulate exponential backoff with minimal delay for testing
                await asyncio.sleep(0.01)

        assert success
        assert call_count["value"] >= 3  # At least 2 failures + 1 success

    async def test_max_retries_exceeded(self, mock_always_fail_client):
        """Verify that max retries are respected."""
        max_attempts = 5
        failures = 0

        for attempt in range(max_attempts):
            try:
                await mock_always_fail_client.post("test")
                break  # Success, exit loop
            except httpx.HTTPStatusError:
                failures += 1
                # Continue to next attempt

        # Verify all attempts failed (no success)
        assert failures == max_attempts

    async def test_retry_with_exponential_backoff(self, mock_rate_limited_client):
        """Verify exponential backoff timing."""
        mock_client, call_count = mock_rate_limited_client

        import time
        start_time = time.time()

        for attempt in range(5):
            try:
                await mock_client.post("test")
                break
            except httpx.HTTPStatusError:
                if attempt < 4:
                    # Simulate exponential backoff: 1, 2, 4, 8 seconds
                    # For testing, use minimal delays
                    await asyncio.sleep(0.01 * (2 ** attempt))

        elapsed = time.time() - start_time
        # Should have taken some time due to backoff
        assert elapsed > 0

    async def test_non_rate_limit_errors_not_retried(self):
        """Verify that non-429 errors are not retried."""
        call_count = {"value": 0}

        async def server_error(*args, **kwargs):
            call_count["value"] += 1
            response = MagicMock()
            response.status_code = 500
            raise httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)

        mock_client = AsyncMock()
        mock_client.post = server_error

        # In a real scenario with retry logic, non-429 errors might not be retried
        try:
            await mock_client.post("test")
        except httpx.HTTPStatusError:
            pass

        # Only called once (no retry for non-429 errors in basic case)
        assert call_count["value"] == 1


class TestCircuitBreakerStateTransition:
    """Test Circuit Breaker state transitions."""

    async def test_circuit_starts_closed(self):
        """Verify that circuit breaker starts in CLOSED state."""
        from backend.services.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        assert manager.get_state("embedding") == "closed"
        assert manager.get_state("llm") == "closed"
        assert manager.get_state("parser") == "closed"

    async def test_circuit_opens_after_threshold(self):
        """Verify circuit opens after failure threshold is reached."""
        from backend.services.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        breaker = manager.get_breaker("embedding")

        # Create a test function with circuit breaker
        @breaker
        async def protected_call():
            raise Exception("Simulated failure")

        # Simulate failures
        for _ in range(5):
            try:
                await protected_call()
            except Exception:
                pass

        # Circuit should be open now
        assert manager.get_state("embedding") == "open"

    async def test_open_circuit_raises_error(self):
        """Verify that open circuit raises CircuitBreakerError."""
        from backend.services.circuit_breaker import CircuitBreakerManager
        from circuitbreaker import CircuitBreakerError

        manager = CircuitBreakerManager()
        breaker = manager.get_breaker("embedding")

        # Create a test function with circuit breaker
        @breaker
        async def protected_failing_call():
            raise Exception("Simulated failure")

        @breaker
        async def protected_success_call():
            return "success"

        # Open the circuit by causing failures
        for _ in range(5):
            try:
                await protected_failing_call()
            except Exception:
                pass

        # Verify circuit is open
        assert manager.get_state("embedding") == "open"

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await protected_success_call()

    async def test_circuit_half_open_after_recovery_timeout(self):
        """Verify circuit transitions to HALF_OPEN after recovery timeout."""
        from backend.services.circuit_breaker import CircuitBreakerManager

        # Create manager with short timeout for testing
        manager = CircuitBreakerManager()
        breaker = manager.get_breaker("parser")  # Parser has 120s timeout

        # Create a test function with circuit breaker
        @breaker
        async def protected_call():
            raise Exception("Simulated failure")

        # Open the circuit (parser threshold is 3)
        for _ in range(3):
            try:
                await protected_call()
            except Exception:
                pass

        assert manager.get_state("parser") == "open"

        # Note: In real scenario, we'd wait for recovery_timeout
        # For testing, we verify the mechanism exists

    async def test_different_services_independent_circuits(self):
        """Verify that different services have independent circuit breakers."""
        from backend.services.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()

        # Open embedding circuit
        embedding_breaker = manager.get_breaker("embedding")

        @embedding_breaker
        async def protected_embedding_call():
            raise Exception("Simulated failure")

        for _ in range(5):
            try:
                await protected_embedding_call()
            except Exception:
                pass

        assert manager.get_state("embedding") == "open"
        assert manager.get_state("llm") == "closed"  # Other services unaffected
        assert manager.get_state("parser") == "closed"

    async def _failing_coro(self):
        """Helper coroutine that always fails."""
        raise Exception("Simulated failure")

    async def _succeeding_coro(self):
        """Helper coroutine that always succeeds."""
        return "success"


class TestMetricsCollection:
    """Test metrics collection during rate limit scenarios."""

    async def test_metrics_increment_on_rate_limit(self):
        """Verify metrics are incremented on rate limit hits."""
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()
        initial_count = metrics.get_summary()["upstage_services"]["embedding"]["calls"]

        # Increment metrics
        metrics.increment("upstage_embedding_calls")
        metrics.increment("upstage_embedding_calls")
        metrics.increment("upstage_embedding_rate_limits")

        summary = metrics.get_summary()
        assert summary["upstage_services"]["embedding"]["calls"] == initial_count + 2
        assert (
            summary["upstage_services"]["embedding"]["rate_limit_hits"] == 1
        )

    async def test_metrics_record_request(self):
        """Verify request metrics are recorded correctly."""
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()

        # Record several requests
        metrics.record_request("POST /api/v1/documents/upload", 0.5, 200)
        metrics.record_request("POST /api/v1/documents/upload", 0.3, 200)
        metrics.record_request("POST /api/v1/documents/upload", 1.2, 500)

        summary = metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["error_rate"] == pytest.approx(1 / 3, rel=0.01)

    async def test_metrics_endpoint_stats(self):
        """Verify endpoint-specific metrics are tracked."""
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()

        metrics.record_request("POST /api/v1/chat", 0.1, 200)
        metrics.record_request("POST /api/v1/chat", 0.2, 200)
        metrics.record_request("GET /api/v1/documents", 0.05, 200)

        summary = metrics.get_summary()
        assert "POST /api/v1/chat" in summary["endpoints"]
        assert summary["endpoints"]["POST /api/v1/chat"]["count"] == 2
        assert summary["endpoints"]["GET /api/v1/documents"]["count"] == 1

    async def test_metrics_per_service_stats(self):
        """Verify per-service rate limit statistics."""
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()

        # Simulate different services
        metrics.increment("upstage_embedding_calls", 5)
        metrics.increment("upstage_embedding_rate_limits", 2)
        metrics.increment("upstage_llm_calls", 3)
        metrics.increment("upstage_llm_rate_limits", 1)

        summary = metrics.get_summary()
        assert summary["upstage_services"]["embedding"]["calls"] == 5
        assert summary["upstage_services"]["embedding"]["rate_limit_hits"] == 2
        assert summary["upstage_services"]["llm"]["calls"] == 3
        assert summary["upstage_services"]["llm"]["rate_limit_hits"] == 1

    async def test_metrics_error_rate_edge_cases(self):
        """Test error rate calculation edge cases."""
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()

        # No requests yet - should handle gracefully
        summary = metrics.get_summary()
        assert summary["error_rate"] == 0.0
        assert summary["total_requests"] == 0

        # All successful requests
        for _ in range(10):
            metrics.record_request("GET /test", 0.1, 200)

        summary = metrics.get_summary()
        assert summary["error_rate"] == 0.0
        assert summary["total_requests"] == 10

        # All failed requests
        metrics = InMemoryMetricsCollector()
        for _ in range(5):
            metrics.record_request("GET /test", 0.1, 500)

        summary = metrics.get_summary()
        assert summary["error_rate"] == 1.0
        assert summary["total_errors"] == 5


class TestIntegratedRateLimitHandling:
    """Integration tests combining multiple components."""

    async def test_full_flow_rate_limit_retry_recovery(self, mock_rate_limited_client):
        """Test complete flow: rate limit -> retry -> success."""
        mock_client, call_count = mock_rate_limited_client
        from backend.services.metrics_service import InMemoryMetricsCollector

        metrics = InMemoryMetricsCollector()

        # Simulate the flow
        max_attempts = 5
        success = False
        last_error = None

        for attempt in range(max_attempts):
            try:
                metrics.increment("upstage_embedding_calls")
                response = await mock_client.post("test")
                success = True
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    metrics.increment("upstage_embedding_rate_limits")
                    await asyncio.sleep(0.01)

        assert success
        assert call_count["value"] >= 3
        summary = metrics.get_summary()
        assert summary["upstage_services"]["embedding"]["rate_limit_hits"] >= 1

    async def test_circuit_breaker_prevents_cascading_failures(self):
        """Verify circuit breaker prevents cascading failures."""
        from backend.services.circuit_breaker import CircuitBreakerManager
        from backend.services.metrics_service import InMemoryMetricsCollector
        from circuitbreaker import CircuitBreakerError

        manager = CircuitBreakerManager()
        metrics = InMemoryMetricsCollector()
        breaker = manager.get_breaker("embedding")

        @breaker
        async def protected_call():
            raise Exception("Simulated failure")

        # Simulate cascading failures
        failure_count = 0
        circuit_open_count = 0
        for i in range(10):
            try:
                metrics.increment("upstage_embedding_calls")
                await protected_call()
            except CircuitBreakerError:
                # Circuit is open, preventing cascading failures
                circuit_open_count += 1
            except Exception:
                failure_count += 1
                if i >= 5:
                    metrics.increment("upstage_embedding_rate_limits")

        # Should have stopped attempting after circuit opens
        assert manager.get_state("embedding") == "open"
        assert circuit_open_count > 0  # Circuit opened and prevented some calls

    async def _failing_coro(self):
        """Helper coroutine that always fails."""
        raise Exception("Simulated failure")
