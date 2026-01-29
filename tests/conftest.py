"""
Pytest configuration and shared fixtures for NexoGraph tests.

This module provides common fixtures for unit and integration tests,
including mocked HTTP clients, metrics collectors, and circuit breakers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from typing import Any, Dict, Tuple

# Fixtures for testing


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Basic Mock httpx client returning successful embedding response."""
    mock = AsyncMock()

    async def mock_post(*args, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.json = AsyncMock(return_value={"embedding": [0.1] * 4096})
        return response

    mock.post = mock_post
    return mock


@pytest.fixture
def mock_rate_limited_client() -> Tuple[AsyncMock, Dict[str, int]]:
    """
    Mock httpx client that simulates rate limiting.

    Returns 429 status code for the first 2 calls, then succeeds.
    Returns a tuple of (mock_client, call_count_dict).
    """
    call_count = {"value": 0}

    async def rate_limited_response(*args, **kwargs):
        call_count["value"] += 1
        response = MagicMock()

        if call_count["value"] <= 2:
            response.status_code = 429
            response.headers = {"Retry-After": "1"}
            response.json = AsyncMock(return_value={"error": "rate_limit_exceeded"})
            # Simulate HTTPStatusError for rate limit
            error = httpx.HTTPStatusError(
                message="429 Too Many Requests",
                request=MagicMock(),
                response=response
            )
            raise error

        response.status_code = 200
        response.headers = {}
        response.json = AsyncMock(return_value={"embedding": [0.1] * 4096})
        return response

    mock = AsyncMock()
    mock.post = rate_limited_response
    return mock, call_count


@pytest.fixture
def mock_always_fail_client() -> AsyncMock:
    """Mock httpx client that always fails with 429 (rate limit)."""
    mock = AsyncMock()

    async def always_fail(*args, **kwargs):
        response = MagicMock()
        response.status_code = 429
        response.headers = {"Retry-After": "60"}
        response.json = AsyncMock(return_value={"error": "rate_limit_exceeded"})

        error = httpx.HTTPStatusError(
            message="429 Too Many Requests",
            request=MagicMock(),
            response=response
        )
        raise error

    mock.post = always_fail
    return mock


@pytest.fixture
def mock_timeout_client() -> AsyncMock:
    """Mock httpx client that raises timeout exception."""
    mock = AsyncMock()

    async def timeout_response(*args, **kwargs):
        raise httpx.TimeoutException("Connection timeout")

    mock.post = timeout_response
    return mock


@pytest.fixture
def mock_connection_error_client() -> AsyncMock:
    """Mock httpx client that raises connection error."""
    mock = AsyncMock()

    async def connection_error(*args, **kwargs):
        raise httpx.ConnectError("Connection failed")

    mock.post = connection_error
    return mock


@pytest.fixture
def mock_successful_parser_response() -> MagicMock:
    """Mock successful document parser response."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json = AsyncMock(return_value={
        "markdown": "# Test Document\n\nThis is test content.",
        "images": []
    })
    return response


@pytest.fixture
def mock_rate_limited_parser_response() -> Tuple[AsyncMock, Dict[str, int]]:
    """
    Mock document parser that rate limits.

    Returns 429 for first call, then succeeds.
    """
    call_count = {"value": 0}

    async def rate_limited_response(*args, **kwargs):
        call_count["value"] += 1
        response = MagicMock()

        if call_count["value"] == 1:
            response.status_code = 429
            response.headers = {"Retry-After": "2"}
            response.json = AsyncMock(return_value={"error": "rate_limit_exceeded"})

            error = httpx.HTTPStatusError(
                message="429 Too Many Requests",
                request=MagicMock(),
                response=response
            )
            raise error

        response.status_code = 200
        response.headers = {}
        response.json = AsyncMock(return_value={
            "markdown": "# Parsed Document",
            "images": []
        })
        return response

    mock = AsyncMock()
    mock.post = rate_limited_response
    return mock, call_count


# Markers for pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "real_api: marks tests as real API tests (run with --run-real-api)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Command-line options


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-real-api",
        action="store_true",
        default=False,
        help="Run real API tests (requires UPSTAGE_API_KEY)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("--run-real-api"):
        return

    skip_real_api = pytest.mark.skip(reason="Real API tests disabled (use --run-real-api to enable)")
    for item in items:
        if "real_api" in item.keywords:
            item.add_marker(skip_real_api)
