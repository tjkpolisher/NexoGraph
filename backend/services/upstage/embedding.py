"""Upstage Embedding service.

This module provides a service for generating embeddings using Upstage's
Solar embedding models (asymmetric embeddings for passage and query).
"""

import asyncio
import logging
from typing import Literal

import httpx
from circuitbreaker import CircuitBreakerError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.services.circuit_breaker import circuit_manager
from backend.services.metrics_service import metrics_collector
from backend.exceptions import CircuitOpenError

logger = logging.getLogger(__name__)


class UpstageEmbeddingError(Exception):
    """Base exception for Upstage Embedding service errors."""

    pass


class UpstageEmbeddingRateLimitError(UpstageEmbeddingError):
    """Exception raised when rate limit is exceeded."""

    pass


class UpstageEmbeddingService:
    """Service for Upstage Solar embedding API.

    This service provides an interface to Upstage's Solar embedding API,
    which supports asymmetric embeddings (separate models for passages and queries).

    Attributes:
        api_key: Upstage API key
        base_url: API base URL (default: https://api.upstage.ai/v1)
        vector_size: Embedding vector dimension (4096 for solar-embedding-1-large)
    """

    VECTOR_SIZE = 4096  # Solar embedding dimension

    # Model names for different use cases
    PASSAGE_MODEL = "solar-embedding-1-large-passage"  # For documents/passages
    QUERY_MODEL = "solar-embedding-1-large-query"  # For search queries

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        max_concurrent_requests: int = 3,
    ) -> None:
        """Initialize Upstage Embedding service.

        Args:
            api_key: Upstage API key (starts with 'up_')
            base_url: API base URL
            max_concurrent_requests: Maximum number of concurrent API requests (default: 3)

        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not api_key.startswith("up_"):
            raise ValueError("Invalid Upstage API key format (must start with 'up_')")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._breaker = circuit_manager.get_breaker("embedding")

        logger.info(
            f"UpstageEmbeddingService initialized "
            f"(max_concurrent_requests: {max_concurrent_requests})"
        )

    async def get_embedding(
        self,
        text: str,
        model_type: Literal["passage", "query"] = "passage",
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model_type: Type of embedding model to use:
                - "passage": For documents/passages (solar-embedding-1-large-passage)
                - "query": For search queries (solar-embedding-1-large-query)

        Returns:
            Embedding vector (dimension: 4096)

        Raises:
            UpstageEmbeddingError: If API call fails
            CircuitOpenError: If circuit breaker is open
            ValueError: If text is empty

        Example:
            >>> service = UpstageEmbeddingService(api_key="up_xxx")
            >>> # Embed a document passage
            >>> passage_vec = await service.get_embedding(
            ...     "This is a document passage",
            ...     model_type="passage"
            ... )
            >>> # Embed a search query
            >>> query_vec = await service.get_embedding(
            ...     "search query",
            ...     model_type="query"
            ... )
        """
        # Record API call
        metrics_collector.increment("upstage_embedding_calls")

        try:
            # Circuit Breaker protection
            return await self._breaker.call_async(
                self._get_embedding_with_retry, text, model_type
            )
        except CircuitBreakerError:
            # Circuit is open
            metrics_collector.increment("upstage_embedding_rate_limits")
            raise CircuitOpenError(
                service="embedding",
                retry_after=self._breaker.recovery_timeout
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, UpstageEmbeddingRateLimitError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _get_embedding_with_retry(
        self,
        text: str,
        model_type: Literal["passage", "query"] = "passage",
    ) -> list[float]:
        """Generate embedding with retry logic (internal method).

        This is the actual implementation wrapped by circuit breaker.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Select appropriate model
        model = self.PASSAGE_MODEL if model_type == "passage" else self.QUERY_MODEL

        url = f"{self.base_url}/solar/embeddings"

        payload = {
            "model": model,
            "input": text.strip(),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    logger.debug(f"Requesting embedding (model: {model})")

                    response = await client.post(url, json=payload, headers=headers)

                    # Handle rate limit (429)
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                            logger.warning(
                                f"Rate limit exceeded. Retry after {wait_time} seconds"
                            )
                            await asyncio.sleep(wait_time)
                        metrics_collector.increment("upstage_embedding_rate_limits")
                        raise UpstageEmbeddingRateLimitError("Rate limit exceeded (429)")

                    response.raise_for_status()

                    result = response.json()

                    # Extract embedding from response
                    embedding = result["data"][0]["embedding"]

                    # Validate dimension
                    if len(embedding) != self.VECTOR_SIZE:
                        raise UpstageEmbeddingError(
                            f"Unexpected embedding dimension: {len(embedding)} "
                            f"(expected {self.VECTOR_SIZE})"
                        )

                    logger.debug(f"Embedding generated (dim: {len(embedding)})")

                    return embedding

        except httpx.HTTPStatusError as e:
            # Handle rate limit separately
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                    logger.warning(f"Rate limit exceeded. Retry after {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                metrics_collector.increment("upstage_embedding_rate_limits")
                raise UpstageEmbeddingRateLimitError("Rate limit exceeded (429)") from e

            error_msg = f"HTTP error from Upstage Embedding API: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"

            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = "Request to Upstage Embedding API timed out"
            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error calling Upstage Embedding API: {e}"
            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

    async def get_embeddings_batch(
        self,
        texts: list[str],
        model_type: Literal["passage", "query"] = "passage",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            model_type: Type of embedding model to use

        Returns:
            List of embedding vectors

        Raises:
            UpstageEmbeddingError: If API call fails
            CircuitOpenError: If circuit breaker is open
            ValueError: If texts list is empty

        Example:
            >>> service = UpstageEmbeddingService(api_key="up_xxx")
            >>> texts = ["Document 1", "Document 2", "Document 3"]
            >>> embeddings = await service.get_embeddings_batch(
            ...     texts,
            ...     model_type="passage"
            ... )
            >>> len(embeddings)  # 3
        """
        # Record API call
        metrics_collector.increment("upstage_embedding_calls")

        try:
            # Circuit Breaker protection
            return await self._breaker.call_async(
                self._get_embeddings_batch_with_retry, texts, model_type
            )
        except CircuitBreakerError:
            # Circuit is open
            metrics_collector.increment("upstage_embedding_rate_limits")
            raise CircuitOpenError(
                service="embedding",
                retry_after=self._breaker.recovery_timeout
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, UpstageEmbeddingRateLimitError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _get_embeddings_batch_with_retry(
        self,
        texts: list[str],
        model_type: Literal["passage", "query"] = "passage",
    ) -> list[list[float]]:
        """Generate embeddings for batch with retry logic (internal method).

        This is the actual implementation wrapped by circuit breaker.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty after stripping")

        # Select appropriate model
        model = self.PASSAGE_MODEL if model_type == "passage" else self.QUERY_MODEL

        url = f"{self.base_url}/solar/embeddings"

        payload = {
            "model": model,
            "input": valid_texts,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    logger.debug(
                        f"Requesting batch embeddings (model: {model}, count: {len(valid_texts)})"
                    )

                    response = await client.post(url, json=payload, headers=headers)

                    # Handle rate limit (429)
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                            logger.warning(
                                f"Rate limit exceeded. Retry after {wait_time} seconds"
                            )
                            await asyncio.sleep(wait_time)
                        metrics_collector.increment("upstage_embedding_rate_limits")
                        raise UpstageEmbeddingRateLimitError("Rate limit exceeded (429)")

                    response.raise_for_status()

                    result = response.json()

                    # Extract embeddings from response
                    embeddings = [item["embedding"] for item in result["data"]]

                    # Validate count and dimensions
                    if len(embeddings) != len(valid_texts):
                        raise UpstageEmbeddingError(
                            f"Expected {len(valid_texts)} embeddings, got {len(embeddings)}"
                        )

                    for i, emb in enumerate(embeddings):
                        if len(emb) != self.VECTOR_SIZE:
                            raise UpstageEmbeddingError(
                                f"Embedding {i} has unexpected dimension: {len(emb)} "
                                f"(expected {self.VECTOR_SIZE})"
                            )

                    logger.info(f"Batch embeddings generated: {len(embeddings)} vectors")

                    return embeddings

        except httpx.HTTPStatusError as e:
            # Handle rate limit separately
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                    logger.warning(f"Rate limit exceeded. Retry after {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                metrics_collector.increment("upstage_embedding_rate_limits")
                raise UpstageEmbeddingRateLimitError("Rate limit exceeded (429)") from e

            error_msg = f"HTTP error from Upstage Embedding API: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"

            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = "Request to Upstage Embedding API timed out"
            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error calling Upstage Embedding API: {e}"
            logger.error(error_msg)
            raise UpstageEmbeddingError(error_msg) from e

    async def health_check(self) -> bool:
        """Check if Upstage Embedding API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple test embedding
            await self.get_embedding("test", model_type="query")
            logger.debug("Upstage Embedding health check: OK")
            return True
        except Exception as e:
            logger.warning(f"Upstage Embedding health check failed: {e}")
            return False
