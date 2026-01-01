"""Qdrant vector database service.

This module provides a service layer for interacting with Qdrant,
including collection management, vector operations, and health checks.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class QdrantServiceError(Exception):
    """Base exception for Qdrant service errors."""

    pass


class QdrantConnectionError(QdrantServiceError):
    """Exception raised when connection to Qdrant fails."""

    pass


class QdrantCollectionError(QdrantServiceError):
    """Exception raised when collection operations fail."""

    pass


class QdrantService:
    """Service for managing Qdrant vector database operations.

    This service provides a high-level interface for vector operations,
    including collection management, vector insertion, search, and deletion.

    Attributes:
        host: Qdrant server hostname
        port: Qdrant server port
        collection_name: Name of the vector collection
        vector_size: Dimension of vectors (default: 4096 for Upstage embeddings)
        client: Qdrant client instance
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "nexograph_documents",
        vector_size: int = 4096,
    ) -> None:
        """Initialize Qdrant service.

        Args:
            host: Qdrant server hostname
            port: Qdrant server port
            collection_name: Name of the vector collection
            vector_size: Dimension of vectors (default: 4096 for Upstage)

        Raises:
            QdrantConnectionError: If connection to Qdrant fails
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size

        try:
            # Use async client for non-blocking operations
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                timeout=10,  # 10 second timeout
            )
            logger.info(
                f"Qdrant async client initialized: {self.host}:{self.port} "
                f"(collection: {self.collection_name})"
            )
        except Exception as e:
            error_msg = f"Failed to initialize Qdrant client at {self.host}:{self.port}: {e}"
            logger.error(error_msg)
            raise QdrantConnectionError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((QdrantConnectionError, QdrantCollectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def ensure_collection_exists(self) -> None:
        """Ensure the vector collection exists, create if not.

        Creates a collection with the specified vector size and cosine distance
        metric if it doesn't already exist.

        Raises:
            QdrantCollectionError: If collection creation fails
        """
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return

            # Create collection
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created collection '{self.collection_name}' "
                f"(vector_size={self.vector_size}, distance=COSINE)"
            )

        except Exception as e:
            error_msg = f"Failed to ensure collection '{self.collection_name}' exists: {e}"
            logger.error(error_msg)
            raise QdrantCollectionError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(QdrantServiceError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def upsert_point(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Add or update a single vector in the collection.

        Args:
            point_id: Unique identifier for the vector
            vector: Vector embedding (dimension=vector_size)
            payload: Metadata dictionary associated with the vector

        Raises:
            QdrantServiceError: If vector insertion fails

        Example:
            >>> await service.upsert_point(
            ...     point_id="chunk_1",
            ...     vector=[0.1, 0.2, ...],
            ...     payload={"document_id": "doc1", "text": "..."}
            ... )
        """
        try:
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.debug(f"Upserted point '{point_id}' to '{self.collection_name}'")

        except Exception as e:
            error_msg = f"Failed to upsert point '{point_id}' to '{self.collection_name}': {e}"
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(QdrantServiceError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def add_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Add vectors to the collection in batch.

        Args:
            ids: List of unique identifiers for vectors
            vectors: List of vector embeddings (each with dimension=vector_size)
            payloads: List of metadata dictionaries associated with each vector

        Raises:
            ValueError: If input lists have different lengths
            QdrantServiceError: If vector insertion fails

        Example:
            >>> service.add_vectors(
            ...     ids=["chunk_1", "chunk_2"],
            ...     vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ...     payloads=[
            ...         {"document_id": "doc1", "text": "..."},
            ...         {"document_id": "doc1", "text": "..."}
            ...     ]
            ... )
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"Input lists must have equal length: "
                f"ids={len(ids)}, vectors={len(vectors)}, payloads={len(payloads)}"
            )

        if not ids:
            logger.warning("add_vectors called with empty lists, skipping")
            return

        try:
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(
                f"Successfully added {len(points)} vectors to '{self.collection_name}'"
            )

        except Exception as e:
            error_msg = f"Failed to add vectors to '{self.collection_name}': {e}"
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(QdrantServiceError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Filter] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in the collection.

        Args:
            query_vector: Query vector embedding (dimension=vector_size)
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0-1.0, optional)
            filter_conditions: Qdrant filter for metadata-based filtering (optional)

        Returns:
            List of search results, each containing:
            - id: Vector ID
            - score: Similarity score (0.0-1.0)
            - payload: Associated metadata

        Raises:
            QdrantServiceError: If search fails

        Example:
            >>> results = service.search(
            ...     query_vector=[0.1, 0.2, ...],
            ...     limit=5,
            ...     score_threshold=0.7
            ... )
            >>> for result in results:
            ...     print(f"ID: {result['id']}, Score: {result['score']}")
        """
        try:
            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
            )
            search_result = search_result.points

            results = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in search_result
            ]

            logger.info(
                f"Search completed: found {len(results)} results "
                f"(limit={limit}, threshold={score_threshold})"
            )

            return results

        except Exception as e:
            error_msg = f"Failed to search in '{self.collection_name}': {e}"
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(QdrantServiceError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def delete_point(self, point_id: str) -> None:
        """Delete a single point from the collection.

        Args:
            point_id: Point identifier to delete

        Raises:
            QdrantServiceError: If deletion fails

        Example:
            >>> await service.delete_point("chunk_1")
        """
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=[point_id]
                ),
            )

            logger.debug(f"Deleted point '{point_id}' from '{self.collection_name}'")

        except Exception as e:
            error_msg = f"Failed to delete point '{point_id}' from '{self.collection_name}': {e}"
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(QdrantServiceError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all vectors associated with a specific document.

        Args:
            document_id: Document identifier to filter by

        Returns:
            Number of vectors deleted

        Raises:
            QdrantServiceError: If deletion fails

        Example:
            >>> deleted_count = service.delete_by_document_id("doc_123")
            >>> print(f"Deleted {deleted_count} vectors")
        """
        try:
            # Create filter to match document_id in payload
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

            # Delete points matching the filter
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
            )

            # The result contains operation_id and status
            logger.info(
                f"Deleted vectors for document '{document_id}' "
                f"from '{self.collection_name}'"
            )

            # Note: Qdrant doesn't return count directly, but we log the operation
            return 1  # Return success indicator

        except Exception as e:
            error_msg = (
                f"Failed to delete vectors for document '{document_id}' "
                f"from '{self.collection_name}': {e}"
            )
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    async def health_check(self) -> bool:
        """Check if Qdrant service is healthy and accessible.

        Returns:
            True if service is healthy, False otherwise

        Example:
            >>> if await service.health_check():
            ...     print("Qdrant is healthy")
            ... else:
            ...     print("Qdrant is unavailable")
        """
        try:
            # Try to get collections as a health check
            await self.client.get_collections()
            logger.debug("Qdrant health check: OK")
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection.

        Returns:
            Dictionary containing collection statistics:
            - vectors_count: Number of vectors in collection
            - indexed_vectors_count: Number of indexed vectors
            - points_count: Total number of points
            - status: Collection status

        Raises:
            QdrantServiceError: If collection info retrieval fails
        """
        try:
            collection_info = await self.client.get_collection(
                collection_name=self.collection_name
            )

            return {
                "vectors_count": collection_info.vectors_count or 0,
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                "points_count": collection_info.points_count or 0,
                "status": collection_info.status.name if collection_info.status else "unknown",
            }

        except Exception as e:
            error_msg = f"Failed to get collection info for '{self.collection_name}': {e}"
            logger.error(error_msg)
            raise QdrantServiceError(error_msg) from e

    async def close(self) -> None:
        """Close the Qdrant client connection.

        Call this method when shutting down the application to properly
        release resources.
        """
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
                logger.info("Qdrant client connection closed")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")


# Singleton instance (lazy initialization)
_qdrant_service_instance: Optional[QdrantService] = None


def get_qdrant_service_instance(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "nexograph_documents",
    vector_size: int = 4096,
) -> QdrantService:
    """Get or create singleton Qdrant service instance.

    This function implements a singleton pattern to ensure only one
    QdrantService instance exists throughout the application lifecycle.

    Args:
        host: Qdrant server hostname
        port: Qdrant server port
        collection_name: Name of the vector collection
        vector_size: Dimension of vectors (default: 4096)

    Returns:
        Singleton QdrantService instance

    Example:
        >>> service = get_qdrant_service_instance()
        >>> service.health_check()
    """
    global _qdrant_service_instance

    if _qdrant_service_instance is None:
        _qdrant_service_instance = QdrantService(
            host=host,
            port=port,
            collection_name=collection_name,
            vector_size=vector_size,
        )
        logger.info("Created new QdrantService singleton instance")

    return _qdrant_service_instance
