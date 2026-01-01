"""LightRAG service for knowledge graph management.

This module integrates LightRAG with Upstage services (LLM and Embedding)
to build and query knowledge graphs from documents.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.services.upstage import UpstageLLMService, UpstageEmbeddingService

logger = logging.getLogger(__name__)


class LightRAGServiceError(Exception):
    """Base exception for LightRAG service errors."""

    pass


class LightRAGService:
    """Service for LightRAG knowledge graph operations.

    This service wraps LightRAG and provides integration with Upstage
    services for LLM and embedding operations.

    Attributes:
        working_dir: Directory for storing graph data
        llm_service: Upstage LLM service instance
        embedding_service: Upstage Embedding service instance
        rag: LightRAG instance (initialized lazily)
    """

    def __init__(
        self,
        working_dir: str,
        llm_service: UpstageLLMService,
        embedding_service: UpstageEmbeddingService,
    ) -> None:
        """Initialize LightRAG service.

        Args:
            working_dir: Directory for storing graph data
            llm_service: Upstage LLM service for text generation
            embedding_service: Upstage Embedding service for vectors

        Raises:
            LightRAGServiceError: If initialization fails
        """
        self.working_dir = Path(working_dir)
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.rag: Optional[Any] = None  # Will be initialized async

        # Create working directory if it doesn't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LightRAGService initialized (working_dir: {working_dir})")

    async def _initialize_rag(self) -> None:
        """Initialize LightRAG instance with custom functions.

        This is called lazily on first use to avoid import errors
        and allow proper async initialization.

        Raises:
            LightRAGServiceError: If LightRAG initialization fails
        """
        if self.rag is not None:
            return  # Already initialized

        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.utils import wrap_embedding_func_with_attrs

            # Define custom LLM function for LightRAG
            async def custom_llm_func(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: list = [],
                keyword_extraction: bool = False,
                **kwargs,
            ) -> str:
                """Custom LLM function using Upstage Solar.

                Args:
                    prompt: User prompt
                    system_prompt: System prompt (optional)
                    history_messages: Conversation history (optional)
                    keyword_extraction: Whether this is for keyword extraction
                    **kwargs: Additional arguments

                Returns:
                    Generated text response
                """
                # Build messages list
                messages = []

                # Add system prompt if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                # Add history messages
                messages.extend(history_messages)

                # Add current prompt
                messages.append({"role": "user", "content": prompt})

                # Use lower temperature for keyword extraction
                temperature = 0.0 if keyword_extraction else kwargs.get("temperature", 0.1)

                # Call Upstage LLM
                response = await self.llm_service.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=kwargs.get("max_tokens", 2048),
                )

                return response

            # Define custom embedding function for LightRAG
            @wrap_embedding_func_with_attrs(
                embedding_dim=4096,  # Upstage Solar Embedding dimension
                max_token_size=8192,
            )
            async def custom_embedding_func(texts: list[str]) -> np.ndarray:
                """Custom embedding function using Upstage Solar Embedding.

                Args:
                    texts: List of texts to embed

                Returns:
                    NumPy array of embedding vectors (shape: [len(texts), 4096])
                """
                if not texts:
                    return np.array([], dtype=np.float32)

                # Use batch embedding for efficiency
                if len(texts) == 1:
                    # Single text
                    embedding = await self.embedding_service.get_embedding(
                        text=texts[0],
                        model_type="passage",  # Use passage model for documents
                    )
                    embeddings = [embedding]
                else:
                    # Multiple texts
                    embeddings = await self.embedding_service.get_embeddings_batch(
                        texts=texts,
                        model_type="passage",
                    )

                # Convert to NumPy array
                return np.array(embeddings, dtype=np.float32)

            # Initialize LightRAG with custom functions
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=custom_llm_func,
                embedding_func=custom_embedding_func,
                # Use default NetworkX graph storage (in-memory)
                # In future phases, can switch to Neo4j
            )

            # Initialize storages (required for LightRAG)
            await self.rag.initialize_storages()

            logger.info("LightRAG instance initialized successfully")

        except ImportError as e:
            error_msg = (
                f"Failed to import LightRAG. Make sure 'lightrag-hku' is installed: {e}"
            )
            logger.error(error_msg)
            raise LightRAGServiceError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to initialize LightRAG: {e}"
            logger.error(error_msg)
            raise LightRAGServiceError(error_msg) from e

    async def insert_document(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert a document into the knowledge graph.

        Args:
            text: Document text content
            metadata: Optional metadata about the document

        Raises:
            LightRAGServiceError: If document insertion fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")

        # Ensure RAG is initialized
        await self._initialize_rag()

        try:
            # Insert document into knowledge graph
            logger.info(
                f"Inserting document into knowledge graph "
                f"(length: {len(text)} chars)"
            )

            await self.rag.ainsert(text)

            logger.info(
                f"Document inserted successfully "
                f"(metadata: {metadata if metadata else 'none'})"
            )

        except Exception as e:
            error_msg = f"Failed to insert document into knowledge graph: {e}"
            logger.error(error_msg)
            raise LightRAGServiceError(error_msg) from e

    async def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        top_k: int = 60,
        chunk_top_k: int = 20,
        only_need_context: bool = False,
    ) -> dict[str, Any]:
        """Query the knowledge graph.

        Args:
            query_text: Query text
            mode: Search mode - "local", "global", "hybrid", "naive", or "mix"
            top_k: Number of top entities/relations to retrieve (default: 60)
            chunk_top_k: Number of text chunks to retrieve (default: 20)
            only_need_context: If True, only return context without LLM response

        Returns:
            Dictionary containing:
            - answer: Generated answer text (if only_need_context=False)
            - context: Retrieved context (if only_need_context=True)
            - mode_used: The search mode that was used

        Raises:
            LightRAGServiceError: If query fails
            ValueError: If query text is empty or mode is invalid
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        valid_modes = ["local", "global", "hybrid", "naive", "mix"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Ensure RAG is initialized
        await self._initialize_rag()

        try:
            from lightrag import QueryParam

            logger.info(
                f"Querying knowledge graph (mode: {mode}, top_k: {top_k}, "
                f"chunk_top_k: {chunk_top_k}, query: '{query_text[:50]}...')"
            )

            # Query the knowledge graph
            result = await self.rag.aquery(
                query=query_text,
                param=QueryParam(
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    only_need_context=only_need_context,
                ),
            )

            logger.info(f"Query completed (mode: {mode}, result length: {len(str(result))})")

            # Format response based on only_need_context flag
            if only_need_context:
                return {
                    "context": str(result),
                    "mode_used": mode,
                }
            else:
                return {
                    "answer": str(result),
                    "context": [],  # LightRAG doesn't expose context directly in answer mode
                    "mode_used": mode,
                }

        except Exception as e:
            error_msg = f"Failed to query knowledge graph: {e}"
            logger.error(error_msg)
            raise LightRAGServiceError(error_msg) from e

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary containing graph statistics:
            - working_dir: Path to working directory
            - initialized: Whether RAG is initialized
            - graph_files: List of graph files in working directory

        Note:
            LightRAG doesn't expose graph statistics directly in 1.4.x,
            so we provide basic file system information.
        """
        stats = {
            "working_dir": str(self.working_dir),
            "initialized": self.rag is not None,
            "graph_files": [],
        }

        # List files in working directory
        if self.working_dir.exists():
            graph_files = list(self.working_dir.glob("*"))
            stats["graph_files"] = [
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size if f.is_file() else 0,
                    "is_dir": f.is_dir(),
                }
                for f in graph_files
            ]

        return stats

    async def health_check(self) -> bool:
        """Check if LightRAG service is healthy.

        Returns:
            True if service is initialized and working directory exists
        """
        try:
            # Check if working directory exists
            if not self.working_dir.exists():
                logger.warning(f"Working directory does not exist: {self.working_dir}")
                return False

            # Check if LightRAG can be initialized
            await self._initialize_rag()

            return self.rag is not None

        except Exception as e:
            logger.warning(f"LightRAG health check failed: {e}")
            return False

    async def clear_graph(self) -> None:
        """Clear all graph data.

        Warning: This will delete all knowledge graph data!
        Use only for testing or cleanup.

        Raises:
            LightRAGServiceError: If clearing fails
        """
        try:
            import shutil

            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.rag = None  # Reset RAG instance
                logger.warning(f"Cleared all graph data from {self.working_dir}")
            else:
                logger.info("Working directory doesn't exist, nothing to clear")

        except Exception as e:
            error_msg = f"Failed to clear graph data: {e}"
            logger.error(error_msg)
            raise LightRAGServiceError(error_msg) from e


# Singleton instance (lazy initialization)
_lightrag_service_instance: Optional[LightRAGService] = None


def get_lightrag_service_instance(
    working_dir: str,
    llm_service: UpstageLLMService,
    embedding_service: UpstageEmbeddingService,
) -> LightRAGService:
    """Get or create singleton LightRAG service instance.

    Args:
        working_dir: Directory for storing graph data
        llm_service: Upstage LLM service
        embedding_service: Upstage Embedding service

    Returns:
        Singleton LightRAGService instance
    """
    global _lightrag_service_instance

    if _lightrag_service_instance is None:
        _lightrag_service_instance = LightRAGService(
            working_dir=working_dir,
            llm_service=llm_service,
            embedding_service=embedding_service,
        )
        logger.info("Created new LightRAGService singleton instance")

    return _lightrag_service_instance
