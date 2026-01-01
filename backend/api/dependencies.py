"""Dependency injection utilities for FastAPI routes.

This module provides reusable dependencies that can be injected into FastAPI
route handlers using the Depends() mechanism.
"""

from fastapi import Depends

from backend.config import Settings, get_settings as _get_settings
from backend.services.qdrant_service import (
    QdrantService,
    get_qdrant_service_instance,
)
from backend.services.upstage import (
    UpstageLLMService,
    UpstageEmbeddingService,
    UpstageDocumentParseService,
)
from backend.services.lightrag_service import (
    LightRAGService,
    get_lightrag_service_instance,
)
from backend.services.document_service import DocumentService
from backend.models.database import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession


async def get_settings() -> Settings:
    """Get application settings as a FastAPI dependency.

    This async function wraps the cached get_settings() to provide a consistent
    dependency injection interface for FastAPI route handlers.

    Returns:
        Settings: Application configuration instance

    Example:
        @router.get("/endpoint")
        async def endpoint(settings: Settings = Depends(get_settings)):
            # Use settings here
            pass
    """
    return _get_settings()


def get_qdrant_service(settings: Settings = Depends(get_settings)) -> QdrantService:
    """Get Qdrant service instance as a FastAPI dependency.

    This function provides a singleton QdrantService instance configured
    with settings from the application configuration.

    Args:
        settings: Application settings (injected dependency)

    Returns:
        QdrantService: Configured Qdrant service instance

    Example:
        @router.post("/search")
        async def search(
            qdrant: QdrantService = Depends(get_qdrant_service)
        ):
            results = qdrant.search(query_vector=[...])
            return results
    """
    return get_qdrant_service_instance(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=4096,  # Upstage embedding dimension
    )


def get_upstage_llm_service(
    settings: Settings = Depends(get_settings),
) -> UpstageLLMService:
    """Get Upstage LLM service instance as a FastAPI dependency.

    Args:
        settings: Application settings (injected dependency)

    Returns:
        UpstageLLMService: Configured Upstage LLM service

    Example:
        @router.post("/generate")
        async def generate(
            llm: UpstageLLMService = Depends(get_upstage_llm_service)
        ):
            response = await llm.chat_completion(messages=[...])
            return response
    """
    return UpstageLLMService(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
    )


def get_upstage_embedding_service(
    settings: Settings = Depends(get_settings),
) -> UpstageEmbeddingService:
    """Get Upstage Embedding service instance as a FastAPI dependency.

    Args:
        settings: Application settings (injected dependency)

    Returns:
        UpstageEmbeddingService: Configured Upstage Embedding service

    Example:
        @router.post("/embed")
        async def embed(
            embedding: UpstageEmbeddingService = Depends(get_upstage_embedding_service)
        ):
            vector = await embedding.get_embedding(text="...", model_type="query")
            return vector
    """
    return UpstageEmbeddingService(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
    )


def get_upstage_document_parse_service(
    settings: Settings = Depends(get_settings),
) -> UpstageDocumentParseService:
    """Get Upstage Document Parse service instance as a FastAPI dependency.

    Args:
        settings: Application settings (injected dependency)

    Returns:
        UpstageDocumentParseService: Configured Upstage Document Parse service

    Example:
        @router.post("/parse")
        async def parse(
            parser: UpstageDocumentParseService = Depends(get_upstage_document_parse_service)
        ):
            result = await parser.parse_document(file_content=..., filename="...")
            return result
    """
    return UpstageDocumentParseService(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
    )


async def get_db() -> AsyncSession:
    """Get async database session as a FastAPI dependency.

    This dependency provides an async SQLAlchemy session for database operations.
    The session is automatically committed on success and rolled back on error.

    Yields:
        AsyncSession: Database session

    Example:
        @router.get("/documents")
        async def get_documents(
            db: AsyncSession = Depends(get_db)
        ):
            result = await db.execute(select(Document))
            documents = result.scalars().all()
            return documents
    """
    async with get_db_session() as session:
        yield session


def get_lightrag_service(
    settings: Settings = Depends(get_settings),
    llm: UpstageLLMService = Depends(get_upstage_llm_service),
    embedding: UpstageEmbeddingService = Depends(get_upstage_embedding_service),
) -> LightRAGService:
    """Get LightRAG service instance as a FastAPI dependency.

    Args:
        settings: Application settings (injected dependency)
        llm: Upstage LLM service (injected dependency)
        embedding: Upstage Embedding service (injected dependency)

    Returns:
        LightRAGService: Configured LightRAG service instance

    Example:
        @router.post("/graph/insert")
        async def insert_to_graph(
            lightrag: LightRAGService = Depends(get_lightrag_service)
        ):
            await lightrag.insert_document(text="...")
            return {"status": "inserted"}
    """
    return get_lightrag_service_instance(
        working_dir=settings.lightrag_working_dir,
        llm_service=llm,
        embedding_service=embedding,
    )


async def get_document_service(
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant_service),
    parser: UpstageDocumentParseService = Depends(get_upstage_document_parse_service),
    embedding: UpstageEmbeddingService = Depends(get_upstage_embedding_service),
    lightrag: LightRAGService = Depends(get_lightrag_service),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Get DocumentService instance as a FastAPI dependency.

    This creates a new DocumentService instance for each request with all
    required services injected.

    Args:
        db: Database session (injected dependency)
        qdrant: Qdrant service (injected dependency)
        parser: Upstage Document Parse service (injected dependency)
        embedding: Upstage Embedding service (injected dependency)
        lightrag: LightRAG service (injected dependency)
        settings: Application settings (injected dependency)

    Returns:
        DocumentService: Configured document service instance

    Example:
        @router.post("/documents/upload")
        async def upload(
            doc_service: DocumentService = Depends(get_document_service)
        ):
            result = await doc_service.process_document(...)
            return result
    """
    return DocumentService(
        db_session=db,
        qdrant_service=qdrant,
        upstage_parser=parser,
        upstage_embedding=embedding,
        lightrag_service=lightrag,
        parsed_dir="./data/parsed",
    )
