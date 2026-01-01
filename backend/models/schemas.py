"""Pydantic schemas for API request/response models.

This module defines all the data models used for API endpoints,
including validation rules, examples, and documentation.
"""

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Type Aliases
# ============================================================================

DocumentCategory = Literal["paper", "blog", "documentation"]
QueryMode = Literal["local", "global", "hybrid", "naive"]
ServiceStatus = Literal["connected", "disconnected", "configured", "unconfigured", "initialized", "not_initialized"]
OverallStatus = Literal["healthy", "degraded", "unhealthy"]
ProcessingStatus = Literal["processing", "completed", "failed"]


# ============================================================================
# Health Check Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Overall system health status
        version: Application version
        services: Status of each service component
    """

    status: OverallStatus = Field(
        description="Overall system health status",
        examples=["healthy"]
    )
    version: str = Field(
        description="Application version",
        examples=["0.1.0"]
    )
    services: dict[str, ServiceStatus] = Field(
        description="Status of each service component",
        examples=[{
            "qdrant": "connected",
            "upstage": "configured",
            "lightrag": "initialized"
        }]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "services": {
                    "qdrant": "connected",
                    "upstage": "configured",
                    "lightrag": "initialized"
                }
            }
        }


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Document upload response model.

    This schema represents both processing and completed states.

    Attributes:
        document_id: Unique identifier for the uploaded document
        status: Processing status (processing/completed/failed)
        message: Status message (optional)
        title: Document title (only for completed)
        chunks_count: Number of text chunks created (only for completed)
        entities_extracted: Number of entities extracted (only for completed)
        processing_time_ms: Processing time in milliseconds (only for completed)
    """

    document_id: str = Field(
        description="Unique identifier for the uploaded document",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    status: ProcessingStatus = Field(
        description="Processing status",
        examples=["processing"]
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message",
        examples=["Document upload started"]
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title (only for completed status)",
        examples=["Attention Is All You Need"]
    )
    chunks_count: Optional[int] = Field(
        default=None,
        description="Number of text chunks created",
        examples=[10]
    )
    entities_extracted: Optional[int] = Field(
        default=None,
        description="Number of entities extracted from the document",
        examples=[25]
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Processing time in milliseconds",
        examples=[3500]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "examples": [
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "processing",
                    "message": "Document upload started"
                },
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "title": "Attention Is All You Need",
                    "chunks_count": 10,
                    "entities_extracted": 25,
                    "processing_time_ms": 3500
                }
            ]
        }


class DocumentListItem(BaseModel):
    """Document list item model.

    Represents a single document in the list view with summary information.

    Attributes:
        id: Unique document identifier
        title: Document title
        category: Document category
        tags: List of tags associated with the document
        chunks_count: Number of text chunks
        created_at: Document creation timestamp
    """

    id: str = Field(
        description="Unique document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    title: str = Field(
        description="Document title",
        examples=["Attention Is All You Need"]
    )
    category: DocumentCategory = Field(
        description="Document category",
        examples=["paper"]
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags associated with the document",
        examples=[["AI", "NLP", "Transformer"]]
    )
    chunks_count: int = Field(
        description="Number of text chunks created from this document",
        examples=[10]
    )
    created_at: datetime = Field(
        description="Document creation timestamp in ISO 8601 format",
        examples=["2025-01-15T10:30:00Z"]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Attention Is All You Need",
                "category": "paper",
                "tags": ["AI", "NLP", "Transformer"],
                "chunks_count": 10,
                "created_at": "2025-01-15T10:30:00Z"
            }
        }


class DocumentListResponse(BaseModel):
    """Document list response with pagination.

    Attributes:
        documents: List of document items
        total: Total number of documents matching the query
        page: Current page number (1-indexed)
        limit: Number of items per page
    """

    documents: list[DocumentListItem] = Field(
        description="List of document items",
        examples=[[]]
    )
    total: int = Field(
        description="Total number of documents matching the query",
        examples=[100]
    )
    page: int = Field(
        ge=1,
        description="Current page number (1-indexed)",
        examples=[1]
    )
    limit: int = Field(
        ge=1,
        le=100,
        description="Number of items per page (max 100)",
        examples=[20]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "title": "Attention Is All You Need",
                        "category": "paper",
                        "tags": ["AI", "NLP"],
                        "chunks_count": 10,
                        "created_at": "2025-01-15T10:30:00Z"
                    }
                ],
                "total": 100,
                "page": 1,
                "limit": 20
            }
        }


class DocumentDetailResponse(BaseModel):
    """Document detail response model.

    Provides comprehensive information about a single document.

    Attributes:
        id: Unique document identifier
        title: Document title
        category: Document category
        tags: List of tags
        original_filename: Original uploaded filename
        parsed_content_preview: First 500 characters of parsed content
        chunks_count: Number of text chunks
        entities: List of extracted entities
        created_at: Document creation timestamp
        file_size_bytes: Original file size in bytes
    """

    id: str = Field(
        description="Unique document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    title: str = Field(
        description="Document title",
        examples=["Attention Is All You Need"]
    )
    category: DocumentCategory = Field(
        description="Document category",
        examples=["paper"]
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags associated with the document",
        examples=[["AI", "NLP", "Transformer"]]
    )
    original_filename: str = Field(
        description="Original uploaded filename",
        examples=["attention_paper.pdf"]
    )
    parsed_content_preview: str = Field(
        description="First 500 characters of parsed content",
        examples=["# Attention Is All You Need\n\nAbstract: The dominant..."]
    )
    chunks_count: int = Field(
        description="Number of text chunks created from this document",
        examples=[10]
    )
    entities: list[str] = Field(
        default_factory=list,
        description="List of entities extracted from the document",
        examples=[["GPT-4", "Transformer", "Attention", "BERT"]]
    )
    created_at: datetime = Field(
        description="Document creation timestamp in ISO 8601 format",
        examples=["2025-01-15T10:30:00Z"]
    )
    file_size_bytes: int = Field(
        description="Original file size in bytes",
        examples=[1024000]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Attention Is All You Need",
                "category": "paper",
                "tags": ["AI", "NLP"],
                "original_filename": "attention_paper.pdf",
                "parsed_content_preview": "# Attention Is All You Need\n\nAbstract: The dominant...",
                "chunks_count": 10,
                "entities": ["GPT-4", "Transformer", "Attention"],
                "created_at": "2025-01-15T10:30:00Z",
                "file_size_bytes": 1024000
            }
        }


# ============================================================================
# Chat Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model for Q&A queries.

    Attributes:
        query: User's question or query
        mode: Search mode (local/global/hybrid/naive)
        top_k: Number of chunks to retrieve (1-20)
        include_sources: Whether to include source information
    """

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="User's question or query",
        examples=["Transformer의 Attention 메커니즘을 설명해줘"]
    )
    mode: QueryMode = Field(
        default="hybrid",
        description="Search mode: local (entities), global (communities), hybrid (both), or naive (simple)",
        examples=["hybrid"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (1-20)",
        examples=[5]
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source document information in response",
        examples=[True]
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not just whitespace.

        Args:
            v: Query string to validate

        Returns:
            Validated query string

        Raises:
            ValueError: If query is empty or only whitespace
        """
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "query": "Transformer의 Attention 메커니즘을 설명해줘",
                "mode": "hybrid",
                "top_k": 5,
                "include_sources": True
            }
        }


class SourceInfo(BaseModel):
    """Source document information for chat responses.

    Attributes:
        document_id: Source document identifier
        document_title: Source document title
        chunk_preview: Preview of the relevant chunk (first 200 chars)
        relevance_score: Relevance score (0.0-1.0)
    """

    document_id: str = Field(
        description="Source document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    document_title: str = Field(
        description="Source document title",
        examples=["Attention Is All You Need"]
    )
    chunk_preview: str = Field(
        description="Preview of the relevant chunk (first 200 characters)",
        examples=["The Transformer uses self-attention mechanisms to process..."]
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score between 0.0 and 1.0",
        examples=[0.95]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_title": "Attention Is All You Need",
                "chunk_preview": "The Transformer uses self-attention mechanisms to process...",
                "relevance_score": 0.95
            }
        }


class ChatResponse(BaseModel):
    """Chat response model with answer and sources.

    Attributes:
        answer: Generated answer to the user's query
        sources: List of source documents (if include_sources=True)
        mode_used: The search mode that was used
        processing_time_ms: Processing time in milliseconds
    """

    answer: str = Field(
        description="Generated answer to the user's query",
        examples=["Transformer의 Attention 메커니즘은 시퀀스의 각 위치에서..."]
    )
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="List of source documents used to generate the answer",
        examples=[[]]
    )
    mode_used: QueryMode = Field(
        description="The search mode that was used for this query",
        examples=["hybrid"]
    )
    processing_time_ms: int = Field(
        ge=0,
        description="Processing time in milliseconds",
        examples=[1200]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "answer": "Transformer의 Attention 메커니즘은 시퀀스의 각 위치에서 다른 모든 위치와의 관계를 계산하여...",
                "sources": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_title": "Attention Is All You Need",
                        "chunk_preview": "The Transformer uses self-attention mechanisms...",
                        "relevance_score": 0.95
                    }
                ],
                "mode_used": "hybrid",
                "processing_time_ms": 1200
            }
        }


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model.

    Attributes:
        error: Error code or type
        message: Human-readable error message
        detail: Additional error details (optional)
    """

    error: str = Field(
        description="Error code or type",
        examples=["invalid_file_type"]
    )
    message: str = Field(
        description="Human-readable error message",
        examples=["Only PDF and Markdown files are supported"]
    )
    detail: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error details (field errors, context, etc.)",
        examples=[{"supported_types": [".pdf", ".md"]}]
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "examples": [
                {
                    "error": "invalid_file_type",
                    "message": "Only PDF and Markdown files are supported",
                    "detail": {
                        "supported_types": [".pdf", ".md"]
                    }
                },
                {
                    "error": "document_not_found",
                    "message": "Document with ID '123' not found",
                    "detail": None
                }
            ]
        }
