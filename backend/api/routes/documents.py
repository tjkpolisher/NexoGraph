"""Document management API endpoints.

This module provides REST API endpoints for document upload, retrieval, and deletion.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_document_service, get_db, get_qdrant_service
from backend.models.database import Document
from backend.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse,
)
from backend.services.document_service import (
    DocumentService,
    DocumentProcessingError,
)
from backend.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["documents"])

# File upload constraints
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".md"}


# ============================================================================
# Helper Functions
# ============================================================================


def validate_file_extension(filename: str) -> None:
    """Validate that file has an allowed extension.

    Args:
        filename: Name of the file to validate

    Raises:
        HTTPException: If file extension is not allowed
    """
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
                "detail": {
                    "uploaded_extension": file_ext,
                    "allowed_extensions": list(ALLOWED_EXTENSIONS),
                },
            },
        )


def validate_file_size(file_size: int) -> None:
    """Validate that file size is within limits.

    Args:
        file_size: Size of file in bytes

    Raises:
        HTTPException: If file is too large
    """
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
                "detail": {
                    "file_size_bytes": file_size,
                    "max_size_bytes": MAX_FILE_SIZE_BYTES,
                    "max_size_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
                },
            },
        )


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="""
    Upload a document (PDF or Markdown) for processing.

    The document will be:
    1. Parsed into markdown (if PDF)
    2. Chunked into smaller pieces
    3. Embedded and stored in vector database
    4. Added to knowledge graph

    Supported formats: PDF (.pdf), Markdown (.md)
    Maximum file size: 50MB
    """,
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    title: Optional[str] = Form(None, description="Document title (defaults to filename)"),
    category: str = Form("paper", description="Document category (paper/blog/documentation)"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Upload and process a document.

    Args:
        file: Uploaded file (PDF or Markdown)
        title: Optional document title (defaults to filename)
        category: Document category (paper/blog/documentation)
        tags: Optional comma-separated tags
        doc_service: Document service (injected dependency)

    Returns:
        DocumentUploadResponse with processing results

    Raises:
        HTTPException 400: Invalid file type
        HTTPException 413: File too large
        HTTPException 500: Processing error
    """
    logger.info(f"Document upload request: {file.filename}")

    # Validate file extension
    validate_file_extension(file.filename)

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # Validate file size
    validate_file_size(file_size)

    # Parse tags
    tags_list = []
    if tags:
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    # Process document
    try:
        result = await doc_service.process_document(
            file_content=file_content,
            filename=file.filename,
            title=title,
            category=category,
            tags=tags_list,
        )

        logger.info(
            f"Document uploaded successfully "
            f"(id: {result.document_id}, filename: {file.filename})"
        )

        return result

    except DocumentProcessingError as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "processing_error",
                "message": "Failed to process document",
                "detail": {"reason": str(e)},
            },
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error during document upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "detail": {"reason": str(e)},
            },
        ) from e


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
    description="""
    Retrieve a paginated list of documents with optional filtering.

    Query parameters:
    - page: Page number (1-indexed, default: 1)
    - limit: Items per page (1-100, default: 20)
    - category: Filter by category (paper/blog/documentation)
    - search: Search in document titles
    """,
)
async def list_documents(
    page: int = 1,
    limit: int = 20,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """List documents with pagination and filtering.

    Args:
        page: Page number (1-indexed, default: 1)
        limit: Items per page (1-100, default: 20)
        category: Optional category filter
        search: Optional search query for titles
        db: Database session (injected dependency)

    Returns:
        DocumentListResponse with paginated results

    Raises:
        HTTPException 400: Invalid query parameters
        HTTPException 500: Database error
    """
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_page", "message": "Page must be >= 1"},
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_limit",
                "message": "Limit must be between 1 and 100",
            },
        )

    try:
        # Build base query
        query = select(Document).where(Document.status == "completed")

        # Apply filters
        if category:
            query = query.where(Document.category == category)

        if search:
            query = query.where(Document.title.ilike(f"%{search}%"))

        # Order by creation date (newest first)
        query = query.order_by(Document.created_at.desc())

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)

        # Execute query
        result = await db.execute(query)
        documents = result.scalars().all()

        # Convert to response format
        document_items = [
            DocumentListItem(
                id=doc.id,
                title=doc.title,
                category=doc.category,
                tags=doc.tags,
                chunks_count=doc.chunks_count,
                created_at=doc.created_at,
            )
            for doc in documents
        ]

        logger.info(
            f"Listed {len(document_items)} documents "
            f"(page: {page}, limit: {limit}, total: {total})"
        )

        return DocumentListResponse(
            documents=document_items,
            total=total,
            page=page,
            limit=limit,
        )

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "database_error",
                "message": "Failed to retrieve documents",
                "detail": {"reason": str(e)},
            },
        ) from e


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="""
    Retrieve detailed information about a specific document.

    Includes:
    - Full metadata
    - Parsed content preview (first 500 characters)
    - Extracted entities (placeholder for Phase 2)
    """,
)
async def get_document_detail(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Get detailed information about a document.

    Args:
        document_id: Document UUID
        db: Database session (injected dependency)

    Returns:
        DocumentDetailResponse with full document information

    Raises:
        HTTPException 404: Document not found
        HTTPException 500: Database or file read error
    """
    try:
        # Fetch document from database
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "document_not_found",
                    "message": f"Document with ID '{document_id}' not found",
                },
            )

        # Read parsed content preview
        parsed_content_preview = ""
        if document.parsed_content_path:
            try:
                parsed_file = Path(document.parsed_content_path)
                if parsed_file.exists():
                    content = parsed_file.read_text(encoding="utf-8")
                    parsed_content_preview = content[:500]
                else:
                    logger.warning(
                        f"Parsed file not found: {document.parsed_content_path}"
                    )
                    parsed_content_preview = "[Parsed content file not found]"
            except Exception as e:
                logger.error(f"Error reading parsed content: {e}")
                parsed_content_preview = "[Error reading parsed content]"

        # TODO: Extract entities from LightRAG in Phase 2
        entities = []

        response = DocumentDetailResponse(
            id=document.id,
            title=document.title,
            category=document.category,
            tags=document.tags,
            original_filename=document.original_filename,
            parsed_content_preview=parsed_content_preview,
            chunks_count=document.chunks_count,
            entities=entities,
            created_at=document.created_at,
            file_size_bytes=document.file_size_bytes,
        )

        logger.info(f"Retrieved document details (id: {document_id})")

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "retrieval_error",
                "message": "Failed to retrieve document details",
                "detail": {"reason": str(e)},
            },
        ) from e


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="""
    Delete a document and all associated data.

    This will:
    1. Remove document record from database
    2. Delete all vector embeddings from Qdrant
    3. Delete parsed markdown file

    Note: LightRAG data is not removed in Phase 1 (no selective deletion support)
    """,
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant_service),
) -> None:
    """Delete a document and all associated data.

    Args:
        document_id: Document UUID to delete
        db: Database session (injected dependency)
        qdrant: Qdrant service (injected dependency)

    Raises:
        HTTPException 404: Document not found
        HTTPException 500: Deletion error
    """
    try:
        # Fetch document from database
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "document_not_found",
                    "message": f"Document with ID '{document_id}' not found",
                },
            )

        logger.info(f"Deleting document (id: {document_id}, title: {document.title})")

        # Step 1: Delete vectors from Qdrant
        try:
            await qdrant.delete_by_document_id(document_id)
            logger.info(f"Deleted vectors from Qdrant for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete vectors from Qdrant: {e}")
            # Continue with deletion even if Qdrant fails

        # Step 2: Delete parsed markdown file
        if document.parsed_content_path:
            try:
                parsed_file = Path(document.parsed_content_path)
                if parsed_file.exists():
                    parsed_file.unlink()
                    logger.info(f"Deleted parsed file: {document.parsed_content_path}")
            except Exception as e:
                logger.error(f"Failed to delete parsed file: {e}")
                # Continue with deletion even if file deletion fails

        # Step 3: Delete database record
        await db.delete(document)
        await db.commit()

        logger.info(f"Document deleted successfully (id: {document_id})")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "deletion_error",
                "message": "Failed to delete document",
                "detail": {"reason": str(e)},
            },
        ) from e
