"""Document processing orchestration service.

This module orchestrates the complete document processing pipeline,
coordinating between parsing, embedding, vector storage, and knowledge graph services.
"""

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.database import Document
from backend.models.schemas import DocumentUploadResponse, ProcessingStatus
from backend.services.upstage.document_parser import UpstageDocumentParseService
from backend.services.upstage.embedding import UpstageEmbeddingService
from backend.services.qdrant_service import QdrantService
from backend.services.lightrag_service import LightRAGService

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""

    pass


# ============================================================================
# Text Chunking Utilities
# ============================================================================


def chunk_markdown_text(
    text: str,
    max_chunk_size: int = 3000,
    overlap_size: int = 200,
) -> list[str]:
    """Chunk markdown text based on sections with overlap.

    This function splits markdown text into chunks, prioritizing section
    boundaries (headers) and ensuring chunks don't exceed max_chunk_size.
    It adds overlap between chunks to maintain context.

    Algorithm:
    1. Split by section headers (#, ##, ###, etc.)
    2. If a section exceeds max_chunk_size, split by paragraphs
    3. Add overlap between consecutive chunks

    Args:
        text: Markdown text to chunk
        max_chunk_size: Maximum characters per chunk (default: 3000)
            Note: Upstage Embedding API has a 4000 token limit.
            With ~2.5-3 chars/token, 3000 chars ≈ 1000-1200 tokens (safe margin)
        overlap_size: Characters to overlap between chunks (default: 200)

    Returns:
        List of text chunks with overlap

    Example:
        >>> text = "# Section 1\\n\\nContent...\\n\\n## Section 2\\n\\nMore content..."
        >>> chunks = chunk_markdown_text(text, max_chunk_size=1000)
        >>> len(chunks)
        2
    """
    if not text or not text.strip():
        return []

    # Pattern to match markdown headers (# Header, ## Header, etc.)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Find all header positions
    headers = []
    for match in header_pattern.finditer(text):
        headers.append({
            "level": len(match.group(1)),  # Number of # symbols
            "title": match.group(2).strip(),
            "start": match.start(),
            "end": match.end(),
        })

    # If no headers found, treat whole text as one section
    if not headers:
        return _chunk_by_paragraphs(text, max_chunk_size, overlap_size)

    # Split text into sections
    sections = []
    for i, header in enumerate(headers):
        # Section starts at header position
        section_start = header["start"]

        # Section ends at next header (or end of text)
        if i + 1 < len(headers):
            section_end = headers[i + 1]["start"]
        else:
            section_end = len(text)

        section_text = text[section_start:section_end].strip()
        sections.append({
            "header": header,
            "text": section_text,
            "size": len(section_text),
        })

    # Process sections into chunks
    chunks = []
    current_chunk = ""

    for section in sections:
        section_text = section["text"]

        # If adding this section exceeds max size, finalize current chunk
        if current_chunk and len(current_chunk) + len(section_text) > max_chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            current_chunk = _get_overlap(current_chunk, overlap_size) + "\n\n"

        # If section itself is too large, split it
        if len(section_text) > max_chunk_size:
            # Finalize current chunk if exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split large section by paragraphs
            section_chunks = _chunk_by_paragraphs(
                section_text, max_chunk_size, overlap_size
            )
            chunks.extend(section_chunks)
        else:
            # Add section to current chunk
            current_chunk += section_text + "\n\n"

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Ensure no chunk is too large (safety check)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            # Force split if still too large
            final_chunks.extend(_chunk_by_paragraphs(chunk, max_chunk_size, overlap_size))
        else:
            final_chunks.append(chunk)

    return final_chunks


def _chunk_by_paragraphs(
    text: str,
    max_chunk_size: int,
    overlap_size: int,
) -> list[str]:
    """Split text into chunks by paragraph boundaries.

    Args:
        text: Text to split
        max_chunk_size: Maximum chunk size
        overlap_size: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Split by double newline (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If adding paragraph exceeds size, finalize chunk
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = _get_overlap(current_chunk, overlap_size) + "\n\n"

        # If single paragraph is too large, split by sentences
        if len(paragraph) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split by sentences (simple split on . ! ?)
            sentences = re.split(r"([.!?]\s+)", paragraph)
            sentence_chunk = ""

            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
                full_sentence = sentence + delimiter

                if sentence_chunk and len(sentence_chunk) + len(full_sentence) > max_chunk_size:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = _get_overlap(sentence_chunk, overlap_size)

                sentence_chunk += full_sentence

            if sentence_chunk.strip():
                chunks.append(sentence_chunk.strip())
        else:
            current_chunk += paragraph + "\n\n"

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _get_overlap(text: str, overlap_size: int) -> str:
    """Get the last overlap_size characters from text.

    Args:
        text: Source text
        overlap_size: Number of characters to extract

    Returns:
        Last overlap_size characters (or less if text is shorter)
    """
    if len(text) <= overlap_size:
        return text

    # Get last overlap_size characters
    overlap = text[-overlap_size:]

    # Try to start at a word boundary
    first_space = overlap.find(" ")
    if first_space > 0 and first_space < 100:  # Don't skip too much
        overlap = overlap[first_space + 1:]

    return overlap


# ============================================================================
# Document Service
# ============================================================================


class DocumentService:
    """Service for orchestrating document processing pipeline.

    This service coordinates the entire document processing workflow:
    1. Parse document with Upstage Document Parse API
    2. Chunk the parsed markdown
    3. Generate embeddings for chunks
    4. Store chunks in Qdrant vector database
    5. Add document to LightRAG knowledge graph
    6. Save metadata to SQL database

    Attributes:
        db_session: SQLAlchemy async session for database operations
        qdrant_service: Qdrant vector database service
        upstage_parser: Upstage Document Parse service
        upstage_embedding: Upstage Embedding service
        lightrag_service: LightRAG knowledge graph service
        parsed_dir: Directory for storing parsed markdown files
    """

    def __init__(
        self,
        db_session: AsyncSession,
        qdrant_service: QdrantService,
        upstage_parser: UpstageDocumentParseService,
        upstage_embedding: UpstageEmbeddingService,
        lightrag_service: LightRAGService,
        parsed_dir: str = "./data/parsed",
    ) -> None:
        """Initialize document service.

        Args:
            db_session: Database session
            qdrant_service: Qdrant service instance
            upstage_parser: Upstage parser service instance
            upstage_embedding: Upstage embedding service instance
            lightrag_service: LightRAG service instance
            parsed_dir: Directory for parsed markdown files
        """
        self.db_session = db_session
        self.qdrant_service = qdrant_service
        self.upstage_parser = upstage_parser
        self.upstage_embedding = upstage_embedding
        self.lightrag_service = lightrag_service

        # Ensure parsed directory exists
        self.parsed_dir = Path(parsed_dir)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DocumentService initialized (parsed_dir: {self.parsed_dir})")

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
        category: str = "paper",
        tags: Optional[list[str]] = None,
    ) -> DocumentUploadResponse:
        """Process a document through the complete pipeline.

        This method orchestrates the entire document processing workflow:
        1. Create document record in database (status: "processing")
        2. Parse document with Upstage Document Parse API
        3. Chunk the parsed markdown text
        4. Generate embeddings and store in Qdrant
        5. Add full document to LightRAG knowledge graph
        6. Update database record (status: "completed")
        7. Save parsed markdown to file

        Args:
            file_content: Binary content of the uploaded file
            filename: Original filename
            title: Document title (defaults to filename if not provided)
            category: Document category (paper/blog/documentation)
            tags: List of tags for the document

        Returns:
            DocumentUploadResponse with processing results

        Raises:
            DocumentProcessingError: If processing fails at any step

        Example:
            >>> service = DocumentService(...)
            >>> with open("paper.pdf", "rb") as f:
            ...     content = f.read()
            >>> result = await service.process_document(
            ...     file_content=content,
            ...     filename="paper.pdf",
            ...     title="Attention Is All You Need",
            ...     category="paper",
            ...     tags=["NLP", "Transformer"]
            ... )
            >>> print(result.status)
            completed
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())

        # Use filename as title if not provided
        if not title:
            title = Path(filename).stem

        # Default tags to empty list
        if tags is None:
            tags = []

        logger.info(
            f"Starting document processing pipeline "
            f"(id: {document_id}, filename: {filename}, title: {title})"
        )

        # Step 1: Create document record in database
        try:
            document = Document(
                id=document_id,
                title=title,
                original_filename=filename,
                category=category,
                tags=tags,
                file_size_bytes=len(file_content),
                status="processing",
                chunks_count=0,
            )

            self.db_session.add(document)
            await self.db_session.flush()  # Flush to get ID assigned

            logger.info(f"Document record created in database (id: {document_id})")

        except Exception as e:
            error_msg = f"Failed to create document record: {e}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

        try:
            # Step 2: Parse document (or use directly for markdown files)
            file_ext = Path(filename).suffix.lower()

            if file_ext == ".md":
                # For markdown files, decode and use content directly
                logger.info(f"Using markdown content directly (id: {document_id})")
                parsed_markdown = file_content.decode("utf-8")
            else:
                # For PDF and other files, parse with Upstage Document Parse API
                logger.info(f"Parsing document with Upstage API (id: {document_id})")

                parsed_markdown = await self.upstage_parser.parse_to_markdown(
                    file_content=file_content,
                    filename=filename,
                )

            logger.info(
                f"Document processed successfully (id: {document_id}, "
                f"markdown length: {len(parsed_markdown)} chars)"
            )

            # Step 3: Chunk the parsed markdown
            logger.info(f"Chunking document (id: {document_id})")

            chunks = chunk_markdown_text(
                text=parsed_markdown,
                max_chunk_size=3000,  # Safe for 4000 token limit (~1000-1200 tokens)
                overlap_size=200,
            )

            logger.info(
                f"Document chunked into {len(chunks)} chunks (id: {document_id})"
            )

            # Validate chunk sizes (safety check for token limit)
            MAX_SAFE_CHUNK_SIZE = 3000  # Characters, ~1000-1200 tokens
            oversized_chunks = [
                (i, len(chunk)) for i, chunk in enumerate(chunks)
                if len(chunk) > MAX_SAFE_CHUNK_SIZE
            ]
            if oversized_chunks:
                logger.warning(
                    f"Found {len(oversized_chunks)} oversized chunks (id: {document_id}). "
                    f"Sizes: {oversized_chunks[:5]}..."
                )

            # Step 4: Generate embeddings and store in Qdrant
            logger.info(f"Generating embeddings for chunks (id: {document_id})")

            chunk_ids = []
            for chunk_index, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if not chunk_text.strip():
                    logger.warning(f"Skipping empty chunk at index {chunk_index}")
                    continue

                # Truncate if still too large (emergency fallback)
                if len(chunk_text) > MAX_SAFE_CHUNK_SIZE:
                    logger.warning(
                        f"Truncating oversized chunk {chunk_index} from "
                        f"{len(chunk_text)} to {MAX_SAFE_CHUNK_SIZE} chars"
                    )
                    chunk_text = chunk_text[:MAX_SAFE_CHUNK_SIZE]

                # Generate embedding
                embedding = await self.upstage_embedding.get_embedding(
                    text=chunk_text,
                    model_type="passage",
                )

                # Create chunk ID (must be a valid UUID for Qdrant)
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)

                # Store in Qdrant
                await self.qdrant_service.upsert_point(
                    point_id=chunk_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "document_title": title,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "category": category,
                        "tags": tags,
                    },
                )

            logger.info(
                f"Embeddings generated and stored in Qdrant "
                f"(id: {document_id}, chunks: {len(chunks)})"
            )

            # Step 5: Add full document to LightRAG knowledge graph
            logger.info(f"Adding document to LightRAG (id: {document_id})")

            await self.lightrag_service.insert_document(
                text=parsed_markdown,
                metadata={
                    "document_id": document_id,
                    "title": title,
                    "category": category,
                    "tags": tags,
                },
            )

            logger.info(f"Document added to LightRAG (id: {document_id})")

            # Step 6: Save parsed markdown to file
            parsed_file_path = self.parsed_dir / f"{document_id}.md"
            parsed_file_path.write_text(parsed_markdown, encoding="utf-8")

            logger.info(f"Parsed markdown saved to {parsed_file_path}")

            # Step 7: Update document record (status: "completed")
            document.status = "completed"
            document.chunks_count = len(chunks)
            document.parsed_content_path = str(parsed_file_path)
            document.error_message = None

            await self.db_session.flush()

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Document processing completed successfully "
                f"(id: {document_id}, time: {processing_time_ms}ms, chunks: {len(chunks)})"
            )

            # Commit transaction
            await self.db_session.commit()

            return DocumentUploadResponse(
                document_id=document_id,
                status="completed",
                title=title,
                chunks_count=len(chunks),
                entities_extracted=None,  # LightRAG doesn't expose this directly
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            # Rollback database transaction
            await self.db_session.rollback()

            error_msg = f"Document processing failed (id: {document_id}): {e}"
            logger.error(error_msg)

            # Update document status to "failed"
            try:
                # Re-fetch document to update status
                result = await self.db_session.execute(
                    select(Document).where(Document.id == document_id)
                )
                failed_doc = result.scalar_one_or_none()

                if failed_doc:
                    failed_doc.status = "failed"
                    failed_doc.error_message = str(e)
                    await self.db_session.commit()
                    logger.info(f"Document status updated to 'failed' (id: {document_id})")

            except Exception as update_error:
                logger.error(
                    f"Failed to update document status to 'failed': {update_error}"
                )

            # Clean up partial data
            try:
                await self._cleanup_partial_data(document_id, chunk_ids if 'chunk_ids' in locals() else [])
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")

            raise DocumentProcessingError(error_msg) from e

    async def _cleanup_partial_data(
        self,
        document_id: str,
        chunk_ids: list[str],
    ) -> None:
        """Clean up partial data after processing failure.

        This method removes any partially processed data:
        - Removes chunks from Qdrant
        - Deletes parsed markdown file

        Args:
            document_id: Document ID to clean up
            chunk_ids: List of chunk IDs to remove from Qdrant

        Note:
            LightRAG data is not cleaned up as it doesn't support
            selective deletion in Phase 1.
        """
        logger.info(f"Cleaning up partial data for document {document_id}")

        # Remove chunks from Qdrant
        if chunk_ids:
            try:
                for chunk_id in chunk_ids:
                    await self.qdrant_service.delete_point(chunk_id)
                logger.info(f"Removed {len(chunk_ids)} chunks from Qdrant")
            except Exception as e:
                logger.warning(f"Failed to remove chunks from Qdrant: {e}")

        # Delete parsed markdown file
        try:
            parsed_file_path = self.parsed_dir / f"{document_id}.md"
            if parsed_file_path.exists():
                parsed_file_path.unlink()
                logger.info(f"Deleted parsed markdown file: {parsed_file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete parsed markdown file: {e}")

        logger.info(f"Cleanup completed for document {document_id}")
