"""Chat (Q&A) API endpoint.

This module provides the chat endpoint for question-answering over the knowledge graph.
It combines vector search (Qdrant) with graph-based retrieval (LightRAG) to generate
accurate answers with source citations.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import (
    get_db,
    get_qdrant_service,
    get_upstage_embedding_service,
    get_upstage_llm_service,
    get_lightrag_service,
)
from backend.models.database import Document
from backend.models.schemas import ChatRequest, ChatResponse, SourceInfo
from backend.services.qdrant_service import QdrantService
from backend.services.upstage.embedding import UpstageEmbeddingService
from backend.services.upstage.llm import UpstageLLMService
from backend.services.lightrag_service import LightRAGService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["chat"])

# System prompt for the Q&A assistant
SYSTEM_PROMPT = """당신은 AI 분야 지식 베이스 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 정확하게 답변하세요.
컨텍스트에 없는 내용은 추측하지 마세요.
답변 시 관련 출처를 언급하세요."""


# ============================================================================
# Helper Functions
# ============================================================================


def build_context_from_chunks(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks.

    Args:
        chunks: List of chunk dictionaries from vector search

    Returns:
        Formatted context string
    """
    if not chunks:
        return "[검색된 컨텍스트가 없습니다]"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.get("payload", {}).get("text", "")
        doc_title = chunk.get("payload", {}).get("document_title", "Unknown")

        context_parts.append(
            f"[출처 {i}: {doc_title}]\n{chunk_text}\n"
        )

    return "\n---\n".join(context_parts)


def build_user_prompt(context: str, query: str) -> str:
    """Build user prompt with context and query.

    Args:
        context: Retrieved context text
        query: User's query

    Returns:
        Formatted user prompt
    """
    return f"""컨텍스트:
{context}

질문: {query}"""


async def fetch_document_titles(
    document_ids: list[str],
    db: AsyncSession,
) -> dict[str, str]:
    """Fetch document titles from database.

    Args:
        document_ids: List of document IDs
        db: Database session

    Returns:
        Dictionary mapping document_id to title
    """
    if not document_ids:
        return {}

    try:
        result = await db.execute(
            select(Document).where(Document.id.in_(document_ids))
        )
        documents = result.scalars().all()

        return {doc.id: doc.title for doc in documents}

    except Exception as e:
        logger.warning(f"Failed to fetch document titles: {e}")
        return {}


# ============================================================================
# API Endpoint
# ============================================================================


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat Q&A",
    description="""
    Ask a question and get an AI-generated answer based on the knowledge base.

    This endpoint:
    1. Searches the vector database for relevant document chunks
    2. Optionally queries the knowledge graph (LightRAG)
    3. Combines results to generate a comprehensive answer
    4. Returns answer with source citations

    Query modes:
    - **local**: Focus on specific entities and relationships
    - **global**: Use global knowledge from the graph
    - **hybrid**: Combine local and global approaches (recommended)
    - **naive**: Simple vector search without graph
    """,
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant_service),
    embedding: UpstageEmbeddingService = Depends(get_upstage_embedding_service),
    llm: UpstageLLMService = Depends(get_upstage_llm_service),
    lightrag: LightRAGService = Depends(get_lightrag_service),
) -> ChatResponse:
    """Process a chat query and return an AI-generated answer.

    This endpoint combines vector search with knowledge graph retrieval
    to provide accurate, source-backed answers to user questions.

    Args:
        request: Chat request with query and parameters
        db: Database session (injected dependency)
        qdrant: Qdrant vector search service (injected dependency)
        embedding: Upstage embedding service (injected dependency)
        llm: Upstage LLM service (injected dependency)
        lightrag: LightRAG service (injected dependency)

    Returns:
        ChatResponse with answer, sources, and metadata

    Raises:
        HTTPException 400: Invalid query or parameters
        HTTPException 500: Processing error
    """
    start_time = time.time()

    logger.info(
        f"Chat query received (mode: {request.mode}, query: '{request.query[:50]}...')"
    )

    try:
        # Step 1: Generate query embedding
        logger.debug("Generating query embedding...")
        query_vector = await embedding.get_embedding(
            text=request.query,
            model_type="query",  # Use query model for search
        )
        logger.debug("Query embedding generated")

        # Step 2: Search Qdrant for similar chunks
        logger.debug(f"Searching Qdrant (top_k={request.top_k})...")
        vector_results = await qdrant.search(
            query_vector=query_vector,
            limit=request.top_k,
            score_threshold=0.3,  # Minimum relevance score
        )
        logger.info(f"Found {len(vector_results)} relevant chunks from Qdrant")

        # Step 3: Query LightRAG knowledge graph
        lightrag_context = ""
        if request.mode != "naive":
            try:
                logger.debug(f"Querying LightRAG (mode: {request.mode})...")

                # Initialize LightRAG if not already done
                if not lightrag.rag:
                    await lightrag._initialize_rag()

                # Query LightRAG - get context only, we'll generate answer ourselves
                lightrag_result = await lightrag.query(
                    query_text=request.query,
                    mode=request.mode,
                    top_k=request.top_k,
                    chunk_top_k=request.top_k,
                    only_need_context=True,  # Get context, not answer
                )

                lightrag_context = lightrag_result.get("context", "")
                logger.info(
                    f"Retrieved context from LightRAG "
                    f"(length: {len(lightrag_context)} chars)"
                )

            except Exception as e:
                logger.warning(f"LightRAG query failed: {e}")
                # Continue without LightRAG context
                lightrag_context = ""

        # Step 4: Build context from both sources
        # Combine Qdrant chunks and LightRAG context
        qdrant_context = build_context_from_chunks(vector_results)

        combined_context = ""
        if qdrant_context and qdrant_context != "[검색된 컨텍스트가 없습니다]":
            combined_context += f"=== Vector Search Results ===\n{qdrant_context}\n\n"

        if lightrag_context:
            combined_context += f"=== Knowledge Graph Context ===\n{lightrag_context}\n"

        if not combined_context:
            combined_context = "[검색된 컨텍스트가 없습니다]"

        logger.debug(f"Combined context length: {len(combined_context)} chars")

        # Step 5: Build prompt
        user_prompt = build_user_prompt(combined_context, request.query)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Step 6: Generate answer with LLM
        logger.debug("Generating answer with Solar LLM...")
        answer = await llm.chat_completion(
            messages=messages,
            temperature=0.1,  # Low temperature for factual answers
            max_tokens=2048,
        )
        logger.info(f"Answer generated (length: {len(answer)} chars)")

        # Step 7: Build source information (if requested)
        sources = []
        if request.include_sources and vector_results:
            # Collect unique document IDs
            doc_ids = list(set(
                chunk.get("payload", {}).get("document_id")
                for chunk in vector_results
                if chunk.get("payload", {}).get("document_id")
            ))

            # Fetch document titles from database
            doc_titles = await fetch_document_titles(doc_ids, db)

            # Build source info for each chunk
            for chunk in vector_results:
                payload = chunk.get("payload", {})
                doc_id = payload.get("document_id")
                chunk_text = payload.get("text", "")

                # Get document title
                doc_title = doc_titles.get(doc_id, payload.get("document_title", "Unknown"))

                # Create source info
                source = SourceInfo(
                    document_id=doc_id or "unknown",
                    document_title=doc_title,
                    chunk_preview=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    relevance_score=float(chunk.get("score", 0.0)),
                )
                sources.append(source)

            logger.info(f"Built source information for {len(sources)} chunks")

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Chat query completed "
            f"(mode: {request.mode}, time: {processing_time_ms}ms)"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            mode_used=request.mode,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "chat_error",
                "message": "Failed to process chat query",
                "detail": {"reason": str(e)},
            },
        ) from e
