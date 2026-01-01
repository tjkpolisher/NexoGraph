"""Test script to verify Pydantic schemas."""

from datetime import datetime
from backend.models.schemas import (
    HealthResponse,
    DocumentUploadResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentDetailResponse,
    ChatRequest,
    SourceInfo,
    ChatResponse,
    ErrorResponse,
)


def test_health_response():
    """Test HealthResponse schema."""
    health = HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "qdrant": "connected",
            "upstage": "configured",
            "lightrag": "initialized"
        }
    )
    print("✓ HealthResponse:", health.model_dump_json(indent=2))


def test_document_schemas():
    """Test document-related schemas."""
    # Upload response (processing)
    upload_processing = DocumentUploadResponse(
        document_id="550e8400-e29b-41d4-a716-446655440000",
        status="processing",
        message="Document upload started"
    )
    print("✓ DocumentUploadResponse (processing):", upload_processing.model_dump_json(indent=2))

    # Upload response (completed)
    upload_completed = DocumentUploadResponse(
        document_id="550e8400-e29b-41d4-a716-446655440000",
        status="completed",
        title="Attention Is All You Need",
        chunks_count=10,
        entities_extracted=25,
        processing_time_ms=3500
    )
    print("✓ DocumentUploadResponse (completed):", upload_completed.model_dump_json(indent=2))

    # List item
    list_item = DocumentListItem(
        id="550e8400-e29b-41d4-a716-446655440000",
        title="Attention Is All You Need",
        category="paper",
        tags=["AI", "NLP"],
        chunks_count=10,
        created_at=datetime.now()
    )
    print("✓ DocumentListItem:", list_item.model_dump_json(indent=2))

    # List response
    list_response = DocumentListResponse(
        documents=[list_item],
        total=1,
        page=1,
        limit=20
    )
    print("✓ DocumentListResponse:", list_response.model_dump_json(indent=2))

    # Detail response
    detail = DocumentDetailResponse(
        id="550e8400-e29b-41d4-a716-446655440000",
        title="Attention Is All You Need",
        category="paper",
        tags=["AI", "NLP"],
        original_filename="attention_paper.pdf",
        parsed_content_preview="# Attention Is All You Need\n\nAbstract...",
        chunks_count=10,
        entities=["GPT-4", "Transformer"],
        created_at=datetime.now(),
        file_size_bytes=1024000
    )
    print("✓ DocumentDetailResponse:", detail.model_dump_json(indent=2))


def test_chat_schemas():
    """Test chat-related schemas."""
    # Request
    request = ChatRequest(
        query="Transformer의 Attention 메커니즘을 설명해줘",
        mode="hybrid",
        top_k=5,
        include_sources=True
    )
    print("✓ ChatRequest:", request.model_dump_json(indent=2))

    # Source info
    source = SourceInfo(
        document_id="550e8400-e29b-41d4-a716-446655440000",
        document_title="Attention Is All You Need",
        chunk_preview="The Transformer uses self-attention...",
        relevance_score=0.95
    )
    print("✓ SourceInfo:", source.model_dump_json(indent=2))

    # Response
    response = ChatResponse(
        answer="Transformer의 Attention 메커니즘은...",
        sources=[source],
        mode_used="hybrid",
        processing_time_ms=1200
    )
    print("✓ ChatResponse:", response.model_dump_json(indent=2))


def test_error_response():
    """Test ErrorResponse schema."""
    error = ErrorResponse(
        error="invalid_file_type",
        message="Only PDF and Markdown files are supported",
        detail={"supported_types": [".pdf", ".md"]}
    )
    print("✓ ErrorResponse:", error.model_dump_json(indent=2))


def test_validation():
    """Test validation rules."""
    print("\n" + "=" * 60)
    print("Testing validation rules...")
    print("=" * 60)

    # Test empty query validation
    try:
        ChatRequest(query="   ", mode="hybrid")
        print("✗ Empty query validation FAILED")
    except ValueError as e:
        print(f"✓ Empty query validation: {e}")

    # Test top_k range validation
    try:
        ChatRequest(query="test", mode="hybrid", top_k=100)
        print("✗ top_k range validation FAILED")
    except ValueError as e:
        print(f"✓ top_k range validation: {e}")

    # Test invalid mode
    try:
        ChatRequest(query="test", mode="invalid")  # type: ignore
        print("✗ Mode validation FAILED")
    except ValueError as e:
        print(f"✓ Mode validation: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pydantic Schemas")
    print("=" * 60)
    print()

    try:
        test_health_response()
        print()
        test_document_schemas()
        print()
        test_chat_schemas()
        print()
        test_error_response()
        print()
        test_validation()
        print()
        print("=" * 60)
        print("All schema tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
