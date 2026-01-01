"""Test script for Upstage API services.

This script tests all three Upstage services:
- LLM (Solar language models)
- Embedding (Solar embedding models)
- Document Parse (Document AI)

Usage:
    conda activate nexograph
    python test_upstage_services.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.services.upstage import (
    UpstageLLMService,
    UpstageEmbeddingService,
    UpstageDocumentParseService,
)


async def test_llm_service():
    """Test Upstage LLM service."""
    print("=" * 60)
    print("Testing UpstageLLMService")
    print("=" * 60)
    print()

    settings = get_settings()

    try:
        service = UpstageLLMService(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        print("✓ LLM Service initialized")
        print(f"  - Model: {service.default_model}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize LLM service: {e}")
        return

    # Test 1: Simple completion
    print("Test 1: Simple completion")
    try:
        response = await service.simple_completion(
            prompt="What is GraphRAG in one sentence?",
            system_prompt="You are a helpful AI assistant.",
            max_tokens=100,
        )
        print(f"✓ Simple completion successful")
        print(f"  Response: {response[:100]}...")
        print()
    except Exception as e:
        print(f"✗ Simple completion failed: {e}")
        print()

    # Test 2: Chat completion with messages
    print("Test 2: Chat completion with messages")
    try:
        messages = [
            {"role": "system", "content": "You are a concise AI expert."},
            {"role": "user", "content": "Explain attention mechanism in 2 sentences."},
        ]
        response = await service.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=150,
        )
        print(f"✓ Chat completion successful")
        print(f"  Response: {response[:150]}...")
        print()
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
        print()

    # Test 3: Health check
    print("Test 3: Health check")
    try:
        is_healthy = await service.health_check()
        if is_healthy:
            print("✓ LLM health check passed")
        else:
            print("✗ LLM health check failed")
        print()
    except Exception as e:
        print(f"✗ Health check error: {e}")
        print()


async def test_embedding_service():
    """Test Upstage Embedding service."""
    print("=" * 60)
    print("Testing UpstageEmbeddingService")
    print("=" * 60)
    print()

    settings = get_settings()

    try:
        service = UpstageEmbeddingService(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        print("✓ Embedding Service initialized")
        print(f"  - Vector size: {service.VECTOR_SIZE}")
        print(f"  - Passage model: {service.PASSAGE_MODEL}")
        print(f"  - Query model: {service.QUERY_MODEL}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize Embedding service: {e}")
        return

    # Test 1: Single passage embedding
    print("Test 1: Generate passage embedding")
    try:
        text = "This is a test document about artificial intelligence and machine learning."
        embedding = await service.get_embedding(text, model_type="passage")
        print(f"✓ Passage embedding generated")
        print(f"  - Dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print()
    except Exception as e:
        print(f"✗ Passage embedding failed: {e}")
        print()

    # Test 2: Single query embedding
    print("Test 2: Generate query embedding")
    try:
        query = "What is artificial intelligence?"
        query_embedding = await service.get_embedding(query, model_type="query")
        print(f"✓ Query embedding generated")
        print(f"  - Dimension: {len(query_embedding)}")
        print(f"  - First 5 values: {query_embedding[:5]}")
        print()
    except Exception as e:
        print(f"✗ Query embedding failed: {e}")
        print()

    # Test 3: Batch embeddings
    print("Test 3: Generate batch embeddings")
    try:
        texts = [
            "First document about AI",
            "Second document about machine learning",
            "Third document about neural networks",
        ]
        embeddings = await service.get_embeddings_batch(texts, model_type="passage")
        print(f"✓ Batch embeddings generated")
        print(f"  - Count: {len(embeddings)}")
        print(f"  - All same dimension: {all(len(e) == service.VECTOR_SIZE for e in embeddings)}")
        print()
    except Exception as e:
        print(f"✗ Batch embeddings failed: {e}")
        print()

    # Test 4: Health check
    print("Test 4: Health check")
    try:
        is_healthy = await service.health_check()
        if is_healthy:
            print("✓ Embedding health check passed")
        else:
            print("✗ Embedding health check failed")
        print()
    except Exception as e:
        print(f"✗ Health check error: {e}")
        print()


async def test_document_parse_service():
    """Test Upstage Document Parse service."""
    print("=" * 60)
    print("Testing UpstageDocumentParseService")
    print("=" * 60)
    print()

    settings = get_settings()

    try:
        service = UpstageDocumentParseService(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        print("✓ Document Parse Service initialized")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize Document Parse service: {e}")
        return

    # Test 1: Parse simple text document
    print("Test 1: Parse simple text document")
    try:
        # Create a simple markdown test document
        test_content = b"""# Test Document

## Introduction
This is a test document for the Upstage Document Parse API.

## Main Content
- Point 1: Testing markdown parsing
- Point 2: Testing element extraction
- Point 3: Testing API integration

## Conclusion
The test is complete.
"""
        result = await service.parse_document(
            file_content=test_content,
            filename="test_doc.md",
            output_formats=["markdown", "text"],
        )

        print("✓ Document parsed successfully")
        print(f"  - Available formats: {list(result.get('content', {}).keys())}")
        print(f"  - Markdown length: {len(result['content'].get('markdown', ''))}")
        print(f"  - Element count: {len(result.get('elements', []))}")
        print()

        # Show first 200 chars of markdown
        markdown = result["content"].get("markdown", "")
        if markdown:
            print(f"  Markdown preview:")
            print(f"  {markdown[:200]}...")
        print()

    except Exception as e:
        print(f"✗ Document parsing failed: {e}")
        print()

    # Test 2: Parse to markdown only
    print("Test 2: Parse to markdown (convenience method)")
    try:
        test_content = b"# Simple Test\n\nThis is a simple test."
        markdown = await service.parse_to_markdown(
            file_content=test_content,
            filename="simple.md",
        )
        print("✓ Markdown parsing successful")
        print(f"  - Content: {markdown[:100]}")
        print()
    except Exception as e:
        print(f"✗ Markdown parsing failed: {e}")
        print()

    # Test 3: Parse with elements
    print("Test 3: Parse with elements")
    try:
        test_content = b"# Title\n\n## Section\n\nParagraph content."
        markdown, elements = await service.parse_with_elements(
            file_content=test_content,
            filename="elements_test.md",
        )
        print("✓ Parsing with elements successful")
        print(f"  - Markdown length: {len(markdown)}")
        print(f"  - Elements count: {len(elements)}")
        if elements:
            print(f"  - First element category: {elements[0].get('category', 'N/A')}")
        print()
    except Exception as e:
        print(f"✗ Parsing with elements failed: {e}")
        print()

    # Test 4: File type validation
    print("Test 4: File type validation")
    from backend.services.upstage.document_parser import (
        is_supported_file_type,
        get_file_type,
    )

    test_files = [
        ("document.pdf", True, "pdf"),
        ("image.jpg", True, "image"),
        ("presentation.pptx", True, "office"),
        ("data.csv", False, "unknown"),
    ]

    for filename, should_be_supported, expected_type in test_files:
        is_supported = is_supported_file_type(filename)
        file_type = get_file_type(filename)

        status = "✓" if is_supported == should_be_supported else "✗"
        print(f"  {status} {filename}: supported={is_supported}, type={file_type}")

    print()


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Upstage Services Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Check settings
    try:
        settings = get_settings()
        print("✓ Settings loaded")
        print(f"  - API Key: {settings.upstage_api_key[:10]}...")
        print(f"  - Base URL: {settings.upstage_base_url}")
        print()
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        sys.exit(1)

    # Run tests
    try:
        await test_llm_service()
        await test_embedding_service()
        await test_document_parse_service()

        print("=" * 60)
        print("✓ All Upstage service tests completed!")
        print("=" * 60)
        print()

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
