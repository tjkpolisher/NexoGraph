"""Test script for LightRAG service.

This script tests the LightRAG integration with Upstage services.

Usage:
    conda activate nexograph
    python test_lightrag_service.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.services.upstage import UpstageLLMService, UpstageEmbeddingService
from backend.services.lightrag_service import LightRAGService


async def test_lightrag_service():
    """Test LightRAG service functionality."""
    print("=" * 60)
    print("Testing LightRAGService")
    print("=" * 60)
    print()

    # Load settings
    settings = get_settings()

    # Initialize Upstage services
    print("Initializing Upstage services...")
    try:
        llm_service = UpstageLLMService(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        embedding_service = UpstageEmbeddingService(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        print("✓ Upstage services initialized")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize Upstage services: {e}")
        return

    # Initialize LightRAG service
    print("Initializing LightRAG service...")
    try:
        # Use test working directory
        test_working_dir = "./data/lightrag_test"

        lightrag_service = LightRAGService(
            working_dir=test_working_dir,
            llm_service=llm_service,
            embedding_service=embedding_service,
        )
        print(f"✓ LightRAG service initialized")
        print(f"  - Working directory: {test_working_dir}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize LightRAG service: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 1: Health check
    print("Test 1: Health check")
    try:
        is_healthy = await lightrag_service.health_check()
        if is_healthy:
            print("✓ LightRAG health check passed")
        else:
            print("⚠ LightRAG health check returned False")
        print()
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print()

    # Test 2: Get stats (before inserting data)
    print("Test 2: Get stats (empty graph)")
    try:
        stats = lightrag_service.get_stats()
        print("✓ Stats retrieved:")
        print(f"  - Working dir: {stats['working_dir']}")
        print(f"  - Initialized: {stats['initialized']}")
        print(f"  - Graph files: {len(stats['graph_files'])}")
        if stats['graph_files']:
            for file_info in stats['graph_files']:
                print(f"    - {file_info['name']} ({file_info['size_bytes']} bytes)")
        print()
    except Exception as e:
        print(f"✗ Get stats failed: {e}")
        print()

    # Test 3: Insert document
    print("Test 3: Insert document into knowledge graph")
    try:
        # Sample document about transformers
        document_text = """
# Transformer Architecture

The Transformer is a deep learning architecture that relies on the attention mechanism.
It was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.

## Key Components

### Self-Attention
Self-attention allows the model to weigh the importance of different parts of the input
when processing each element. It computes attention scores between all pairs of positions.

### Multi-Head Attention
Multi-head attention runs multiple attention mechanisms in parallel, allowing the model
to attend to information from different representation subspaces.

### Positional Encoding
Since transformers don't have inherent notion of sequence order, positional encodings
are added to the input embeddings to inject information about position.

## Applications
Transformers are used in:
- Natural Language Processing (BERT, GPT)
- Computer Vision (Vision Transformer)
- Speech Recognition
- Protein Structure Prediction (AlphaFold)
"""

        print("Inserting document (this may take a minute)...")
        print(f"  Document length: {len(document_text)} characters")

        await lightrag_service.insert_document(
            text=document_text,
            metadata={"title": "Transformer Architecture", "category": "paper"}
        )

        print("✓ Document inserted successfully")
        print()

    except Exception as e:
        print(f"✗ Document insertion failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 4: Get stats (after inserting data)
    print("Test 4: Get stats (after insertion)")
    try:
        stats = lightrag_service.get_stats()
        print("✓ Stats retrieved:")
        print(f"  - Graph files: {len(stats['graph_files'])}")
        for file_info in stats['graph_files']:
            size_kb = file_info['size_bytes'] / 1024
            print(f"    - {file_info['name']}: {size_kb:.2f} KB")
        print()
    except Exception as e:
        print(f"✗ Get stats failed: {e}")
        print()

    # Test 5: Query with different modes
    print("Test 5: Query knowledge graph")

    test_queries = [
        ("What is a Transformer?", "hybrid"),
        ("Explain self-attention mechanism", "local"),
        ("What are the applications of transformers?", "global"),
    ]

    for query_text, mode in test_queries:
        print(f"\nQuery: '{query_text}' (mode: {mode})")
        try:
            print("Querying (this may take a minute)...")

            result = await lightrag_service.query(
                query_text=query_text,
                mode=mode,
                top_k=3,
            )

            print(f"✓ Query successful")
            print(f"  - Mode used: {result['mode_used']}")
            print(f"  - Answer preview: {result['answer'][:200]}...")
            print()

        except Exception as e:
            print(f"✗ Query failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Test 6: Insert another document
    print("Test 6: Insert second document")
    try:
        document_text_2 = """
# BERT: Bidirectional Encoder Representations from Transformers

BERT is a transformer-based model for natural language understanding.
It was developed by Google AI Language in 2018.

## Key Innovation
BERT uses bidirectional training, meaning it looks at context from both left and right
during pre-training. This is different from previous models that were unidirectional.

## Pre-training Tasks
1. Masked Language Modeling (MLM): Randomly mask some tokens and predict them
2. Next Sentence Prediction (NSP): Predict if two sentences follow each other

## Impact
BERT achieved state-of-the-art results on many NLP tasks and inspired many variants
like RoBERTa, ALBERT, and DistilBERT.
"""

        await lightrag_service.insert_document(
            text=document_text_2,
            metadata={"title": "BERT Overview", "category": "paper"}
        )

        print("✓ Second document inserted")
        print()

    except Exception as e:
        print(f"✗ Second document insertion failed: {e}")
        print()

    # Test 7: Query with more context
    print("Test 7: Query with multiple documents in graph")
    try:
        result = await lightrag_service.query(
            query_text="Compare Transformer and BERT architectures",
            mode="hybrid",
        )

        print("✓ Multi-document query successful")
        print(f"  - Answer:")
        print(f"    {result['answer'][:400]}...")
        print()

    except Exception as e:
        print(f"✗ Multi-document query failed: {e}")
        print()

    # Test 8: Error handling - empty text
    print("Test 8: Error handling (empty text)")
    try:
        await lightrag_service.insert_document(text="   ")
        print("✗ Should have raised ValueError for empty text")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
    print()

    # Test 9: Error handling - invalid mode
    print("Test 9: Error handling (invalid query mode)")
    try:
        await lightrag_service.query(query_text="test", mode="invalid_mode")
        print("✗ Should have raised ValueError for invalid mode")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
    print()

    # Cleanup
    print("Cleanup: Clearing test graph data")
    try:
        await lightrag_service.clear_graph()
        print("✓ Test graph data cleared")
        print()
    except Exception as e:
        print(f"⚠ Cleanup failed: {e}")
        print()

    print("=" * 60)
    print("✓ All LightRAG tests completed!")
    print("=" * 60)
    print()
    print("Note: LightRAG operations can be slow on first run.")
    print("Subsequent runs will be faster due to caching.")


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "LightRAG Service Test Suite" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print("⚠ WARNING: These tests will call Upstage APIs")
    print("  and may consume API credits.")
    print()

    await test_lightrag_service()


if __name__ == "__main__":
    asyncio.run(main())
