"""Test script for Qdrant service functionality.

This script tests all methods of the QdrantService class.
Make sure Qdrant is running before executing this script.

Usage:
    conda activate nexograph
    python test_qdrant_service.py
"""

import sys
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.services.qdrant_service import (
    QdrantService,
    QdrantConnectionError,
    QdrantServiceError,
)


def generate_random_vector(size: int = 4096) -> list[float]:
    """Generate a random vector for testing.

    Args:
        size: Vector dimension

    Returns:
        Random vector
    """
    return [random.uniform(-1, 1) for _ in range(size)]


def test_qdrant_service() -> None:
    """Test all QdrantService functionality."""
    print("=" * 60)
    print("Testing QdrantService")
    print("=" * 60)
    print()

    # Test 1: Initialize service
    print("Test 1: Initialize Qdrant service")
    try:
        service = QdrantService(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=4096,
        )
        print("✓ Service initialized")
        print(f"  - Host: {service.host}")
        print(f"  - Port: {service.port}")
        print(f"  - Collection: {service.collection_name}")
        print(f"  - Vector Size: {service.vector_size}")
        print()
    except QdrantConnectionError as e:
        print(f"✗ Failed to initialize service: {e}")
        print()
        print("Make sure Qdrant is running:")
        print("  docker-compose up -d")
        sys.exit(1)

    # Test 2: Health check
    print("Test 2: Health check")
    if service.health_check():
        print("✓ Health check passed")
        print()
    else:
        print("✗ Health check failed")
        sys.exit(1)

    # Test 3: Ensure collection exists
    print("Test 3: Ensure collection exists")
    try:
        service.ensure_collection_exists()
        print(f"✓ Collection '{service.collection_name}' created/verified")
        print()
    except Exception as e:
        print(f"✗ Failed to ensure collection: {e}")
        sys.exit(1)

    # Test 4: Get collection info
    print("Test 4: Get collection info")
    try:
        info = service.get_collection_info()
        print("✓ Collection info retrieved:")
        print(f"  - Vectors: {info['vectors_count']}")
        print(f"  - Indexed: {info['indexed_vectors_count']}")
        print(f"  - Points: {info['points_count']}")
        print(f"  - Status: {info['status']}")
        print()
    except Exception as e:
        print(f"✗ Failed to get collection info: {e}")
        print()

    # Test 5: Add vectors
    print("Test 5: Add vectors")
    try:
        test_vectors = [
            generate_random_vector(4096),
            generate_random_vector(4096),
            generate_random_vector(4096),
        ]
        test_ids = ["test_vec_1", "test_vec_2", "test_vec_3"]
        test_payloads = [
            {
                "document_id": "test_doc_1",
                "chunk_id": 0,
                "text": "This is a test chunk about AI and machine learning.",
            },
            {
                "document_id": "test_doc_1",
                "chunk_id": 1,
                "text": "This chunk discusses neural networks and transformers.",
            },
            {
                "document_id": "test_doc_2",
                "chunk_id": 0,
                "text": "A different document about natural language processing.",
            },
        ]

        service.add_vectors(
            ids=test_ids,
            vectors=test_vectors,
            payloads=test_payloads,
        )
        print(f"✓ Added {len(test_vectors)} vectors")
        print()
    except Exception as e:
        print(f"✗ Failed to add vectors: {e}")
        print()

    # Test 6: Search vectors
    print("Test 6: Search for similar vectors")
    try:
        query_vector = test_vectors[0]  # Use first vector as query
        results = service.search(
            query_vector=query_vector,
            limit=3,
            score_threshold=0.5,
        )
        print(f"✓ Search completed: found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  Result {i}:")
            print(f"    - ID: {result['id']}")
            print(f"    - Score: {result['score']:.4f}")
            print(f"    - Document ID: {result['payload'].get('document_id', 'N/A')}")
        print()
    except Exception as e:
        print(f"✗ Search failed: {e}")
        print()

    # Test 7: Delete vectors by document_id
    print("Test 7: Delete vectors by document_id")
    try:
        deleted = service.delete_by_document_id("test_doc_1")
        print(f"✓ Deleted vectors for document 'test_doc_1'")
        print()

        # Verify deletion with search
        results_after = service.search(
            query_vector=query_vector,
            limit=10,
        )
        remaining_docs = set(r['payload'].get('document_id') for r in results_after)
        print(f"  Remaining documents: {remaining_docs}")
        if "test_doc_1" not in remaining_docs:
            print("  ✓ Verified: test_doc_1 vectors deleted")
        else:
            print("  ⚠ Warning: test_doc_1 still present")
        print()
    except Exception as e:
        print(f"✗ Delete failed: {e}")
        print()

    # Test 8: Test error handling (invalid input)
    print("Test 8: Test error handling")
    try:
        # Try to add vectors with mismatched lengths
        service.add_vectors(
            ids=["id1"],
            vectors=[[0.1, 0.2]],
            payloads=[{"key": "value"}, {"key2": "value2"}],  # Mismatch!
        )
        print("✗ Error handling failed: should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")
    except Exception as e:
        print(f"⚠ Unexpected error type: {e}")
    print()

    # Cleanup: Delete test collection
    print("Cleanup: Deleting test collection")
    try:
        service.client.delete_collection(collection_name=service.collection_name)
        print(f"✓ Test collection '{service.collection_name}' deleted")
        print()
    except Exception as e:
        print(f"⚠ Could not delete test collection: {e}")
        print()

    # Test 9: Singleton pattern
    print("Test 9: Test singleton pattern")
    try:
        from backend.services.qdrant_service import get_qdrant_service_instance

        instance1 = get_qdrant_service_instance(
            host="localhost",
            port=6333,
            collection_name="singleton_test",
        )
        instance2 = get_qdrant_service_instance(
            host="localhost",
            port=6333,
            collection_name="singleton_test",
        )

        if instance1 is instance2:
            print("✓ Singleton pattern working: same instance returned")
        else:
            print("✗ Singleton pattern broken: different instances")
        print()
    except Exception as e:
        print(f"✗ Singleton test failed: {e}")
        print()

    print("=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_qdrant_service()
