"""Integration test script for Nexograph.

This script tests the complete system flow:
1. Health check
2. Document upload
3. Document list
4. Chat query
5. Document deletion

Usage:
    conda activate nexograph
    # Make sure backend server is running: uvicorn backend.main:app --reload
    python scripts/test_integration.py
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TEST_TIMEOUT = 120.0


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"ℹ️  {message}")


async def test_health_check():
    """Test health check endpoint."""
    print_header("Test 1: Health Check")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")

            if response.status_code == 200:
                data = response.json()
                print_success("Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Services: {data.get('services')}")
                return True
            else:
                print_error(f"Health check failed: HTTP {response.status_code}")
                return False

    except Exception as e:
        print_error(f"Health check failed: {e}")
        print_info("Make sure the backend server is running:")
        print_info("  uvicorn backend.main:app --reload --port 8000")
        return False


async def test_document_upload():
    """Test document upload endpoint."""
    print_header("Test 2: Document Upload")

    # Find test document
    test_file = Path("data/test_papers/test_rag_document.md")

    if not test_file.exists():
        print_error(f"Test file not found: {test_file}")
        return None

    print_info(f"Uploading: {test_file}")

    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/markdown")}
                data = {
                    "title": "Test RAG Document",
                    "category": "paper",
                    "tags": "RAG,Test,GraphRAG",
                }

                print_info("Processing document (this may take a minute)...")
                start_time = time.time()

                response = await client.post(
                    f"{API_BASE_URL}/documents/upload",
                    files=files,
                    data=data,
                )

                elapsed = time.time() - start_time

                if response.status_code == 201:
                    result = response.json()
                    print_success(f"Document uploaded in {elapsed:.1f}s")
                    print(f"   Document ID: {result.get('document_id')}")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Chunks: {result.get('chunks_count')}")
                    print(f"   Processing time: {result.get('processing_time_ms')}ms")
                    return result.get('document_id')
                else:
                    print_error(f"Upload failed: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    return None

    except httpx.TimeoutException:
        print_error("Upload timeout (>120s)")
        print_info("Document processing may take longer for large files")
        return None

    except Exception as e:
        print_error(f"Upload failed: {e}")
        return None


async def test_document_list():
    """Test document list endpoint."""
    print_header("Test 3: Document List")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/documents",
                params={"page": 1, "limit": 10},
            )

            if response.status_code == 200:
                data = response.json()
                docs = data.get("documents", [])
                total = data.get("total", 0)

                print_success(f"Retrieved {len(docs)} documents (total: {total})")

                for i, doc in enumerate(docs[:3], 1):
                    print(f"   {i}. {doc.get('title')}")
                    print(f"      Category: {doc.get('category')}, Chunks: {doc.get('chunks_count')}")

                return True
            else:
                print_error(f"List failed: HTTP {response.status_code}")
                return False

    except Exception as e:
        print_error(f"List failed: {e}")
        return False


async def test_chat_query(document_id: str = None):
    """Test chat query endpoint."""
    print_header("Test 4: Chat Query")

    queries = [
        "What is RAG?",
        "How does LightRAG work?",
        "What are the benefits of graph-based RAG?",
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "query": query,
                    "mode": "hybrid",
                    "top_k": 5,
                    "include_sources": True,
                }

                start_time = time.time()
                response = await client.post(
                    f"{API_BASE_URL}/chat",
                    json=payload,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    print_success(f"Answer received in {elapsed:.1f}s")
                    print(f"   Answer: {answer[:200]}...")
                    print(f"   Sources: {len(sources)}")

                    if sources:
                        for i, source in enumerate(sources[:2], 1):
                            print(f"      {i}. {source.get('document_title')} (score: {source.get('relevance_score', 0):.2f})")

                else:
                    print_error(f"Chat failed: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")

        except Exception as e:
            print_error(f"Chat failed: {e}")

        await asyncio.sleep(1)  # Small delay between queries

    return True


async def test_delete_document(document_id: str):
    """Test document deletion endpoint."""
    print_header("Test 5: Document Deletion")

    if not document_id:
        print_info("Skipping deletion (no document ID)")
        return True

    print_info(f"Deleting document: {document_id}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(
                f"{API_BASE_URL}/documents/{document_id}"
            )

            if response.status_code == 204:
                print_success("Document deleted successfully")
                return True
            else:
                print_error(f"Deletion failed: HTTP {response.status_code}")
                return False

    except Exception as e:
        print_error(f"Deletion failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "NEXOGRAPH INTEGRATION TESTS" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")

    # Test 1: Health Check
    health_ok = await test_health_check()
    if not health_ok:
        print()
        print_error("Backend server is not running or unhealthy")
        print_info("Please start the server and try again:")
        print_info("  uvicorn backend.main:app --reload --port 8000")
        sys.exit(1)

    # Test 2: Document Upload
    document_id = await test_document_upload()

    # Test 3: Document List
    await test_document_list()

    # Test 4: Chat Query
    await test_chat_query(document_id)

    # Test 5: Delete Document (optional, cleanup)
    # Uncomment to clean up test document
    # if document_id:
    #     await test_delete_document(document_id)

    # Summary
    print_header("Test Summary")
    print_success("All integration tests completed!")
    print()
    print("ℹ️  Note: Test document was not deleted for manual inspection")
    print("   You can delete it via Streamlit UI or API if needed")
    print()


if __name__ == "__main__":
    asyncio.run(main())
