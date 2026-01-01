"""Test script for database operations.

This script tests CRUD operations on the Document model.

Usage:
    conda activate nexograph
    python test_database.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, update, delete
from backend.models.database import Document, get_db_session, init_db


async def test_database_operations():
    """Test all database CRUD operations."""
    print("=" * 60)
    print("Testing Database Operations")
    print("=" * 60)
    print()

    # Initialize database (create tables if they don't exist)
    print("Initializing database...")
    try:
        await init_db()
        print("✓ Database initialized")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return

    # Test 1: Create (INSERT)
    print("Test 1: Create new documents")
    try:
        async with get_db_session() as session:
            # Create test documents
            doc1 = Document(
                title="Attention Is All You Need",
                original_filename="attention_paper.pdf",
                category="paper",
                tags=["AI", "NLP", "Transformer"],
                chunks_count=15,
                file_size_bytes=1024000,
                status="completed",
                parsed_content_path="data/parsed/attention_paper.md",
            )

            doc2 = Document(
                title="GraphRAG Overview",
                original_filename="graphrag_blog.md",
                category="blog",
                tags=["GraphRAG", "Knowledge Graph"],
                chunks_count=8,
                file_size_bytes=512000,
                status="processing",
            )

            session.add(doc1)
            session.add(doc2)
            await session.commit()

            print(f"✓ Created 2 documents")
            print(f"  - Document 1 ID: {doc1.id}")
            print(f"  - Document 2 ID: {doc2.id}")
            print()

            # Store IDs for later tests
            doc1_id = doc1.id
            doc2_id = doc2.id

    except Exception as e:
        print(f"✗ Create operation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Read (SELECT)
    print("Test 2: Read documents")
    try:
        async with get_db_session() as session:
            # Get all documents
            result = await session.execute(select(Document))
            all_docs = result.scalars().all()

            print(f"✓ Retrieved {len(all_docs)} documents")
            for doc in all_docs:
                print(f"  - {doc.title} ({doc.category}, {doc.status})")
            print()

            # Get specific document by ID
            result = await session.execute(
                select(Document).where(Document.id == doc1_id)
            )
            doc = result.scalar_one_or_none()

            if doc:
                print(f"✓ Retrieved document by ID: {doc.title}")
                print(f"  - Tags: {doc.tags}")
                print(f"  - Chunks: {doc.chunks_count}")
                print(f"  - Created: {doc.created_at}")
                print()
            else:
                print("✗ Document not found")
                print()

    except Exception as e:
        print(f"✗ Read operation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Filter queries
    print("Test 3: Filter queries")
    try:
        async with get_db_session() as session:
            # Filter by category
            result = await session.execute(
                select(Document).where(Document.category == "paper")
            )
            papers = result.scalars().all()
            print(f"✓ Found {len(papers)} paper(s)")

            # Filter by status
            result = await session.execute(
                select(Document).where(Document.status == "completed")
            )
            completed = result.scalars().all()
            print(f"✓ Found {len(completed)} completed document(s)")

            # Filter by tags (JSON contains)
            # Note: SQLite JSON support is limited, so we'll skip complex JSON queries
            print()

    except Exception as e:
        print(f"✗ Filter query failed: {e}")
        print()

    # Test 4: Update
    print("Test 4: Update document")
    try:
        async with get_db_session() as session:
            # Update document status
            await session.execute(
                update(Document)
                .where(Document.id == doc2_id)
                .values(
                    status="completed",
                    chunks_count=10,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

            # Verify update
            result = await session.execute(
                select(Document).where(Document.id == doc2_id)
            )
            updated_doc = result.scalar_one()

            print(f"✓ Updated document: {updated_doc.title}")
            print(f"  - New status: {updated_doc.status}")
            print(f"  - New chunks_count: {updated_doc.chunks_count}")
            print(f"  - Updated at: {updated_doc.updated_at}")
            print()

    except Exception as e:
        print(f"✗ Update operation failed: {e}")
        print()

    # Test 5: Pagination
    print("Test 5: Pagination")
    try:
        async with get_db_session() as session:
            # Add more test documents for pagination
            for i in range(3, 8):
                doc = Document(
                    title=f"Test Document {i}",
                    original_filename=f"test_{i}.pdf",
                    category="paper",
                    tags=["test"],
                    chunks_count=5,
                    file_size_bytes=100000,
                    status="completed",
                )
                session.add(doc)
            await session.commit()

            # Paginated query (page 1, limit 3)
            result = await session.execute(
                select(Document)
                .order_by(Document.created_at.desc())
                .limit(3)
                .offset(0)
            )
            page1_docs = result.scalars().all()

            print(f"✓ Page 1: {len(page1_docs)} documents")
            for doc in page1_docs:
                print(f"  - {doc.title}")

            # Paginated query (page 2, limit 3)
            result = await session.execute(
                select(Document)
                .order_by(Document.created_at.desc())
                .limit(3)
                .offset(3)
            )
            page2_docs = result.scalars().all()

            print(f"✓ Page 2: {len(page2_docs)} documents")
            for doc in page2_docs:
                print(f"  - {doc.title}")
            print()

    except Exception as e:
        print(f"✗ Pagination failed: {e}")
        print()

    # Test 6: Count
    print("Test 6: Count documents")
    try:
        async with get_db_session() as session:
            from sqlalchemy import func

            # Count all documents
            result = await session.execute(select(func.count(Document.id)))
            total_count = result.scalar()

            print(f"✓ Total documents: {total_count}")

            # Count by category
            result = await session.execute(
                select(func.count(Document.id)).where(Document.category == "paper")
            )
            paper_count = result.scalar()

            print(f"✓ Paper documents: {paper_count}")
            print()

    except Exception as e:
        print(f"✗ Count query failed: {e}")
        print()

    # Test 7: to_dict() method
    print("Test 7: Convert to dictionary")
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == doc1_id)
            )
            doc = result.scalar_one()

            doc_dict = doc.to_dict()
            print(f"✓ Converted document to dict:")
            print(f"  Keys: {list(doc_dict.keys())}")
            print(f"  Title: {doc_dict['title']}")
            print(f"  Category: {doc_dict['category']}")
            print()

    except Exception as e:
        print(f"✗ to_dict() failed: {e}")
        print()

    # Test 8: Delete
    print("Test 8: Delete documents")
    try:
        async with get_db_session() as session:
            # Delete test documents (cleanup)
            await session.execute(
                delete(Document).where(Document.title.like("Test Document%"))
            )
            await session.commit()

            # Count remaining
            result = await session.execute(select(func.count(Document.id)))
            remaining = result.scalar()

            print(f"✓ Deleted test documents")
            print(f"  - Remaining documents: {remaining}")
            print()

    except Exception as e:
        print(f"✗ Delete operation failed: {e}")
        print()

    # Cleanup all test data
    print("Cleanup: Removing all test data")
    try:
        async with get_db_session() as session:
            await session.execute(delete(Document))
            await session.commit()
            print("✓ All test data removed")
            print()
    except Exception as e:
        print(f"⚠ Cleanup failed: {e}")
        print()

    print("=" * 60)
    print("✓ All database tests completed!")
    print("=" * 60)


async def main():
    """Run all tests."""
    await test_database_operations()


if __name__ == "__main__":
    asyncio.run(main())
