"""Database initialization script.

This script creates the SQLite database and all tables.
It also creates the necessary data directories.

Usage:
    python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.models.database import Base, Document, async_engine, init_db


def create_data_directories() -> None:
    """Create necessary data directories if they don't exist."""
    settings = get_settings()

    # Parse database URL to get the path
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        if db_path.startswith("./"):
            db_path = db_path[2:]

        db_file = Path(db_path)
        db_dir = db_file.parent

        # Create database directory
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {db_dir}")

    # Create other data directories
    data_dirs = [
        Path("data/uploads"),
        Path("data/parsed"),
        Path("data/test_papers"),
    ]

    for data_dir in data_dirs:
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {data_dir}")


async def main() -> None:
    """Initialize database and create tables."""
    print("=" * 60)
    print("NexoGraph - Database Initialization")
    print("=" * 60)
    print()

    # Load settings
    try:
        settings = get_settings()
        print("✓ Settings loaded")
        print(f"  - Database URL: {settings.database_url}")
        print()
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        sys.exit(1)

    # Create data directories
    try:
        print("Creating data directories...")
        create_data_directories()
        print()
    except Exception as e:
        print(f"✗ Failed to create directories: {e}")
        sys.exit(1)

    # Initialize database
    try:
        print("Initializing database...")
        await init_db()
        print("✓ Database initialized successfully")
        print()
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify tables were created
    try:
        print("Verifying database tables...")
        async with async_engine.connect() as conn:
            # Check if tables exist
            tables = await conn.run_sync(
                lambda sync_conn: sync_conn.dialect.get_table_names(sync_conn)
            )

            print(f"✓ Found {len(tables)} table(s):")
            for table in tables:
                print(f"  - {table}")
            print()

            # Verify Document table structure
            if "documents" in tables:
                print("Document table columns:")
                from sqlalchemy import inspect

                def get_columns(sync_conn):
                    inspector = inspect(sync_conn)
                    return inspector.get_columns("documents")

                columns = await conn.run_sync(get_columns)
                for col in columns:
                    print(f"  - {col['name']}: {col['type']}")
                print()

    except Exception as e:
        print(f"⚠ Could not verify tables: {e}")
        print()

    # Test database operations
    try:
        print("Testing database operations...")
        from backend.models.database import get_db_session
        from sqlalchemy import select

        async with get_db_session() as session:
            # Try to query documents (should be empty)
            result = await session.execute(select(Document))
            documents = result.scalars().all()
            print(f"✓ Database query successful")
            print(f"  - Current document count: {len(documents)}")
            print()

    except Exception as e:
        print(f"⚠ Database test query failed: {e}")
        print()

    # Cleanup
    try:
        from backend.models.database import close_db
        await close_db()
        print("✓ Database connections closed")
        print()
    except Exception as e:
        print(f"⚠ Error closing database: {e}")
        print()

    print("=" * 60)
    print("✓ Database initialization complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start Qdrant: docker-compose up -d")
    print("  2. Initialize Qdrant: python scripts/setup_qdrant.py")
    print("  3. Start API server: uvicorn backend.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
