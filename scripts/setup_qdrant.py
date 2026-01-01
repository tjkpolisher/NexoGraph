"""Setup script for Qdrant vector database.

This script initializes the Qdrant collection with the correct configuration.
Run this script before starting the application for the first time.

Usage:
    python scripts/setup_qdrant.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.services.qdrant_service import QdrantService, QdrantConnectionError


async def main() -> None:
    """Initialize Qdrant collection for NexoGraph."""
    print("=" * 60)
    print("NexoGraph - Qdrant Setup")
    print("=" * 60)
    print()

    # Load settings
    try:
        settings = get_settings()
        print(f"✓ Settings loaded")
        print(f"  - Qdrant Host: {settings.qdrant_host}")
        print(f"  - Qdrant Port: {settings.qdrant_port}")
        print(f"  - Collection: {settings.qdrant_collection_name}")
        print()
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        sys.exit(1)

    # Initialize Qdrant service
    try:
        print("Connecting to Qdrant...")
        service = QdrantService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_size=4096,  # Upstage embedding dimension
        )
        print(f"✓ Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        print()
    except QdrantConnectionError as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        print()
        print("Make sure Qdrant is running:")
        print("  docker-compose up -d")
        sys.exit(1)

    # Health check
    try:
        if await service.health_check():
            print("✓ Qdrant health check passed")
            print()
        else:
            print("✗ Qdrant health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Health check error: {e}")
        sys.exit(1)

    # Ensure collection exists
    try:
        print(f"Ensuring collection '{settings.qdrant_collection_name}' exists...")
        await service.ensure_collection_exists()
        print(f"✓ Collection '{settings.qdrant_collection_name}' is ready")
        print()
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        sys.exit(1)

    # Get collection info
    try:
        info = await service.get_collection_info()
        print("Collection Information:")
        print(f"  - Vectors Count: {info['vectors_count']}")
        print(f"  - Indexed Vectors: {info['indexed_vectors_count']}")
        print(f"  - Points Count: {info['points_count']}")
        print(f"  - Status: {info['status']}")
        print()
    except Exception as e:
        print(f"⚠ Could not retrieve collection info: {e}")
        print()

    print("=" * 60)
    print("✓ Qdrant setup completed successfully!")
    print("=" * 60)
    print()
    print("You can now start the NexoGraph API server:")
    print("  uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    asyncio.run(main())
