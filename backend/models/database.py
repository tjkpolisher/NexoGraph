"""SQLAlchemy database models and session management.

This module defines the database models and provides async session management
for SQLite using SQLAlchemy 2.0 with aiosqlite.
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from backend.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Document(Base):
    """Document metadata model.

    Stores metadata about uploaded documents including processing status,
    file information, and extracted content statistics.

    Attributes:
        id: Unique document identifier (UUID)
        title: Document title
        original_filename: Original uploaded filename
        category: Document category (paper/blog/documentation)
        tags: List of tags (stored as JSON)
        chunks_count: Number of text chunks created from document
        file_size_bytes: Original file size in bytes
        status: Processing status (processing/completed/failed)
        parsed_content_path: Path to parsed markdown file (optional)
        error_message: Error message if processing failed (optional)
        created_at: Document creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "documents"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique document identifier (UUID)",
    )

    # Basic information
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Document title",
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Original uploaded filename",
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="paper",
        index=True,
        comment="Document category (paper/blog/documentation)",
    )

    # Metadata
    tags: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of tags (JSON array)",
    )

    chunks_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of text chunks created from document",
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Original file size in bytes",
    )

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="processing",
        index=True,
        comment="Processing status (processing/completed/failed)",
    )

    parsed_content_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to parsed markdown file",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if processing failed",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Document creation timestamp",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last update timestamp",
    )

    def __repr__(self) -> str:
        """String representation of Document."""
        return (
            f"<Document(id={self.id}, title='{self.title}', "
            f"status={self.status}, category={self.category})>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary.

        Returns:
            Dictionary representation of document
        """
        return {
            "id": self.id,
            "title": self.title,
            "original_filename": self.original_filename,
            "category": self.category,
            "tags": self.tags,
            "chunks_count": self.chunks_count,
            "file_size_bytes": self.file_size_bytes,
            "status": self.status,
            "parsed_content_path": self.parsed_content_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Database engine and session configuration

def get_database_url() -> str:
    """Get database URL from settings.

    Returns:
        Database URL for SQLAlchemy

    Note:
        Converts sqlite:/// to sqlite+aiosqlite:/// for async support
    """
    settings = get_settings()
    db_url = settings.database_url

    # Convert to async SQLite URL if needed
    if db_url.startswith("sqlite:///"):
        db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

    return db_url


# Create async engine
async_engine: AsyncEngine = create_async_engine(
    get_database_url(),
    echo=False,  # Set to True for SQL query logging
    future=True,
    pool_pre_ping=True,  # Verify connections before using
)

# Create async session factory
async_session_maker = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get async database session.

    This is an async context manager that provides a database session
    and handles proper cleanup.

    Yields:
        AsyncSession: Database session

    Example:
        async with get_db_session() as session:
            result = await session.execute(select(Document))
            documents = result.scalars().all()
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database by creating all tables.

    This should be called during application startup or via init script.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all database tables.

    Warning: This will delete all data! Use only for testing or cleanup.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """Close database engine and cleanup connections.

    This should be called during application shutdown.
    """
    await async_engine.dispose()
