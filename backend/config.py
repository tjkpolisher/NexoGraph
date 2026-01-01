"""Configuration management for NexoGraph backend.

This module uses pydantic-settings to load configuration from environment variables
and .env files. It also validates that the correct conda environment is active.
"""

import os
import warnings
from functools import lru_cache
from pydantic_settings import BaseSettings


def check_conda_env() -> None:
    """Check if the correct conda environment is active.

    Validates that the CONDA_DEFAULT_ENV environment variable matches the expected
    'nexograph' environment. Issues a warning if there's a mismatch.

    This is a development helper to catch environment setup issues early.
    """
    expected_env = "nexograph"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != expected_env:
        warnings.warn(
            f"Expected conda env '{expected_env}', but got '{current_env or 'None'}'. "
            f"Run: conda activate {expected_env}",
            UserWarning,
            stacklevel=2,
        )


# Check conda environment on module import
check_conda_env()


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This class defines all configuration parameters for the NexoGraph application.
    Values are loaded from environment variables or the .env file.

    Attributes:
        upstage_api_key: API key for Upstage services (required)
        upstage_base_url: Base URL for Upstage API endpoints
        qdrant_host: Hostname for Qdrant vector database
        qdrant_port: Port for Qdrant vector database
        qdrant_collection_name: Name of the Qdrant collection for documents
        app_env: Application environment (development/production)
        app_debug: Enable debug mode
        app_version: Application version string
        database_url: SQLite database connection URL
        lightrag_working_dir: Working directory for LightRAG graph data
    """

    # Upstage API Configuration
    upstage_api_key: str
    upstage_base_url: str = "https://api.upstage.ai/v1"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "nexograph_documents"

    # Application Configuration
    app_env: str = "development"
    app_debug: bool = True
    app_version: str = "0.1.0"

    # Database Configuration
    database_url: str = "sqlite:///./data/db/nexograph.db"

    # LightRAG Configuration
    lightrag_working_dir: str = "./data/lightrag"

    class Config:
        """Pydantic configuration for Settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings instance.

    This function uses LRU cache to ensure only one Settings instance is created,
    implementing the singleton pattern. The settings are loaded once and reused
    across the application.

    Returns:
        Settings: Cached application configuration instance

    Raises:
        ValidationError: If required environment variables are missing or invalid
    """
    return Settings()
