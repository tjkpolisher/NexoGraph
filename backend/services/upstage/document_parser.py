"""Upstage Document Parse service.

This module provides a service for parsing documents (PDF, images, etc.)
using Upstage's Document AI API with OCR and layout analysis capabilities.
"""

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class UpstageDocumentParseError(Exception):
    """Base exception for Upstage Document Parse service errors."""

    pass


class UpstageDocumentParseService:
    """Service for Upstage Document Parse API.

    This service provides an interface to Upstage's Document AI API,
    which can parse PDFs and images with OCR, layout analysis, and
    structured output in multiple formats.

    Attributes:
        api_key: Upstage API key
        base_url: API base URL (default: https://api.upstage.ai/v1)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
    ) -> None:
        """Initialize Upstage Document Parse service.

        Args:
            api_key: Upstage API key (starts with 'up_')
            base_url: API base URL

        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not api_key.startswith("up_"):
            raise ValueError("Invalid Upstage API key format (must start with 'up_')")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        logger.info("UpstageDocumentParseService initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def parse_document(
        self,
        file_content: bytes,
        filename: str,
        output_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse a document using Upstage Document AI.

        Args:
            file_content: Binary content of the file to parse
            filename: Original filename (used for determining file type)
            output_formats: List of output formats to request
                (default: ["markdown", "html", "text"])
                Options: "markdown", "html", "text"

        Returns:
            Dictionary containing:
            - content: Dict with requested formats (markdown, html, text)
            - elements: List of document elements with metadata
            - metadata: Document metadata (page count, etc.)

        Raises:
            UpstageDocumentParseError: If API call fails
            ValueError: If file content is empty

        Example:
            >>> service = UpstageDocumentParseService(api_key="up_xxx")
            >>> with open("document.pdf", "rb") as f:
            ...     content = f.read()
            >>> result = await service.parse_document(content, "document.pdf")
            >>> markdown = result["content"]["markdown"]
            >>> elements = result["elements"]
        """
        if not file_content:
            raise ValueError("File content cannot be empty")

        if output_formats is None:
            output_formats = ["markdown"]  # Default to markdown only

        url = f"{self.base_url}/document-ai/document-parse"

        # Prepare multipart form data
        files = {
            "document": (filename, file_content),
        }

        data = {
            "output_formats": ",".join(output_formats),  # Comma-separated formats
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Use longer timeout for document parsing (can be slow for large docs)
            async with httpx.AsyncClient(timeout=120.0) as client:
                logger.info(f"Parsing document: {filename} (size: {len(file_content)} bytes)")

                response = await client.post(
                    url,
                    files=files,
                    data=data,
                    headers=headers,
                )
                response.raise_for_status()

                result = response.json()

                # Validate response structure
                if "content" not in result:
                    raise UpstageDocumentParseError(
                        "Invalid response: missing 'content' field"
                    )

                logger.info(
                    f"Document parsed successfully: {filename} "
                    f"(formats: {list(result.get('content', {}).keys())})"
                )

                return result

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Upstage Document Parse API: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"

            logger.error(error_msg)
            raise UpstageDocumentParseError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = f"Request to Upstage Document Parse API timed out for {filename}"
            logger.error(error_msg)
            raise UpstageDocumentParseError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error parsing document {filename}: {e}"
            logger.error(error_msg)
            raise UpstageDocumentParseError(error_msg) from e

    async def parse_to_markdown(
        self,
        file_content: bytes,
        filename: str,
    ) -> str:
        """Parse document and return only markdown content (convenience method).

        Args:
            file_content: Binary content of the file
            filename: Original filename

        Returns:
            Parsed document in markdown format

        Raises:
            UpstageDocumentParseError: If API call fails or markdown not available

        Example:
            >>> service = UpstageDocumentParseService(api_key="up_xxx")
            >>> with open("paper.pdf", "rb") as f:
            ...     content = f.read()
            >>> markdown = await service.parse_to_markdown(content, "paper.pdf")
            >>> print(markdown[:100])
        """
        result = await self.parse_document(
            file_content=file_content,
            filename=filename,
            output_formats=["markdown"],
        )

        markdown = result.get("content", {}).get("markdown")
        if not markdown:
            raise UpstageDocumentParseError(
                "Markdown content not available in parse result"
            )

        return markdown

    async def parse_with_elements(
        self,
        file_content: bytes,
        filename: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Parse document and return markdown with element metadata.

        Args:
            file_content: Binary content of the file
            filename: Original filename

        Returns:
            Tuple of (markdown_content, elements_list)
            - markdown_content: Full document in markdown
            - elements_list: List of document elements with metadata

        Example:
            >>> service = UpstageDocumentParseService(api_key="up_xxx")
            >>> with open("paper.pdf", "rb") as f:
            ...     content = f.read()
            >>> markdown, elements = await service.parse_with_elements(
            ...     content, "paper.pdf"
            ... )
            >>> for elem in elements:
            ...     print(f"{elem['category']}: {elem['content'][:50]}")
        """
        result = await self.parse_document(
            file_content=file_content,
            filename=filename,
            output_formats=["markdown"],
        )

        markdown = result.get("content", {}).get("markdown", "")
        elements = result.get("elements", [])

        return markdown, elements

    async def health_check(self) -> bool:
        """Check if Upstage Document Parse API is accessible.

        Returns:
            True if API is healthy, False otherwise

        Note:
            This makes a minimal API call to verify connectivity.
            Returns False if API is unreachable or credentials are invalid.
        """
        try:
            # Create a minimal test document (1-byte text file)
            test_content = b"test"
            await self.parse_document(
                file_content=test_content,
                filename="test.txt",
                output_formats=["text"],
            )
            logger.debug("Upstage Document Parse health check: OK")
            return True
        except Exception as e:
            logger.warning(f"Upstage Document Parse health check failed: {e}")
            return False


# Utility functions for file validation

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported by Document Parse API.

    Args:
        filename: Filename to check

    Returns:
        True if file type is supported

    Supported formats:
        - PDF (.pdf)
        - Images (.jpg, .jpeg, .png, .bmp, .tiff, .tif)
        - Office docs (.docx, .pptx, .xlsx) - limited support
    """
    supported_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".docx",
        ".pptx",
        ".xlsx",
    }

    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in supported_extensions)


def get_file_type(filename: str) -> str:
    """Get file type category from filename.

    Args:
        filename: Filename to categorize

    Returns:
        File type category: "pdf", "image", "office", or "unknown"
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return "pdf"
    elif filename_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
        return "image"
    elif filename_lower.endswith((".docx", ".pptx", ".xlsx")):
        return "office"
    else:
        return "unknown"
