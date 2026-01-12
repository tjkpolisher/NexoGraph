"""Token counting utilities using tiktoken.

This module provides utilities for counting tokens in text using tiktoken,
which is essential for accurate token limit enforcement with LLM APIs.
"""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)

# Global tokenizer instance (cached for performance)
_tokenizer: Optional[tiktoken.Encoding] = None


def get_tokenizer() -> tiktoken.Encoding:
    """Get or create tokenizer instance (cached for performance).

    Uses o200k_base encoding which is used by:
    - GPT-4o and later models
    - GPT-5 series (when available)

    Returns:
        Tiktoken encoding instance

    Note:
        The tokenizer is cached globally to avoid repeated initialization overhead.
    """
    global _tokenizer
    if _tokenizer is None:
        logger.debug("Initializing o200k_base tokenizer")
        _tokenizer = tiktoken.get_encoding("o200k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens

    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("The quick brown fox jumps over the lazy dog")
        10
    """
    if not text:
        return 0

    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    *,
    suffix: str = "..."
) -> str:
    """Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        suffix: Suffix to add to truncated text (default: "...")

    Returns:
        Truncated text that fits within max_tokens (including suffix)

    Example:
        >>> long_text = "This is a very long text that exceeds the token limit..."
        >>> truncated = truncate_to_token_limit(long_text, max_tokens=10)
        >>> count_tokens(truncated) <= 10
        True
    """
    if not text:
        return text

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    # If already within limit, return as-is
    if len(tokens) <= max_tokens:
        return text

    # Calculate tokens available for content (reserve space for suffix)
    suffix_tokens = tokenizer.encode(suffix)
    available_tokens = max_tokens - len(suffix_tokens)

    if available_tokens <= 0:
        logger.warning(
            f"Token limit ({max_tokens}) too small for suffix ({len(suffix_tokens)} tokens)"
        )
        return suffix[:max_tokens] if max_tokens > 0 else ""

    # Truncate to available tokens and decode
    truncated_tokens = tokens[:available_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)

    return truncated_text + suffix


def split_text_by_tokens(
    text: str,
    max_tokens_per_chunk: int,
    overlap_tokens: int = 0
) -> list[str]:
    """Split text into chunks by token count.

    Args:
        text: Text to split
        max_tokens_per_chunk: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks (default: 0)

    Returns:
        List of text chunks, each within max_tokens_per_chunk

    Example:
        >>> text = "Long document text..."
        >>> chunks = split_text_by_tokens(text, max_tokens_per_chunk=1000, overlap_tokens=50)
        >>> all(count_tokens(chunk) <= 1000 for chunk in chunks)
        True
    """
    if not text:
        return []

    if overlap_tokens >= max_tokens_per_chunk:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be less than "
            f"max_tokens_per_chunk ({max_tokens_per_chunk})"
        )

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    # If text fits in one chunk, return as-is
    if len(tokens) <= max_tokens_per_chunk:
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Extract chunk tokens
        end_idx = start_idx + max_tokens_per_chunk
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move to next chunk with overlap
        start_idx = end_idx - overlap_tokens

        # Prevent infinite loop if overlap is too large
        if start_idx <= end_idx - max_tokens_per_chunk:
            start_idx = end_idx

    return chunks


def estimate_tokens_from_chars(char_count: int) -> int:
    """Estimate token count from character count.

    This is a rough approximation:
    - English text: ~4 chars per token
    - Code: ~3-4 chars per token
    - Non-English: varies widely

    Args:
        char_count: Number of characters

    Returns:
        Estimated token count

    Note:
        This is only an approximation. Use count_tokens() for accurate counts.
    """
    # Conservative estimate: 3.5 characters per token
    return int(char_count / 3.5) + 1
