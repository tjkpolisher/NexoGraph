"""Upstage LLM (Solar) service.

This module provides a service for interacting with Upstage's Solar language models
via their OpenAI-compatible API.
"""

import logging
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class UpstageLLMError(Exception):
    """Base exception for Upstage LLM service errors."""

    pass


class UpstageLLMService:
    """Service for Upstage Solar language model API.

    This service provides an interface to Upstage's Solar LLM API,
    which is compatible with OpenAI's chat completion API.

    Attributes:
        api_key: Upstage API key
        base_url: API base URL (default: https://api.upstage.ai/v1)
        default_model: Default model to use (default: solar-pro)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        default_model: str = "solar-pro2",
    ) -> None:
        """Initialize Upstage LLM service.

        Args:
            api_key: Upstage API key (starts with 'up_')
            base_url: API base URL
            default_model: Default model to use (solar-pro or solar-mini)

        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not api_key.startswith("up_"):
            raise ValueError("Invalid Upstage API key format (must start with 'up_')")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

        logger.info(f"UpstageLLMService initialized (model: {self.default_model})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Generate a chat completion using Solar LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0, default: 0.1)
            max_tokens: Maximum tokens to generate (default: 2048)
            model: Model to use (default: self.default_model)
            stream: Whether to stream the response (default: False)

        Returns:
            Generated text response

        Raises:
            UpstageLLMError: If API call fails
            ValueError: If messages format is invalid

        Example:
            >>> service = UpstageLLMService(api_key="up_xxx")
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is GraphRAG?"}
            ... ]
            >>> response = await service.chat_completion(messages)
        """
        # Validate messages
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")

        # Prepare request
        model_name = model or self.default_model
        url = f"{self.base_url}/solar/chat/completions"

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.debug(f"Sending chat completion request (model: {model_name})")

                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()

                # Extract content from response
                content = result["choices"][0]["message"]["content"]

                logger.info(
                    f"Chat completion successful "
                    f"(tokens: {result.get('usage', {}).get('total_tokens', 'N/A')})"
                )

                return content

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Upstage LLM API: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"

            logger.error(error_msg)
            raise UpstageLLMError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = "Request to Upstage LLM API timed out"
            logger.error(error_msg)
            raise UpstageLLMError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error calling Upstage LLM API: {e}"
            logger.error(error_msg)
            raise UpstageLLMError(error_msg) from e

    async def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Simple completion with a single prompt (convenience method).

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Example:
            >>> service = UpstageLLMService(api_key="up_xxx")
            >>> response = await service.simple_completion(
            ...     prompt="What is attention mechanism?",
            ...     system_prompt="You are an AI expert."
            ... )
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """Check if Upstage LLM API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple test completion
            messages = [{"role": "user", "content": "Hello"}]
            await self.chat_completion(messages, max_tokens=10)
            logger.debug("Upstage LLM health check: OK")
            return True
        except Exception as e:
            logger.warning(f"Upstage LLM health check failed: {e}")
            return False
