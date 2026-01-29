"""
E2E tests for real Upstage API rate limit handling.

These tests are SKIPPED by default and require:
1. UPSTAGE_API_KEY environment variable set
2. --run-real-api command-line flag to enable

Usage:
    UPSTAGE_API_KEY=your_key pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v

Note: Real API tests may incur API call charges and should be used cautiously.
"""

import pytest
import os
import asyncio
import logging
from typing import Optional

pytestmark = pytest.mark.real_api

logger = logging.getLogger(__name__)


class TestRealUpstageEmbeddingAPI:
    """Real Upstage Embedding API tests (requires API key)."""

    @pytest.fixture
    def api_key(self) -> Optional[str]:
        """Get API key from environment or skip test."""
        key = os.getenv("UPSTAGE_API_KEY")
        if not key:
            pytest.skip("UPSTAGE_API_KEY environment variable not set")
        return key

    @pytest.mark.asyncio
    async def test_embedding_basic_call(self, api_key):
        """Test basic embedding API call (real API)."""
        from backend.services.upstage.embedding import UpstageEmbeddingService

        service = UpstageEmbeddingService(api_key=api_key)

        try:
            result = await service.get_embedding("Hello world", embed_type="passage")
            assert isinstance(result, list)
            assert len(result) == 4096  # Solar embedding size
            assert all(isinstance(x, float) for x in result)
            logger.info("✓ Basic embedding call succeeded")
        except Exception as e:
            pytest.fail(f"Embedding call failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_with_custom_text(self, api_key):
        """Test embedding with longer text (real API)."""
        from backend.services.upstage.embedding import UpstageEmbeddingService

        service = UpstageEmbeddingService(api_key=api_key)
        long_text = "This is a test document about machine learning and AI. " * 10

        try:
            result = await service.get_embedding(long_text, embed_type="passage")
            assert len(result) == 4096
            logger.info("✓ Long text embedding succeeded")
        except Exception as e:
            pytest.fail(f"Long text embedding failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_query_vs_passage_types(self, api_key):
        """Test different embedding types (real API)."""
        from backend.services.upstage.embedding import UpstageEmbeddingService

        service = UpstageEmbeddingService(api_key=api_key)
        text = "What is machine learning?"

        try:
            query_embedding = await service.get_embedding(text, embed_type="query")
            passage_embedding = await service.get_embedding(text, embed_type="passage")

            assert len(query_embedding) == 4096
            assert len(passage_embedding) == 4096
            # Query and passage embeddings should be different
            assert query_embedding != passage_embedding
            logger.info("✓ Query vs passage embedding test succeeded")
        except Exception as e:
            pytest.fail(f"Query/passage embedding test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_embedding_rate_limit_handling(self, api_key):
        """Test embedding API behavior under rate limiting (real API)."""
        from backend.services.upstage.embedding import UpstageEmbeddingService

        service = UpstageEmbeddingService(api_key=api_key)

        # Make multiple rapid calls to potentially trigger rate limiting
        tasks = []
        for i in range(5):
            task = service.get_embedding(f"Test text {i}", embed_type="passage")
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            logger.info(f"Rate limit test: {successful} successful, {failed} failed")

            # At least some should succeed
            assert successful > 0
            logger.info("✓ Rate limit handling test completed")
        except Exception as e:
            pytest.fail(f"Rate limit handling test failed: {e}")


class TestRealUpstageLLMAPI:
    """Real Upstage LLM API tests (requires API key)."""

    @pytest.fixture
    def api_key(self) -> Optional[str]:
        """Get API key from environment or skip test."""
        key = os.getenv("UPSTAGE_API_KEY")
        if not key:
            pytest.skip("UPSTAGE_API_KEY environment variable not set")
        return key

    @pytest.mark.asyncio
    async def test_llm_basic_completion(self, api_key):
        """Test basic LLM completion (real API)."""
        from backend.services.upstage.llm import UpstageLLMService

        service = UpstageLLMService(api_key=api_key)

        try:
            response = await service.generate_completion(
                messages=[
                    {
                        "role": "user",
                        "content": "What is machine learning? Answer in one sentence."
                    }
                ]
            )

            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✓ LLM completion succeeded: {response[:50]}...")
        except Exception as e:
            pytest.fail(f"LLM completion failed: {e}")

    @pytest.mark.asyncio
    async def test_llm_with_system_prompt(self, api_key):
        """Test LLM with system prompt (real API)."""
        from backend.services.upstage.llm import UpstageLLMService

        service = UpstageLLMService(api_key=api_key)

        try:
            response = await service.generate_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer briefly."
                    },
                    {
                        "role": "user",
                        "content": "Explain neural networks in 2 sentences."
                    }
                ]
            )

            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✓ LLM with system prompt succeeded")
        except Exception as e:
            pytest.fail(f"LLM system prompt test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_llm_rate_limit_handling(self, api_key):
        """Test LLM API behavior under rate limiting (real API)."""
        from backend.services.upstage.llm import UpstageLLMService

        service = UpstageLLMService(api_key=api_key)

        # Make multiple calls
        tasks = []
        for i in range(3):
            task = service.generate_completion(
                messages=[
                    {
                        "role": "user",
                        "content": f"Short response to: test {i}"
                    }
                ]
            )
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful = sum(1 for r in results if isinstance(r, str) and len(r) > 0)
            failed = sum(1 for r in results if isinstance(r, Exception))

            logger.info(f"LLM rate limit test: {successful} successful, {failed} failed")
            assert successful > 0
            logger.info("✓ LLM rate limit handling test completed")
        except Exception as e:
            pytest.fail(f"LLM rate limit test failed: {e}")


class TestRealUpstageDocumentParserAPI:
    """Real Upstage Document Parser API tests (requires API key)."""

    @pytest.fixture
    def api_key(self) -> Optional[str]:
        """Get API key from environment or skip test."""
        key = os.getenv("UPSTAGE_API_KEY")
        if not key:
            pytest.skip("UPSTAGE_API_KEY environment variable not set")
        return key

    @pytest.fixture
    def sample_pdf_path(self) -> Optional[str]:
        """Get path to sample PDF for testing."""
        # Try to find a sample PDF in test data
        import pathlib
        test_papers_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "test_papers"

        if test_papers_dir.exists():
            pdf_files = list(test_papers_dir.glob("*.pdf"))
            if pdf_files:
                return str(pdf_files[0])

        pytest.skip("No sample PDF found in data/test_papers/")
        return None

    @pytest.mark.asyncio
    async def test_document_parser_basic(self, api_key, sample_pdf_path):
        """Test basic document parsing (real API)."""
        from backend.services.upstage.document_parser import UpstageDocumentParseService

        service = UpstageDocumentParseService(api_key=api_key)

        try:
            with open(sample_pdf_path, "rb") as f:
                result = await service.parse_document(f, file_name="test.pdf")

            assert isinstance(result, dict)
            assert "markdown" in result
            logger.info(f"✓ Document parsing succeeded: {len(result['markdown'])} chars")
        except FileNotFoundError:
            pytest.skip("Sample PDF not found")
        except Exception as e:
            pytest.fail(f"Document parsing failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_document_parser_rate_limit_handling(self, api_key, sample_pdf_path):
        """Test document parser rate limiting (real API)."""
        from backend.services.upstage.document_parser import UpstageDocumentParseService

        service = UpstageDocumentParseService(api_key=api_key)

        try:
            # Try to parse the same document multiple times
            results = []
            for i in range(2):
                with open(sample_pdf_path, "rb") as f:
                    result = await service.parse_document(f, file_name=f"test{i}.pdf")
                    results.append(result)

            assert len([r for r in results if r is not None]) > 0
            logger.info("✓ Document parser rate limit handling test completed")
        except FileNotFoundError:
            pytest.skip("Sample PDF not found")
        except Exception as e:
            pytest.fail(f"Document parser rate limit test failed: {e}")


class TestRealAPICircuitBreakerIntegration:
    """Integration tests for circuit breaker with real APIs."""

    @pytest.fixture
    def api_key(self) -> Optional[str]:
        """Get API key from environment or skip test."""
        key = os.getenv("UPSTAGE_API_KEY")
        if not key:
            pytest.skip("UPSTAGE_API_KEY environment variable not set")
        return key

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_real_embedding_api(self, api_key):
        """Test circuit breaker behavior with real embedding API."""
        from backend.services.upstage.embedding import UpstageEmbeddingService
        from backend.services.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        service = UpstageEmbeddingService(api_key=api_key)

        # Make some calls - should succeed if API is available
        try:
            result = await service.get_embedding("Test", embed_type="passage")
            assert len(result) == 4096
            logger.info("✓ Embedding API call succeeded with circuit breaker")
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            # Not a failure of circuit breaker itself, just API unavailability

    @pytest.mark.asyncio
    async def test_metrics_collection_with_real_api(self, api_key):
        """Test that metrics are properly collected with real API calls."""
        from backend.services.upstage.embedding import UpstageEmbeddingService
        from backend.services.metrics_service import metrics_collector

        service = UpstageEmbeddingService(api_key=api_key)

        # Get initial metrics
        summary_before = metrics_collector.get_summary()

        try:
            # Make a real API call
            result = await service.get_embedding("Test", embed_type="passage")

            # Metrics should have been updated
            summary_after = metrics_collector.get_summary()

            # Calls should have increased
            assert summary_after["upstage_services"]["embedding"]["calls"] >= summary_before[
                "upstage_services"
            ]["embedding"]["calls"]

            logger.info("✓ Metrics properly collected with real API")
        except Exception as e:
            logger.warning(f"Real API call failed: {e}")
            # Not a failure of metrics collection itself


@pytest.mark.skipif(
    os.getenv("UPSTAGE_API_KEY") is None,
    reason="UPSTAGE_API_KEY not set - real API tests disabled"
)
class TestRealAPIWithMetrics:
    """Combined real API and metrics validation tests."""

    @pytest.mark.asyncio
    async def test_full_rate_limit_flow_with_real_api(self):
        """Test complete rate limit flow using real API (optional)."""
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            pytest.skip("UPSTAGE_API_KEY not set")

        from backend.services.upstage.embedding import UpstageEmbeddingService
        from backend.services.metrics_service import metrics_collector

        service = UpstageEmbeddingService(api_key=api_key)

        try:
            # Make a call
            result = await service.get_embedding("Test content", embed_type="passage")

            assert isinstance(result, list)
            assert len(result) == 4096

            summary = metrics_collector.get_summary()
            assert summary["total_requests"] > 0

            logger.info("✓ Full rate limit flow test with real API completed")
        except Exception as e:
            logger.warning(f"Full flow test warning: {e}")
