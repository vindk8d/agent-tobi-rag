"""
OpenAI Embeddings module for the Salesperson Copilot RAG system.
Handles OpenAI text embedding models with rate limiting and error handling.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from config import get_settings


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    text: str
    token_count: int
    model: str
    processing_time_ms: int


class OpenAIEmbeddings:
    """
    OpenAI Embeddings service for text embedding models.
    Handles rate limiting, error handling, and batch processing.
    """

    def __init__(self):
        self.settings = None  # Will be loaded asynchronously
        self.client: Optional[openai.AsyncOpenAI] = None  # Will be initialized asynchronously
        self.model: str = "text-embedding-3-small"  # Default model
        self.batch_size: int = 100  # Default batch size
        self.tokenizer = None  # Will be initialized asynchronously

    async def _init_tokenizer_async(self, model: str):
        """Initialize tokenizer asynchronously to avoid blocking calls."""
        def _get_tokenizer():
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base if model-specific encoding not found
                return tiktoken.get_encoding("cl100k_base")

        # Move tokenizer initialization to thread to avoid os.getcwd() blocking calls
        return await asyncio.to_thread(_get_tokenizer)

    async def _ensure_initialized(self):
        """Ensure the client is initialized asynchronously."""
        if self.client is None:
            self.settings = await get_settings()

            # Initialize OpenAI async client with proper configuration
            self.client = openai.AsyncOpenAI(
                api_key=self.settings.openai.api_key,
                timeout=60.0,
                max_retries=3
            )

            self.model = self.settings.openai.embedding_model or "text-embedding-3-small"
            self.batch_size = min(self.settings.rag.embedding_batch_size or 100, 50)

            # Initialize tokenizer asynchronously to prevent blocking calls
            self.tokenizer = await self._init_tokenizer_async(self.model)

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer asynchronously"""
        await self._ensure_initialized()

        def _count_tokens():
            if self.tokenizer is None:
                # Fallback tokenizer if not initialized yet
                try:
                    import tiktoken
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                    return len(tokenizer.encode(text))
                except Exception:
                    # Rough estimation if tiktoken fails
                    return int(len(text.split()) * 1.3)  # Approximate tokens per word
            return len(self.tokenizer.encode(text))

        # Move token counting to thread to avoid any potential blocking calls
        return await asyncio.to_thread(_count_tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer (sync fallback)"""
        if self.tokenizer is None:
            # Fallback tokenizer if not initialized yet
            try:
                import tiktoken
                tokenizer = tiktoken.get_encoding("cl100k_base")
                return len(tokenizer.encode(text))
            except Exception:
                # Rough estimation if tiktoken fails
                return int(len(text.split()) * 1.3)  # Approximate tokens per word
        return len(self.tokenizer.encode(text))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    async def _create_embedding(self, text: str) -> List[float]:
        """Create a single embedding with retry logic"""
        await self._ensure_initialized()

        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        response = await self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    async def embed_single_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text string.
        Returns an EmbeddingResult with embedding, metadata, and timing.
        """
        start_time = time.time()

        await self._ensure_initialized()

        # Count tokens asynchronously
        token_count = await self.count_tokens_async(text)

        # Create embedding
        embedding = await self._create_embedding(text)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return EmbeddingResult(
            embedding=embedding,
            text=text,
            token_count=token_count,
            model=self.model,
            processing_time_ms=processing_time_ms
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts.
        Handles rate limiting and processes in configured batch sizes.
        """
        await self._ensure_initialized()

        results = []

        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_start_time = time.time()

            # Count tokens for each text in batch asynchronously
            token_counts = await asyncio.gather(
                *[self.count_tokens_async(text) for text in batch]
            )

            # Create embeddings for the batch
            batch_embeddings = await asyncio.gather(
                *[self._create_embedding(text) for text in batch]
            )

            batch_processing_time_ms = int((time.time() - batch_start_time) * 1000)

            # Create results for this batch
            for j, (text, embedding, token_count) in enumerate(zip(batch, batch_embeddings, token_counts)):
                results.append(EmbeddingResult(
                    embedding=embedding,
                    text=text,
                    token_count=token_count,
                    model=self.model,
                    processing_time_ms=batch_processing_time_ms // len(batch)  # Distribute time
                ))

        return results

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Legacy method for compatibility.
        Generate embeddings for a list of documents and return just the embeddings.
        """
        results = await self.embed_batch(documents)
        return [result.embedding for result in results]

    async def embed_query(self, query: str) -> List[float]:
        """
        Legacy method for compatibility.
        Generate embedding for a query string and return just the embedding.
        """
        result = await self.embed_single_text(query)
        return result.embedding

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding model statistics and configuration"""
        # Model-specific dimensions
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }

        model = self.model or "text-embedding-3-small"  # Default if not initialized
        batch_size = self.batch_size or 100  # Default if not initialized

        return {
            "model": model,
            "max_tokens": 8191,
            "embedding_dimensions": model_dimensions.get(model, 1536),
            "batch_size": batch_size,
            "supported_features": [
                "batch_processing",
                "token_counting",
                "rate_limiting",
                "retry_logic"
            ]
        }


class EmbeddingValidator:
    """Utility class for validating embeddings"""

    @staticmethod
    def validate_embedding_dimensions(embedding: List[float], expected_dim: int) -> bool:
        """Validate embedding has correct dimensions"""
        return len(embedding) == expected_dim

    @staticmethod
    def validate_embedding_values(embedding: List[float]) -> bool:
        """Validate embedding values are reasonable"""
        if not embedding:
            return False

        # Check for NaN or infinite values
        for value in embedding:
            if not isinstance(value, (int, float)) or not (-1 <= value <= 1):
                return False

        return True

    @staticmethod
    def calculate_embedding_magnitude(embedding: List[float]) -> float:
        """Calculate the magnitude (norm) of an embedding vector"""
        return sum(x**2 for x in embedding) ** 0.5


# Convenience functions for easy usage
async def embed_text(text: str) -> List[float]:
    """Simple function to embed a single text and return just the embedding vector"""
    embedder = OpenAIEmbeddings()
    result = await embedder.embed_single_text(text)
    return result.embedding


async def embed_documents_simple(texts: List[str]) -> List[List[float]]:
    """Simple function to embed multiple texts and return just the embedding vectors"""
    embedder = OpenAIEmbeddings()
    results = await embedder.embed_batch(texts)
    return [result.embedding for result in results]


def test_embedding_connection() -> bool:
    """Test the OpenAI embedding connection"""
    try:
        embedder = OpenAIEmbeddings()

        # Test with a simple text
        import asyncio

        async def test():
            result = await embedder.embed_single_text("Test embedding connection")
            return result

        result = asyncio.run(test())

        print(f"✅ Embedding test successful!")
        print(f"✅ Model: {result.model}")
        print(f"✅ Dimensions: {len(result.embedding)}")
        print(f"✅ Token count: {result.token_count}")
        print(f"✅ Processing time: {result.processing_time_ms}ms")

        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    test_embedding_connection()
