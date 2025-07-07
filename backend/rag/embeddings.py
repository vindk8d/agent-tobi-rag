"""
OpenAI Embeddings module for the Salesperson Copilot RAG system.
Handles text-embedding-3-large model integration with rate limiting and error handling.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from backend.config import get_settings


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
    OpenAI Embeddings service configured for text-embedding-3-large.
    Handles rate limiting, error handling, and batch processing.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.openai.api_key)
        self.model = self.settings.openai.embedding_model
        self.batch_size = self.settings.rag.embedding_batch_size
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base if model-specific encoding not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer"""
        return len(self.tokenizer.encode(text))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    async def _create_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts with retry logic.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except openai.RateLimitError as e:
            print(f"⚠️ Rate limit hit, retrying... {e}")
            raise
        except openai.APIConnectionError as e:
            print(f"⚠️ API connection error, retrying... {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error in embedding creation: {e}")
            raise
    
    async def embed_single_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        """
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Count tokens
        token_count = self.count_tokens(text)
        
        # Check token limit (text-embedding-3-large has 8191 token limit)
        if token_count > 8191:
            raise ValueError(f"Text too long: {token_count} tokens (max 8191)")
        
        # Generate embedding
        embeddings = await self._create_embedding([text])
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return EmbeddingResult(
            embedding=embeddings[0],
            text=text,
            token_count=token_count,
            model=self.model,
            processing_time_ms=processing_time
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts.
        Automatically handles batching if input is too large.
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches to avoid rate limits and memory issues
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _process_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process a single batch of texts"""
        start_time = time.time()
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        
        if not valid_texts:
            return []
        
        # Extract just the texts for embedding
        text_list = [text for _, text in valid_texts]
        
        # Count tokens for each text
        token_counts = [self.count_tokens(text) for text in text_list]
        
        # Check for texts that are too long
        for i, (token_count, text) in enumerate(zip(token_counts, text_list)):
            if token_count > 8191:
                print(f"⚠️ Skipping text {i}: {token_count} tokens (too long)")
                continue
        
        # Generate embeddings
        embeddings = await self._create_embedding(text_list)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Create results
        results = []
        for (original_index, text), embedding, token_count in zip(valid_texts, embeddings, token_counts):
            results.append(EmbeddingResult(
                embedding=embedding,
                text=text,
                token_count=token_count,
                model=self.model,
                processing_time_ms=processing_time
            ))
        
        return results
    
    async def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document dictionaries.
        Expected format: [{"id": str, "content": str, "metadata": dict}, ...]
        """
        if not documents:
            return []
        
        # Extract content for embedding
        contents = [doc.get("content", "") for doc in documents]
        
        # Generate embeddings
        embedding_results = await self.embed_batch(contents)
        
        # Combine results with original documents
        enhanced_documents = []
        for doc, result in zip(documents, embedding_results):
            enhanced_doc = doc.copy()
            enhanced_doc.update({
                "embedding": result.embedding,
                "token_count": result.token_count,
                "embedding_model": result.model,
                "processing_time_ms": result.processing_time_ms
            })
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding model statistics and configuration"""
        return {
            "model": self.model,
            "max_tokens": 8191,
            "embedding_dimensions": 3072 if "3-large" in self.model else 1536,
            "batch_size": self.batch_size,
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
    def validate_embedding_dimensions(embedding: List[float], expected_dim: int = 3072) -> bool:
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