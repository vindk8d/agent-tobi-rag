"""
Semantic retriever for similarity search with configurable threshold and source attribution.
"""
from typing import List, Dict, Any, Optional
from .embeddings import OpenAIEmbeddings
from .vector_store import SupabaseVectorStore
from core.config import get_settings
import asyncio
import logging

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Semantic retriever for similarity search with configurable threshold and source attribution.
    """
    def __init__(self):
        self.embedder = OpenAIEmbeddings()
        self.vector_store = SupabaseVectorStore()
        self.settings = None  # Will be loaded asynchronously


    async def retrieve(self, query: str, threshold: Optional[float] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query using vector similarity search.
        Returns a list of dicts with content, similarity, and source metadata.
        """
        # Load settings asynchronously to avoid blocking calls
        if self.settings is None:
            self.settings = await get_settings()

        # 1. Embed the query
        embedding_result = await self.embedder.embed_single_text(query)
        embedding = embedding_result.embedding
        # 2. Use threshold and top_k from config if not provided
        # Use more permissive default threshold (0.3) for better recall
        config_threshold = getattr(self.settings.rag, 'similarity_threshold', 0.3) if self.settings else 0.3
        final_threshold: float = threshold if threshold is not None else config_threshold
        
        config_top_k = getattr(self.settings.rag, 'max_retrieved_documents', 10) if self.settings else 10
        final_top_k: int = top_k if top_k is not None else config_top_k
        # 3. Search
        results = await self.vector_store.similarity_search(embedding, threshold=final_threshold, top_k=final_top_k)
        # 4. Add source attribution and ensure consistent format
        for r in results:
            # Ensure source attribution from metadata
            metadata = r.get("metadata", {})
            r["source"] = (
                metadata.get("source") or 
                metadata.get("original_filename") or 
                metadata.get("storage_path") or 
                f"Document {r.get('document_chunk_id', 'Unknown')}"
            )
            
            # Ensure all expected fields are present
            r.setdefault("content", "")
            r.setdefault("similarity", 0.0)
            
        logger.info(f"[RAG] Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
