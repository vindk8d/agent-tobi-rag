"""
Semantic retriever for similarity search with configurable threshold and source attribution.
"""
from typing import List, Dict, Any, Optional
from .embeddings import OpenAIEmbeddings
from .vector_store import SupabaseVectorStore
from ..config import get_settings
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
        threshold = threshold if threshold is not None else self.settings.rag.similarity_threshold
        top_k = top_k if top_k is not None else self.settings.rag.max_retrieved_documents
        # 3. Search
        results = await self.vector_store.similarity_search(embedding, threshold=threshold, top_k=top_k)
        # 4. Add source attribution
        for r in results:
            r["source"] = r.get("metadata", {}).get("source")
        return results 