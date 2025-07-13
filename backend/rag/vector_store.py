"""
Supabase Vector Store operations for RAG system: upsert, similarity search, and hybrid search.
"""

from typing import List, Dict, Any, Optional
from ..database import db_client
import logging
import asyncio

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """
    Vector store for storing and searching embeddings in Supabase (pgvector).
    """
    def __init__(self):
        self.client = None  # Will be initialized asynchronously

    async def _ensure_client(self):
        """Ensure the database client is initialized asynchronously."""
        if self.client is None:
            self.client = await db_client.async_client()

    async def upsert_embedding(self, document_chunk_id: str, embedding: List[float], model_name: str = "text-embedding-3-small", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert an embedding for a document chunk.
        Returns the embedding row id.
        """
        try:
            await self._ensure_client()
            data = {
                "document_chunk_id": document_chunk_id,
                "embedding": embedding,
                "model_name": model_name,
                "metadata": metadata or {},
            }
            
            # Use async thread wrapper for non-blocking operation
            result = await asyncio.to_thread(
                lambda: self.client.table("embeddings").insert(data).execute()
            )
            
            if result.data:
                return result.data[0]["id"]
            else:
                logger.error(f"Failed to upsert embedding for document chunk {document_chunk_id}")
                return ""
        except Exception as e:
            logger.error(f"Failed to upsert embedding: {e}")
            return ""

    async def get_all_embeddings_with_content(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all embeddings with their associated document content.
        """
        try:
            await self._ensure_client()
            # Join embeddings with document_chunks to get content
            result = await asyncio.to_thread(
                lambda: self.client.table("embeddings").select(
                    "id, document_chunk_id, embedding, model_name, metadata, "
                    "document_chunks(content, title, metadata, chunk_index, data_source_id, "
                    "data_sources(name, url, metadata))"
                ).limit(limit).execute()
            )
            
            # Format the results
            formatted_results = []
            for row in result.data:
                chunk_data = row.get("document_chunks", {})
                source_data = chunk_data.get("data_sources", {}) if chunk_data else {}
                formatted_results.append({
                    "id": row["id"],
                    "document_chunk_id": row["document_chunk_id"],
                    "embedding": row["embedding"],
                    "model_name": row["model_name"],
                    "embedding_metadata": row["metadata"],
                    "content": chunk_data.get("content", ""),
                    "title": chunk_data.get("title", ""),
                    "chunk_metadata": chunk_data.get("metadata", {}),
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "data_source_id": chunk_data.get("data_source_id", ""),
                    "source_name": source_data.get("name", ""),
                    "source_url": source_data.get("url", ""),
                    "source_metadata": source_data.get("metadata", {})
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to get all embeddings with content: {e}")
            return []

    async def get_embeddings_by_source(self, source_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get embeddings for a specific data source.
        """
        try:
            await self._ensure_client()
            # Get data source ID first
            ds_result = await asyncio.to_thread(
                lambda: self.client.table("data_sources").select("id").eq("name", source_name).execute()
            )
            
            if not ds_result.data:
                return []
            
            data_source_ids = [ds["id"] for ds in ds_result.data]
            
            # Get document chunks for these data sources
            chunks_result = await asyncio.to_thread(
                lambda: self.client.table("document_chunks").select("id").in_("data_source_id", data_source_ids).execute()
            )
            
            if not chunks_result.data:
                return []
            
            chunk_ids = [chunk["id"] for chunk in chunks_result.data]
            
            # Get embeddings for these chunks
            embeddings_result = await asyncio.to_thread(
                lambda: self.client.table("embeddings").select("id, document_chunk_id, model_name, created_at").in_("document_chunk_id", chunk_ids).limit(limit).execute()
            )
            
            return embeddings_result.data if embeddings_result.data else []
        except Exception as e:
            logger.error(f"Failed to get embeddings by source: {e}")
            return []

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        """
        try:
            await self._ensure_client()
            
            # Get basic counts
            embeddings_result = await asyncio.to_thread(
                lambda: self.client.table("embeddings").select("id, document_chunk_id, model_name, created_at").execute()
            )
            
            if not embeddings_result.data:
                return {}
            
            embeddings = embeddings_result.data
            unique_chunks = set(emb["document_chunk_id"] for emb in embeddings)
            models = set(emb["model_name"] for emb in embeddings)
            
            # Get date range
            dates = [emb["created_at"] for emb in embeddings if emb["created_at"]]
            
            return {
                "total_embeddings": len(embeddings),
                "unique_document_chunks": len(unique_chunks),
                "unique_models": len(models),
                "oldest_embedding": min(dates) if dates else None,
                "newest_embedding": max(dates) if dates else None,
                "models_used": list(models)
            }
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {}

    async def similarity_search(self, query_embedding: List[float], threshold: float = 0.8, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform similarity search using the database's similarity_search function.
        """
        try:
            await self._ensure_client()
            
            # Use the existing similarity_search function
            # Note: Database function uses > instead of >=, so we subtract a small epsilon
            adjusted_threshold = max(0.0, threshold - 0.001)
            result = await asyncio.to_thread(
                lambda: self.client.rpc("similarity_search", {
                    "query_embedding": query_embedding,
                    "match_threshold": adjusted_threshold,
                    "match_count": top_k
                }).execute()
            )
            
            # Format results to match expected structure
            formatted_results = []
            for row in result.data:
                formatted_results.append({
                    "id": row["id"],
                    "document_chunk_id": row["document_chunk_id"],
                    "content": row["content"],
                    "similarity": row["similarity"],
                    "metadata": row["metadata"]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []

    async def hybrid_search(self, query_embedding: List[float], query_text: str, threshold: float = 0.8, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using the database's hybrid_search function.
        """
        try:
            await self._ensure_client()
            
            # Use the existing hybrid_search function
            # Note: Database function uses > instead of >=, so we subtract a small epsilon
            adjusted_threshold = max(0.0, threshold - 0.001)
            result = await asyncio.to_thread(
                lambda: self.client.rpc("hybrid_search", {
                    "query_text": query_text,
                    "query_embedding": query_embedding,
                    "match_threshold": adjusted_threshold,
                    "match_count": top_k
                }).execute()
            )
            
            # Format results to match expected structure
            formatted_results = []
            for row in result.data:
                formatted_results.append({
                    "id": row["id"],
                    "document_chunk_id": row["document_chunk_id"],
                    "content": row["content"],
                    "similarity": row["similarity"],
                    "keyword_rank": row["keyword_rank"],
                    "combined_score": row["combined_score"],
                    "metadata": row["metadata"]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return [] 