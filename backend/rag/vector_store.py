"""
Supabase Vector Store operations for RAG system: upsert, similarity search, and hybrid search.
"""

from typing import List, Dict, Any, Optional
from database import db_client
import logging

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """
    Vector store for storing and searching embeddings in Supabase (pgvector).
    """
    def __init__(self):
        self.client = db_client.client

    def upsert_embedding(self, document_chunk_id: str, embedding: List[float], model_name: str = "text-embedding-3-small", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert an embedding for a document chunk.
        Returns the embedding row id.
        """
        data = {
            "document_chunk_id": document_chunk_id,
            "embedding": embedding,
            "model_name": model_name,
            "metadata": metadata or {},
        }
        
        try:
            # Correct Supabase client syntax for insert with select
            result = self.client.table("embeddings").insert(data).execute()
            if result.data and len(result.data) > 0:
                # Get the ID from the inserted record
                return str(result.data[0]["id"])
        except Exception as e:
            logger.error(f"Failed to insert embedding for document_chunk_id={document_chunk_id}: {e}")
        
        logger.warning(f"Embedding insertion failed for document_chunk_id={document_chunk_id}")
        return ""

    def get_all_embeddings_with_content(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all embeddings with their associated document content for auditing.
        Returns a list of dictionaries with embedding and document data.
        """
        try:
            # Get embeddings with basic info
            embeddings_result = self.client.table("embeddings").select("id, document_chunk_id, model_name, created_at").limit(limit).order("created_at", desc=True).execute()
            
            if not embeddings_result.data:
                return []
            
            # Get document chunks info
            chunk_ids = [emb["document_chunk_id"] for emb in embeddings_result.data]
            chunks_result = self.client.table("document_chunks").select("id, title, content, document_type, word_count, character_count, chunk_index, data_source_id").in_("id", chunk_ids).execute()
            
            # Get data sources info
            if chunks_result.data:
                source_ids = [chunk["data_source_id"] for chunk in chunks_result.data]
                sources_result = self.client.table("data_sources").select("id, name, url, source_type").in_("id", source_ids).execute()
                
                # Create lookup dictionaries
                chunks_lookup = {chunk["id"]: chunk for chunk in chunks_result.data}
                sources_lookup = {source["id"]: source for source in sources_result.data}
                
                # Combine the data
                combined_data = []
                for emb in embeddings_result.data:
                    chunk = chunks_lookup.get(emb["document_chunk_id"])
                    if chunk:
                        source = sources_lookup.get(chunk["data_source_id"])
                        combined_data.append({
                            "embedding_id": emb["id"],
                            "document_chunk_id": emb["document_chunk_id"],
                            "model_name": emb["model_name"],
                            "embedding_created_at": emb["created_at"],
                            "title": chunk.get("title"),
                            "content": chunk.get("content"),
                            "document_type": chunk.get("document_type"),
                            "word_count": chunk.get("word_count"),
                            "character_count": chunk.get("character_count"),
                            "chunk_index": chunk.get("chunk_index"),
                            "source_name": source.get("name") if source else None,
                            "source_url": source.get("url") if source else None,
                            "source_type": source.get("source_type") if source else None
                        })
                
                return combined_data
            
            return []
        except Exception as e:
            logger.error(f"Failed to get all embeddings with content: {e}")
            return []

    def get_embeddings_by_source(self, source_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get embeddings for documents from a specific data source.
        """
        try:
            # Get data source ID first
            ds_result = self.client.table("data_sources").select("id").ilike("name", f"%{source_name}%").execute()
            
            if not ds_result.data:
                return []
            
            data_source_ids = [ds["id"] for ds in ds_result.data]
            
            # Get document chunks for these data sources
            chunks_result = self.client.table("document_chunks").select("id").in_("data_source_id", data_source_ids).execute()
            
            if not chunks_result.data:
                return []
            
            chunk_ids = [chunk["id"] for chunk in chunks_result.data]
            
            # Get embeddings for these chunks
            embeddings_result = self.client.table("embeddings").select("id, document_chunk_id, model_name, created_at").in_("document_chunk_id", chunk_ids).limit(limit).execute()
            
            return embeddings_result.data if embeddings_result.data else []
        except Exception as e:
            logger.error(f"Failed to get embeddings by source: {e}")
            return []

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        """
        query = """
        SELECT 
            COUNT(*) as total_embeddings,
            COUNT(DISTINCT document_chunk_id) as unique_document_chunks,
            COUNT(DISTINCT model_name) as unique_models,
            MIN(created_at) as oldest_embedding,
            MAX(created_at) as newest_embedding,
            array_agg(DISTINCT model_name) as models_used
        FROM embeddings
        """
        result = self.client.rpc("sql", {"query": query}).execute()
        return result.data[0] if result.data else {}

    def similarity_search(self, query_embedding: List[float], match_threshold: float = 0.8, match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a similarity search using the similarity_search function in Supabase.
        Returns a list of matching documents with similarity scores.
        """
        try:
            result = self.client.rpc("similarity_search", {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []

    def hybrid_search(self, query_text: str, query_embedding: List[float], match_threshold: float = 0.8, match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (keyword + vector) using the hybrid_search function in Supabase.
        Returns a list of matching documents with combined scores.
        """
        try:
            result = self.client.rpc("hybrid_search", {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return [] 