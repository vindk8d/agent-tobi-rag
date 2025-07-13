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

    def upsert_embedding(self, document_id: str, embedding: List[float], model_name: str = "text-embedding-3-small", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert an embedding for a document chunk.
        Returns the embedding row id.
        """
        data = {
            "document_id": document_id,
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
            logger.error(f"Failed to insert embedding for document_id={document_id}: {e}")
        
        logger.warning(f"Embedding insertion failed for document_id={document_id}")
        return ""

    def get_all_embeddings_with_content(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all embeddings with their associated document content for auditing.
        Returns a list of dictionaries with embedding and document data.
        """
        query = f"""
        SELECT 
            e.id as embedding_id,
            e.document_id,
            e.model_name,
            e.created_at as embedding_created_at,
            d.title,
            d.content,
            d.document_type,
            d.word_count,
            d.character_count,
            d.chunk_index,
            ds.name as source_name,
            ds.url as source_url,
            ds.source_type
        FROM embeddings e
        JOIN documents d ON e.document_id = d.id
        LEFT JOIN data_sources ds ON d.data_source_id = ds.id
        ORDER BY e.created_at DESC
        LIMIT {limit}
        """
        result = self.client.rpc("sql", {"query": query}).execute()
        return result.data if result.data else []

    def get_embeddings_by_source(self, source_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get embeddings for documents from a specific data source.
        """
        query = f"""
        SELECT 
            e.id as embedding_id,
            e.document_id,
            e.model_name,
            d.title,
            LEFT(d.content, 200) as content_preview,
            d.document_type,
            d.word_count,
            ds.name as source_name,
            ds.url as source_url
        FROM embeddings e
        JOIN documents d ON e.document_id = d.id
        JOIN data_sources ds ON d.data_source_id = ds.id
        WHERE ds.name ILIKE '%{source_name}%'
        ORDER BY e.created_at DESC
        LIMIT {limit}
        """
        result = self.client.rpc("sql", {"query": query}).execute()
        return result.data if result.data else []

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        """
        query = """
        SELECT 
            COUNT(*) as total_embeddings,
            COUNT(DISTINCT document_id) as unique_documents,
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
        sql = f"""
            SELECT * FROM similarity_search(
                ARRAY{query_embedding}::vector(1536),
                {match_threshold},
                {match_count}
            );
        """
        result = self.client.rpc("sql", {"query": sql}).execute()
        return result.data if result.data else []

    def hybrid_search(self, query_text: str, query_embedding: List[float], match_threshold: float = 0.8, match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (keyword + vector) using the hybrid_search function in Supabase.
        Returns a list of matching documents with combined scores.
        """
        sql = f"""
            SELECT * FROM hybrid_search(
                '{query_text.replace("'", "''")}',
                ARRAY{query_embedding}::vector(1536),
                {match_threshold},
                {match_count}
            );
        """
        result = self.client.rpc("sql", {"query": sql}).execute()
        return result.data if result.data else [] 