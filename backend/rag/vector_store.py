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