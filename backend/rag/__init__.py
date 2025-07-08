"""
RAG (Retrieval-Augmented Generation) system components
"""

# Import existing embeddings module
from .embeddings import OpenAIEmbeddings

# Future imports will be added when implementing other RAG components
# from .document_loader import DocumentLoader
# from .vector_store import VectorStore
# from .retriever import SemanticRetriever

__all__ = [
    "OpenAIEmbeddings",
    "get_embedding_client",
]

__version__ = "1.0.0" 