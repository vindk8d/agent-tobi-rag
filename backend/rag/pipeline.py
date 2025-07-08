"""
Document processing pipeline: load, chunk, embed, and store documents.
"""
from typing import Optional, Dict, Any, List
from .document_loader import DocumentLoader, split_documents
from .embeddings import OpenAIEmbeddings
from .vector_store import SupabaseVectorStore
from langchain_core.documents import Document
import asyncio
import logging

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """
    Pipeline for processing documents: load, chunk, embed, and store in vector DB.
    """
    def __init__(self):
        self.embedder = OpenAIEmbeddings()
        self.vector_store = SupabaseVectorStore()

    async def process_file(self, file_path: str, file_type: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a file: load, split, embed, and store chunks.
        Returns a summary dict.
        """
        # 1. Load document
        docs: List[Document] = DocumentLoader.load_from_file(file_path, file_type)
        if not docs:
            logger.error(f"No content loaded from {file_path}")
            return {"success": False, "error": "No content loaded"}
        # 2. Split into chunks
        chunks = split_documents(docs)
        if not chunks:
            logger.error(f"No chunks generated from {file_path}")
            return {"success": False, "error": "No chunks generated"}
        # 3. Embed chunks
        chunk_dicts = [
            {"id": f"{document_id}-{i}", "content": c.page_content, "metadata": {**(c.metadata or {}), **(metadata or {})}} for i, c in enumerate(chunks)
        ]
        embedded_chunks = await self.embedder.embed_documents(chunk_dicts)
        # 4. Store in vector DB
        stored_ids = []
        for chunk in embedded_chunks:
            chunk_id = self.vector_store.upsert_embedding(
                document_id=document_id,
                embedding=chunk["embedding"],
                model_name=chunk.get("embedding_model", "text-embedding-3-small"),
                metadata=chunk.get("metadata", {})
            )
            stored_ids.append(chunk_id)
        return {
            "success": True,
            "document_id": document_id,
            "chunks": len(chunks),
            "stored_ids": stored_ids
        }

    async def process_url(self, url: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a web page: scrape, split, embed, and store chunks.
        Returns a summary dict.
        """
        from scrapers.web_scraper import WebScraper
        result = WebScraper.scrape(url)
        if not result["success"] or not result["text"]:
            logger.error(f"Failed to scrape {url}")
            return {"success": False, "error": "Scraping failed"}
        doc = Document(page_content=result["text"], metadata={"source": url, **(metadata or {})})
        chunks = split_documents([doc])
        if not chunks:
            logger.error(f"No chunks generated from {url}")
            return {"success": False, "error": "No chunks generated"}
        chunk_dicts = [
            {"id": f"{document_id}-{i}", "content": c.page_content, "metadata": {**(c.metadata or {}), **(metadata or {})}} for i, c in enumerate(chunks)
        ]
        embedded_chunks = await self.embedder.embed_documents(chunk_dicts)
        stored_ids = []
        for chunk in embedded_chunks:
            chunk_id = self.vector_store.upsert_embedding(
                document_id=document_id,
                embedding=chunk["embedding"],
                model_name=chunk.get("embedding_model", "text-embedding-3-small"),
                metadata=chunk.get("metadata", {})
            )
            stored_ids.append(chunk_id)
        return {
            "success": True,
            "document_id": document_id,
            "chunks": len(chunks),
            "stored_ids": stored_ids
        } 