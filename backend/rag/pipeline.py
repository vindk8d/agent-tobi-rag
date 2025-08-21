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

    def _store_document_chunk(self, source_id: str, chunk_index: int, chunk: Document, metadata: Optional[Dict[str, Any]] = None, vehicle_id: Optional[str] = None, document_type: Optional[str] = None, is_document_centric: bool = False) -> str:
        """
        Store a document chunk in the document_chunks table.
        Returns the chunk document ID.
        
        Args:
            source_id: Either data_source_id or document_id depending on is_document_centric
            is_document_centric: If True, source_id is a document_id; if False, it's a data_source_id
        """
        from core.database import db_client

        # Calculate content stats
        content = chunk.page_content
        word_count = len(content.split()) if content else 0
        character_count = len(content) if content else 0

        # Prepare document chunk data based on approach
        if is_document_centric:
            chunk_data = {
                "document_id": source_id,  # Link to document (documents-centric)
                "vehicle_id": vehicle_id,  # Link to vehicle (if applicable)
                "title": chunk.metadata.get('title', f"Chunk {chunk_index}"),
                "content": content,
                "document_type": document_type or chunk.metadata.get('document_type', 'text'),
                "chunk_index": chunk_index,
                "word_count": word_count,
                "character_count": character_count,
                "status": "completed",
                "metadata": {**(chunk.metadata or {}), **(metadata or {})}
            }
        else:
            chunk_data = {
                "data_source_id": source_id,  # Link to data source (legacy)
                "vehicle_id": vehicle_id,  # Link to vehicle (if applicable)
                "title": chunk.metadata.get('title', f"Chunk {chunk_index}"),
                "content": content,
                "document_type": document_type or chunk.metadata.get('document_type', 'text'),
                "chunk_index": chunk_index,
                "word_count": word_count,
                "character_count": character_count,
                "status": "completed",
                "metadata": {**(chunk.metadata or {}), **(metadata or {})}
            }

        try:
            result = db_client.client.table("document_chunks").insert(chunk_data).execute()
            if result.data:
                return result.data[0]["id"]
            else:
                logger.error(f"Failed to store document chunk {chunk_index}")
                return ""
        except Exception as e:
            logger.error(f"Error storing document chunk {chunk_index}: {e}")
            return ""

    def _get_or_create_data_source(self, url: str, name: Optional[str] = None) -> str:
        """
        Get existing data source by URL or create a new one.
        Returns the data source ID.
        """
        from core.database import db_client

        try:
            # Try to find existing data source by URL
            result = db_client.client.table("data_sources").select("*").eq("url", url).execute()

            if result.data:
                return result.data[0]["id"]

            # Create new data source if not found
            data_source_data = {
                "name": name or f"Source for {url}",
                "source_type": "website",
                "url": url,
                "status": "active"
            }

            result = db_client.client.table("data_sources").insert(data_source_data).execute()
            if result.data:
                return result.data[0]["id"]
            else:
                logger.error(f"Failed to create data source for {url}")
                return ""

        except Exception as e:
            logger.error(f"Error getting/creating data source for {url}: {e}")
            return ""


    async def process_file(self, file_path: str, file_type: str, source_id: str, metadata: Optional[Dict[str, Any]] = None, vehicle_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a file: load, split, embed, and store chunks.
        
        Args:
            source_id: Either data_source_id or document_id depending on upload method
            metadata: Should contain 'upload_method' to determine if documents-centric
        
        Returns a summary dict.
        """
        from core.database import db_client
        
        # Determine if this is documents-centric (vehicle upload) or data-sources-centric (legacy)
        is_document_centric = metadata and metadata.get('upload_method') == 'vehicle_specification_upload'
        
        # Get document_type from appropriate table
        try:
            if is_document_centric:
                # Get document_type from documents table
                doc_result = db_client.client.table('documents').select('document_type').eq('id', source_id).execute()
                document_type = doc_result.data[0]['document_type'] if doc_result.data else 'vehicle_specification'
            else:
                # Get document_type from data_sources table (legacy)
                ds_result = db_client.client.table('data_sources').select('document_type').eq('id', source_id).execute()
                document_type = ds_result.data[0]['document_type'] if ds_result.data else 'text'
        except Exception as e:
            logger.warning(f"Could not get document_type from source {source_id}: {e}")
            document_type = 'vehicle_specification' if is_document_centric else 'text'
        # 1. Load document
        docs: List[Document] = DocumentLoader.load_from_file(file_path, file_type)
        if not docs:
            logger.error(f"No content loaded from {file_path}")
            return {"success": False, "error": "No content loaded"}

        # 2. Split into chunks (use section-based chunking for vehicle specifications)
        chunking_method = "section_based" if (metadata and metadata.get('upload_method') == 'vehicle_specification_upload') else "recursive"
        chunks = await split_documents(docs, chunking_method=chunking_method)
        if not chunks:
            logger.error(f"No chunks generated from {file_path}")
            return {"success": False, "error": "No chunks generated"}

        # 3. Store chunks in documents table and prepare for embedding
        chunk_dicts = []
        stored_chunk_ids = []

        for i, chunk in enumerate(chunks):
            # Enhance chunk with vehicle context if vehicle_id is provided
            enhanced_chunk = chunk
            if vehicle_id and metadata and metadata.get('vehicle_info'):
                vehicle_info = metadata['vehicle_info']
                vehicle_context = f"Vehicle: {vehicle_info.get('brand', '')} {vehicle_info.get('model', '')} {vehicle_info.get('year', '')}\n\n"
                
                # Prepend vehicle context to chunk content for better embeddings
                enhanced_content = vehicle_context + chunk.page_content
                enhanced_chunk = Document(
                    page_content=enhanced_content,
                    metadata={
                        **(chunk.metadata or {}),
                        'vehicle_id': vehicle_id,
                        'vehicle_brand': vehicle_info.get('brand'),
                        'vehicle_model': vehicle_info.get('model'),
                        'vehicle_year': vehicle_info.get('year'),
                        'has_vehicle_context': True
                    }
                )
            
            # Store the chunk content in document_chunks table
            chunk_doc_id = self._store_document_chunk(source_id, i, enhanced_chunk, metadata, vehicle_id, document_type, is_document_centric)
            if chunk_doc_id:
                stored_chunk_ids.append(chunk_doc_id)
                chunk_dicts.append({
                    "id": chunk_doc_id,
                    "content": enhanced_chunk.page_content,
                    "metadata": {**(enhanced_chunk.metadata or {}), **(metadata or {})}
                })

        if not chunk_dicts:
            logger.error(f"No chunks stored for {file_path}")
            return {"success": False, "error": "No chunks stored"}

        # 4. Embed chunks
        # Extract just the text content for embedding
        texts_to_embed = [chunk_dict["content"] for chunk_dict in chunk_dicts]
        embeddings = await self.embedder.embed_documents(texts_to_embed)

        # Combine embeddings with chunk metadata
        embedded_chunks = []
        for i, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
            embedded_chunks.append({
                "id": chunk_dict["id"],
                "content": chunk_dict["content"],
                "embedding": embedding,
                "metadata": chunk_dict["metadata"],
                "embedding_model": getattr(self.embedder, 'model', None) or "text-embedding-3-small"
            })

        # 5. Store embeddings in vector DB
        embedding_ids = []
        for chunk in embedded_chunks:
            chunk_id = await self.vector_store.upsert_embedding(
                document_chunk_id=chunk["id"],  # Use the chunk document ID
                embedding=chunk["embedding"],
                model_name=chunk.get("embedding_model", "text-embedding-3-small"),
                metadata=chunk.get("metadata", {})
            )
            embedding_ids.append(chunk_id)

        return {
            "success": True,
            "source_id": source_id,
            "source_type": "document" if is_document_centric else "data_source",
            "file_path": file_path,
            "chunks": len(chunks),
            "stored_chunk_ids": stored_chunk_ids,
            "embedding_ids": embedding_ids
        }

# ============================================================================
# DEPRIORITIZED CODE - Website scraping functionality has been deprioritized
# ============================================================================

    # DEPRIORITIZED: Website scraping functionality has been deprioritized
    # async def process_url(self, url: str, data_source_name: str = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """
    #     Process a web page: scrape, split, embed, and store chunks.
    #     Returns a summary dict.
    #     """
    #     from scrapers.web_scraper import WebScraper

    #     # Get or create data source
    #     data_source_id = self._get_or_create_data_source(url, data_source_name)
    #     if not data_source_id:
    #         logger.error(f"Failed to get/create data source for {url}")
    #         return {"success": False, "error": "Data source creation failed"}

    #     result = WebScraper.scrape(url)
    #     if not result["success"] or not result["text"]:
    #         logger.error(f"Failed to scrape {url}")
    #         return {"success": False, "error": "Scraping failed"}

    #     # Create document with scraped content
    #     doc = Document(
    #         page_content=result["text"],
    #         metadata={
    #             "source": url,
    #             "document_type": "web_page",
    #             "title": result.get("title", f"Scraped content from {url}"),
    #             **(metadata or {})
    #         }
    #     )

    #     # Split into chunks
    #     chunks = split_documents([doc])
    #     if not chunks:
    #         logger.error(f"No chunks generated from {url}")
    #         return {"success": False, "error": "No chunks generated"}

    #     # Store chunks in documents table and prepare for embedding
    #     chunk_dicts = []
    #     stored_chunk_ids = []

    #     for i, chunk in enumerate(chunks):
    #         # Store the chunk content in documents table
    #         chunk_doc_id = self._store_document_chunk(data_source_id, i, chunk, metadata)
    #         if chunk_doc_id:
    #             stored_chunk_ids.append(chunk_doc_id)
    #             chunk_dicts.append({
    #                 "id": chunk_doc_id,
    #                 "content": chunk.page_content,
    #                 "metadata": {**(chunk.metadata or {}), **(metadata or {})}
    #             })

    #     if not chunk_dicts:
    #         logger.error(f"No chunks stored for {url}")
    #         return {"success": False, "error": "No chunks stored"}

    #     # Embed chunks
    #     embedded_chunks = await self.embedder.embed_documents(chunk_dicts)

    #     # Store embeddings in vector DB
    #     embedding_ids = []
    #     for chunk in embedded_chunks:
    #         chunk_id = self.vector_store.upsert_embedding(
    #             document_id=chunk["id"],  # Use the chunk document ID
    #             embedding=chunk["embedding"],
    #             model_name=chunk.get("embedding_model", "text-embedding-3-small"),
    #             metadata=chunk.get("metadata", {})
    #         )
    #         embedding_ids.append(chunk_id)

    #     return {
    #         "success": True,
    #         "data_source_id": data_source_id,
    #         "url": url,
    #         "chunks": len(chunks),
    #         "stored_chunk_ids": stored_chunk_ids,
    #         "embedding_ids": embedding_ids
    #     }

    # DEPRIORITIZED: Website scraping functionality has been deprioritized
    # async def process_website(self,
    #                          base_url: str,
    #                          data_source_name: str = None,
    #                          max_pages: int = 50,
    #                          max_depth: int = 3,
    #                          delay: float = 1.0,
    #                          include_patterns: Optional[List[str]] = None,
    #                          exclude_patterns: Optional[List[str]] = None,
    #                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """
    #     Process an entire website: crawl child links, scrape, split, embed, and store chunks.
    #     Uses Spider to discover and crawl all related pages.

    #     Args:
    #         base_url: The root URL to start crawling from
    #         data_source_name: Name for the data source
    #         max_pages: Maximum number of pages to crawl
    #         max_depth: Maximum crawl depth
    #         delay: Delay between requests (seconds)
    #         include_patterns: URL patterns to include (e.g., ['/docs/', '/api/'])
    #         exclude_patterns: URL patterns to exclude (e.g., ['/admin/', '.pdf'])
    #         metadata: Additional metadata to store with documents

    #     Returns:
    #         Dictionary with crawl results and processing summary
    #     """
    #     from scrapers.spider_crawler import SpiderCrawler

    #     # Get or create data source
    #     data_source_id = self._get_or_create_data_source(base_url, data_source_name)
    #     if not data_source_id:
    #         logger.error(f"Failed to get/create data source for {base_url}")
    #         return {"success": False, "error": "Data source creation failed"}

    #     # Initialize Spider crawler
    #     crawler = SpiderCrawler(
    #         max_pages=max_pages,
    #         max_depth=max_depth,
    #         delay=delay,
    #         include_patterns=include_patterns,
    #         exclude_patterns=exclude_patterns
    #     )

    #     # Crawl the website
    #     logger.info(f"Starting website crawl for {base_url}")
    #     crawl_result = crawler.crawl_website(base_url, data_source_name)

    #     if not crawl_result["success"]:
    #         logger.error(f"Failed to crawl website {base_url}: {crawl_result.get('error', 'Unknown error')}")
    #         return {"success": False, "error": f"Crawling failed: {crawl_result.get('error', 'Unknown error')}"}

    #     documents = crawl_result.get("documents", [])
    #     if not documents:
    #         logger.error(f"No documents found during crawl of {base_url}")
    #         return {"success": False, "error": "No documents found"}

    #     # Process each document
    #     all_chunk_dicts = []
    #     all_stored_chunk_ids = []
    #     all_embedding_ids = []
    #     processed_pages = 0

    #     for doc in documents:
    #         try:
    #             # Split document into chunks
    #             chunks = split_documents([doc])
    #             if not chunks:
    #                 logger.warning(f"No chunks generated from {doc.metadata.get('source', 'unknown URL')}")
    #                 continue

    #             # Store chunks in documents table and prepare for embedding
    #             doc_chunk_dicts = []
    #             doc_stored_chunk_ids = []

    #             for i, chunk in enumerate(chunks):
    #                 # Store the chunk content in documents table
    #                 chunk_doc_id = self._store_document_chunk(data_source_id, len(all_stored_chunk_ids) + i, chunk, metadata)
    #                 if chunk_doc_id:
    #                     doc_stored_chunk_ids.append(chunk_doc_id)
    #                     doc_chunk_dicts.append({
    #                         "id": chunk_doc_id,
    #                         "content": chunk.page_content,
    #                         "metadata": {**(chunk.metadata or {}), **(metadata or {})}
    #                     })

    #             if doc_chunk_dicts:
    #                 all_chunk_dicts.extend(doc_chunk_dicts)
    #                 all_stored_chunk_ids.extend(doc_stored_chunk_ids)
    #                 processed_pages += 1

    #         except Exception as e:
    #             logger.error(f"Error processing document from {doc.metadata.get('source', 'unknown URL')}: {e}")
    #             continue

    #     if not all_chunk_dicts:
    #         logger.error(f"No chunks stored for website {base_url}")
    #         return {"success": False, "error": "No chunks stored"}

    #     # Embed all chunks
    #     logger.info(f"Embedding {len(all_chunk_dicts)} chunks from {processed_pages} pages")
    #     embedded_chunks = await self.embedder.embed_documents(all_chunk_dicts)

    #     # Store embeddings in vector DB
    #     for chunk in embedded_chunks:
    #         chunk_id = self.vector_store.upsert_embedding(
    #             document_id=chunk["id"],  # Use the chunk document ID
    #             embedding=chunk["embedding"],
    #             model_name=chunk.get("embedding_model", "text-embedding-3-small"),
    #             metadata=chunk.get("metadata", {})
    #         )
    #         all_embedding_ids.append(chunk_id)

    #     return {
    #         "success": True,
    #         "data_source_id": data_source_id,
    #         "base_url": base_url,
    #         "crawl_results": crawl_result,
    #         "processed_pages": processed_pages,
    #         "total_pages_found": len(documents),
    #         "total_chunks": len(all_chunk_dicts),
    #         "stored_chunk_ids": all_stored_chunk_ids,
    #         "embedding_ids": all_embedding_ids,
    #         "crawl_time": crawl_result.get("crawl_time", 0),
    #         "discovered_urls": crawl_result.get("discovered_urls", 0)
    #     }
