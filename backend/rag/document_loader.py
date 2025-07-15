"""
Document loader utilities for PDFs, Word docs, and web content using LangChain and standard libraries.
"""

from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import get_settings_sync

class DocumentLoader:
    """
    Unified document loader for PDFs, Word docs, and web content.
    Returns a list of LangChain Document objects.
    """

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load a PDF file and return a list of Document objects (one per page)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_word(file_path: str) -> List[Document]:
        """Load a Word (.docx) file and return a list of Document objects (one per section)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Word file not found: {file_path}")
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()

    @staticmethod
    def load_html_from_string(html: str, url: Optional[str] = None) -> List[Document]:
        """Load HTML content from a string and return a Document object."""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        metadata = {"source": url} if url else {}
        return [Document(page_content=text, metadata=metadata)]

    @staticmethod
    def load_markdown_from_string(md: str, source: Optional[str] = None) -> List[Document]:
        """Load Markdown content from a string and return a Document object."""
        # Optionally, use the markdown package to convert to HTML, then extract text
        import markdown as mdlib
        html = mdlib.markdown(md)
        return DocumentLoader.load_html_from_string(html, url=source)

    @staticmethod
    def load_from_file(file_path: str, file_type: str) -> List[Document]:
        """Dispatch loading based on file type ('pdf', 'word', 'html', 'markdown', 'text')."""
        if file_type == "pdf":
            return DocumentLoader.load_pdf(file_path)
        elif file_type == "word":
            return DocumentLoader.load_word(file_path)
        elif file_type == "html":
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            return DocumentLoader.load_html_from_string(html, url=file_path)
        elif file_type == "markdown":
            with open(file_path, "r", encoding="utf-8") as f:
                md = f.read()
            return DocumentLoader.load_markdown_from_string(md, source=file_path)
        elif file_type == "text":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": file_path})]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

async def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split a list of LangChain Document objects into text chunks using RecursiveCharacterTextSplitter.
    Uses chunk size and overlap from config if not provided.
    """
    from backend.config import get_settings
    settings = await get_settings()
    chunk_size = chunk_size or settings.rag.chunk_size
    chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
    
    # Move the actual splitting to a thread to avoid blocking
    import asyncio
    def _split():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)
    
    return await asyncio.to_thread(_split)

# ============================================================================
# DEPRIORITIZED CODE - Website scraping functionality has been deprioritized
# ============================================================================

# DEPRIORITIZED: Website scraping functionality has been deprioritized
# from langchain_community.document_loaders import WebBaseLoader

# @staticmethod
# def load_web(url: str) -> List[Document]:
#     """Load web content from a URL and return a list of Document objects (single chunk)."""
#     # Use LangChain's WebBaseLoader for robust web page loading
#     loader = WebBaseLoader(url)
#     return loader.load() 