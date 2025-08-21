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
from core.config import get_settings_sync


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
        # Keep the raw markdown content to preserve section headers for chunking
        metadata = {"source": source} if source else {}
        return [Document(page_content=md, metadata=metadata)]

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


def split_by_sections(documents: List[Document], section_header: str = "##") -> List[Document]:
    """
    Split documents by section headers (e.g., ## for markdown H2 headers).
    This is ideal for vehicle specifications that are organized by sections.
    """
    section_chunks = []
    
    for doc in documents:
        content = doc.page_content
        metadata = doc.metadata.copy()
        
        # Split by section headers (handle both \n## and line-start ##)
        sections = content.split(f"\n{section_header} ")
        
        # If no splits found, try splitting by section headers at line start
        if len(sections) == 1:
            sections = content.split(f"{section_header} ")
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Add back the section header (except for the first section)
            if i > 0:
                section = f"{section_header} {section}"
            
            # Extract section title from the first line
            lines = section.strip().split('\n')
            if lines and lines[0].startswith(section_header):
                section_title = lines[0].replace(section_header, '').strip()
                section_metadata = {
                    **metadata,
                    'section_title': section_title,
                    'section_index': i,
                    'chunking_method': 'section_based'
                }
            else:
                section_metadata = {
                    **metadata,
                    'section_index': i,
                    'chunking_method': 'section_based'
                }
            
            section_chunks.append(Document(
                page_content=section.strip(),
                metadata=section_metadata
            ))
    
    return section_chunks


async def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    chunking_method: str = "recursive"
) -> List[Document]:
    """
    Split a list of LangChain Document objects into text chunks.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk (uses config default if not provided)
        chunk_overlap: Overlap between chunks (uses config default if not provided)
        chunking_method: Method to use ('recursive' or 'section_based')
    
    Returns:
        List of chunked documents
    """
    if chunking_method == "section_based":
        # Use section-based chunking for vehicle specifications
        return split_by_sections(documents)
    
    # Default recursive character text splitting
    from core.config import get_settings
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
