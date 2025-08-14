"""
RAG (Retrieval-Augmented Generation) Tools

Tools for document retrieval and knowledge-based question answering using
the company's document database and knowledge base.
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Core imports
from langchain_core.tools import tool
from langsmith import traceable

# Toolbox imports
from .toolbox import get_appropriate_llm

logger = logging.getLogger(__name__)

# =============================================================================
# RAG INFRASTRUCTURE
# =============================================================================

# Global retriever instance
_retriever = None

async def _ensure_retriever():
    """Ensure the retriever is initialized and return it."""
    global _retriever
    
    if _retriever is None:
        try:
            # Import from the correct location based on original tools
            try:
                from rag.retriever import SemanticRetriever
                _retriever = SemanticRetriever()
                logger.info("[RAG] SemanticRetriever initialized successfully")
            except ImportError:
                try:
                    from rag.retriever import SemanticRetriever
                    _retriever = SemanticRetriever()
                    logger.info("[RAG] SemanticRetriever initialized successfully")
                except ImportError:
                    # Fallback: create a mock retriever for now
                    class MockRetriever:
                        async def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.7):
                            return []
                    _retriever = MockRetriever()
                    logger.warning("[RAG] Using mock retriever - SemanticRetriever not available")
                    return _retriever
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize retriever: {e}")
            raise
    
    return _retriever

async def _get_rag_llm():
    """Get LLM optimized for RAG responses."""
    try:
        # Use the general-purpose LLM selector
        return await get_appropriate_llm("document analysis and knowledge retrieval")
    except Exception as e:
        logger.error(f"[RAG] Error getting RAG LLM: {e}")
        raise

# =============================================================================
# RAG QUERY PARAMETERS AND TOOLS
# =============================================================================

class SimpleRAGParams(BaseModel):
    """Parameters for simple RAG tool."""
    question: str = Field(..., description="The user's question to answer using RAG")
    top_k: int = Field(default=5, description="Number of documents to retrieve")

@tool(args_schema=SimpleRAGParams)
@traceable(name="simple_rag")
async def simple_rag(question: str, top_k: int = 5) -> str:
    """
    Simple RAG pipeline that retrieves documents and generates responses.

    This tool searches through the company's knowledge base and document collection
    to find relevant information and generate comprehensive answers to user questions.

    **Use this tool for:**
    - Questions about company policies, procedures, and guidelines
    - Product information and specifications not in the CRM database
    - Technical documentation and how-to guides
    - Company history, mission, and values
    - Industry knowledge and best practices
    - General business information and FAQs

    **Do NOT use this tool for:**
    - Real-time CRM data queries (use simple_query_crm_data instead)
    - Customer-specific information (use CRM tools)
    - Live inventory or pricing data (use CRM tools)
    - Transactional operations (use appropriate business tools)

    Args:
        question: The user's question to answer
        top_k: Number of documents to retrieve (default: 5)

    Returns:
        Generated response based on retrieved documents
    """
    try:
        logger.info(f"[RAG] Processing question: {question[:100]}...")
        
        # Get retriever
        retriever = await _ensure_retriever()
        if not retriever:
            return "❌ Document retrieval system is currently unavailable. Please try again later or contact support."

        # Retrieve documents
        documents = await retriever.retrieve(
            query=question,
            top_k=top_k,
            threshold=0.7
        )

        # Handle no documents found
        if not documents:
            return f"""🔍 **No Relevant Documents Found**

I couldn't find any relevant documents to answer your question: '{question}'

**You might want to try:**
- Rephrasing your question with different keywords
- Asking about our business data using CRM tools instead
- Checking if the relevant documents have been uploaded to our knowledge base
- Contacting support if you believe this information should be available

**Alternative:** If you're looking for customer data, vehicle inventory, or sales information, try using the CRM query tools instead."""

        # Format context from documents
        context_parts = []
        source_list = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.get('source', f'Document {i}')
            content = doc.get('content', 'No content available')
            
            context_parts.append(f"Document {i} ({source}):\n{content}")
            source_list.append(f"• {source}")

        context = "\n\n".join(context_parts)
        sources = "\n".join(source_list)

        # Generate response using RAG LLM
        llm = await _get_rag_llm()
        
        rag_prompt = f"""You are a helpful assistant answering questions based on retrieved documents.

User Question: {question}

Retrieved Documents:
{context}

Please provide a comprehensive answer based on the retrieved documents. Follow these guidelines:

1. **Answer the question directly and thoroughly**
2. **Use information from the documents to support your answer**
3. **If multiple documents provide relevant information, synthesize them**
4. **Be specific and cite relevant details from the documents**
5. **If the documents don't fully answer the question, acknowledge what's missing**
6. **Maintain a helpful and professional tone**
7. **Structure your response clearly with headers or bullet points if helpful**

Your response should be informative and directly address the user's question using the available document content."""

        response = await llm.ainvoke([{"role": "user", "content": rag_prompt}])
        
        # Format final response with sources
        final_response = f"""{response.content}

---
**Sources:**
{sources}"""

        logger.info(f"[RAG] ✅ Successfully generated response using {len(documents)} documents")
        return final_response

    except Exception as e:
        logger.error(f"Error in simple_rag: {e}")
        return f"""❌ **Document Search Error**

I encountered an issue while searching for that information: {str(e)}

**Please try:**
- Rephrasing your question
- Using more specific keywords
- Asking for help from support if the issue persists

**Alternative:** For real-time business data, try using our CRM query tools instead."""

# =============================================================================
# DOCUMENT MANAGEMENT HELPERS
# =============================================================================

async def get_document_stats() -> Dict[str, Any]:
    """Get statistics about the document collection."""
    try:
        retriever = await _ensure_retriever()
        if not retriever:
            return {"status": "unavailable", "error": "Retriever not available"}
        
        # Get basic stats from retriever if available
        if hasattr(retriever, 'get_stats'):
            stats = await retriever.get_stats()
            return {"status": "available", "stats": stats}
        else:
            return {"status": "available", "stats": "Statistics not available"}
            
    except Exception as e:
        logger.error(f"[RAG] Error getting document stats: {e}")
        return {"status": "error", "error": str(e)}

async def search_documents_by_source(source_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for documents by source pattern."""
    try:
        retriever = await _ensure_retriever()
        if not retriever:
            return []
        
        # This would depend on the retriever implementation
        # For now, return empty list as this is a helper function
        logger.info(f"[RAG] Searching documents by source pattern: {source_pattern}")
        return []
        
    except Exception as e:
        logger.error(f"[RAG] Error searching documents by source: {e}")
        return []

# =============================================================================
# RAG QUALITY AND DEBUGGING TOOLS
# =============================================================================

async def test_rag_retrieval(question: str, top_k: int = 3) -> Dict[str, Any]:
    """Test RAG retrieval for debugging purposes (internal use)."""
    try:
        retriever = await _ensure_retriever()
        if not retriever:
            return {"status": "error", "message": "Retriever not available"}
        
        # Retrieve documents
        documents = await retriever.retrieve(
            query=question,
            top_k=top_k,
            threshold=0.5  # Lower threshold for testing
        )
        
        # Format results for debugging
        results = {
            "status": "success",
            "question": question,
            "retrieved_count": len(documents),
            "documents": []
        }
        
        for i, doc in enumerate(documents):
            results["documents"].append({
                "rank": i + 1,
                "source": doc.get('source', 'Unknown'),
                "content_preview": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                "score": doc.get('score', 'N/A')
            })
        
        return results
        
    except Exception as e:
        logger.error(f"[RAG] Error in test retrieval: {e}")
        return {"status": "error", "message": str(e)}

def format_rag_response(question: str, documents: List[Dict[str, Any]], generated_response: str) -> str:
    """Format a comprehensive RAG response with sources and metadata."""
    if not documents:
        return f"""🔍 **No Documents Found**

I couldn't find relevant documents for: "{question}"

{generated_response}"""
    
    # Build source list
    sources = []
    for i, doc in enumerate(documents, 1):
        source = doc.get('source', f'Document {i}')
        sources.append(f"{i}. {source}")
    
    # Format final response
    return f"""{generated_response}

---
**📚 Sources Referenced:**
{chr(10).join(sources)}

**📊 Search Results:** Found {len(documents)} relevant document(s)"""
