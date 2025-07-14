"""
Tools for the RAG agent using @tool decorators.
These tools will be available for the agent to call dynamically.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from langsmith import traceable

from rag.retriever import SemanticRetriever
from config import get_settings

logger = logging.getLogger(__name__)

# Initialize components that tools will use
_retriever = None
_settings = None

def _get_retriever():
    """Get or create the retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever

async def _get_settings():
    """Get or create the settings instance asynchronously."""
    global _settings
    if _settings is None:
        _settings = await get_settings()
    return _settings


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="The search query to find relevant documents")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve (optional)")
    threshold: Optional[float] = Field(default=None, description="Similarity threshold for filtering results (optional)")


class SourceFormattingInput(BaseModel):
    """Input schema for source formatting tool."""
    sources: List[Dict[str, Any]] = Field(description="List of source documents with metadata from semantic_search results", default=[])


class ContextBuildingInput(BaseModel):
    """Input schema for context building tool."""
    documents: List[Dict[str, Any]] = Field(description="List of retrieved documents to build context from", default=[])


class ConversationSummaryInput(BaseModel):
    """Input schema for conversation summary tool."""
    conversation_history: List[Dict[str, str]] = Field(description="List of conversation messages with role and content", default=[])


@tool("semantic_search", args_schema=SemanticSearchInput, return_direct=False)
@traceable(name="semantic_search_tool")
async def semantic_search(query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> str:
    """
    Search for relevant documents using semantic similarity.
    
    Use this tool when you need to find documents related to a user's question.
    Returns a JSON string with search results including content, sources, and similarity scores.
    """
    try:
        retriever = _get_retriever()
        settings = await _get_settings()
        
        # Use provided parameters or defaults from config
        top_k = top_k or settings.rag_max_retrieved_documents
        threshold = threshold or settings.rag_similarity_threshold
        
        logger.info(f"Performing semantic search for: {query}")
        
        # Retrieve documents
        documents = await retriever.retrieve(
            query=query,
            threshold=threshold,
            top_k=top_k
        )
        
        if not documents:
            return json.dumps({
                "query": query,
                "num_results": 0,
                "documents": [],
                "message": "No relevant documents found for the query."
            })
        
        # Format results for agent consumption
        results = {
            "query": query,
            "num_results": len(documents),
            "documents": []
        }
        
        for doc in documents:
            results["documents"].append({
                "content": doc.get("content", ""),
                "source": doc.get("source", "Unknown"),
                "similarity": doc.get("similarity", 0.0),
                "metadata": doc.get("metadata", {})
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return json.dumps({
            "error": f"Error performing semantic search: {str(e)}",
            "query": query,
            "num_results": 0,
            "documents": []
        })


@tool("format_sources", args_schema=SourceFormattingInput, return_direct=False)
@traceable(name="format_sources_tool")
def format_sources(sources: List[Dict[str, Any]] = None) -> str:
    """
    Format document sources for user-friendly display.
    
    Use this tool to create a formatted list of sources with similarity scores.
    Input should be a list of source dictionaries from semantic_search results.
    
    Expected format: {"sources": [{"source": "source_name", "similarity": 0.85, ...}, ...]}
    """
    try:
        if sources is None:
            sources = []
        
        if not sources or not isinstance(sources, list):
            logger.warning(f"format_sources called with invalid sources: {type(sources)}")
            return "No sources available."
        
        if len(sources) == 0:
            return "No sources found."
        
        formatted_sources = []
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
            if not isinstance(source, dict):
                logger.warning(f"Invalid source format at index {i}: {type(source)}")
                continue
                
            source_name = source.get("source", "Unknown source")
            similarity = source.get("similarity", 0.0)
            
            # Handle different similarity formats
            if isinstance(similarity, str):
                try:
                    similarity = float(similarity)
                except ValueError:
                    similarity = 0.0
            
            formatted_sources.append(f"{i}. {source_name} (similarity: {similarity:.2f})")
        
        if not formatted_sources:
            return "No valid sources to format."
        
        return "\n".join(formatted_sources)
        
    except Exception as e:
        logger.error(f"Error formatting sources: {str(e)}")
        return f"Error formatting sources: {str(e)}"


@tool("build_context", args_schema=ContextBuildingInput, return_direct=False)
@traceable(name="build_context_tool")
def build_context(documents: List[Dict[str, Any]] = None) -> str:
    """
    Build a formatted context string from retrieved documents.
    
    Use this tool to create a context string from search results that can be used
    to answer user questions. Input should be a list of document dictionaries from semantic_search.
    """
    try:
        if documents is None:
            documents = []
        
        if not documents or not isinstance(documents, list):
            logger.info(f"build_context called with: {type(documents)}")
            return "No documents available to build context."
        
        if len(documents) == 0:
            return "No documents provided to build context."
        
        context_parts = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Invalid document format at index {i}: {type(doc)}")
                continue
                
            content = doc.get("content", "").strip()
            source = doc.get("source", "Unknown source")
            
            if content:
                context_parts.append(f"From {source}:\n{content}")
        
        if not context_parts:
            return "No content available from documents."
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error building context: {str(e)}")
        return f"Error building document context: {str(e)}"


@tool("get_conversation_summary", args_schema=ConversationSummaryInput, return_direct=False)
@traceable(name="get_conversation_summary_tool")
def get_conversation_summary(conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Get a summary of the recent conversation for context.
    
    Use this tool to understand the conversation context before answering questions.
    If no conversation history is provided, returns a default message.
    """
    try:
        if conversation_history is None:
            conversation_history = []
        
        if not conversation_history or not isinstance(conversation_history, list):
            logger.info(f"get_conversation_summary called with: {type(conversation_history)}")
            return "No previous conversation available."
        
        if len(conversation_history) == 0:
            return "No previous conversation available."
        
        # Get last 5 messages for context
        recent_messages = conversation_history[-5:]
        
        summary_parts = []
        for msg in recent_messages:
            if not isinstance(msg, dict):
                logger.warning(f"Invalid message format: {type(msg)}")
                continue
                
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:  # Only include messages with content
                summary_parts.append(f"{role.capitalize()}: {content}")
        
        if not summary_parts:
            return "No meaningful conversation history found."
        
        return "Recent conversation:\n" + "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error getting conversation summary: {str(e)}")
        return f"Error getting conversation summary: {str(e)}"


def get_all_tools():
    """
    Get all available tools for the RAG agent.
    
    Returns a list of tool functions that can be bound to the LLM.
    """
    return [
        semantic_search,
        format_sources,
        build_context,
        get_conversation_summary
    ]


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()] 