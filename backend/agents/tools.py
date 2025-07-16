"""
Tools for RAG agents - organized for reusability across different agents.

This module provides:
1. Generic tools (reusable across any agent)
2. Configurable SQL tools (adaptable to different domains)
3. Domain-specific tools (CRM-focused example)
"""

import json
import logging
import asyncio
import os
import re
import concurrent.futures
import threading
import html
import urllib.parse
import functools
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple

from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from pydantic.v1 import BaseModel, Field
from langsmith import traceable
import sqlalchemy
from sqlalchemy import create_engine, text
import sqlparse
import sqlparse.tokens
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, DML
import numpy as np

from backend.rag.retriever import SemanticRetriever
from backend.rag.embeddings import OpenAIEmbeddings
from backend.config import get_settings

logger = logging.getLogger(__name__)

# =============================================================================
# GENERIC TOOL INFRASTRUCTURE (Reusable across any agent)
# =============================================================================

# Initialize components that tools will use
_retriever = None
_settings = None
_embeddings = None

# Simple cache for search results (in production, use Redis or similar)
_search_cache = {}
_cache_max_size = 50


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


async def _get_embeddings():
    """Get or create the embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings()
    return _embeddings


def _get_cache_key(query: str, top_k: int, threshold: float) -> str:
    """Generate a cache key for search results."""
    return hashlib.md5(f"{query}:{top_k}:{threshold}".encode()).hexdigest()


def _get_cached_result(cache_key: str) -> Optional[str]:
    """Get cached result if available."""
    return _search_cache.get(cache_key)


def _cache_result(cache_key: str, result: str):
    """Cache a search result with LRU eviction."""
    global _search_cache
    
    # Simple LRU: if cache is full, remove oldest entry
    if len(_search_cache) >= _cache_max_size:
        oldest_key = next(iter(_search_cache))
        del _search_cache[oldest_key]
    
    _search_cache[cache_key] = result


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0


# =============================================================================
# GENERIC TOOLS (Reusable across any agent)
# =============================================================================

@tool("semantic_search")
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
        
        # Check cache first
        cache_key = _get_cache_key(query, top_k, threshold)
        cached_result = _get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for: {query}")
            return cached_result
        
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
        
        # Cache the result before returning
        result_json = json.dumps(results, indent=2)
        _cache_result(cache_key, result_json)
        
        return result_json
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return json.dumps({
            "error": f"Error performing semantic search: {str(e)}",
            "query": query,
            "num_results": 0,
            "documents": []
        })


@tool("format_sources")
@traceable(name="format_sources_tool")
def format_sources(sources: Optional[List[Dict[str, Any]]] = None) -> str:
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


@tool("build_context")
@traceable(name="build_context_tool")
def build_context(documents: Optional[List[Dict[str, Any]]] = None) -> str:
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


# =============================================================================
# CONFIGURABLE SQL INFRASTRUCTURE (Adaptable to different domains)
# =============================================================================

class SQLDomainConfig:
    """Configuration for different SQL domains."""
    
    def __init__(self, 
                 tables: List[str],
                 schema_docs: Dict[str, Any],
                 validation_cache: Optional[Dict[str, List[str]]] = None,
                 high_cardinality_columns: Optional[Dict[str, List[str]]] = None):
        self.tables = tables
        self.schema_docs = schema_docs
        self.validation_cache = validation_cache or {}
        self.high_cardinality_columns = high_cardinality_columns or {}


# Global SQL database connection
_sql_database = None


async def _get_sql_database():
    """Get or create the SQL database connection with restricted table access."""
    global _sql_database
    if _sql_database is None:
        settings = await _get_settings()
        
        # Extract project reference from Supabase URL
        project_ref = settings.supabase_url.split('//')[1].split('.')[0]
        
        # Get database password from settings
        db_password = settings.supabase_db_password
        if not db_password:
            logger.warning("SUPABASE_DB_PASSWORD not set. SQL tools will not be available.")
            return None
        
        # Use Supabase transaction pooler for reliable connections
        db_url = f"postgresql://postgres.{project_ref}:{db_password}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
        
        try:
            # Create SQLDatabase with restricted table access
            _sql_database = SQLDatabase.from_uri(
                database_uri=db_url,
                include_tables=CRM_CONFIG.tables,  # Only allow access to configured tables
                sample_rows_in_table_info=2,
                custom_table_info=None,
                view_support=True
            )
            
            logger.info(f"Connected to SQL database with restricted access to tables: {CRM_CONFIG.tables}")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQL database: {e}")
            raise
    
    return _sql_database


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATIONS (CRM Example)
# =============================================================================

# CRM Domain Configuration
CRM_CONFIG = SQLDomainConfig(
    tables=[
        "branches", "employees", "customers", "vehicles", 
        "opportunities", "transactions", "pricing", "activities"
    ],
    schema_docs={
        "database_overview": {
            "description": "CRM Sales Management System for automotive dealerships",
            "business_domain": "Vehicle sales, customer management, and opportunity tracking",
            "key_metrics": ["sales_performance", "conversion_rates", "revenue", "customer_satisfaction"]
        },
        "key_relationships": {
            "pricing_queries": "vehicles.id = pricing.vehicle_id (ALWAYS JOIN for price questions)",
            "sales_performance": "employees → opportunities → transactions",
            "customer_journey": "customers → opportunities → transactions"
        }
    },
    validation_cache={
        "employee_names": ["John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wang"],
        "vehicle_brands": ["Toyota", "Honda", "Ford", "Nissan", "BMW", "Mercedes", "Audi"],
        "vehicle_models": ["Camry", "RAV4", "Civic", "CR-V", "F-150", "Altima", "Prius"]
    },
    high_cardinality_columns={
        "customers": ["name", "company"],
        "employees": ["name"],
        "vehicles": ["brand", "model"],
        "branches": ["name", "brand"],
        "activities": ["subject", "description"]
    }
)


# =============================================================================
# DOMAIN-SPECIFIC TOOLS (CRM Example)
# =============================================================================

@tool("query_crm_data")
@traceable(name="query_crm_data_tool")
async def query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database with natural language questions about sales, customers, inventory, performance, and PRICING.
    
    This tool converts business questions into SQL queries and executes them safely against the CRM database.
    
    **IMPORTANT: Use this tool for ANY question about data that exists in our CRM system, including:**
    
    PRICING & VEHICLE DATA:
    - Vehicle prices (e.g., "How much is the Prius?", "What's the price of the Honda Civic?")
    - Price comparisons between models
    - Vehicle specifications (brand, model, year, type, power, etc.)
    - Inventory availability and stock levels
    
    SALES & BUSINESS DATA:
    - Sales performance by employee, branch, time period
    - Customer analysis and lifetime value  
    - Sales pipeline and conversion rates
    - Activity tracking and follow-up management
    - Revenue and transaction analysis
    
    PEOPLE & ORGANIZATION:
    - Employee information (positions, names, hierarchy)
    - Customer details and business relationships
    - Branch and location data
    
    Args:
        question: Natural language business question
        time_period: Optional time filter (e.g., "last 30 days", "this quarter", "this year")
        
    Returns:
        Formatted query results with business insights
    """
    try:
        logger.info(f"Processing CRM query: {question}")
        if time_period:
            logger.info(f"Time period filter: {time_period}")
        
        # Simple SQL generation for demo purposes
        # In production, you would use the full SQL generation logic
        db = await _get_sql_database()
        if db is None:
            return "❌ **Database connection not available. Check configuration.**"
        
        # For now, return a simple response indicating the tool is working
        return f"📊 **CRM Query Processed**\n\n**Question:** {question}\n\n**Note:** This is a simplified version. The full SQL generation and execution logic would be implemented here using the configurable domain infrastructure."
        
    except Exception as e:
        error_msg = f"❌ **Error processing CRM query:** {str(e)}"
        logger.error(f"Error in query_crm_data: {str(e)}")
        return error_msg


# =============================================================================
# TOOL REGISTRY (Available tools for agents)
# =============================================================================

def get_all_tools():
    """
    Get all available tools for the RAG agent.
    
    Returns a list of tool functions that can be bound to the LLM.
    """
    return [
        semantic_search,
        format_sources,
        build_context,
        query_crm_data
    ]


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]


def get_generic_tools():
    """Get only the generic tools (reusable across any agent)."""
    return [
        semantic_search,
        format_sources,
        build_context
    ]


def get_domain_tools():
    """Get only the domain-specific tools (CRM-focused)."""
    return [
        query_crm_data
    ]