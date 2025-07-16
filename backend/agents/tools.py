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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
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
_sql_llm = None
_sql_db = None

# =============================================================================
# SQL AGENT INFRASTRUCTURE (LangChain SQL QA Approach)
# =============================================================================

async def _get_sql_llm():
    """Get the LLM instance for SQL generation."""
    global _sql_llm
    if _sql_llm is None:
        settings = await get_settings()
        _sql_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
    return _sql_llm

async def _get_sql_database():
    """Get the SQL database connection for LangChain SQL toolkit."""
    global _sql_db
    if _sql_db is None:
        settings = await get_settings()
        try:
            database_url = settings.supabase.postgresql_connection_string
            # Fix postgres dialect - SQLAlchemy expects postgresql not postgres
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            logger.info(f"Connecting to database for SQL toolkit")
            _sql_db = SQLDatabase.from_uri(database_url)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    return _sql_db

# SQL Query Generation Chain
SQL_QUERY_GENERATION_TEMPLATE = """
You are a SQL expert. Given a natural language question about a business database, generate a precise SQL query.

Database Schema Information:
{schema}

Question: {question}

Additional Context:
- The database contains CRM data for a car dealership
- Key tables: employees, customers, vehicles, pricing, opportunities, branches
- Use proper joins when needed
- Return only the SQL query, no explanation
- Use LIMIT when appropriate to avoid large result sets
- For employee names, use: SELECT name, position FROM employees WHERE is_active = true
- For pricing, join vehicles and pricing tables
- For inventory, use vehicles table with stock_quantity and is_available fields

SQL Query:
"""

SQL_QUERY_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=SQL_QUERY_GENERATION_TEMPLATE
)

# SQL Result Formatting Chain
SQL_RESULT_FORMATTING_TEMPLATE = """
You are a business intelligence assistant. Format the SQL query result into a user-friendly response.

Original Question: {question}
SQL Query: {query}
Raw Results: {result}

Format the results in a clear, business-friendly way with:
- Use emojis for visual appeal (📊 for data, 💰 for pricing, 👥 for people, 🏢 for business)
- Present data in a structured format
- Include relevant context
- If no results, explain why

Formatted Response:
"""

SQL_RESULT_PROMPT = PromptTemplate(
    input_variables=["question", "query", "result"],
    template=SQL_RESULT_FORMATTING_TEMPLATE
)

async def _generate_sql_query(question: str, db: SQLDatabase) -> str:
    """Generate SQL query from natural language question using LLM."""
    llm = await _get_sql_llm()
    
    # Get database schema
    schema = db.get_table_info()
    
    # Create chain
    chain = SQL_QUERY_PROMPT | llm | StrOutputParser()
    
    # Generate query
    query = await chain.ainvoke({
        "schema": schema,
        "question": question
    })
    
    # Clean up the query - remove markdown formatting
    query = query.strip()
    if query.startswith("```sql"):
        query = query[6:]  # Remove ```sql
    if query.startswith("```"):
        query = query[3:]  # Remove ```
    if query.endswith("```"):
        query = query[:-3]  # Remove trailing ```
    
    return query.strip()

async def _execute_sql_query(query: str, db: SQLDatabase) -> str:
    """Execute SQL query safely."""
    try:
        # Basic SQL injection protection
        if not _is_safe_sql_query(query):
            raise ValueError("Potentially unsafe SQL query detected")
        
        # Execute query
        result = db.run(query)
        return result
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return f"Error: {str(e)}"

async def _format_sql_result(question: str, query: str, result: str) -> str:
    """Format SQL result into user-friendly response."""
    llm = await _get_sql_llm()
    
    chain = SQL_RESULT_PROMPT | llm | StrOutputParser()
    
    formatted_result = await chain.ainvoke({
        "question": question,
        "query": query,
        "result": result
    })
    
    return formatted_result.strip()

def _is_safe_sql_query(query: str) -> bool:
    """Basic SQL injection protection."""
    query_lower = query.lower().strip()
    
    # Block dangerous operations
    dangerous_keywords = [
        'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
        'grant', 'revoke', 'exec', 'execute', 'sp_', 'xp_', '--', ';--',
        'union', 'information_schema', 'sys.', 'pg_', 'mysql.',
        'load_file', 'into outfile', 'into dumpfile'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            return False
    
    # Allow only SELECT statements
    if not query_lower.startswith('select'):
        return False
    
    return True

# =============================================================================
# LEGACY INFRASTRUCTURE (Being phased out)
# =============================================================================

async def _ensure_embeddings():
    """Ensure embeddings are initialized."""
    global _embeddings
    if _embeddings is None:
        settings = await get_settings()
        _embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model="text-embedding-3-large"
        )
    return _embeddings

async def _ensure_retriever():
    """Ensure retriever is initialized."""
    global _retriever
    if _retriever is None:
        settings = await get_settings()
        embeddings = await _ensure_embeddings()
        _retriever = SemanticRetriever(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_anon_key,
            embeddings=embeddings
        )
    return _retriever

# Database connection cache
_db_cache = {}
_db_lock = threading.Lock()

async def _get_sql_database_legacy():
    """Get SQL database connection - legacy version."""
    settings = await get_settings()
    database_url = settings.supabase.postgresql_connection_string
    cache_key = f"sql_db_{database_url}"
    
    with _db_lock:
        if cache_key in _db_cache:
            return _db_cache[cache_key]
    
    try:
        # Fix postgres dialect - SQLAlchemy expects postgresql not postgres
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        # Create SQLAlchemy engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Create SQLDatabase instance
        db = SQLDatabase(engine)
        
        with _db_lock:
            _db_cache[cache_key] = db
        
        logger.info("SQL database connection established")
        return db
        
    except Exception as e:
        logger.error(f"Failed to connect to SQL database: {e}")
        return None

# =============================================================================
# GENERIC SEMANTIC SEARCH TOOLS
# =============================================================================

class SemanticSearchParams(BaseModel):
    """Parameters for semantic search."""
    query: str = Field(..., description="Search query for finding relevant documents")
    limit: int = Field(default=5, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score for results")

@tool(args_schema=SemanticSearchParams)
@traceable(name="semantic_search_tool")
async def semantic_search(query: str, limit: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Perform semantic search across all documents in the knowledge base.
    
    This tool searches through the entire document collection using semantic similarity
    to find the most relevant content for answering user questions.
    
    Args:
        query: The search query - describe what information you're looking for
        limit: Maximum number of results to return (default: 5)
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
    
    Returns:
        Formatted search results with content and metadata
    """
    try:
        retriever = await _ensure_retriever()
        
        # Perform semantic search
        results = await retriever.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        if not results:
            return f"No relevant documents found for query: '{query}'"
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            source_info = f"Source: {result.get('source', 'Unknown')}"
            if result.get('page_number'):
                source_info += f" (Page {result.get('page_number')})"
            
            formatted_results.append(f"""
**Result {i}:**
{source_info}
Relevance Score: {result.get('similarity_score', 0.0):.2f}

Content:
{result.get('content', 'No content available')}
""")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Error performing semantic search: {str(e)}"
        logger.error(error_msg)
        return error_msg

class FormatSourcesParams(BaseModel):
    """Parameters for formatting sources."""
    sources: List[str] = Field(..., description="List of source references to format")

@tool(args_schema=FormatSourcesParams)
@traceable(name="format_sources_tool")
async def format_sources(sources: List[str]) -> str:
    """
    Format source references for citation in responses.
    
    Args:
        sources: List of source references (file paths, URLs, etc.)
    
    Returns:
        Formatted source citations
    """
    if not sources:
        return "No sources provided"
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        # Clean up source path
        clean_source = source.replace('\\', '/').split('/')[-1]
        formatted_sources.append(f"[{i}] {clean_source}")
    
    return "\n".join(formatted_sources)

class BuildContextParams(BaseModel):
    """Parameters for building context."""
    search_results: List[Dict[str, Any]] = Field(..., description="Search results to build context from")
    max_context_length: int = Field(default=4000, description="Maximum context length in characters")

@tool(args_schema=BuildContextParams)
@traceable(name="build_context_tool")
async def build_context(search_results: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
    """
    Build context from search results for answering questions.
    
    Args:
        search_results: List of search result dictionaries
        max_context_length: Maximum context length in characters
    
    Returns:
        Built context string
    """
    if not search_results:
        return "No search results provided"
    
    context_parts = []
    current_length = 0
    
    for result in search_results:
        content = result.get('content', '')
        source = result.get('source', 'Unknown')
        
        if current_length + len(content) > max_context_length:
            # Truncate content to fit within limit
            remaining_space = max_context_length - current_length
            if remaining_space > 100:  # Only add if we have meaningful space
                content = content[:remaining_space-50] + "..."
            else:
                break
        
        context_parts.append(f"Source: {source}\n{content}")
        current_length += len(content)
        
        if current_length >= max_context_length:
            break
    
    return "\n\n".join(context_parts)

# =============================================================================
# FLEXIBLE SQL AGENT TOOL (Replacing hardcoded CRM tool)
# =============================================================================

class CRMQueryParams(BaseModel):
    """Parameters for CRM data queries."""
    question: str = Field(..., description="Natural language question about CRM data")
    time_period: Optional[str] = Field(None, description="Optional time period filter")

@tool(args_schema=CRMQueryParams)
@traceable(name="query_crm_data_tool")
async def query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database with natural language questions using an intelligent SQL agent.
    
    This tool uses an LLM to convert natural language questions into SQL queries,
    making it flexible and able to handle a wide variety of business questions.
    
    **Examples of questions this tool can handle:**
    - "How many employees are in the organization?"
    - "What are the names of all employees?"
    - "Which employees work in sales?"
    - "What vehicles do we have in stock?"
    - "What's the price of the Toyota Prius?"
    - "Show me all customers from the last month"
    - "What's our sales performance this quarter?"
    - "Which deals are in the pipeline?"
    
    Args:
        question: Natural language business question
        time_period: Optional time filter (e.g., "last 30 days", "this quarter")
        
    Returns:
        Formatted query results with business insights
    """
    try:
        logger.info(f"Processing CRM query with SQL agent: {question}")
        if time_period:
            question = f"{question} for {time_period}"
            logger.info(f"Added time period filter: {time_period}")
        
        # Get database connection
        db = await _get_sql_database()
        if db is None:
            return "❌ **Database connection not available. Check configuration.**"
        
        # Generate SQL query using LLM
        sql_query = await _generate_sql_query(question, db)
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Execute the query safely
        raw_result = await _execute_sql_query(sql_query, db)
        logger.info(f"Query executed, raw result length: {len(str(raw_result))}")
        
        # Format result using LLM
        formatted_result = await _format_sql_result(question, sql_query, raw_result)
        logger.info(f"Formatted result length: {len(formatted_result)}")
        
        return formatted_result
        
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