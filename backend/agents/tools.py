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
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
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

from rag.retriever import SemanticRetriever
from rag.embeddings import OpenAIEmbeddings
from config import get_settings

logger = logging.getLogger(__name__)

# =============================================================================
# DYNAMIC MODEL SELECTION INFRASTRUCTURE
# =============================================================================

class QueryComplexity(Enum):
    """Enum for classifying query complexity levels"""
    SIMPLE = "simple"      # GPT-3.5-turbo appropriate
    COMPLEX = "complex"    # GPT-4 recommended

class ModelSelector:
    """
    Handles dynamic model selection based on query complexity.
    Uses simple heuristic-based classification for simplicity and portability.
    """
    
    def __init__(self, settings):
        """Initialize with settings containing model configurations"""
        self.settings = settings
        self.simple_model = getattr(settings, 'openai_simple_model', 'gpt-3.5-turbo')
        self.complex_model = getattr(settings, 'openai_complex_model', 'gpt-4')
        
        # Complex query indicators
        self.complex_keywords = [
            "analyze", "compare", "explain why", "reasoning", 
            "strategy", "recommendation", "pros and cons",
            "multiple", "various", "different approaches",
            "evaluate", "assessment", "implications",
            "complex", "sophisticated", "comprehensive"
        ]
        
        # Simple query indicators  
        self.simple_keywords = [
            "what is", "who is", "when", "where", "how much",
            "price", "contact", "address", "status", "list",
            "show me", "find", "get", "display"
        ]
        
        logger.info(f"ModelSelector initialized: simple={self.simple_model}, complex={self.complex_model}")
    
    def classify_query_complexity(self, messages: List[BaseMessage]) -> QueryComplexity:
        """
        Classify query complexity using simple heuristics.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            QueryComplexity enum value
        """
        if not messages:
            return QueryComplexity.SIMPLE
        
        # Get the latest user message content
        latest_message = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                latest_message = msg.content.lower()
                break
        
        if not latest_message:
            return QueryComplexity.SIMPLE
        
        # Length heuristic - longer queries tend to be more complex
        if len(latest_message) > 200:
            logger.debug(f"Query classified as COMPLEX due to length: {len(latest_message)}")
            return QueryComplexity.COMPLEX
        
        # Complex keyword matching
        complex_matches = sum(1 for keyword in self.complex_keywords if keyword in latest_message)
        simple_matches = sum(1 for keyword in self.simple_keywords if keyword in latest_message)
        
        if complex_matches > 0:
            logger.debug(f"Query classified as COMPLEX due to keywords: {complex_matches} matches")
            return QueryComplexity.COMPLEX
        
        if simple_matches > 0:
            logger.debug(f"Query classified as SIMPLE due to keywords: {simple_matches} matches")
            return QueryComplexity.SIMPLE
        
        # Default to simple for unclear cases
        logger.debug("Query classified as SIMPLE (default)")
        return QueryComplexity.SIMPLE
    
    def get_model_for_query(self, messages: List[BaseMessage]) -> str:
        """
        Get the appropriate model name for a query.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Model name string
        """
        complexity = self.classify_query_complexity(messages)
        model = self.complex_model if complexity == QueryComplexity.COMPLEX else self.simple_model
        
        logger.info(f"Selected model: {model} for complexity: {complexity.value}")
        return model
    
    def get_model_for_context(self, tool_calls: List, query_length: int, has_retrieved_docs: bool = False) -> str:
        """
        Get model based on context and tool usage for more granular control.
        
        Args:
            tool_calls: List of tool calls in the query
            query_length: Length of the query text
            has_retrieved_docs: Whether documents were retrieved
            
        Returns:
            Model name string
        """
        # Complex: Multiple tools or SQL queries
        if len(tool_calls) > 1:
            logger.debug("Selected COMPLEX model due to multiple tool calls")
            return self.complex_model
        
        # Complex: CRM queries (business reasoning needed)
        if any("query_crm_data" in str(tool) for tool in tool_calls):
            logger.debug("Selected COMPLEX model due to CRM query")
            return self.complex_model
            
        # Complex: Long queries with retrieved documents
        if query_length > 150 and has_retrieved_docs:
            logger.debug("Selected COMPLEX model due to long query with documents")
            return self.complex_model
            
        logger.debug("Selected SIMPLE model (default context)")
        return self.simple_model

# =============================================================================
# GENERIC TOOL INFRASTRUCTURE (Reusable across any agent)
# =============================================================================

# Initialize components that tools will use
_retriever = None
_settings = None
_embeddings = None
_sql_db = None

# =============================================================================
# SQL AGENT INFRASTRUCTURE (LangChain SQL QA Approach)
# =============================================================================

async def _get_sql_llm(question: str = None):
    """
    Get the LLM instance for SQL generation with dynamic model selection.
    
    Args:
        question: Natural language query to assess complexity
        
    Returns:
        ChatOpenAI instance with appropriate model
    """
    settings = await get_settings()
    
    # Use dynamic model selection if question is provided
    if question:
        # Create a simple message-like structure for the ModelSelector
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=question)]
        
        # Initialize ModelSelector with current settings
        model_selector = ModelSelector(settings)
        selected_model = model_selector.get_model_for_query(messages)
        
        logger.info(f"SQL LLM using model: {selected_model} for query: {question[:50]}...")
    else:
        # Fallback to complex model for SQL operations without context
        selected_model = getattr(settings, 'openai_complex_model', 'gpt-4')
        logger.info(f"SQL LLM using default complex model: {selected_model}")
    
    # Create LLM instance with selected model
    return ChatOpenAI(
        model=selected_model,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )

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
            
            # Try direct connection first (avoid threading complications)
            try:
                import psycopg2  # Ensure psycopg2 is available
                _sql_db = SQLDatabase.from_uri(database_url)
                logger.info("SQL database connection established successfully")
            except ImportError as ie:
                logger.error(f"psycopg2 import failed in main thread: {ie}")
                return None
                
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

CRITICAL SQL RULES:
- NEVER use subqueries that return multiple rows in scalar contexts (WHERE x = (SELECT...))
- Use proper JOINs instead of subqueries when possible
- Always add LIMIT 100 unless user asks for "all" data
- Use explicit column names, avoid SELECT *
- For employee lookups, JOIN the employees table directly
- Test your logic: if subquery could return multiple rows, use IN, EXISTS, or JOIN

Additional Context:
- The database contains CRM data for a car dealership
- Key tables: employees, customers, vehicles, pricing, opportunities, branches
- Use proper joins when needed
- Return only the SQL query, no explanation
- For pricing, join vehicles and pricing tables
- For inventory, use vehicles table with stock_quantity and is_available fields

EXAMPLES OF CORRECT PATTERNS:
❌ BAD: WHERE e.name = (SELECT name FROM employees WHERE is_active = true)
✅ GOOD: WHERE e.is_active = true

❌ BAD: WHERE customer_id = (SELECT id FROM customers WHERE name LIKE '%Smith%')
✅ GOOD: JOIN customers c ON o.customer_id = c.id WHERE c.name LIKE '%Smith%'

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
    llm = await _get_sql_llm(question)
    
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
    """Execute SQL query safely with comprehensive error handling."""
    try:
        # Basic SQL injection protection
        if not _is_safe_sql_query(query):
            raise ValueError("SQL_INJECTION_DETECTED")
        
        # SQL quality validation
        validation_error = _validate_sql_query(query)
        if validation_error:
            logger.warning(f"SQL validation warning: {validation_error}")
            # Log but continue - let DB handle the actual error
        
        # Execute query with timeout protection
        result = db.run(query)
        return result
        
    except ValueError as e:
        if "SQL_INJECTION_DETECTED" in str(e):
            logger.error(f"SQL injection attempt blocked: {query}")
            return "SQL_INJECTION_ERROR"
        else:
            logger.error(f"Value error in SQL execution: {e}")
            return f"SQL_VALUE_ERROR: {str(e)}"
            
    except Exception as e:
        error_str = str(e).lower()
        
        # Categorize different types of SQL errors for specific fallbacks
        if "connection" in error_str or "connect" in error_str:
            logger.error(f"Database connection error: {e}")
            return "DB_CONNECTION_ERROR"
            
        elif "syntax" in error_str or "near" in error_str:
            logger.error(f"SQL syntax error: {e}")
            return f"SQL_SYNTAX_ERROR: {str(e)}"
            
        elif "permission" in error_str or "access" in error_str or "denied" in error_str:
            logger.error(f"Database permission error: {e}")
            return "DB_PERMISSION_ERROR"
            
        elif "timeout" in error_str or "timed out" in error_str:
            logger.error(f"Query timeout error: {e}")
            return "SQL_TIMEOUT_ERROR"
            
        elif "does not exist" in error_str or "not found" in error_str:
            logger.error(f"Schema/table not found error: {e}")
            return f"SQL_SCHEMA_ERROR: {str(e)}"
            
        elif "column" in error_str and ("unknown" in error_str or "not exist" in error_str):
            logger.error(f"Column reference error: {e}")
            return f"SQL_COLUMN_ERROR: {str(e)}"
            
        elif "cardinality" in error_str and "more than one row" in error_str:
            logger.error(f"Subquery cardinality error: {e}")
            return f"SQL_SUBQUERY_ERROR: {str(e)}"
            
        else:
            logger.error(f"General SQL execution error: {e}")
            return f"SQL_GENERAL_ERROR: {str(e)}"

async def _format_sql_result(question: str, query: str, result: str) -> str:
    """Format SQL result into user-friendly response with comprehensive fallback handling."""
    
    # Handle different types of SQL errors with specific fallback responses
    if result.startswith("SQL_") or result.startswith("DB_"):
        return await _handle_sql_error_fallback(question, query, result)
    
    # Handle empty results
    if not result or result.strip() == "" or result == "[]" or result == "()":
        return await _handle_empty_result_fallback(question, query)
    
    # CONTEXT WINDOW PROTECTION: Check result size before LLM processing
    settings = await get_settings()
    
    # Configurable limits (fallback to defaults if not set)
    MAX_RESULT_SIZE_FOR_SIMPLE_MODEL = getattr(settings, 'max_result_size_simple_model', 50000)
    FORCE_COMPLEX_MODEL_SIZE = getattr(settings, 'force_complex_model_size', 20000)
    MAX_DISPLAY_LENGTH = getattr(settings, 'max_display_length', 10000)
    
    result_size = len(str(result))
    
    if result_size > MAX_RESULT_SIZE_FOR_SIMPLE_MODEL:
        logger.warning(f"Result size ({result_size} bytes) too large for simple model formatting. Using fallback.")
        return await _handle_large_result_fallback(question, query, result, result_size)
    
    # Format successful results using LLM
    try:
        # For large results, force complex model to handle bigger context
        if result_size > FORCE_COMPLEX_MODEL_SIZE:
            llm = await _get_sql_llm()  # Use default complex model
            logger.info(f"Using complex model for large result formatting ({result_size} bytes)")
        else:
            llm = await _get_sql_llm(question)  # Use dynamic selection
            
        chain = SQL_RESULT_PROMPT | llm | StrOutputParser()
        
        formatted_result = await chain.ainvoke({
            "question": question,
            "query": query,
            "result": result
        })
        
        return formatted_result.strip()
        
    except Exception as e:
        logger.error(f"Error formatting SQL result with LLM: {e}")
        # Fallback to basic formatting if LLM fails
        return f"📊 **Query Results for:** {question}\n\n**Data Found:**\n{result}\n\n*Note: Advanced formatting temporarily unavailable*"

async def _handle_sql_error_fallback(question: str, query: str, error_code: str) -> str:
    """Provide user-friendly fallback responses for different SQL error types."""
    
    fallback_responses = {
        "SQL_INJECTION_ERROR": {
            "message": "🔒 **Security Notice:** Your query contains potentially unsafe elements and was blocked for security reasons.",
            "suggestion": "Please rephrase your question using simple, descriptive language. For example: 'How many employees are there?' or 'What vehicles are in stock?'"
        },
        
        "DB_CONNECTION_ERROR": {
            "message": "🔌 **Database Connection Issue:** I'm temporarily unable to connect to the business database.",
            "suggestion": "This is likely a temporary issue. Please try your question again in a moment, or ask about information from our documents instead."
        },
        
        "DB_PERMISSION_ERROR": {
            "message": "🚫 **Access Restricted:** I don't have permission to access the requested data.",
            "suggestion": "I can help with general business questions. Try asking about employee count, available vehicles, or pricing information."
        },
        
        "SQL_TIMEOUT_ERROR": {
            "message": "⏱️ **Query Taking Too Long:** Your request is taking longer than expected to process.",
            "suggestion": "Try asking for a smaller subset of data or be more specific about what you're looking for."
        }
    }
    
    # Handle syntax and schema errors with more specific messages
    if error_code.startswith("SQL_SYNTAX_ERROR"):
        return f"""❌ **Query Processing Issue:** I had trouble understanding how to search for that information.

🤔 **What you asked:** {question}

💡 **Suggestion:** Try rephrasing your question more simply. For example:
• "How many employees do we have?"
• "What's the price of Toyota Camry?"
• "Show me vehicles in stock"
• "How many customers are there?"

I'm better with direct questions about counts, names, prices, and availability."""

    elif error_code.startswith("SQL_SCHEMA_ERROR"):
        return f"""📋 **Information Not Found:** The specific data you're looking for might not be available in our system.

🤔 **What you asked:** {question}

💡 **Available information I can help with:**
• Employee information (names, positions, departments)  
• Vehicle inventory (models, prices, availability)
• Customer data (counts, company information)
• Sales opportunities and performance
• Branch locations

Try asking about one of these topics instead."""

    elif error_code.startswith("SQL_COLUMN_ERROR"):
        return f"""🔍 **Data Field Not Found:** I couldn't find the specific information field you're asking about.

🤔 **What you asked:** {question}

💡 **Try asking about these available details:**
• Employee: names, positions, departments, active status
• Vehicles: brand, model, type, price, stock quantity
• Customers: names, companies, contact info
• Sales: opportunities, stages, revenue, activities

Rephrase your question using these terms for better results."""

    elif error_code.startswith("SQL_SUBQUERY_ERROR"):
        return f"""⚡ **Query Processing Error:** I generated a database query with a technical issue (subquery returned multiple values).

🤔 **What you asked:** {question}

💡 **Let me try a simpler approach:**
• "Show me leads" → I'll look for new opportunities
• "List employees" → I'll show active staff members  
• "What vehicles are available" → I'll show current inventory

Please ask your question again, and I'll use a better query structure."""

    # Use predefined fallback responses for other errors
    error_type = error_code.split(":")[0]  # Get the error type without details
    fallback = fallback_responses.get(error_type)
    
    if fallback:
        return f"""{fallback['message']}

🤔 **What you asked:** {question}

💡 **Suggestion:** {fallback['suggestion']}"""
    
    # General fallback for unknown errors
    return f"""❌ **Temporary Issue:** I encountered a problem while searching for that information.

🤔 **What you asked:** {question}

💡 **What you can try:**
• Ask the question in a different way
• Try asking about employee count, vehicle inventory, or customer information
• Check if you meant to ask about something from our uploaded documents instead

If this continues, please let me know and I'll help you find the information another way."""

async def _handle_empty_result_fallback(question: str, query: str) -> str:
    """Provide helpful responses when SQL queries return no results."""
    
    question_lower = question.lower()
    
    # Detect what type of information they were looking for
    if any(word in question_lower for word in ["employee", "staff", "worker", "person"]):
        category = "employees"
        suggestions = [
            "How many employees are there?",
            "What employees work in sales?",
            "Show me all active employees"
        ]
    elif any(word in question_lower for word in ["vehicle", "car", "truck", "inventory"]):
        category = "vehicles" 
        suggestions = [
            "What vehicles are in stock?",
            "Show me Toyota vehicles",
            "What's the price of Honda Civic?"
        ]
    elif any(word in question_lower for word in ["customer", "client", "buyer"]):
        category = "customers"
        suggestions = [
            "How many customers do we have?", 
            "Show me business customers",
            "What companies are our customers?"
        ]
    elif any(word in question_lower for word in ["sale", "opportunity", "deal", "revenue"]):
        category = "sales data"
        suggestions = [
            "What opportunities are in the pipeline?",
            "Show me won deals",
            "How much revenue this quarter?"
        ]
    else:
        category = "that information"
        suggestions = [
            "How many employees are there?",
            "What vehicles are available?", 
            "Show me customer information"
        ]
    
    return f"""📭 **No Results Found:** I couldn't find any {category} matching your request.

🤔 **What you asked:** {question}

💡 **This could mean:**
• The specific criteria you mentioned don't match our current data
• The information might be stored differently than expected
• There might be a typo in your question

🔍 **Try these similar questions:**
{chr(10).join([f'• "{suggestion}"' for suggestion in suggestions])}

Or let me know if you'd like to search our uploaded documents instead!"""

async def _handle_large_result_fallback(question: str, query: str, result: str, result_size: int) -> str:
    """Fallback for when the result is too large for LLM processing."""
    logger.warning(f"Result size ({result_size} bytes) too large for LLM formatting. Using fallback.")
    
    # Get configurable display limit
    settings = await get_settings()
    MAX_DISPLAY_LENGTH = getattr(settings, 'max_display_length', 10000)
    
    # Truncate very large results for better display
    if result_size > MAX_DISPLAY_LENGTH:
        truncated_result = result[:MAX_DISPLAY_LENGTH]
        rows_shown = truncated_result.count('\n') + 1
        return f"""📊 **Query Results for:** {question}

**Data Found (showing first {rows_shown} rows):**
{truncated_result}

... *[Result truncated - {result_size:,} bytes total. Result too large for full display.]*

💡 **Tips to get smaller results:**
• Add more specific filters to your question
• Ask for specific columns instead of all data
• Use time ranges or limit criteria"""
    
    return f"""📊 **Query Results for:** {question}

**Data Found:**
{result}

*Note: The result was too large for advanced formatting. Please refer to the raw data.*"""

def _is_safe_sql_query(query: str) -> bool:
    """Basic SQL injection protection with word boundary checks."""
    query_lower = query.lower().strip()
    
    # Allow only SELECT statements
    if not query_lower.startswith('select'):
        logger.error(f"Only SELECT statements are allowed, got: {query_lower[:50]}")
        return False
    
    import re
    
    # Block dangerous SQL operations using word boundaries to avoid false positives
    dangerous_patterns = [
        r'\bdrop\s+table\b',     # DROP TABLE
        r'\bdelete\s+from\b',    # DELETE FROM
        r'\btruncate\s+table\b', # TRUNCATE TABLE
        r'\balter\s+table\b',    # ALTER TABLE
        r'\bcreate\s+table\b',   # CREATE TABLE (not just 'create')
        r'\binsert\s+into\b',    # INSERT INTO
        r'\bupdate\s+\w+\s+set\b', # UPDATE table SET
        r'\bgrant\b',            # GRANT permissions
        r'\brevoke\b',           # REVOKE permissions
        r'\bunion\s+select\b',   # UNION SELECT injection
        r';--',                  # Comment injection
        r'--\s*$',              # End line comments
        r'/\*.*?\*/',           # Block comments
        r'\bexec\s*\(',         # EXEC()
        r'\bexecute\s*\(',      # EXECUTE()
        r'\bsp_\w+',            # System stored procedures
        r'\bxp_\w+',            # Extended stored procedures
        r'\binformation_schema\b', # Schema inspection
        r'\bsys\.',             # System tables
        r'\bpg_\w+',            # PostgreSQL system functions
        r'\bmysql\.',           # MySQL system references
        r'\bload_file\s*\(',    # File operations
        r'\binto\s+outfile\b',  # File writing
        r'\binto\s+dumpfile\b', # File writing
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower):
            logger.error(f"Potentially dangerous SQL pattern detected: {pattern} in query: {query_lower[:100]}")
            return False
    
    return True

def _validate_sql_query(query: str) -> Optional[str]:
    """
    Validate SQL query for common errors that cause execution failures.
    Returns warning message if issues found, None if okay.
    """
    query_lower = query.lower()
    
    # Check for scalar subquery issues
    if 'where ' in query_lower and '= (' in query_lower and 'select ' in query_lower:
        # Look for pattern: WHERE column = (SELECT ...)
        import re
        scalar_subquery_pattern = r'where\s+\w+\.\w+\s*=\s*\(\s*select\s+\w+'
        if re.search(scalar_subquery_pattern, query_lower):
            return "Potential scalar subquery issue - subquery might return multiple rows"
    
    # Check for missing LIMIT in potentially large queries
    if 'limit ' not in query_lower:
        # Tables that typically have large datasets
        large_tables = ['customers', 'opportunities', 'vehicles', 'messages', 'activities']
        for table in large_tables:
            if table in query_lower and 'count(' not in query_lower:
                return f"Query on {table} table without LIMIT might return large dataset"
    
    # Check for SELECT * usage
    if 'select *' in query_lower:
        return "SELECT * might return more data than needed - consider specifying columns"
    
    return None

# =============================================================================
# RAG LCEL CHAINS (Modern LangChain approach)
# =============================================================================

# RAG Retrieval Chain Template
RAG_RETRIEVAL_TEMPLATE = """
Based on the user's question, create a focused search query to find the most relevant documents.

User Question: {question}

Create a concise search query that captures the key concepts and intent:
"""

RAG_RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=RAG_RETRIEVAL_TEMPLATE
)

# RAG Generation Chain Template  
RAG_GENERATION_TEMPLATE = """
You are a helpful sales assistant. Answer the user's question based on the provided context from company documents.

Context from documents:
{context}

User Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain relevant information, say so clearly
- Provide specific details and examples from the context when available
- Be conversational and helpful
- If referencing specific sources, mention them naturally

Answer:
"""

RAG_GENERATION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_GENERATION_TEMPLATE
)

# Context Formatting Template
RAG_CONTEXT_TEMPLATE = """
Document {index}: {source}
Content: {content}
Relevance Score: {score}

"""

async def _get_rag_llm():
    """Get the LLM instance for RAG generation."""
    settings = await get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        openai_api_key=settings.openai_api_key
    )

async def _create_retrieval_chain():
    """Create LCEL chain for document retrieval."""
    llm = await _get_rag_llm()
    retriever = await _ensure_retriever()
    
    async def retrieval_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LCEL-compatible retrieval function."""
        question = inputs["question"]
        
        # Generate optimized search query
        search_query_chain = RAG_RETRIEVAL_PROMPT | llm | StrOutputParser()
        optimized_query = await search_query_chain.ainvoke({"question": question})
        
        # Perform semantic search
        results = await retriever.retrieve(
            query=optimized_query.strip(),
            top_k=inputs.get("top_k", 5),
            threshold=inputs.get("threshold", 0.7)
        )
        
        return {
            "question": question,
            "documents": results,
            "search_query": optimized_query.strip()
        }
    
    return retrieval_chain

async def _create_generation_chain():
    """Create LCEL chain for response generation."""
    llm = await _get_rag_llm()
    
    async def generation_chain(inputs: Dict[str, Any]) -> str:
        """LCEL-compatible generation function."""
        question = inputs["question"]
        documents = inputs.get("documents", [])
        
        # Format context from retrieved documents
        if documents:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                formatted_context = RAG_CONTEXT_TEMPLATE.format(
                    index=i,
                    source=doc.get('source', 'Unknown'),
                    content=doc.get('content', 'No content'),
                    score=doc.get('similarity_score', 0.0)
                )
                context_parts.append(formatted_context)
            
            context = "\n".join(context_parts)
        else:
            context = "No relevant documents found."
        
        # Generate response
        generation_chain = RAG_GENERATION_PROMPT | llm | StrOutputParser()
        response = await generation_chain.ainvoke({
            "context": context,
            "question": question
        })
        
        return response
    
    return generation_chain

async def _create_rag_chain():
    """Create complete RAG chain combining retrieval and generation."""
    retrieval_chain = await _create_retrieval_chain()
    generation_chain = await _create_generation_chain()
    
    async def rag_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Complete RAG pipeline using LCEL pattern."""
        # Step 1: Retrieval
        retrieval_result = await retrieval_chain(inputs)
        
        # Step 2: Generation 
        generation_inputs = {
            "question": retrieval_result["question"],
            "documents": retrieval_result["documents"]
        }
        
        response = await generation_chain(generation_inputs)
        
        return {
            "question": retrieval_result["question"],
            "search_query": retrieval_result["search_query"],
            "documents": retrieval_result["documents"],
            "response": response,
            "sources": [doc.get('source', 'Unknown') for doc in retrieval_result["documents"]]
        }
    
    return rag_chain

# =============================================================================
# LEGACY INFRASTRUCTURE (Being phased out)
# =============================================================================

async def _ensure_embeddings():
    """Ensure embeddings are initialized."""
    global _embeddings
    if _embeddings is None:
        # OpenAIEmbeddings loads settings asynchronously, no parameters needed
        _embeddings = OpenAIEmbeddings()
    return _embeddings

async def _ensure_retriever():
    """Ensure retriever is initialized."""
    global _retriever
    if _retriever is None:
        # SemanticRetriever loads its own settings and dependencies
        _retriever = SemanticRetriever()
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

# Legacy RAG tools (semantic_search, format_sources, build_context) have been removed
# and replaced with modern LCEL-based tools (lcel_rag, lcel_retrieval, lcel_generation)
# for better performance, query optimization, and clearer tool selection

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
# LCEL-BASED RAG TOOLS (Modern LangChain tools using LCEL chains)
# =============================================================================

class LCELRAGParams(BaseModel):
    """Parameters for LCEL-based RAG tool."""
    question: str = Field(..., description="The user's question to answer using RAG")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")

@tool(args_schema=LCELRAGParams)
@traceable(name="lcel_rag_tool")
async def lcel_rag(question: str, top_k: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Complete RAG pipeline using LCEL chains for retrieval and generation.
    
    This tool uses modern LangChain Expression Language (LCEL) patterns to:
    1. Optimize the search query using an LLM
    2. Retrieve relevant documents using semantic search  
    3. Generate a comprehensive response based on retrieved context
    
    Args:
        question: The user's question to answer
        top_k: Number of documents to retrieve (default: 5)
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
    
    Returns:
        Generated response based on retrieved documents
    """
    try:
        # Create RAG chain
        rag_chain = await _create_rag_chain()
        
        # Execute RAG pipeline
        result = await rag_chain({
            "question": question,
            "top_k": top_k, 
            "threshold": similarity_threshold
        })
        
        # Check if no documents were found and provide fallback
        if not result.get('documents') or len(result.get('sources', [])) == 0:
            return await _handle_rag_no_documents_fallback(question, result.get('search_query', question))
        
        # Check for low-relevance results  
        if result.get('documents') and all(doc.get('similarity_score', 0) < 0.5 for doc in result['documents']):
            return await _handle_rag_low_relevance_fallback(question, result)
        
        # Format successful results
        response = f"""**Answer:** {result['response']}

**Search Query Used:** {result['search_query']}

**Sources ({len(result['sources'])}):**
{chr(10).join([f"• {source}" for source in result['sources']])}"""
        
        return response
        
    except Exception as e:
        # Handle different types of RAG errors
        return await _handle_rag_error_fallback(question, str(e))

async def _handle_rag_no_documents_fallback(question: str, search_query: str) -> str:
    """Provide helpful fallback when no relevant documents are found."""
    
    # Suggest alternative approaches based on question content
    question_lower = question.lower()
    
    # Check if it might be a business data question better suited for CRM
    business_keywords = [
        "employee", "staff", "customer", "vehicle", "car", "price", "inventory",
        "sales", "revenue", "opportunity", "deal", "branch", "performance"
    ]
    
    if any(keyword in question_lower for keyword in business_keywords):
        crm_suggestion = f'I might be able to find this information in our business database instead. Try asking: "Check our business data for {question.lower()}"'
    else:
        crm_suggestion = "If this is about business data (employees, inventory, sales), I can search our business database instead."
    
    return f"""📄 **No Relevant Documents Found**

🤔 **What you asked:** {question}

🔍 **Search attempts:**
• Optimized search query: "{search_query}"
• Searched through all uploaded documents
• No content met the relevance threshold for your question

💡 **What you can try:**

**1. Rephrase your question:**
• Use different keywords or synonyms
• Be more general (e.g., "customer support" instead of "technical assistance")
• Try asking about the broader topic first

**2. Alternative data sources:**
• {crm_suggestion}
• Ask if we have documents about this specific topic

**3. Upload relevant documents:**
• If you have documents about this topic, upload them to the system
• I'll be able to help better with the right source material

**Example rephrasing:**
Instead of: "{question}"
Try: "{_suggest_alternative_question(question)}"

Would you like me to try a different approach or help you find information another way?"""

async def _handle_rag_low_relevance_fallback(question: str, result: dict) -> str:
    """Handle cases where documents were found but with low relevance scores."""
    
    documents = result.get('documents', [])
    best_score = max([doc.get('similarity_score', 0) for doc in documents]) if documents else 0
    
    return f"""📄 **Limited Relevant Information Found**

🤔 **What you asked:** {question}

⚠️ **What I found:**
• Found {len(documents)} document(s) but with low relevance (best match: {best_score:.1%})
• The available content doesn't closely match your specific question
• This might not fully answer what you're looking for

💡 **Suggestions:**

**1. Try broader terms:**
• "{_suggest_broader_question(question)}"

**2. Check business data:**
• If this is about operations, employees, or sales, try asking about our business database

**3. Be more specific:**
• Add more context to help me find better matches
• Mention specific document types or topics you're thinking of

Would you like me to show you the limited results I found, or try a different approach?"""

async def _handle_rag_error_fallback(question: str, error_message: str) -> str:
    """Handle various RAG pipeline errors with helpful fallbacks."""
    
    error_lower = error_message.lower()
    
    # Categorize errors for specific responses
    if "connection" in error_lower or "network" in error_lower:
        error_type = "connection"
        message = "🔌 **Connection Issue:** I'm having trouble accessing the document search system."
        suggestion = "This is likely temporary. Please try your question again in a moment."
        
    elif "timeout" in error_lower or "timed out" in error_lower:
        error_type = "timeout"
        message = "⏱️ **Search Timeout:** Your question is taking longer than expected to process."
        suggestion = "Try making your question more specific or breaking it into smaller parts."
        
    elif "embedding" in error_lower or "vector" in error_lower:
        error_type = "embedding"
        message = "🧠 **Processing Issue:** I'm having trouble understanding how to search for that information."
        suggestion = "Try rephrasing your question using simpler terms or different keywords."
        
    elif "api" in error_lower or "openai" in error_lower:
        error_type = "api"
        message = "🤖 **AI Service Issue:** The AI processing service is temporarily unavailable."
        suggestion = "This should resolve shortly. Meanwhile, I can help you search our business database."
        
    else:
        error_type = "general"
        message = "❌ **Search Issue:** I encountered a problem while searching for that information."
        suggestion = "Let me try to help you find the information in a different way."
    
    return f"""{message}

🤔 **What you asked:** {question}

💡 **What you can do:**
• {suggestion}
• Try asking about information from our business database instead
• Ask me to help you with a different question for now

🔄 **Alternative approaches:**
• Search business data: Ask about employees, customers, vehicles, or sales
• Try later: Technical issues are usually temporary
• Rephrase: Use different words to describe what you're looking for

I'm still here to help! What else can I assist you with?"""

def _suggest_alternative_question(original_question: str) -> str:
    """Suggest an alternative phrasing for better document search."""
    question_lower = original_question.lower()
    
    # Simple heuristics for better question phrasing
    if "how to" in question_lower:
        return original_question.replace("how to", "process for").replace("How to", "Process for")
    elif "what is" in question_lower:
        return original_question.replace("what is", "information about").replace("What is", "Information about")
    elif len(original_question.split()) > 10:
        # For long questions, suggest shorter version
        words = original_question.split()
        return " ".join(words[:6]) + "..."
    else:
        # For short questions, suggest adding context
        return f"general information about {original_question.lower()}"

def _suggest_broader_question(original_question: str) -> str:
    """Suggest a broader version of the question for better matches."""
    question_lower = original_question.lower()
    
    # Extract key topics and make them broader
    if "specific" in question_lower or "particular" in question_lower:
        return question_lower.replace("specific", "general").replace("particular", "general")
    elif any(word in question_lower for word in ["exactly", "precisely", "detailed"]):
        words = original_question.split()
        filtered_words = [w for w in words if w.lower() not in ["exactly", "precisely", "detailed"]]
        return " ".join(filtered_words)
    else:
        # Add broader context words
        key_topics = []
        if "support" in question_lower:
            key_topics.append("customer support")
        elif "policy" in question_lower:
            key_topics.append("company policies")
        elif "process" in question_lower:
            key_topics.append("business processes")
        
        if key_topics:
            return f"information about {key_topics[0]}"
        else:
            return f"general information about {question_lower}"

class LCELRetrievalParams(BaseModel):
    """Parameters for LCEL-based retrieval tool."""
    question: str = Field(..., description="The question for which to retrieve relevant documents")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")

@tool(args_schema=LCELRetrievalParams)
@traceable(name="lcel_retrieval_tool")
async def lcel_retrieval(question: str, top_k: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Retrieve relevant documents using LCEL-based retrieval chain.
    
    Uses LangChain Expression Language to optimize search queries and retrieve
    the most relevant documents for a given question.
    
    Args:
        question: The question for document retrieval
        top_k: Number of documents to retrieve (default: 5)
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
    
    Returns:
        Formatted retrieved documents with metadata
    """
    try:
        # Create retrieval chain
        retrieval_chain = await _create_retrieval_chain()
        
        # Execute retrieval
        result = await retrieval_chain({
            "question": question,
            "top_k": top_k,
            "threshold": similarity_threshold
        })
        
        # Handle no documents found with comprehensive fallback
        if not result["documents"]:
            return await _handle_rag_no_documents_fallback(question, result.get('search_query', question))
        
        formatted_results = []
        formatted_results.append(f"**Search Query:** {result['search_query']}")
        formatted_results.append(f"**Retrieved Documents ({len(result['documents'])}):**\n")
        
        for i, doc in enumerate(result["documents"], 1):
            source_info = f"Source: {doc.get('source', 'Unknown')}"
            if doc.get('page_number'):
                source_info += f" (Page {doc.get('page_number')})"
                
            formatted_results.append(f"""**Document {i}:**
{source_info}
Relevance Score: {doc.get('similarity_score', 0.0):.2f}

Content:
{doc.get('content', 'No content available')}
""")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in LCEL retrieval: {str(e)}")
        return await _handle_rag_error_fallback(question, str(e))

class LCELGenerationParams(BaseModel):
    """Parameters for LCEL-based generation tool."""
    question: str = Field(..., description="The user's question to answer")
    documents: List[Dict[str, Any]] = Field(..., description="Retrieved documents to use as context")

@tool(args_schema=LCELGenerationParams)
@traceable(name="lcel_generation_tool")
async def lcel_generation(question: str, documents: List[Dict[str, Any]]) -> str:
    """
    Generate response using LCEL-based generation chain.
    
    Uses LangChain Expression Language to generate responses based on
    retrieved documents and user questions.
    
    Args:
        question: The user's question to answer
        documents: Retrieved documents to use as context
    
    Returns:
        Generated response based on the provided context
    """
    try:
        # Create generation chain
        generation_chain = await _create_generation_chain()
        
        # Execute generation
        response = await generation_chain({
            "question": question,
            "documents": documents
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in LCEL generation: {str(e)}")
        # For generation errors, provide a different type of fallback
        return f"""💭 **Response Generation Issue**

🤔 **What you asked:** {question}

❌ **Problem:** I found relevant documents but had trouble generating a comprehensive response.

💡 **What happened:**
• Retrieved {len(documents)} document(s) successfully  
• Processing issue occurred during response generation
• This is typically a temporary technical issue

🔄 **You can try:**
• Ask the question again - generation issues are usually temporary
• Try rephrasing your question slightly differently
• Ask for specific aspects of the topic instead of the full question

📄 **Raw information available:** {len(documents)} documents were found but couldn't be properly formatted.

Would you like me to try again or help you with a different question?"""

# =============================================================================
# TOOL REGISTRY (Available tools for agents)
# =============================================================================

def get_all_tools():
    """
    Get all available tools for the RAG agent.
    
    Returns a list of tool functions that can be bound to the LLM.
    """
    return [
        # LCEL-based modern tools (primary RAG functionality)
        lcel_rag,
        lcel_retrieval, 
        lcel_generation,
        # Domain-specific tools
        query_crm_data
    ]

def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]

def get_generic_tools():
    """Get only the generic tools (reusable across any agent)."""
    return [
        # LCEL-based modern tools (primary RAG functionality)
        lcel_rag,
        lcel_retrieval,
        lcel_generation
    ]

def get_domain_tools():
    """Get only the domain-specific tools (CRM-focused)."""
    return [
        query_crm_data
    ]