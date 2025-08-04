"""
Tools for Sales Copilot

This module provides:
1. User context management (consolidated from user_context.py)
2. Simplified SQL tools following LangChain best practices
3. Modern LCEL-based RAG tools
4. Natural language capabilities over hard-coded logic

Following the principle: Use LLM intelligence rather than keyword matching or complex logic.
"""

import logging
import contextvars
import json
import uuid
from datetime import datetime, date
from enum import Enum
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from pydantic.v1 import BaseModel, Field
from langsmith import traceable
from sqlalchemy import create_engine, text
from rag.retriever import SemanticRetriever
from core.config import get_settings
from agents.hitl import request_approval, request_input, request_selection
import re

# NOTE: LangGraph interrupt functionality is now handled by the centralized HITL system in hitl.py
# No direct interrupt handling needed in individual tools - they use dedicated HITL request tools instead



logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONNECTION UTILITIES
# =============================================================================

async def _get_sql_engine():
    """Create a SQL engine for database operations in tools."""
    settings = await get_settings()
    database_url = settings.supabase.postgresql_connection_string

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    # Settings optimized for tool execution
    engine = create_engine(
        database_url,
        pool_size=2,
        max_overflow=3,
        pool_timeout=20,
        pool_recycle=1800,
        pool_pre_ping=True,
        pool_reset_on_return='commit',
    )
    
    return engine

# =============================================================================
# MODEL SELECTION AND QUERY COMPLEXITY
# =============================================================================

class QueryComplexity(Enum):
    """Enumeration for query complexity levels."""
    SIMPLE = "simple"
    COMPLEX = "complex"


class ModelSelector:
    """Selects appropriate model based on query complexity."""

    def __init__(self, settings):
        self.settings = settings
        self.simple_model = getattr(settings, 'openai_simple_model', 'gpt-3.5-turbo')
        self.complex_model = getattr(settings, 'openai_complex_model', 'gpt-4')

        self.complex_keywords = [
            "analyze", "compare", "explain why", "reasoning",
            "strategy", "recommendation", "pros and cons"
        ]

        self.simple_keywords = [
            "what is", "who is", "when", "where", "how much",
            "price", "contact", "address", "status"
        ]

    def classify_query_complexity(self, messages):
        """Classify the complexity of a query based on the latest human message."""
        if not messages:
            return QueryComplexity.SIMPLE

        latest_message = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                latest_message = msg.content.lower()
                break

        if not latest_message:
            return QueryComplexity.SIMPLE

        # Length heuristic
        if len(latest_message) > 200:
            return QueryComplexity.COMPLEX

        # Keyword matching
        return QueryComplexity.COMPLEX if any(keyword in latest_message for keyword in self.complex_keywords) else QueryComplexity.SIMPLE

    def get_model_for_query(self, messages):
        """Get the appropriate model for the given query."""
        complexity = self.classify_query_complexity(messages)
        return self.complex_model if complexity == QueryComplexity.COMPLEX else self.simple_model

# =============================================================================
# USER CONTEXT MANAGEMENT (Consolidated from user_context.py)
# =============================================================================

# Context variables for user information
current_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_user_id', default=None)
current_conversation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_conversation_id', default=None)
current_user_type: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_user_type', default=None)
# Add context variable for state-based employee ID  
current_employee_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_employee_id', default=None)


class UserContext:
    """Context manager for setting user context during tool execution"""


    def __init__(self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, user_type: Optional[str] = None, employee_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_type = user_type
        self.employee_id = employee_id
        self.user_id_token = None
        self.conversation_id_token = None
        self.user_type_token = None
        self.employee_id_token = None


    def __enter__(self):
        # Set context variables
        if self.user_id:
            self.user_id_token = current_user_id.set(self.user_id)
            logger.debug(f"Set user context: user_id={self.user_id}")

        if self.conversation_id:
            self.conversation_id_token = current_conversation_id.set(self.conversation_id)
            logger.debug(f"Set user context: conversation_id={self.conversation_id}")

        if self.user_type:
            self.user_type_token = current_user_type.set(self.user_type)
            logger.debug(f"Set user context: user_type={self.user_type}")

        if self.employee_id:
            self.employee_id_token = current_employee_id.set(self.employee_id)
            logger.debug(f"Set user context: employee_id={self.employee_id}")

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        if self.user_id_token:
            current_user_id.reset(self.user_id_token)
            logger.debug("Reset user_id context")

        if self.conversation_id_token:
            current_conversation_id.reset(self.conversation_id_token)
            logger.debug("Reset conversation_id context")

        if self.user_type_token:
            current_user_type.reset(self.user_type_token)
            logger.debug("Reset user_type context")

        if self.employee_id_token:
            current_employee_id.reset(self.employee_id_token)
            logger.debug("Reset employee_id context")


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context"""
    return current_user_id.get()


def get_current_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context"""
    return current_conversation_id.get()


def get_current_user_type() -> Optional[str]:
    """Get the current user type from context (employee, customer, unknown)"""
    return current_user_type.get()


def get_current_employee_id_from_context() -> Optional[str]:
    """Get the current employee ID directly from context (set by agent state)"""
    return current_employee_id.get()


async def get_current_employee_id() -> Optional[str]:
    """
    Get the current employee ID - streamlined to use state-based approach.
    
    This function now primarily uses the employee_id set by user_verification_node in the agent state.
    The agent state is passed through UserContext, making this much more reliable.

    Returns:
        Optional[str]: Employee ID if user is an employee, None otherwise
    """
    # Get employee_id directly from context (set by agent state via UserContext)
    employee_id = get_current_employee_id_from_context()
    if employee_id:
        logger.debug(f"Found employee_id from state context: {employee_id}")
        return str(employee_id)
    
    # If not available from context, user is likely not an employee or context not set
    logger.debug("No employee_id available from context - user may not be an employee")
    return None


def get_user_context() -> dict:
    """Get all current user context as a dictionary"""
    return {
        "user_id": current_user_id.get(),
        "conversation_id": current_conversation_id.get(),
        "user_type": current_user_type.get()
    }

# =============================================================================
# SIMPLIFIED MODEL SELECTION (Natural Language Based)
# =============================================================================

async def _get_appropriate_llm(question: str = None) -> ChatOpenAI:
    """
    Get appropriate LLM using natural language assessment rather than keyword matching.
    Let the LLM decide its own capability needs.
    """
    settings = await get_settings()

    # Simple approach: Use GPT-4 for business questions, GPT-3.5 for simple lookups
    # Let the model's natural language understanding decide complexity
    if question and len(question) > 100:
        # Longer questions likely need more reasoning
        model = getattr(settings, 'openai_complex_model', 'gpt-4')
    else:
        # Default to efficient model
        model = getattr(settings, 'openai_simple_model', 'gpt-3.5-turbo')

    return ChatOpenAI(
        model=model,
        temperature=0,  # Deterministic for SQL
        openai_api_key=settings.openai_api_key
    )


async def _get_sql_llm(question: str = None) -> ChatOpenAI:
    """
    Get appropriate SQL LLM using ModelSelector for dynamic model selection.
    """
    settings = await get_settings()

    if question:
        # Use ModelSelector to determine appropriate model
        selector = ModelSelector(settings)
        messages = [type('Message', (), {'type': 'human', 'content': question})]
        model = selector.get_model_for_query(messages)
    else:
        # Default to complex model for SQL operations when no question provided
        model = getattr(settings, 'openai_complex_model', 'gpt-4')

    return ChatOpenAI(
        model=model,
        temperature=0,  # Deterministic for SQL
        openai_api_key=settings.openai_api_key
    )

# =============================================================================
# DATABASE CONNECTION MANAGEMENT (EXECUTION-SCOPED)
# =============================================================================

# Import the execution-scoped connection manager
from .memory import (
    create_execution_scoped_sql_database,
    get_connection_statistics,
    log_connection_status
)

async def _get_sql_database() -> Optional[SQLDatabase]:
    """Get SQL database connection using execution-scoped management."""
    try:
        # Use execution-scoped database that will be automatically cleaned up
        sql_db = await create_execution_scoped_sql_database()
        if sql_db:
            logger.debug("Created execution-scoped SQL database connection")
        return sql_db
    except Exception as e:
        logger.error(f"Failed to create execution-scoped SQL database: {e}")
        return None

# Legacy compatibility functions (deprecated)
async def close_database_connections():
    """
    DEPRECATED: This function is kept for backward compatibility but does nothing.
    Connection cleanup is now handled automatically by ExecutionScope.
    """
    logger.warning("close_database_connections() is deprecated. Use ExecutionScope for automatic cleanup.")

async def get_connection_pool_status():
    """Get current connection pool status for monitoring."""
    try:
        return await get_connection_statistics()
    except Exception as e:
        logger.error(f"Error getting connection pool status: {e}")
        return None

# =============================================================================
# CONVERSATION CONTEXT RETRIEVAL (Keep this enhancement)
# =============================================================================

async def _get_conversation_context(conversation_id: str, limit: int = 6) -> str:
    """
    Retrieve recent conversation context to help SQL generation.
    This is a good enhancement to keep from the original implementation.
    """
    try:
        if not conversation_id:
            return ""

        engine = await _get_sql_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = :conversation_id
                ORDER BY created_at DESC
                LIMIT :limit
                """),
                {"conversation_id": conversation_id, "limit": limit}
            )

            messages = result.fetchall()
            if not messages:
                return ""

            context_lines = []
            for msg in reversed(messages):  # Chronological order
                role, content, created_at = msg
                role_display = "User" if role == "human" else "Assistant"
                context_lines.append(f"{role_display}: {content}")

            return "Recent conversation:\n" + "\n".join(context_lines)

    except Exception as e:
        logger.error(f"Error retrieving conversation context: {e}")
        return ""

# =============================================================================
# SIMPLIFIED SQL QUERY GENERATION (LangChain Best Practice)
# =============================================================================

async def _generate_sql_query_simple(question: str, db: SQLDatabase, user_type: str = "employee") -> str:
    """
    Generate SQL query using LangChain best practice approach with user type-based access control.
    No hard-coded keywords - let the LLM use natural language understanding.
    
    Args:
        question: The user's question
        db: Database connection
        user_type: User type for access control (employee, customer, admin)
    """
    try:
        # Get conversation context (keep this enhancement)
        conversation_id = get_current_conversation_id()
        conversation_context = ""
        if conversation_id:
            conversation_context = await _get_conversation_context(conversation_id)

        # Get current employee ID for employee-specific queries
        employee_id = await get_current_employee_id()
        employee_context = ""
        if employee_id:
            employee_context = f"\nCurrent user employee ID: {employee_id}"

        # Get appropriate LLM using natural language assessment
        llm = await _get_appropriate_llm(question)

        # User type-specific system templates
        if user_type == "customer":
            system_template = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.

IMPORTANT - CUSTOMER ACCESS RESTRICTIONS:
You can ONLY query these tables:
- vehicles (for vehicle specifications, models, features)
- pricing (for price information)

You MUST NOT query these restricted tables:
- employees (FORBIDDEN)
- opportunities (FORBIDDEN)
- customers (FORBIDDEN)
- activities (FORBIDDEN)
- branches (FORBIDDEN)

Database Schema (CUSTOMER ACCESS ONLY):
{table_info}

{context}

Guidelines for CUSTOMER queries:
- Query for at most 5 results using LIMIT unless specified otherwise
- Only query columns needed to answer the question about vehicles and pricing
- Use proper JOIN conditions between vehicles and pricing tables when needed
- Be careful with column names and table references
- Focus only on vehicle specifications, features, models, and pricing
- If asked about employees, deals, opportunities, or customer records, respond with "Access denied - customers can only query vehicle and pricing information"

Question: {question}
SQL Query:"""
        else:
            # Employee/admin template (existing functionality)
            system_template = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.

Database Schema:
{table_info}

{context}

{employee_context}

Guidelines:
- Query for at most 5 results using LIMIT unless specified otherwise
- Only query columns needed to answer the question
- Use proper JOIN conditions when needed
- Be careful with column names and table references
- If the question references something from recent conversation, use that context
- IMPORTANT: If a current user employee ID is provided, it means the user is asking about THEIR data
- For employee-related queries (leads, opportunities, deals, pipeline, prospects), filter by opportunity_salesperson_ae_id = current employee ID
- For sales/transactions, filter by opportunity_salesperson_ae_id in transactions table
- For activities, filter by opportunity_salesperson_ae_id in activities table

Question: {question}
SQL Query:"""

        # Include context based on user type
        context_section = ""
        if conversation_context:
            context_section = f"\nRecent conversation context:\n{conversation_context}\n"

        employee_section = ""
        if employee_context and user_type in ["employee", "admin"]:
            employee_section = employee_context

        prompt = ChatPromptTemplate.from_template(system_template)

        # User type-specific chain configuration
        if user_type == "customer":
            # Customer chain - no employee context
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "table_info": lambda _: _get_customer_table_info(db),
                    "context": lambda _: context_section,
                    "dialect": lambda _: db.dialect,
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        else:
            # Employee/admin chain - full access
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "table_info": lambda _: db.get_table_info(),
                    "context": lambda _: context_section,
                    "employee_context": lambda _: employee_section,
                "dialect": lambda _: db.dialect
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        result = await chain.ainvoke(question)

        # Basic cleanup
        query = result.strip()
        if query.startswith("```sql"):
            query = query[6:]
        if query.endswith("```"):
            query = query[:-3]

        return query.strip()

    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        raise

# =============================================================================
# SIMPLIFIED QUERY EXECUTION WITH BASIC ERROR HANDLING
# =============================================================================

async def _execute_sql_query_simple(query: str, db: SQLDatabase) -> str:
    """
    Execute SQL query with simple, graceful error handling.
    No complex error categorization - keep it simple.
    """
    try:
        # Basic safety check
        query_lower = query.lower().strip()
        if not query_lower.startswith('select'):
            return "ERROR: Only SELECT queries are allowed"

        # Execute with LangChain's built-in safety
        result = db.run(query)
        return result

    except Exception as e:
        # Simple error handling - let the LLM understand the error naturally
        error_msg = str(e)
        logger.error(f"SQL execution error: {error_msg}")
        return f"SQL_ERROR: {error_msg}"

# =============================================================================
# SIMPLIFIED RESULT FORMATTING
# =============================================================================

async def _format_sql_result_simple(question: str, query: str, result: str) -> str:
    """
    Format SQL result using simple, natural language approach.
    No complex fallback logic - trust the LLM to handle edge cases.
    """
    # Handle errors gracefully
    if result.startswith("SQL_ERROR:") or result.startswith("ERROR:"):
        return f"I encountered an issue: {result}. Could you please rephrase your question?"

    # Handle empty results
    if not result or result.strip() in ["", "[]", "()"]:
        return f"I couldn't find any data matching your question: '{question}'. The query might need different criteria."

    try:
        # Simple formatting prompt - let LLM handle complexity naturally
        llm = await _get_appropriate_llm(question)

        format_prompt = """Given this question and SQL result, provide a helpful answer.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Please provide a clear, conversational response based on the data:"""

        prompt = ChatPromptTemplate.from_template(format_prompt)
        chain = prompt | llm | StrOutputParser()

        formatted = await chain.ainvoke({
            "question": question,
            "query": query,
            "result": result
        })

        return formatted.strip()

    except Exception as e:
        logger.error(f"Error formatting result: {e}")
        # Simple fallback
        return f"Query results: {result}"

# =============================================================================
# SIMPLIFIED MAIN SQL TOOL
# =============================================================================

class SimpleCRMQueryParams(BaseModel):
    """Simplified parameters for CRM queries."""
    question: str = Field(..., description="Natural language question about CRM data")
    time_period: Optional[str] = Field(None, description="Optional time period filter")

@tool(args_schema=SimpleCRMQueryParams)
@traceable(name="simple_query_crm_data")
async def simple_query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database using simplified, natural language approach with user type-based access control.

    This tool follows LangChain best practices:
    - Relies on LLM natural language understanding
    - No hard-coded keyword matching
    - Simple, graceful error handling
    - Context-aware through conversation history
    - User-aware through employee context
    - Table-level access control based on user type

    Args:
        question: Natural language business question
        time_period: Optional time period filter

    Returns:
        Natural language response based on data
    """
    try:
        # Check user type for access control
        user_type = get_current_user_type()
        logger.info(f"[CRM_QUERY] Processing CRM query for user_type: {user_type}, question: {question}")

        # Add time period to question if provided
        if time_period:
            question = f"{question} for {time_period}"

        # Get database connection
        db = await _get_sql_database()
        if not db:
            return "Sorry, I cannot access the database right now. Please try again later."

        # Generate SQL query using simplified approach with user type filtering
        sql_query = await _generate_sql_query_simple(question, db, user_type)
        logger.info(f"[CRM_QUERY] Generated query for {user_type}: {sql_query}")

        # Table-based access control for customers (most secure approach)
        if user_type == "customer":
            # Define tables that customers are NOT allowed to access
            restricted_tables_for_customers = [
                "employees", "employee", 
                "opportunities", "opportunity",
                "customers", "customer", 
                "sales_activities", "sales_activity",
                "branches", "branch",
                "performance_metrics", "performance", 
                "commissions", "commission",
                "leads", "lead",
                "deals", "deal",
                "activities", "activity"
            ]
            
            # Check if generated SQL query accesses any restricted tables
            sql_query_lower = sql_query.lower()
            for restricted_table in restricted_tables_for_customers:
                if restricted_table in sql_query_lower:
                    logger.warning(f"[CRM_QUERY] Customer user attempted to access restricted table '{restricted_table}' with query: {question}")
                    return """I apologize, but I can only help you with vehicle specifications and pricing information. 
                    
I cannot access information about employees, customer records, sales opportunities, or other business data.

Please ask me about:
- Vehicle models, features, and specifications
- Pricing information
- Available inventory
- Vehicle comparisons

How can I help you find the right vehicle for your needs?"""

        # Execute query
        raw_result = await _execute_sql_query_simple(sql_query, db)

        # Format result
        formatted_result = await _format_sql_result_simple(question, sql_query, raw_result)

        return formatted_result

    except Exception as e:
        logger.error(f"Error in simple_query_crm_data: {e}")
        return f"I encountered an issue while processing your question. Please try rephrasing it or ask for help."


def _get_customer_table_info(db: SQLDatabase) -> str:
    """
    Get table information filtered for customer access.
    Only includes vehicles and pricing tables.
    """
    try:
        # Get full table info
        full_table_info = db.get_table_info()
        
        # Filter for customer-allowed tables only
        allowed_tables = ["vehicles", "pricing"]
        filtered_lines = []
        current_table = None
        include_line = False
        
        for line in full_table_info.split('\n'):
            line_lower = line.lower()
            
            # Check if this line starts a new table definition
            if 'create table' in line_lower:
                table_found = False
                for table in allowed_tables:
                    if table in line_lower:
                        include_line = True
                        current_table = table
                        table_found = True
                        break
                if not table_found:
                    include_line = False
                    current_table = None
            
            # Include line if we're in an allowed table or it's a general schema line
            if include_line or (not line.strip() or line.startswith('/*') or 'sqlite_stat' not in line_lower):
                filtered_lines.append(line)
                
        filtered_info = '\n'.join(filtered_lines)
        
        if not filtered_info.strip():
            return """
Customer Access: Limited to vehicle specifications and pricing information only.

Available tables:
- vehicles: Vehicle models, specifications, features, inventory
- pricing: Price information, base prices, discounts, fees
"""
        
        return filtered_info
        
    except Exception as e:
        logger.error(f"Error filtering table info for customer: {e}")
        return """
Customer Access: Limited to vehicle specifications and pricing information only.

Available tables:
- vehicles: Vehicle models, specifications, features, inventory  
- pricing: Price information, base prices, discounts, fees
"""


# =============================================================================
# RAG INFRASTRUCTURE (Simplified)
# =============================================================================

_retriever = None
_embeddings = None


async def _ensure_retriever():
    """Ensure retriever is initialized."""
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever


async def _get_rag_llm():
    """Get the LLM instance for RAG generation."""
    settings = await get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        openai_api_key=settings.openai_api_key
    )

# =============================================================================
# CUSTOMER MESSAGING TOOL (Employee Only)
# =============================================================================

# =============================================================================
# LEGACY FUNCTIONS - DEPRECATED (Replaced by revolutionary 3-field HITL system)
# =============================================================================
# The following functions were part of the old confirmation system and have been
# replaced by the revolutionary 3-field HITL architecture (hitl_phase, hitl_prompt, hitl_context)
# using dedicated HITL request tools with _handle_confirmation_approved(),
# _handle_confirmation_denied(), and _deliver_customer_message().
# 
# These functions are kept temporarily for test compatibility but should not be
# used in new code. They will be removed in a future cleanup.
# =============================================================================

# DEPRECATED: Use _deliver_customer_message() instead
# async def _deliver_message_via_chat(customer_info: dict, formatted_message: str, message_type: str, sender_employee_id: str) -> dict:
#     """DEPRECATED: Legacy message delivery function. Use _deliver_customer_message() instead."""

# DEPRECATED: Tracking is now handled within _deliver_customer_message()
# async def _track_message_delivery(customer_info: dict, delivery_result: dict, message_type: str, sender_employee_id: str) -> dict:
#     """DEPRECATED: Legacy message tracking function. Tracking is now built into _deliver_customer_message()."""


def _format_customer_data(customer_data: dict) -> dict:
    """
    Format customer data consistently across all lookup functions.
    
    Args:
        customer_data: Raw customer data from database
        
    Returns:
        Formatted customer dictionary
    """
    # Split name into first and last name for backward compatibility
    full_name = (customer_data.get("name") or "").strip()
    name_parts = full_name.split(maxsplit=1) if full_name else ["", ""]
    first_name = name_parts[0] if len(name_parts) > 0 else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""
    
    return {
        "customer_id": str(customer_data["id"]),
        "id": str(customer_data["id"]),
        "name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "email": customer_data.get("email") or "",
        "phone": customer_data.get("phone") or customer_data.get("mobile_number") or "",
        "mobile_number": customer_data.get("mobile_number") or "",
        "company": customer_data.get("company") or "",
        "is_for_business": customer_data.get("is_for_business") or False,
        "created_at": customer_data.get("created_at"),
        "updated_at": customer_data.get("updated_at")
    }



async def _list_active_customers(limit: int = 50) -> List[dict]:
    """
    List customers for reference (used in debugging and validation).
    
    Args:
        limit: Maximum number of customers to return
        
    Returns:
        List of customer dictionaries with basic info
    """
    try:
        engine = await _get_sql_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, name, email, phone, mobile_number
                    FROM customers 
                    ORDER BY name
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            
            customers = []
            for row in result:
                # Split name into first and last name for backward compatibility
                full_name = (row[1] or "").strip()
                name_parts = full_name.split(maxsplit=1) if full_name else ["", ""]
                first_name = name_parts[0] if len(name_parts) > 0 else ""
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                
                customer = {
                    "customer_id": str(row[0]),
                    "id": str(row[0]),
                    "name": full_name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": row[2] or "",
                    "phone": row[3] or row[4] or "",  # Use phone or mobile_number, whichever is available
                    "mobile_number": row[4] or "",
                    "display_name": full_name or "Unnamed Customer"
                }
                customers.append(customer)
            
            logger.debug(f"Found {len(customers)} customers")
            return customers

    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        return []



def _validate_message_content(message_content: str, message_type: str) -> dict:
    """
    Validate message content for basic requirements only.
    
    Simplified validation focusing on essential checks:
    - Empty message prevention
    - Security checks (email leakage prevention)
    
    Args:
        message_content: The message content to validate
        message_type: The type of message (kept for compatibility)
        
    Returns:
        Dict with validation results: {"valid": bool, "errors": list, "warnings": list}
    """
    errors = []
    warnings = []
    
    # Essential check #1: Prevent empty messages
    if not message_content or not message_content.strip():
        errors.append("Message content cannot be empty")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    content = message_content.strip()
    
    # Essential check #2: Security - prevent accidental email leakage
    if "@" in content and "email" not in content.lower():
        warnings.append("Message contains @ symbol - ensure you're not accidentally including internal email addresses")
    
    # That's it! Keep it simple.
    is_valid = len(errors) == 0
    
    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "character_count": len(content),
        "word_count": len(content.split())
    }


def _format_message_by_type(message_content: str, message_type: str, customer_info: dict) -> str:
    """
    Format message content based on message type and customer information.
    
    Args:
        message_content: Original message content
        message_type: Type of message (follow_up, information, promotional, support)
        customer_info: Customer information for personalization
        
    Returns:
        Formatted message with appropriate structure and tone
    """
    customer_name = f"{customer_info.get('first_name', '').strip()} {customer_info.get('last_name', '').strip()}".strip()
    if not customer_name:
        customer_name = "Valued Customer"
    
    # Message templates by type
    if message_type == "follow_up":
        formatted = f"""Dear {customer_name},

I hope this message finds you well. I'm following up on our recent interaction.

{message_content}

Please don't hesitate to reach out if you have any questions or need further assistance.

Best regards"""
    
    elif message_type == "information":
        formatted = f"""Dear {customer_name},

I wanted to share some important information with you:

{message_content}

If you have any questions about this information, please feel free to contact us.

Best regards"""
    
    elif message_type == "promotional":
        formatted = f"""Dear {customer_name},

We have an exciting opportunity that might interest you:

{message_content}

This offer is available for a limited time. Contact us to learn more!

Best regards"""
    
    elif message_type == "support":
        formatted = f"""Dear {customer_name},

Thank you for reaching out. I'm here to help with your inquiry:

{message_content}

Please let me know if you need any clarification or have additional questions.

Best regards"""
    
    else:
        # Default professional format
        formatted = f"""Dear {customer_name},

{message_content}

Please feel free to contact us if you have any questions.

Best regards"""
    
    return formatted

class TriggerCustomerMessageParams(BaseModel):
    """Parameters for triggering customer outreach messages."""
    customer_id: str = Field(..., description="Customer identifier: UUID, name, or email address")
    message_content: str = Field(..., description="Content of the message to send")
    message_type: str = Field(default="follow_up", description="Type of message: follow_up, information, promotional, support")

@tool(args_schema=TriggerCustomerMessageParams)
@traceable(name="trigger_customer_message")
async def trigger_customer_message(customer_id: str, message_content: str, message_type: str = "follow_up") -> str:
    """
    Prepare a customer message for human-in-the-loop confirmation (Employee Only).
    
    This tool validates inputs, prepares the message, and returns confirmation data
    for the agent node to populate in AgentState for state-driven HITL routing.
    
    Args:
        customer_id: The customer identifier (UUID, name, or email address) to message
        message_content: The content of the message to send
        message_type: Type of message (follow_up, information, promotional, support)
    
    Returns:
        Confirmation request data with CONFIRMATION_REQUIRED indicator for state-driven routing
    """
    try:
        # Check if user is an employee by verifying employee_id is available
        sender_employee_id = await get_current_employee_id()
        if not sender_employee_id:
            logger.warning("[CUSTOMER_MESSAGE] Non-employee user attempted to use customer messaging tool")
            return "I apologize, but customer messaging is only available to employees. Please contact your administrator if you need assistance."
        
        # Validate inputs
        if not customer_id or not customer_id.strip():
            return "Error: Customer identifier is required to send a message."
        
        # Validate message type
        valid_types = ["follow_up", "information", "promotional", "support"]
        if message_type not in valid_types:
            message_type = "follow_up"  # Default fallback
            
        # Simple validation check
        validation_result = _validate_message_content(message_content, message_type)
        if not validation_result["valid"]:
            # For our simplified validation, failures are usually empty messages
            # Return a simple error message instead of complex HITL flow
            error_msg = "❌ **Message Validation Failed:**\n"
            for error in validation_result["errors"]:
                error_msg += f"• {error}\n"
            if validation_result["warnings"]:
                error_msg += "\n⚠️ **Suggestions:**\n"
                for warning in validation_result["warnings"]:
                    error_msg += f"• {warning}\n"
            return error_msg.strip()
        
        logger.info(f"[CUSTOMER_MESSAGE] Employee {sender_employee_id} preparing message for customer {customer_id}")
        
        # Validate customer exists and get customer information
        customer_info = await _lookup_customer(customer_id)
            
        if not customer_info:
            # Customer not found - use HITL input request for clarification
            not_found_prompt = f"""❓ **Customer Not Found**

I couldn't find a customer matching "{customer_id}".

Could you provide any additional information to help me locate the right customer? 

For example:
- A different spelling of their name
- Their email address or phone number
- Company name (for business customers)
- Or any other details you remember

Just share whatever comes to mind - any detail could help me find them."""
            
            # Context for handling the input response
            context_data = {
                "tool": "trigger_customer_message", 
                "action": "customer_lookup_retry",
                "original_customer_id": customer_id,
                "message_content": message_content,
                "message_type": message_type,
                "sender_employee_id": sender_employee_id,
                "requested_by": get_current_user_id(),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[CUSTOMER_MESSAGE] Customer '{customer_id}' not found - requesting additional information via HITL")
            
            # REVOLUTIONARY: Use dedicated request_input() tool for customer lookup clarification
            return request_input(
                prompt=not_found_prompt,
                input_type="customer_identifier", 
                context=context_data,
                validation_hints=[
                    "Any additional detail helps - even partial information is useful",
                    "Don't worry if you only remember some details",
                    "I'll search based on whatever you can provide"
                ]
            )
        
        logger.info(f"[CUSTOMER_MESSAGE] Validated customer: {customer_info.get('name', 'Unknown')} ({customer_info.get('email', 'no-email')})")
        
        # Format the message with professional templates
        formatted_message = _format_message_by_type(message_content, message_type, customer_info)
        
        # Create confirmation prompt for the user
        customer_name = customer_info.get('name', 'Unknown Customer')
        customer_email = customer_info.get('email', 'no-email')
        message_type_display = message_type.replace('_', ' ').title()
        
        confirmation_prompt = f"""🔄 **Customer Message Confirmation**

**To:** {customer_name} ({customer_email})
**Type:** {message_type_display}
**Message:** {message_content}

**Formatted Preview:**
{formatted_message}

Do you want to send this message to the customer?"""
        
        # Prepare context data for post-confirmation processing
        context_data = {
            "tool": "trigger_customer_message",
            "customer_info": customer_info,
            "message_content": message_content,
            "formatted_message": formatted_message,
            "message_type": message_type,
            "validation_result": validation_result,
            "customer_id": customer_id,
            "sender_employee_id": sender_employee_id,
            "requested_by": get_current_user_id(),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[CUSTOMER_MESSAGE] Message prepared for HITL confirmation - Customer: {customer_name}")
        
        # REVOLUTIONARY: Use dedicated request_approval() tool for standardized HITL interaction
        return request_approval(
            prompt=confirmation_prompt,
            context=context_data,
            approve_text="send",
            deny_text="cancel"
        )

    except Exception as e:
        logger.error(f"Error in trigger_customer_message: {e}")
        return f"Sorry, I encountered an error while preparing your customer message request. Please try again or contact support if the issue persists."


# [ELIMINATED] gather_further_details and related functions removed due to redundancy
# These functions have been replaced by business-specific recursive collection tools
# that use the revolutionary 3-field HITL architecture (hitl_phase, hitl_prompt, hitl_context)
# and manage their own collection state with better context-aware validation.

# ============================================================================= 
# REVOLUTIONARY UNIVERSAL CONVERSATION ANALYSIS UTILITY (Task 9.2)
# =============================================================================

async def extract_fields_from_conversation(
    state: Dict[str, Any],
    field_definitions: Dict[str, str],
    tool_name: str = "unknown"
) -> Dict[str, str]:
    """
    REVOLUTIONARY: Universal LLM-powered function to extract already-provided information from conversation.
    
    This universal helper eliminates redundant questions by intelligently analyzing conversation
    context to identify information customers have already provided. Can be used by ANY tool
    that needs to collect information from users.
    
    ARCHITECTURE: Directly pulls from agent state (messages + conversation_summary) for uniform
    integration with agents' context window and memory management system.
    
    Key Benefits:
    - Eliminates frustrating redundant questions
    - Improves user experience dramatically  
    - Reusable across all collection tools
    - Uses fast/cheap models for cost effectiveness
    - Consistent LLM-powered analysis
    - Uniform with agents' memory management
    - Handles complex natural language expressions
    
    Args:
        state: Agent state containing messages and conversation_summary
        field_definitions: Dict mapping field names to descriptions
                          e.g., {"budget": "Budget range or maximum amount", 
                                "timeline": "When they need the vehicle"}
        tool_name: Name of calling tool for logging purposes
        
    Returns:
        Dictionary of extracted fields and their values from conversation
        
    Examples:
        State: {"messages": [...], "conversation_summary": "Customer interested in SUV..."}
        Fields: {"budget": "Budget range", "vehicle_type": "Type of vehicle", "primary_use": "How they'll use it"}
        Returns: {"budget": "under $50,000", "vehicle_type": "SUV", "primary_use": "daily commuting"}
    """
    try:
        # REVOLUTIONARY: Extract conversation context directly from agent state
        messages = state.get("messages", [])
        conversation_summary = state.get("conversation_summary", "")
        
        # Build conversation context from messages + summary
        conversation_parts = []
        
        # Add conversation summary if available (provides long-term context)
        if conversation_summary:
            conversation_parts.append(f"CONVERSATION SUMMARY: {conversation_summary}")
        
        # Add recent messages (provides immediate context)
        if messages:
            recent_messages = messages[-10:]  # Last 10 messages for context
            for msg in recent_messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    if msg.type == 'human':
                        conversation_parts.append(f"USER: {msg.content}")
                    elif msg.type == 'ai':
                        conversation_parts.append(f"ASSISTANT: {msg.content}")
        
        conversation_context = "\n".join(conversation_parts)
        
        if not conversation_context or len(conversation_context.strip()) < 10:
            logger.info(f"[CONVERSATION_EXTRACT] No meaningful context for {tool_name}")
            return {}
            
        if not field_definitions:
            logger.warning(f"[CONVERSATION_EXTRACT] No field definitions provided for {tool_name}")
            return {}
            
        logger.info(f"[CONVERSATION_EXTRACT] 🧠 Analyzing conversation for {tool_name} with {len(field_definitions)} fields")
        logger.info(f"[CONVERSATION_EXTRACT] Context sources: {len(messages)} messages, summary: {'yes' if conversation_summary else 'no'}")
        
        # REVOLUTIONARY: Use fast/cheap model for cost-effective analysis
        settings = await get_settings()
        # Use simple model (gpt-3.5-turbo) instead of expensive models for extraction
        model = getattr(settings, 'openai_simple_model', 'gpt-3.5-turbo')
        
        llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Low temperature for precise extraction
            api_key=settings.openai_api_key
        )
        
        logger.info(f"[CONVERSATION_EXTRACT] Using cost-effective model: {model}")
        
        # Build dynamic field descriptions for the prompt
        field_descriptions = "\n".join([
            f"- {field}: {description}" 
            for field, description in field_definitions.items()
        ])
        
        # Universal extraction prompt that works for any tool/fields
        extraction_prompt = ChatPromptTemplate.from_template("""
You are an expert at extracting specific information from customer conversations.

Analyze the following conversation and extract any information that relates to the specified fields.

FIELDS TO EXTRACT:
{field_descriptions}

CONVERSATION:
{conversation_context}

INSTRUCTIONS:
- Extract ONLY information that was clearly stated by the customer
- Use the customer's exact wording when possible
- If a field wasn't mentioned or implied clearly, don't include it
- Look for natural language expressions (e.g., "under $50k" for budget, "ASAP" for timeline)
- Be conservative - only extract what you're confident about

Return your response as a JSON object with only the fields that were clearly provided:
{{
  "field_name": "extracted value if clearly mentioned",
  "another_field": "another extracted value if clearly mentioned"
}}

If no clear information was found for any fields, return: {{}}
""")
        
        # Execute the extraction
        chain = extraction_prompt | llm | StrOutputParser()
        result = await chain.ainvoke({
            "conversation_context": conversation_context,
            "field_descriptions": field_descriptions
        })
        
        # Parse and validate the JSON response
        try:
            extracted_data = json.loads(result.strip())
            if isinstance(extracted_data, dict):
                # Filter out empty values and validate against field definitions
                filtered_data = {}
                for field, value in extracted_data.items():
                    if field in field_definitions and value and str(value).strip():
                        filtered_data[field] = str(value).strip()
                
                if filtered_data:
                    logger.info(f"[CONVERSATION_EXTRACT] ✅ Extracted for {tool_name}: {list(filtered_data.keys())}")
                    for field, value in filtered_data.items():
                        logger.info(f"[CONVERSATION_EXTRACT]   └─ {field}: '{value}'")
                else:
                    logger.info(f"[CONVERSATION_EXTRACT] ℹ️ No clear information found for {tool_name}")
                
                return filtered_data
            else:
                logger.warning(f"[CONVERSATION_EXTRACT] ⚠️ LLM returned non-dict for {tool_name}")
                return {}
                
        except json.JSONDecodeError as e:
            logger.warning(f"[CONVERSATION_EXTRACT] ⚠️ Failed to parse JSON for {tool_name}: {e}")
            logger.debug(f"[CONVERSATION_EXTRACT] Raw response: {result[:200]}...")
            return {}
            
    except Exception as e:
        logger.error(f"[CONVERSATION_EXTRACT] ❌ Error analyzing conversation for {tool_name}: {e}")
        return {}


# ============================================================================= 
# REVOLUTIONARY TOOL-MANAGED RECURSIVE COLLECTION IMPLEMENTATION (Task 9.1)
# =============================================================================

class CollectSalesRequirementsParams(BaseModel):
    """Parameters for collecting comprehensive sales requirements from customers."""
    customer_identifier: str = Field(..., description="Customer name, ID, email, or phone to identify the customer")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Previously collected requirements data")
    collection_mode: str = Field(default="tool_managed", description="Collection mode - always 'tool_managed' for this revolutionary approach")
    current_field: str = Field(default="", description="Current field being collected (used for recursive calls)")
    user_response: str = Field(default="", description="User's response to the current field request (used for recursive calls)")
    conversation_context: str = Field(default="", description="Recent conversation messages for intelligent pre-population (REVOLUTIONARY: Task 9.2)")


@tool(args_schema=CollectSalesRequirementsParams)
@traceable(name="collect_sales_requirements")
async def collect_sales_requirements(
    customer_identifier: str,
    collected_data: Dict[str, Any] = None,
    collection_mode: str = "tool_managed",
    current_field: str = "",
    user_response: str = "",
    conversation_context: str = ""
) -> str:
    """
    REVOLUTIONARY: Example tool-managed recursive collection for sales requirements.
    
    This tool demonstrates the new approach where tools manage their own collection 
    state, determine completion themselves, and get re-called by the agent with user responses.
    
    KEY REVOLUTIONARY FEATURES (Task 9.2):
    - Intelligent conversation analysis to pre-populate already-provided information
    - Avoids redundant questions by extracting data from conversation history
    - Only asks for genuinely missing information
    - Provides user feedback about pre-populated data
    
    Required Fields: budget, timeline, vehicle_type, primary_use, financing_preference
    
    Args:
        customer_identifier: Customer name, ID, email, or phone
        collected_data: Previously collected data (managed by tool)
        collection_mode: Always "tool_managed" for this approach
        current_field: Current field being collected (for recursive calls)
        user_response: User response (for recursive calls)
        conversation_context: Recent conversation messages for intelligent pre-population
        
    Returns:
        Either complete requirements data or HITL_REQUIRED for next field
        
    Examples:
        Initial call with conversation context:
        collect_sales_requirements("john.doe@email.com", conversation_context="I'm looking for an SUV under $50,000")
        -> Pre-populates budget and vehicle_type, only asks for remaining fields
        
        Recursive call:
        collect_sales_requirements("john.doe@email.com", collected_data={...}, current_field="timeline", user_response="within 2 weeks")
    """
    try:
        if collected_data is None:
            collected_data = {}
            
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Starting collection for customer: {customer_identifier}")
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Current field: '{current_field}', User response: '{user_response}'")
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Collected so far: {list(collected_data.keys())}")
        
        # REVOLUTIONARY: Intelligent conversation analysis for pre-population (Task 9.2)
        # Only analyze on first call (when no data collected yet)
        if not collected_data:
            logger.info("[COLLECT_SALES_REQUIREMENTS] 🧠 Analyzing conversation for already-provided information...")
            
            # Define what fields this tool needs
            field_definitions = {
                "budget": "Budget range or maximum amount (e.g., 'under $50,000', '$40k-60k', 'around 45000')",
                "timeline": "When they need the vehicle (e.g., 'within a month', 'by summer', 'ASAP', 'no rush')",
                "vehicle_type": "Type of vehicle (e.g., 'SUV', 'sedan', 'pickup truck', 'convertible')",
                "primary_use": "How they plan to use it (e.g., 'family trips', 'daily commuting', 'work hauling')",
                "financing_preference": "Payment method (e.g., 'cash', 'financing', 'lease', 'loan')"
            }
            
            # SIMPLIFIED: For now, get conversation context from the conversation_context parameter
            # TODO: In the future, this will be automatically extracted from agent state
            if conversation_context:
                # Build a simple state-like dict for the helper function
                mock_state = {
                    "messages": [],  # This would come from agent state
                    "conversation_summary": conversation_context  # Use provided context as summary
                }
                
                # Use the universal helper to extract already-provided information
                pre_populated_data = await extract_fields_from_conversation(
                    mock_state, 
                    field_definitions, 
                    "collect_sales_requirements"
                )
                
                if pre_populated_data:
                    collected_data.update(pre_populated_data)
                    logger.info(f"[COLLECT_SALES_REQUIREMENTS] ✅ Pre-populated from conversation: {list(pre_populated_data.keys())}")
            else:
                logger.info("[COLLECT_SALES_REQUIREMENTS] ℹ️ No conversation context provided - skipping pre-population")
        
        # REVOLUTIONARY: Handle user response for recursive collection
        if current_field and user_response:
            logger.info(f"[COLLECT_SALES_REQUIREMENTS] 🔄 Processing user response for field: {current_field}")
            collected_data[current_field] = user_response.strip()
        
        # Define required fields and check completion
        required_fields = {
            "budget": "Budget range for the purchase",
            "timeline": "When you need the vehicle", 
            "vehicle_type": "Type of vehicle you're interested in",
            "primary_use": "How you plan to use the vehicle",
            "financing_preference": "Preferred payment method"
        }
        
        # Find next missing field
        missing_fields = [field for field in required_fields.keys() if field not in collected_data]
        
        if not missing_fields:
            # REVOLUTIONARY: Collection complete with intelligent pre-population awareness
            logger.info("[COLLECT_SALES_REQUIREMENTS] 🎉 Collection COMPLETE - all requirements gathered")
            
            # Check if any data was pre-populated from conversation analysis
            pre_populated_count = len([f for f in required_fields.keys() if f in collected_data and not current_field])
            
            if pre_populated_count > 0 and conversation_context:
                acknowledgment = f"\n\n💡 I noticed you already mentioned {pre_populated_count} requirement{'s' if pre_populated_count > 1 else ''} in our conversation, so I didn't need to ask again!"
            else:
                acknowledgment = ""
            
            requirements_summary = f"""✅ **Sales Requirements Collected Successfully**

**Customer:** {customer_identifier}

**Requirements:**
• **Budget:** {collected_data['budget']}
• **Timeline:** {collected_data['timeline']}
• **Vehicle Type:** {collected_data['vehicle_type']}
• **Primary Use:** {collected_data['primary_use']}
• **Financing:** {collected_data['financing_preference']}{acknowledgment}

I now have all the information needed to provide you with personalized recommendations and pricing."""

            return requirements_summary
        
        # Request next missing field using dedicated HITL request tool
        next_field = missing_fields[0]
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Requesting field: {next_field} ({len(missing_fields)} remaining)")
        
        field_prompts = {
            "budget": "What's your budget range for this purchase? (e.g., '$50,000 - $70,000')",
            "timeline": "When do you need the vehicle? (e.g., 'within 2 weeks', 'by summer')",
            "vehicle_type": "What type of vehicle are you looking for? (e.g., 'SUV', 'sedan', 'pickup truck')",
            "primary_use": "How do you plan to use this vehicle? (e.g., 'daily commuting', 'family trips')",
            "financing_preference": "How would you prefer to pay? (e.g., 'cash', 'financing', 'lease')"
        }
        
        field_prompt = field_prompts.get(next_field, f"Please provide information about {next_field}")
        progress_info = f"({len(collected_data)}/{len(required_fields)} requirements collected)"
        
        return request_input(
            prompt=f"""📋 **Sales Requirements Collection** {progress_info}

{field_prompt}

This helps me find vehicles that perfectly match your needs.""",
            input_type="sales_requirement",
            context={
                "source_tool": "collect_sales_requirements",
                "collection_mode": "tool_managed",
                "customer_identifier": customer_identifier,
                "collected_data": collected_data,
                "current_field": next_field,
                "required_fields": required_fields,
                "missing_fields": missing_fields
            },
            validation_hints=["Please provide as much detail as you're comfortable sharing"]
        )

    except Exception as e:
        logger.error(f"[COLLECT_SALES_REQUIREMENTS] Error: {e}")  
        return f"Sorry, I encountered an error while collecting your requirements. Please try again or let me know if you need assistance."


async def _handle_confirmation_approved(context_data: dict) -> str:
    """
    Process approved customer message confirmations from the HITL system.
    
    This function is called when the HITL system determines that a confirmation
    has been approved. It handles the actual message delivery and provides
    structured feedback to the user.
    
    Args:
        context_data: Context data from the HITL request containing all message details
        
    Returns:
        String response with delivery results and feedback
        
    Raises:
        Exception: If message delivery fails
    """
    try:
        # Extract message details from context
        customer_info = context_data.get("customer_info", {})
        customer_name = customer_info.get("name", "Unknown Customer")
        customer_email = customer_info.get("email", "no-email")
        message_content = context_data.get("message_content", "")
        formatted_message = context_data.get("formatted_message", "")
        message_type = context_data.get("message_type", "follow_up")
        sender_employee_id = context_data.get("sender_employee_id")
        
        logger.info(f"[HANDLE_CONFIRMATION_APPROVED] Processing approved message for {customer_name}")
        
        # Attempt message delivery
        delivery_result = await _deliver_customer_message(
            customer_info=customer_info,
            message_content=message_content,
            formatted_message=formatted_message,
            message_type=message_type,
            sender_employee_id=sender_employee_id
        )
        
        if delivery_result.get("success"):
            # Success response with detailed feedback
            success_message = f"""✅ **Message Delivered Successfully!**

**To:** {customer_name} ({customer_email})
**Type:** {message_type.replace('_', ' ').title()}
**Status:** Delivered
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Message Sent:**
{message_content}

The customer will receive your message shortly."""
            
            logger.info(f"[HANDLE_CONFIRMATION_APPROVED] Message delivered successfully to {customer_name}")
            return success_message
            
        else:
            # Delivery failed
            error_details = delivery_result.get("error", "Unknown delivery error")
            
            failure_message = f"""❌ **Message Delivery Failed**

**To:** {customer_name} ({customer_email})
**Type:** {message_type.replace('_', ' ').title()}
**Status:** Failed
**Error:** {error_details}

**Troubleshooting:**
• Verify customer contact information is correct
• Check system connectivity
• Try again in a few moments

Please contact technical support if the issue persists."""
            
            logger.error(f"[HANDLE_CONFIRMATION_APPROVED] Message delivery failed for {customer_name}: {error_details}")
            return failure_message
    
    except Exception as e:
        error_msg = f"Error processing approved customer message: {str(e)}"
        logger.error(f"[HANDLE_CONFIRMATION_APPROVED] {error_msg}")
        
        # Return user-friendly error message
        return f"""❌ **Processing Error**

An error occurred while processing your approved message request.

**Error Details:** {str(e)}

Please try your request again or contact technical support if the issue continues."""


async def _deliver_customer_message(
    customer_info: dict,
    message_content: str,
    formatted_message: str,
    message_type: str,
    sender_employee_id: str
) -> dict:
    """
    Deliver a customer message through the appropriate channels.
    
    This function handles the actual delivery mechanism and can be extended
    for different channels (in-system, email, SMS, etc.).
    
    Args:
        customer_info: Customer information dictionary
        message_content: Original message content
        formatted_message: Professionally formatted message
        message_type: Type of message (follow_up, information, etc.)
        sender_employee_id: ID of the employee sending the message
        
    Returns:
        Dictionary with delivery results:
        - success: bool - Whether delivery was successful
        - method: str - Delivery method used
        - message_id: str - Unique message identifier (if successful)
        - error: str - Error message (if failed)
    """
    try:
        customer_name = customer_info.get("name", "Unknown")
        customer_id = customer_info.get("id") or customer_info.get("customer_id")
        
        logger.info(f"[DELIVER_CUSTOMER_MESSAGE] Attempting delivery to {customer_name} (ID: {customer_id})")
        
        # Create message record for tracking and customer viewing
        message_id = await _create_customer_message_record(
            customer_info=customer_info,
            message_content=formatted_message,
            message_type=message_type,
            sender_employee_id=sender_employee_id
        )
        
        if message_id:
            # TODO: Implement additional delivery channels
            # await _send_email_notification(customer_info, formatted_message)
            # await _send_sms_notification(customer_info, formatted_message)
            # await _send_push_notification(customer_info, formatted_message)
            
            logger.info(f"[DELIVER_CUSTOMER_MESSAGE] Message delivered successfully - ID: {message_id}")
            
            return {
                "success": True,
                "method": "in_system_message",
                "message_id": message_id,
                "error": None
            }
        else:
            error_msg = "Failed to create message record"
            logger.error(f"[DELIVER_CUSTOMER_MESSAGE] {error_msg}")
            
            return {
                "success": False,
                "method": None,
                "message_id": None,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"Message delivery error: {str(e)}"
        logger.error(f"[DELIVER_CUSTOMER_MESSAGE] {error_msg}")
        
        return {
            "success": False,
            "method": None,
            "message_id": None,
            "error": error_msg
        }


async def _create_customer_message_record(
    customer_info: dict,
    message_content: str,
    message_type: str,
    sender_employee_id: str
) -> Optional[str]:
    """
    Create a message record in the database for the customer to view.
    
    Args:
        customer_info: Customer information dictionary
        message_content: Message content to store
        message_type: Type of message
        sender_employee_id: ID of the sending employee
        
    Returns:
        Message ID if successful, None if failed
    """
    try:
        from core.database import db_client
        import uuid
        
        # Get customer UUID
        customer_id = customer_info.get("id") or customer_info.get("customer_id")
        if not customer_id:
            logger.error("[CREATE_MESSAGE_RECORD] No valid customer ID found")
            return None
        
        # Create or get user record for the customer
        user_id = await _get_or_create_customer_user(customer_id, customer_info)
        if not user_id:
            logger.error(f"[CREATE_MESSAGE_RECORD] Failed to get user for customer {customer_id}")
            return None
        
        # Create conversation record
        conversation_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        # Insert conversation
        conversation_data = {
            "id": conversation_id,
            "user_id": user_id,
            "title": f"Message from {message_type.replace('_', ' ').title()}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        conv_result = db_client.client.table("conversations").insert(conversation_data).execute()
        if not conv_result.data:
            logger.error("[CREATE_MESSAGE_RECORD] Failed to create conversation")
            return None
        
        # Insert message
        message_data = {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": message_content,
            "metadata": {
                "message_type": message_type,
                "sender_employee_id": sender_employee_id,
                "customer_id": customer_id,
                "delivery_timestamp": datetime.now().isoformat()
            },
            "created_at": datetime.now().isoformat()
        }
        
        msg_result = db_client.client.table("messages").insert(message_data).execute()
        if msg_result.data:
            logger.info(f"[CREATE_MESSAGE_RECORD] Created message record {message_id} for customer {customer_info.get('name')}")
            return message_id
        else:
            logger.error("[CREATE_MESSAGE_RECORD] Failed to create message record")
            return None
            
    except Exception as e:
        logger.error(f"[CREATE_MESSAGE_RECORD] Error creating message record: {str(e)}")
        return None


async def _get_or_create_customer_user(customer_id: str, customer_info: dict) -> Optional[str]:
    """
    Get or create a user record for a customer.
    
    Args:
        customer_id: Customer UUID from customers table
        customer_info: Customer information dictionary
        
    Returns:
        User ID (UUID) if successful, None otherwise
    """
    try:
        from core.database import db_client
        
        # Check for existing user
        existing_user = db_client.client.table("users").select("id").eq("customer_id", customer_id).execute()
        
        if existing_user.data:
            user_id = existing_user.data[0]["id"]
            logger.debug(f"[GET_OR_CREATE_USER] Found existing user {user_id} for customer {customer_id}")
            return user_id
        
        # Create new user record
        user_data = {
            "email": customer_info.get("email", ""),
            "display_name": customer_info.get("name", "Unknown Customer"),
            "user_type": "customer",
            "customer_id": customer_id,
            "is_active": True,
            "is_verified": False,
            "metadata": {
                "customer_info": {
                    "phone": customer_info.get("phone", ""),
                    "company": customer_info.get("company", ""),
                    "is_for_business": customer_info.get("is_for_business", False)
                }
            }
        }
        
        user_result = db_client.client.table("users").insert(user_data).execute()
        
        if user_result.data:
            user_id = user_result.data[0]["id"]
            logger.info(f"[GET_OR_CREATE_USER] Created user {user_id} for customer {customer_id}")
            return user_id
        else:
            logger.error(f"[GET_OR_CREATE_USER] Failed to create user for customer {customer_id}")
            return None
        
    except Exception as e:
        logger.error(f"[GET_OR_CREATE_USER] Error: {str(e)}")
        return None


async def _handle_confirmation_denied(context_data: dict) -> str:
    """
    Process denied customer message confirmations from the HITL system.
    
    This function is called when the HITL system determines that a confirmation
    has been denied/cancelled. It provides user-friendly feedback without
    performing any message delivery.
    
    Args:
        context_data: Context data from the HITL request containing message details
        
    Returns:
        String response with cancellation confirmation and details
    """
    try:
        # Extract message details from context for feedback
        customer_info = context_data.get("customer_info", {})
        customer_name = customer_info.get("name", "Unknown Customer")
        customer_email = customer_info.get("email", "no-email")
        message_content = context_data.get("message_content", "")
        message_type = context_data.get("message_type", "follow_up")
        sender_employee_id = context_data.get("sender_employee_id")
        
        logger.info(f"[HANDLE_CONFIRMATION_DENIED] Processing denied message for {customer_name} by employee {sender_employee_id}")
        
        # Create cancellation response with details
        cancellation_message = f"""🚫 **Message Cancelled**

**To:** {customer_name} ({customer_email})
**Type:** {message_type.replace('_', ' ').title()}
**Status:** Cancelled by User
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Cancelled Message:**
{message_content}

No message was sent to the customer. You can create a new message request anytime."""
        
        logger.info(f"[HANDLE_CONFIRMATION_DENIED] Message cancelled for {customer_name} - no delivery performed")
        return cancellation_message
    
    except Exception as e:
        error_msg = f"Error processing denied customer message: {str(e)}"
        logger.error(f"[HANDLE_CONFIRMATION_DENIED] {error_msg}")
        
        # Return user-friendly error message
        return f"""🚫 **Message Cancellation Confirmed**

Your customer message request has been cancelled as requested.

If you encounter any issues or need to send a message later, please try again or contact technical support."""


async def _handle_input_received(context_data: dict, user_input: str) -> str:
    """
    Process customer identifier inputs from HITL system and retry customer lookup.
    
    This function is called when the HITL system receives user input in response
    to a customer-not-found scenario. It attempts to find the customer using the
    additional information provided and either proceeds with message confirmation
    or requests more details.
    
    Args:
        context_data: Context data from the HITL request containing original request details
        user_input: Additional customer information provided by the user
        
    Returns:
        String response - either new confirmation request or helpful feedback
    """
    try:
        # Extract original request details from context
        original_customer_id = context_data.get("original_customer_id", "")
        message_content = context_data.get("message_content", "")
        message_type = context_data.get("message_type", "follow_up")
        sender_employee_id = context_data.get("sender_employee_id")
        
        logger.info(f"[HANDLE_INPUT_RECEIVED] Processing customer lookup retry with input: '{user_input[:50]}...' for employee {sender_employee_id}")
        
        # Validate user input
        if not user_input or not user_input.strip():
            return """❓ **No Additional Information Provided**

Please provide some additional customer details to help me locate them:
- Customer name variations or full name
- Email address or phone number  
- Company name (for business customers)
- Any other identifying information

Try again with more specific details."""
        
        # Attempt customer lookup with the new information
        customer_info = await _lookup_customer(user_input.strip())
        
        if customer_info:
            # Customer found! Proceed with the original message confirmation flow
            customer_name = customer_info.get('name', 'Unknown Customer')
            customer_email = customer_info.get('email', 'no-email')
            
            logger.info(f"[HANDLE_INPUT_RECEIVED] Customer found: {customer_name} - proceeding with message confirmation")
            
            # Format the message with professional templates
            formatted_message = _format_message_by_type(message_content, message_type, customer_info)
            
            # Create confirmation prompt for the user
            message_type_display = message_type.replace('_', ' ').title()
            
            confirmation_prompt = f"""✅ **Customer Found!**

I found the customer using your additional details.

🔄 **Customer Message Confirmation**

**To:** {customer_name} ({customer_email})
**Type:** {message_type_display}
**Message:** {message_content}

**Formatted Preview:**
{formatted_message}

Do you want to send this message to the customer?"""
            
            # Prepare context data for confirmation processing
            context_data_new = {
                "tool": "trigger_customer_message",
                "customer_info": customer_info,
                "message_content": message_content,
                "formatted_message": formatted_message,
                "message_type": message_type,
                "customer_id": customer_info.get("id") or customer_info.get("customer_id"),
                "sender_employee_id": sender_employee_id,
                "requested_by": context_data.get("requested_by"),
                "timestamp": datetime.now().isoformat()
            }
            
            # REVOLUTIONARY: Return new confirmation request using dedicated request_approval() tool
            return request_approval(
                prompt=confirmation_prompt,
                context=context_data_new,
                approve_text="send",
                deny_text="cancel"
            )
        
        else:
            # Customer still not found - provide helpful feedback with suggestions
            search_suggestions = _generate_search_suggestions(user_input, original_customer_id)
            
            not_found_response = f"""❓ **Customer Still Not Found**

I searched for a customer using:
- Original identifier: "{original_customer_id}"
- Additional details: "{user_input}"

But couldn't find a match in our system.

**Possible next steps:**
{search_suggestions}

**Alternative options:**
• Double-check the customer information for accuracy
• Search in the CRM system directly if available
• Contact the customer to confirm their details
• Create a new customer record if they're new to our system

Would you like to try again with different information, or would you prefer to handle this differently?"""
            
            logger.info(f"[HANDLE_INPUT_RECEIVED] Customer still not found after retry - providing suggestions")
            return not_found_response
    
    except Exception as e:
        error_msg = f"Error processing customer identifier input: {str(e)}"
        logger.error(f"[HANDLE_INPUT_RECEIVED] {error_msg}")
        
        # Return user-friendly error message
        return f"""❌ **Search Error**

An error occurred while searching for the customer with your additional information.

**Error Details:** {str(e)}

Please try again with the customer information, or contact technical support if the issue persists."""


async def _lookup_customer(customer_identifier: str) -> Optional[dict]:
    """
    Enhanced customer lookup using multiple search strategies.
    
    Attempts to find customers using various fields and fuzzy matching techniques.
    
    Args:
        customer_identifier: The customer UUID, name, email, or other identifier to look up
        
    Returns:
        Customer info dictionary if found, None otherwise
    """
    try:
        from core.database import db_client
        
        # Clean and prepare search input
        search_terms = customer_identifier.lower().strip()
        
        logger.info(f"[CUSTOMER_LOOKUP] Searching with: '{search_terms}'")
        
        # Ensure client is initialized
        client = db_client.client
        # Strategy 1: Check if input looks like a UUID first
        if len(search_terms.replace("-", "")) == 32 and search_terms.count("-") == 4:
            try:
                result = client.table("customers").select("*").eq("id", customer_identifier).execute()
                if result.data:
                    customer_data = result.data[0]
                    customer = _format_customer_data(customer_data)
                    logger.info(f"[CUSTOMER_LOOKUP] Found customer via UUID: {customer.get('name')}")
                    return customer
            except Exception as e:
                logger.debug(f"[CUSTOMER_LOOKUP] UUID search failed: {e}")
        
        # Strategy 2: Direct field matches with fuzzy matching
        search_strategies = [
            ("name", f"%{search_terms}%"),
            ("email", f"%{search_terms}%"), 
            ("phone", f"%{search_terms.replace(' ', '').replace('-', '')}%"),
            ("company", f"%{search_terms}%")
        ]
        
        for field, pattern in search_strategies:
            try:
                result = client.table("customers").select("*").filter(field, "ilike", pattern).limit(1).execute()
                if result.data:
                    customer_data = result.data[0]
                    customer = _format_customer_data(customer_data)
                    logger.info(f"[CUSTOMER_LOOKUP] Found customer via {field}: {customer.get('name')}")
                    return customer
            except Exception as e:
                logger.debug(f"[CUSTOMER_LOOKUP] Search by {field} failed: {e}")
                continue
        
        # Strategy 3: Multi-word search (split terms and search each)
        if " " in search_terms:
            words = search_terms.split()
            for word in words:
                if len(word) >= 3:  # Only search meaningful words
                    try:
                        result = client.table("customers").select("*").filter("name", "ilike", f"%{word}%").limit(1).execute()
                        if result.data:
                            customer_data = result.data[0]
                            customer = _format_customer_data(customer_data)
                            logger.info(f"[CUSTOMER_LOOKUP] Found customer via word '{word}': {customer.get('name')}")
                            return customer
                    except Exception as e:
                        logger.debug(f"[CUSTOMER_LOOKUP] Word search for '{word}' failed: {e}")
                        continue
        
        # Strategy 4: Email domain search (if input looks like email)
        if "@" in search_terms:
            domain = search_terms.split("@")[-1] if "@" in search_terms else ""
            if domain:
                try:
                    result = client.table("customers").select("*").filter("email", "ilike", f"%@{domain}").limit(1).execute()
                    if result.data:
                        customer_data = result.data[0]
                        customer = _format_customer_data(customer_data)
                        logger.info(f"[CUSTOMER_LOOKUP] Found customer via email domain '{domain}': {customer.get('name')}")
                        return customer
                except Exception as e:
                    logger.debug(f"[CUSTOMER_LOOKUP] Domain search failed: {e}")
        
        logger.info(f"[CUSTOMER_LOOKUP] No customer found with any strategy")
        return None
        
    except Exception as e:
        logger.error(f"[CUSTOMER_LOOKUP] Error in customer lookup: {e}")
        return None


def _generate_search_suggestions(user_input: str, original_id: str) -> str:
    """
    Generate helpful search suggestions based on the failed search attempts.
    
    Args:
        user_input: The user input that failed to find a customer
        original_id: The original identifier that failed
        
    Returns:
        Formatted string with search suggestions
    """
    suggestions = []
    
    # Analyze input to provide targeted suggestions
    input_lower = user_input.lower()
    
    if "@" in user_input:
        suggestions.append("• Try just the name part of the email address")
        suggestions.append("• Check if the email domain is correct")
    
    if any(char.isdigit() for char in user_input):
        suggestions.append("• If that's a phone number, try without formatting (spaces, dashes)")
        suggestions.append("• Try searching by name instead of number")
    
    if " " in user_input:
        suggestions.append("• Try just the first or last name separately")
        suggestions.append("• Check for alternative spellings or nicknames")
    
    if len(user_input) < 3:
        suggestions.append("• Provide more details - longer names or additional information")
    
    # General suggestions
    suggestions.extend([
        "• Try the customer's company name if they're a business customer",
        "• Check for recent spelling variations or name changes",
        "• Look for partial matches in your CRM system directly"
    ])
    
    return "\n".join(suggestions)


# DEPRECATED: Use _handle_confirmation_approved() and _handle_confirmation_denied() instead
# async def _handle_customer_message_confirmation(hitl_context: dict, human_response: str) -> str:
#     """DEPRECATED: Legacy confirmation handler. Use _handle_confirmation_approved() and _handle_confirmation_denied() instead."""


# =============================================================================
# SIMPLIFIED RAG TOOLS
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

    Args:
        question: The user's question to answer
        top_k: Number of documents to retrieve (default: 5)

    Returns:
        Generated response based on retrieved documents
    """
    try:
        # Get retriever
        retriever = await _ensure_retriever()

        # Retrieve documents
        documents = await retriever.retrieve(
            query=question,
            top_k=top_k,
            threshold=0.7
        )

        # Handle no documents found
        if not documents:
            return f"I couldn't find any relevant documents to answer your question: '{question}'. You might want to try asking about our business data instead, or check if relevant documents have been uploaded to the system."

        # Format context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}: {doc.get('source', 'Unknown')}\nContent: {doc.get('content', 'No content')}\n")

        context = "\n".join(context_parts)

        # Generate response using LLM
        llm = await _get_rag_llm()

        prompt_template = """You are a helpful assistant. Answer the user's question based on the provided context from company documents.

Context from documents:
{context}

User Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain relevant information, say so clearly
- Provide specific details and examples from the context when available
- Be conversational and helpful
- If referencing sources, mention them naturally

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        response = await chain.ainvoke({
            "context": context,
            "question": question
        })

        # Add source information
        sources = [doc.get('source', 'Unknown') for doc in documents]
        response += f"\n\n**Sources:** {', '.join(sources)}"

        return response

    except Exception as e:
        logger.error(f"Error in simple_rag: {e}")
        return f"I encountered an issue while searching for that information. Please try rephrasing your question or ask for help."

# =============================================================================
# TOOL REGISTRY (Simplified)
# =============================================================================

def get_simple_sql_tools():
    """Get simplified SQL tools following LangChain best practices."""
    return [simple_query_crm_data]


def get_simple_rag_tools():
    """Get simplified RAG tools."""
    return [simple_rag]


def get_all_tools():
    """Get all available tools for the RAG agent."""
    return [
        simple_query_crm_data,
        simple_rag,
        trigger_customer_message,
        collect_sales_requirements,  # REVOLUTIONARY: Tool-managed recursive collection with 3-field HITL architecture (Task 9.1)
        # gather_further_details ELIMINATED - replaced by business-specific tools using revolutionary 3-field approach
    ]


def get_tools_for_user_type(user_type: str = "employee"):
    """Get tools filtered by user type for access control."""
    if user_type == "customer":
        # Customers get full RAG access but restricted CRM access
        return [
            simple_query_crm_data,  # Will be internally filtered for table access
            simple_rag,  # Full access
            collect_sales_requirements,  # REVOLUTIONARY: Tool-managed recursive collection with 3-field HITL architecture
            # gather_further_details ELIMINATED - replaced by business-specific tools using revolutionary 3-field approach
        ]
    elif user_type in ["employee", "admin"]:
        # Employees get full access to all tools including customer messaging
        return [
            simple_query_crm_data,  # Full access
            simple_rag,  # Full access  
            trigger_customer_message,  # Employee only - customer outreach tool
            collect_sales_requirements,  # REVOLUTIONARY: Tool-managed recursive collection with 3-field HITL architecture
            # gather_further_details ELIMINATED - replaced by business-specific tools using revolutionary 3-field approach
        ]
    else:
        # Unknown users get no tools
        return []


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]
