"""
Tools for RAG agents - Simplified and consolidated approach.

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
from typing import Optional, List
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
from agents.hitl import HITLRequest
import re

# NOTE: LangGraph interrupt functionality is now handled by the centralized HITL system in hitl.py
# No direct interrupt handling needed in individual tools - they use HITLRequest format instead



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
    Generate SQL query using minimal context + optional tools approach.
    Follows LangChain best practices - start minimal, let LLM request more via tools.
    """
    try:
        # Get LLM with tool calling capability
        llm = await _get_appropriate_llm(question)
        
        # Make tools available for LLM to use when needed
        tools = [get_detailed_schema, get_recent_conversation_context]
        llm_with_tools = llm.bind_tools(tools)

        # Simple, minimal prompt templates
        if user_type == "customer":
            template = """You are a PostgreSQL expert. Create a SELECT query for: {question}

{minimal_schema}

CUSTOMER RESTRICTIONS: Only query vehicles and pricing tables.

Available tools (use if needed):
- get_detailed_schema(table_names) - Get full schema for specific tables
- get_recent_conversation_context() - Get context if question references previous chat

Guidelines:
- SELECT only, LIMIT 5
- Use exact column names from schema above
- If you need more schema details, use get_detailed_schema tool

Question: {question}
SQL Query:"""
        else:
            template = """You are a PostgreSQL expert. Create a SELECT query for: {question}

{minimal_schema}

Available tools (use if needed):
- get_detailed_schema(table_names) - Get full schema for specific tables  
- get_recent_conversation_context() - Get context if question references previous chat

Guidelines:
- SELECT only, LIMIT 5
- Use proper JOIN conditions when needed
- For employee queries, filter by opportunity_salesperson_ae_id when relevant
- If you need more info, use the tools

Question: {question}  
SQL Query:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Simple chain with minimal context
        chain = (
            {
                "question": RunnablePassthrough(),
                "minimal_schema": lambda _: get_minimal_schema_info(user_type),
            }
            | prompt
            | llm_with_tools
            | StrOutputParser()
        )

        result = await chain.ainvoke(question)
        
        # Clean up SQL
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

def get_minimal_schema_info(user_type: str) -> str:
    """Return minimal schema - just table names and key columns to reduce context size."""
    
    if user_type == "customer":
        return """Available Tables:
• vehicles: id, brand, model, year, type, color, is_available, stock_quantity
• pricing: id, vehicle_id, base_price, final_price, is_active
Note: Price info is in pricing table, linked via vehicle_id"""
    
    # Employee/admin get minimal table info
    return """Available Tables:
• customers: id, name, email, phone, status, created_at
• opportunities: id, customer_id, title, status, amount, opportunity_salesperson_ae_id
• employees: id, name, email, department, role_title
• vehicles: id, brand, model, year, type, color, is_available, stock_quantity
• pricing: id, vehicle_id, base_price, final_price, is_active
• activities: id, opportunity_id, activity_type, activity_date, notes
• branches: id, name, location, manager_id
• transactions: id, opportunity_id, amount, transaction_date, status
Note: Price info is in pricing table, linked to vehicles via vehicle_id"""

@tool
async def get_detailed_schema(table_names: str) -> str:
    """Get detailed schema for specific tables when LLM needs more info.
    
    Args:
        table_names: Comma-separated list of table names (e.g. "customers,opportunities")
    """
    try:
        db = await _get_sql_database()
        tables_list = [name.strip() for name in table_names.split(',')]
        return db.get_table_info(tables_list)
    except Exception as e:
        return f"Error getting schema for {table_names}: {e}"

@tool  
async def get_recent_conversation_context() -> str:
    """Get recent conversation context when question references previous discussion."""
    try:
        conversation_id = get_current_conversation_id()
        if not conversation_id:
            return "No conversation context available"
        return await _get_conversation_context(conversation_id, limit=3)
    except Exception as e:
        return f"Error getting conversation context: {e}"

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
# LEGACY FUNCTIONS - DEPRECATED (Replaced by new HITL system)
# =============================================================================
# The following functions were part of the old confirmation system and have been
# replaced by the new HITLRequest-based system with _handle_confirmation_approved(),
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
            
            # Use HITLRequest.input_request() for customer lookup clarification
            return HITLRequest.input_request(
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
        
        # Use HITLRequest.confirmation() for standardized HITL interaction
        return HITLRequest.confirmation(
            prompt=confirmation_prompt,
            context=context_data,
            approve_text="send",
            deny_text="cancel"
        )

    except Exception as e:
        logger.error(f"Error in trigger_customer_message: {e}")
        return f"Sorry, I encountered an error while preparing your customer message request. Please try again or contact support if the issue persists."


class GatherFurtherDetailsParams(BaseModel):
    """Parameters for gathering additional information from the user."""
    information_needed: str = Field(..., description="Description of what information is needed from the user")
    context: str = Field(default="", description="Context about why this information is needed")
    input_type: str = Field(default="information", description="Type of input being requested (information, details, clarification, etc.)")
    validation_hints: List[str] = Field(default_factory=list, description="Optional hints to help the user provide the right information")


@tool(args_schema=GatherFurtherDetailsParams)
@traceable(name="gather_further_details")
async def gather_further_details(
    information_needed: str, 
    context: str = "", 
    input_type: str = "information",
    validation_hints: List[str] = None
) -> str:
    """
    Request additional information from the user through a HITL interaction.
    
    This tool allows agents to gather specific details, clarifications, or information
    from users when the current context is insufficient to proceed. It uses the
    standardized HITL input request format for consistency.
    
    Args:
        information_needed: Clear description of what information is needed
        context: Optional context explaining why this information is required
        input_type: Type of input being requested (information, details, clarification, etc.)
        validation_hints: Optional list of hints to help the user provide the right information
        
    Returns:
        HITL input request for the user to provide the needed information
        
    Examples:
        - gather_further_details("the customer's preferred contact method", "to send them updates", "contact_preference")
        - gather_further_details("more details about the issue", "to better assist you", "problem_description") 
        - gather_further_details("the specific vehicle model you're interested in", "to show you accurate pricing and availability", "vehicle_preference")
    """
    try:
        # Validate required input
        if not information_needed or not information_needed.strip():
            return "Error: Please specify what information is needed from the user."
        
        # Check user type for appropriate language
        user_type = get_current_user_type()
        user_id = get_current_user_id()
        
        logger.info(f"[GATHER_FURTHER_DETAILS] Requesting information from {user_type} user: {information_needed}")
        
        # Create user-friendly prompt
        if context:
            prompt = f"""ℹ️ **Additional Information Needed**

I need {information_needed} {context}.

Please provide this information so I can better assist you."""
        else:
            prompt = f"""ℹ️ **Additional Information Needed**

Could you please provide {information_needed}?

This will help me give you the most accurate and helpful response."""
        
        # Prepare context data for handling the response
        context_data = {
            "tool": "gather_further_details",
            "information_needed": information_needed,
            "original_context": context,
            "input_type": input_type,
            "requested_by": user_id,
            "user_type": user_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Default validation hints if none provided
        if validation_hints is None:
            validation_hints = [
                "Please be as specific as possible",
                "Any additional details you can provide will be helpful"
            ]
        
        # Add context-specific hints
        if user_type == "customer":
            validation_hints.append("Don't worry if you're not sure about technical details - just describe it in your own words")
        elif user_type in ["employee", "admin"]:
            validation_hints.append("Include any relevant IDs, names, or system references if applicable")
        
        logger.info(f"[GATHER_FURTHER_DETAILS] Created HITL input request for: {information_needed}")
        
        # Use HITLRequest.input_request() for standardized interaction
        return HITLRequest.input_request(
            prompt=prompt,
            input_type=input_type,
            context=context_data,
            validation_hints=validation_hints
        )
        
    except Exception as e:
        logger.error(f"Error in gather_further_details: {e}")
        return f"Sorry, I encountered an error while requesting additional information. Please try rephrasing your request or contact support if the issue persists."


async def _handle_information_gathered(context_data: dict, user_input: str) -> str:
    """
    Process information gathered from the user via gather_further_details tool.
    
    This function handles the response when users provide additional information
    through the gather_further_details HITL interaction.
    
    Args:
        context_data: Context data from the original gather_further_details request
        user_input: The information provided by the user
        
    Returns:
        Confirmation message with the gathered information
    """
    try:
        information_needed = context_data.get("information_needed", "information")
        original_context = context_data.get("original_context", "")
        input_type = context_data.get("input_type", "information")
        user_type = context_data.get("user_type", "unknown")
        
        logger.info(f"[HANDLE_INFORMATION_GATHERED] Processing gathered {input_type} from {user_type} user")
        
        # Validate input
        if not user_input or not user_input.strip():
            return f"""❓ **No Information Provided**

I still need {information_needed} to help you properly.

Please provide this information, or let me know if you'd like to approach this differently."""
        
        # Create confirmation response
        response_message = f"""✅ **Information Received**

Thank you for providing {information_needed}.

**You provided:** {user_input.strip()}

I now have the information I need to better assist you. Let me help you with that."""
        
        logger.info(f"[HANDLE_INFORMATION_GATHERED] Successfully processed user input for: {information_needed}")
        return response_message

    except Exception as e:
        error_msg = f"Error processing gathered information: {str(e)}"
        logger.error(f"[HANDLE_INFORMATION_GATHERED] {error_msg}")
        
        return f"""✅ **Information Received**

Thank you for providing the additional information. I'll use this to help you better.

If you have any other questions or need further assistance, please let me know."""


async def _handle_confirmation_approved(context_data: dict) -> str:
    """
    Process approved customer message confirmations from the HITL system.
    
    This function is called when the HITL system determines that a confirmation
    has been approved. It handles the actual message delivery and provides
    structured feedback to the user.
    
    Args:
        context_data: Context data from the HITLRequest containing all message details
        
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
        context_data: Context data from the HITLRequest containing message details
        
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
        context_data: Context data from the HITLRequest containing original request details
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
            
            # Return new confirmation request using HITLRequest
            return HITLRequest.confirmation(
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
# async def _handle_customer_message_confirmation(confirmation_data: dict, human_response: str) -> str:
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
        gather_further_details,
        get_detailed_schema,
        get_recent_conversation_context
    ]


def get_tools_for_user_type(user_type: str = "employee"):
    """Get tools filtered by user type for access control."""
    if user_type == "customer":
        # Customers get full RAG access but restricted CRM access
        return [
            simple_query_crm_data,  # Will be internally filtered for table access
            simple_rag,  # Full access
            gather_further_details,  # General information gathering
            get_detailed_schema  # Can get vehicle schema details
        ]
    elif user_type in ["employee", "admin"]:
        # Employees get full access to all tools including customer messaging
        return [
            simple_query_crm_data,  # Full access
            simple_rag,  # Full access  
            trigger_customer_message,  # Employee only - customer outreach tool
            gather_further_details,  # General information gathering
            get_detailed_schema,  # Can get detailed schema
            get_recent_conversation_context  # Can access conversation history
        ]
    else:
        # Unknown users get no tools
        return []


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]
