"""
Tools for Sales Copilot

This module provides:
1. User context management (consolidated from user_context.py)
2. Simplified SQL tools following LangChain best practices
3. Modern LCEL-based RAG tools
4. Natural language capabilities over hard-coded logic

Following the principle: Use LLM intelligence rather than keyword matching or complex logic.
"""

import asyncio
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

# Optional imports for PDF generation (may fail in test environments)
try:
    from core.pdf_generator import generate_quotation_pdf
    from core.storage import upload_quotation_pdf, create_signed_quotation_url
    PDF_GENERATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PDF generation not available: {e}")
    PDF_GENERATION_AVAILABLE = False
    # Provide dummy functions for testing
    def generate_quotation_pdf(*args, **kwargs):
        return b"dummy_pdf_content"
    def upload_quotation_pdf(*args, **kwargs):
        return "dummy_path"
    def create_signed_quotation_url(*args, **kwargs):
        return "https://dummy.url/quotation.pdf"

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
    """Parameters for CRM database information lookup and analysis (NOT for quotation generation)."""
    question: str = Field(..., description="Natural language question about CRM data: customer details, vehicle inventory, sales analytics, employee info, etc. (NOT for creating quotations)")
    time_period: Optional[str] = Field(None, description="Optional time period filter for data analysis: 'last 30 days', 'this quarter', '2024', etc.")

@tool(args_schema=SimpleCRMQueryParams)
@traceable(name="simple_query_crm_data")
async def simple_query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database for information lookup and analysis (NOT for generating quotations).
    
    **Use this tool for:**
    - Looking up customer information and contact details
    - Searching vehicle inventory and specifications  
    - Analyzing sales data, performance metrics, and trends
    - Checking opportunity status and pipeline information
    - Finding employee information and branch details
    - General CRM data exploration and reporting
    
    **Do NOT use this tool for:**
    - Creating customer quotations (use generate_quotation instead)
    - Generating PDF documents or official quotes
    - Processing quotation approvals or HITL workflows

    This tool provides read-only database access with user type-based security:
    - Employees: Full CRM database access
    - Customers: Limited to vehicle and pricing information only

    Args:
        question: Natural language question about CRM data (e.g., "Show me Toyota vehicles under 1.5M")
        time_period: Optional time period filter (e.g., "last 30 days", "this quarter")

    Returns:
        Natural language response with requested information
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
        
        # DEBUG: Log context during tool execution
        logger.info(f"🔍 [TRIGGER_CUSTOMER_MESSAGE_CONTEXT] customer_id param: {customer_id}")
        logger.info(f"🔍 [TRIGGER_CUSTOMER_MESSAGE_CONTEXT] sender_employee_id: {sender_employee_id}")
        logger.info(f"🔍 [TRIGGER_CUSTOMER_MESSAGE_CONTEXT] current context: {get_user_context()}")
        
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
        if not customer_identifier or not customer_identifier.strip():
            logger.info("[CUSTOMER_LOOKUP] Empty, None, or whitespace-only customer_identifier provided")
            return None
        search_terms = customer_identifier.lower().strip()
        
        logger.info(f"[CUSTOMER_LOOKUP] Searching with: '{search_terms}'")
        
        # DEBUG: Test database connection during agent execution
        logger.info(f"🔍 [DATABASE_DEBUG] db_client type: {type(db_client)}")
        logger.info(f"🔍 [DATABASE_DEBUG] db_client._client before init: {type(db_client._client)}")
        
        # Ensure client is initialized
        client = db_client.client
        
        # DEBUG: Test basic connectivity with async wrapper
        logger.info(f"🔍 [DATABASE_DEBUG] client after init: {type(client)}")
        try:
            test_query = await asyncio.to_thread(lambda: client.table("customers").select("count").execute())
            logger.info(f"🔍 [DATABASE_DEBUG] Basic connectivity test successful, customer table accessible")
        except Exception as e:
            logger.error(f"🔍 [DATABASE_DEBUG] Basic connectivity test FAILED: {e}")
        # Strategy 1: Check if input looks like a UUID first
        if len(search_terms.replace("-", "")) == 32 and search_terms.count("-") == 4:
            try:
                result = await asyncio.to_thread(lambda: client.table("customers").select("*").eq("id", customer_identifier).execute())
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
                logger.info(f"🔍 [DATABASE_DEBUG] Searching {field} with pattern: {pattern}")
                # Wrap synchronous database call with asyncio.to_thread for LangGraph compatibility
                result = await asyncio.to_thread(
                    lambda: client.table("customers").select("*").filter(field, "ilike", pattern).limit(1).execute()
                )
                logger.info(f"🔍 [DATABASE_DEBUG] Query result: {len(result.data)} records found")
                if result.data:
                    logger.info(f"🔍 [DATABASE_DEBUG] Found customer data: {result.data[0]}")
                    customer_data = result.data[0]
                    customer = _format_customer_data(customer_data)
                    logger.info(f"[CUSTOMER_LOOKUP] Found customer via {field}: {customer.get('name')}")
                    return customer
                else:
                    logger.info(f"🔍 [DATABASE_DEBUG] No results for {field} search with pattern {pattern}")
            except Exception as e:
                logger.error(f"🔍 [DATABASE_DEBUG] Search by {field} failed: {e}")
                logger.debug(f"[CUSTOMER_LOOKUP] Search by {field} failed: {e}")
                continue
        
        # Strategy 3: Multi-word search (split terms and search each)
        if " " in search_terms:
            words = search_terms.split()
            for word in words:
                if len(word) >= 3:  # Only search meaningful words
                    try:
                        result = await asyncio.to_thread(
                            lambda: client.table("customers").select("*").filter("name", "ilike", f"%{word}%").limit(1).execute()
                        )
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
                    result = await asyncio.to_thread(
                        lambda: client.table("customers").select("*").filter("email", "ilike", f"%@{domain}").limit(1).execute()
                    )
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
        error_msg = str(e)
        logger.error(f"[CUSTOMER_LOOKUP] Error in customer lookup: {e}")
        
        # Enhanced error logging with context
        logger.error(f"[CUSTOMER_LOOKUP] Search term: '{customer_identifier}', Error: {error_msg}")
        
        # For critical database errors, we still return None but with better logging
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"[CUSTOMER_LOOKUP] Database connection issue during lookup: {error_msg}")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            logger.warning(f"[CUSTOMER_LOOKUP] Database access permission issue: {error_msg}")
        
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


# =============================================================================
# VEHICLE LOOKUP HELPER (Task 4.2)
# =============================================================================

async def _lookup_vehicle_by_criteria(criteria: Dict[str, Any], limit: int = 20) -> List[dict]:
    """
    Lookup vehicles using flexible criteria.

    Supported keys in criteria:
      - make (maps to 'brand')
      - model
      - type
      - year (int or str)
      - color
      - is_available (bool)
      - min_stock (int) → stock_quantity >= min_stock

    Returns a list of vehicles with key fields for quotation building.
    """
    try:
        # Basic validation and normalization
        if criteria is None or not isinstance(criteria, dict):
            logger.warning("[VEHICLE_LOOKUP] Invalid criteria provided; returning empty result")
            return []
        if not isinstance(limit, int) or limit <= 0:
            limit = 20
        from core.database import db_client

        client = db_client.client

        # Start base query
        def build_query():
            q = client.table("vehicles").select(
                "id, brand, model, year, type, color, is_available, stock_quantity"
            )

            # Map and normalize filters
            make = criteria.get("make") or criteria.get("brand")
            if make:
                q = q.filter("brand", "ilike", f"%{str(make).strip()}%")

            if criteria.get("model"):
                q = q.filter("model", "ilike", f"%{str(criteria['model']).strip()}%")

            if criteria.get("type"):
                q = q.filter("type", "ilike", f"%{str(criteria['type']).strip()}%")

            if criteria.get("year"):
                # Accept numeric or string year; cast to string for ilike for tolerance (exact preferred when int)
                year_val = criteria["year"]
                if isinstance(year_val, int):
                    q = q.eq("year", year_val)
                else:
                    q = q.filter("year", "ilike", f"%{str(year_val).strip()}%")

            if criteria.get("color"):
                q = q.filter("color", "ilike", f"%{str(criteria['color']).strip()}%")

            if "is_available" in criteria and criteria["is_available"] is not None:
                q = q.eq("is_available", bool(criteria["is_available"]))

            if criteria.get("min_stock") is not None:
                try:
                    min_stock = int(criteria["min_stock"])
                    q = q.gte("stock_quantity", min_stock)
                except Exception:
                    # Ignore invalid min_stock values
                    pass

            # Prefer available and in-stock items first, then most recent year
            # Note: desc=True for booleans puts True values first
            q = q.order("is_available", desc=True).order("stock_quantity", desc=True).order("year", desc=True)

            # Apply a sane limit
            q = q.limit(max(1, min(limit, 100)))
            return q

        # Execute synchronously in a thread
        result = await asyncio.to_thread(lambda: build_query().execute())

        vehicles: List[dict] = []
        for row in result.data or []:
            vehicles.append(
                {
                    "id": row.get("id"),
                    "brand": row.get("brand"),
                    "model": row.get("model"),
                    "year": row.get("year"),
                    "type": row.get("type"),
                    "color": row.get("color"),
                    "is_available": row.get("is_available"),
                    "stock_quantity": row.get("stock_quantity"),
                    # Convenience display field
                    "display_name": f"{row.get('brand','')} {row.get('model','')} {row.get('year','')}".strip(),
                }
            )

        logger.info(f"[VEHICLE_LOOKUP] Criteria={criteria} → {len(vehicles)} result(s)")
        return vehicles

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[VEHICLE_LOOKUP] Error looking up vehicles: {e}")
        
        # Enhanced error logging with context
        logger.error(f"[VEHICLE_LOOKUP] Criteria: {criteria}, Limit: {limit}, Error: {error_msg}")
        
        # Categorize and log different types of errors for better debugging
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"[VEHICLE_LOOKUP] Database connection issue during vehicle search: {error_msg}")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            logger.warning(f"[VEHICLE_LOOKUP] Database access permission issue: {error_msg}")
        elif "syntax" in error_msg.lower() or "sql" in error_msg.lower():
            logger.error(f"[VEHICLE_LOOKUP] SQL query issue with criteria {criteria}: {error_msg}")
        
        return []

# DEPRECATED: Use _handle_confirmation_approved() and _handle_confirmation_denied() instead
# async def _handle_customer_message_confirmation(hitl_context: dict, human_response: str) -> str:
#     """DEPRECATED: Legacy confirmation handler. Use _handle_confirmation_approved() and _handle_confirmation_denied() instead."""


# =============================================================================
# VEHICLE REQUIREMENTS PARSING HELPERS (Task 5.8)
# =============================================================================

async def _parse_vehicle_requirements_with_llm(
    requirements: str, 
    extracted_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parse vehicle requirements using LLM intelligence instead of hard-coded rules.
    
    This approach is scalable, flexible, and can handle any brand/model without code changes.
    Follows the project principle: "Use LLM intelligence rather than keyword matching or complex logic."
    
    Args:
        requirements: Natural language vehicle requirements
        extracted_context: Additional context from conversation
    
    Returns:
        Dict with structured vehicle criteria for database lookup
    """
    try:
        settings = await get_settings()
        llm = ChatOpenAI(
            model=settings.openai_simple_model,
            temperature=0.1
        )
        
        # Create parsing prompt that extracts structured vehicle criteria
        parsing_prompt = ChatPromptTemplate.from_template("""
You are a vehicle requirements parser. Extract structured search criteria from the user's requirements.

Requirements: {requirements}
Additional Context: {context}

Extract the following information and return ONLY a JSON object:
{{
    "make": "exact brand name if mentioned (e.g., Toyota, Honda, Ford, BMW, Mercedes-Benz, etc.)",
    "model": "exact model name if mentioned (e.g., Camry, Civic, F-150, X5, etc.)",
    "type": "vehicle type if mentioned (e.g., SUV, Sedan, Pickup, Hatchback, Coupe, Crossover, etc.)",
    "year": "year or year range if mentioned (e.g., 2023, 2022-2024)",
    "color": "color preference if mentioned",
    "transmission": "transmission type if mentioned (Manual, Automatic, CVT)",
    "fuel_type": "fuel type if mentioned (Gasoline, Diesel, Hybrid, Electric)",
    "price_range": "budget or price range if mentioned",
    "quantity": "number of vehicles needed if mentioned",
    "special_features": "any special features or requirements mentioned"
}}

Rules:
- Only include fields that are actually mentioned or can be inferred
- Use null for fields not mentioned
- Be flexible with brand names (handle variations like "Benz" -> "Mercedes-Benz", "Beemer" -> "BMW")
- Normalize vehicle types to standard categories
- Extract quantity even if written as text ("two cars" -> 2)
- Handle abbreviations (CR-V, F-150, etc.)
- Consider context from previous conversation

Return only the JSON object, no other text.
""")
        
        # Execute the parsing
        chain = parsing_prompt | llm | StrOutputParser()
        result = await chain.ainvoke({
            "requirements": requirements,
            "context": json.dumps(extracted_context, default=str)
        })
        
        # Parse the JSON response
        try:
            criteria = json.loads(result.strip())
            # Clean up null values and empty strings
            criteria = {k: v for k, v in criteria.items() if v is not None and v != "" and v != "null"}
            logger.info(f"[VEHICLE_PARSING] Extracted criteria: {criteria}")
            return criteria
        except json.JSONDecodeError:
            logger.warning(f"[VEHICLE_PARSING] Failed to parse LLM response as JSON: {result}")
            return {}
            
    except Exception as e:
        logger.error(f"[VEHICLE_PARSING] Error parsing vehicle requirements: {e}")
        return {}


async def _get_available_makes_and_models() -> Dict[str, Any]:
    """
    Dynamically fetch available makes and models from the database.
    This ensures we can handle any brand/model in our inventory without code changes.
    
    Returns:
        Dict mapping makes to their available models and types
    """
    try:
        engine = await _get_sql_engine()
        
        query = """
        SELECT DISTINCT 
            make,
            model,
            type,
            COUNT(*) as available_count
        FROM vehicles 
        WHERE stock_quantity > 0 
        GROUP BY make, model, type
        ORDER BY make, model
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            inventory = {}
            
            for row in result:
                make = row.make
                if make not in inventory:
                    inventory[make] = {"models": set(), "types": set()}
                
                inventory[make]["models"].add(row.model)
                inventory[make]["types"].add(row.type)
            
            # Convert sets to lists for JSON serialization
            for make in inventory:
                inventory[make]["models"] = list(inventory[make]["models"])
                inventory[make]["types"] = list(inventory[make]["types"])
            
            logger.info(f"[INVENTORY_LOOKUP] Found {len(inventory)} makes in inventory")
            return inventory
            
    except Exception as e:
        logger.error(f"[INVENTORY_LOOKUP] Error fetching available inventory: {e}")
        return {}


async def _enhance_vehicle_criteria_with_fuzzy_matching(
    criteria: Dict[str, Any], 
    available_inventory: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use fuzzy matching to handle variations in brand/model names.
    This helps match user input like "honda" to "Honda" or "crv" to "CR-V".
    
    Args:
        criteria: Original criteria from LLM parsing
        available_inventory: Available makes/models from database
    
    Returns:
        Enhanced criteria with corrected make/model names
    """
    enhanced_criteria = criteria.copy()
    
    if criteria.get("make") and available_inventory:
        # Find closest matching make
        available_makes = list(available_inventory.keys())
        make_match = _find_closest_match(criteria["make"], available_makes)
        if make_match:
            enhanced_criteria["make"] = make_match
            logger.info(f"[FUZZY_MATCHING] Matched make '{criteria['make']}' -> '{make_match}'")
            
            # If we found a make match, try to match the model
            if criteria.get("model") and make_match in available_inventory:
                available_models = available_inventory[make_match]["models"]
                model_match = _find_closest_match(criteria["model"], available_models)
                if model_match:
                    enhanced_criteria["model"] = model_match
                    logger.info(f"[FUZZY_MATCHING] Matched model '{criteria['model']}' -> '{model_match}'")
    
    return enhanced_criteria


def _find_closest_match(target: str, options: List[str], threshold: int = 70) -> Optional[str]:
    """
    Simple fuzzy matching using string similarity.
    
    Args:
        target: String to match
        options: List of possible matches
        threshold: Minimum similarity score (0-100)
    
    Returns:
        Best matching option or None if no good match found
    """
    from difflib import SequenceMatcher
    
    if not target or not options:
        return None
    
    best_match = None
    best_score = 0
    
    target_lower = target.lower().strip()
    
    for option in options:
        option_lower = option.lower().strip()
        
        # Exact match gets priority
        if target_lower == option_lower:
            return option
        
        # Calculate similarity score
        score = SequenceMatcher(None, target_lower, option_lower).ratio() * 100
        
        # Also check if target is contained in option (for abbreviations)
        if target_lower in option_lower or option_lower in target_lower:
            score = max(score, 85)  # Boost score for substring matches
        
        if score > threshold and score > best_score:
            best_score = score
            best_match = option
    
    return best_match


def _generate_inventory_suggestions(available_inventory: Dict[str, Any]) -> str:
    """
    Generate a formatted string of available inventory for user-friendly messages.
    
    Args:
        available_inventory: Dictionary of available makes and their models
    
    Returns:
        Formatted string showing available brands and popular models
    """
    if not available_inventory:
        return "Please contact us for current inventory availability."
    
    suggestions = []
    for make, details in sorted(available_inventory.items()):
        models = details.get("models", [])
        # Show first few models as examples
        model_examples = ", ".join(sorted(models)[:3])
        if len(models) > 3:
            model_examples += f" (and {len(models) - 3} more)"
        
        suggestions.append(f"• **{make}**: {model_examples}")
    
    return "\n".join(suggestions)


# =============================================================================
# HITL RESUME LOGIC HANDLERS (Tasks 5.3.1.3-5.3.1.9)
# =============================================================================

async def _handle_quotation_resume(
    customer_identifier: str,
    vehicle_requirements: str,
    additional_notes: Optional[str],
    quotation_validity_days: int,
    quotation_state: Dict[str, Any],
    current_step: str,
    user_response: str,
    conversation_context: str
) -> str:
    """
    Handle resuming quotation generation from HITL interactions.
    
    This function implements the multi-step HITL flow support by:
    1. Processing user responses for specific steps
    2. Updating quotation state with new information
    3. Continuing to the next step or completing the quotation
    
    Args:
        customer_identifier: Customer identifier
        vehicle_requirements: Vehicle requirements
        additional_notes: Additional notes
        quotation_validity_days: Validity period
        quotation_state: Preserved intermediate state
        current_step: Current HITL step being resumed
        user_response: User's response to HITL prompt
        conversation_context: Conversation context
        
    Returns:
        Either continuation of quotation process or next HITL request
    """
    try:
        logger.info(f"[QUOTATION_RESUME] Processing step: {current_step}")
        logger.info(f"[QUOTATION_RESUME] User response: {user_response}")
        
        # Process user response based on current step
        if current_step == "customer_lookup":
            return await _resume_customer_lookup(
                customer_identifier, quotation_state, user_response
            )
        elif current_step == "vehicle_requirements":
            return await _resume_vehicle_requirements(
                vehicle_requirements, quotation_state, user_response
            )
        elif current_step == "employee_data":
            return await _resume_employee_data(
                quotation_state, user_response
            )
        elif current_step == "missing_information":
            return await _resume_missing_information(
                quotation_state, user_response
            )
        elif current_step == "pricing_issues":
            return await _resume_pricing_issues(
                quotation_state, user_response
            )
        elif current_step == "quotation_approval":
            return await _resume_quotation_approval(
                quotation_state, user_response
            )
        else:
            logger.error(f"[QUOTATION_RESUME] Unknown step: {current_step}")
            return f"❌ Error: Unknown quotation step '{current_step}'. Please start the quotation process again."
            
    except Exception as e:
        logger.error(f"[QUOTATION_RESUME] Error processing step {current_step}: {e}")
        return f"❌ Error processing your response. Please try again or start a new quotation."


async def _resume_customer_lookup(
    customer_identifier: str, quotation_state: Dict[str, Any], user_response: str
) -> str:
    """Resume customer lookup step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing customer lookup response")
    
    # Parse user response for customer information
    # This could be email, phone, company name, or other identifiers
    quotation_state["customer_lookup_response"] = user_response.strip()
    
    # Try to lookup customer with new information
    customer_data = await _lookup_customer(user_response.strip())
    if customer_data:
        quotation_state["customer_data"] = customer_data
        logger.info("[QUOTATION_RESUME] Customer found with provided information")
        
        # Continue with quotation process - call main function without resume params
        return await generate_quotation(
            customer_identifier=user_response.strip(),
            vehicle_requirements=quotation_state.get("vehicle_requirements", ""),
            additional_notes=quotation_state.get("additional_notes"),
            quotation_validity_days=quotation_state.get("quotation_validity_days", 30),
            quotation_state=quotation_state,
            conversation_context=quotation_state.get("conversation_context", "")
        )
    else:
        # Still no customer found - request more specific information
        return request_input(
            prompt=f"""🔍 **Customer Lookup - Additional Information Needed**

I couldn't find a customer record with "{user_response}". 

Please provide:
• **Email address** (most reliable)
• **Phone number** with area code
• **Full company name** (for business customers)
• **Customer ID** (if known)

This helps me locate the correct customer record for the quotation.""",
            input_type="customer_lookup",
            context={
                "source_tool": "generate_quotation",
                "current_step": "customer_lookup",
                "customer_identifier": customer_identifier,
                "quotation_state": quotation_state,
                "previous_attempts": quotation_state.get("customer_lookup_attempts", 0) + 1
            }
        )


async def _resume_vehicle_requirements(
    vehicle_requirements: str, quotation_state: Dict[str, Any], user_response: str
) -> str:
    """Resume vehicle requirements step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing vehicle requirements response")
    
    # Update vehicle requirements with user response
    updated_requirements = f"{vehicle_requirements} {user_response}".strip()
    quotation_state["updated_vehicle_requirements"] = updated_requirements
    
    # Continue with quotation process using updated requirements
    return await generate_quotation(
        customer_identifier=quotation_state.get("customer_identifier", ""),
        vehicle_requirements=updated_requirements,
        additional_notes=quotation_state.get("additional_notes"),
        quotation_validity_days=quotation_state.get("quotation_validity_days", 30),
        quotation_state=quotation_state,
        conversation_context=quotation_state.get("conversation_context", "")
    )


async def _resume_employee_data(quotation_state: Dict[str, Any], user_response: str) -> str:
    """Resume employee data step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing employee data response")
    
    # Parse employee information from user response
    quotation_state["employee_data_response"] = user_response.strip()
    
    # Continue with quotation process
    return await generate_quotation(
        customer_identifier=quotation_state.get("customer_identifier", ""),
        vehicle_requirements=quotation_state.get("vehicle_requirements", ""),
        additional_notes=quotation_state.get("additional_notes"),
        quotation_validity_days=quotation_state.get("quotation_validity_days", 30),
        quotation_state=quotation_state,
        conversation_context=quotation_state.get("conversation_context", "")
    )


async def _resume_missing_information(quotation_state: Dict[str, Any], user_response: str) -> str:
    """Resume missing information step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing missing information response")
    
    # Parse and store missing information
    quotation_state["missing_info_response"] = user_response.strip()
    
    # Continue with quotation process
    return await generate_quotation(
        customer_identifier=quotation_state.get("customer_identifier", ""),
        vehicle_requirements=quotation_state.get("vehicle_requirements", ""),
        additional_notes=quotation_state.get("additional_notes"),
        quotation_validity_days=quotation_state.get("quotation_validity_days", 30),
        quotation_state=quotation_state,
        conversation_context=quotation_state.get("conversation_context", "")
    )


async def _resume_pricing_issues(quotation_state: Dict[str, Any], user_response: str) -> str:
    """Resume pricing issues step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing pricing issues response")
    
    # Parse pricing decision from user response
    quotation_state["pricing_decision"] = user_response.strip()
    
    # Continue with quotation process
    return await generate_quotation(
        customer_identifier=quotation_state.get("customer_identifier", ""),
        vehicle_requirements=quotation_state.get("vehicle_requirements", ""),
        additional_notes=quotation_state.get("additional_notes"),
        quotation_validity_days=quotation_state.get("quotation_validity_days", 30),
        quotation_state=quotation_state,
        conversation_context=quotation_state.get("conversation_context", "")
    )


async def _resume_quotation_approval(quotation_state: Dict[str, Any], user_response: str) -> str:
    """Resume quotation approval step with user-provided information."""
    logger.info("[QUOTATION_RESUME] Processing quotation approval response")
    
    # Use LLM-driven interpretation instead of keyword matching
    context = {
        "source_tool": "generate_quotation",
        "current_step": "quotation_approval",
        "interaction_type": "approval_request"
    }
    
    # Use the centralized HITL system for LLM-driven intent interpretation
    from agents.hitl import _interpret_user_intent_with_llm
    user_intent = await _interpret_user_intent_with_llm(user_response, context)
    
    if user_intent == "approval":
        logger.info(f"[QUOTATION_RESUME] Quotation approved by user: '{user_response}' - proceeding with PDF generation")
        
        try:
            # Extract data from quotation state
            customer_data = quotation_state.get("customer_data", {})
            vehicle_pricing = quotation_state.get("vehicle_pricing", [])
            employee_data = quotation_state.get("employee_data", {})
            additional_notes = quotation_state.get("additional_notes", "")
            quotation_validity_days = quotation_state.get("quotation_validity_days", 30)
            
            # Generate PDF quotation
            logger.info("[QUOTATION_RESUME] Generating PDF quotation...")
            pdf_content = await generate_quotation_pdf(
                customer_data=customer_data,
                vehicle_pricing=vehicle_pricing,
                employee_data=employee_data,
                additional_notes=additional_notes,
                validity_days=quotation_validity_days
            )
            
            # Upload to Supabase storage
            logger.info("[QUOTATION_RESUME] Uploading PDF to storage...")
            
            # Generate unique filename
            from datetime import datetime
            import uuid
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            customer_name = customer_data.get("name", "customer").replace(" ", "_")
            filename = f"quotation_{customer_name}_{timestamp}_{str(uuid.uuid4())[:8]}.pdf"
            
            # Upload PDF
            upload_result = await upload_quotation_pdf(
                pdf_content=pdf_content,
                filename=filename,
                customer_id=customer_data.get("id"),
                employee_id=quotation_state.get("employee_id")
            )
            
            if upload_result.get("success"):
                # Create shareable link
                logger.info("[QUOTATION_RESUME] Creating shareable link...")
                shareable_url = await create_signed_quotation_url(
                    storage_path=filename,
                    expires_in_seconds=48 * 3600
                )
                
                # Calculate totals for summary
                total_amount = sum(item["pricing"].get("final_price", 0) for item in vehicle_pricing)
                vehicle_count = len(vehicle_pricing)
                
                success_message = f"""🎉 **QUOTATION GENERATED SUCCESSFULLY**

📄 **Quotation Details:**
- **Document**: Professional PDF quotation created
- **Customer**: {customer_data.get('name', 'N/A')}
- **Vehicles**: {vehicle_count} option{'s' if vehicle_count != 1 else ''}
- **Total Value**: ₱{total_amount:,.2f}
- **Valid Until**: {(datetime.now().replace(day=datetime.now().day + quotation_validity_days)).strftime('%B %d, %Y')}

🔗 **Access Information:**
- **Shareable Link**: {shareable_url}
- **Link Validity**: 48 hours from now
- **File Size**: {len(pdf_content) / 1024:.1f} KB

📋 **Next Steps:**
✅ Quotation saved to CRM system
✅ Document stored securely in cloud storage  
✅ Shareable link generated for customer access
✅ Quotation tracking activated for follow-up

**Share the link above with your customer to provide them access to download their personalized quotation PDF.**

*The quotation has been automatically recorded in our system for tracking and follow-up purposes.*"""

                logger.info(f"[QUOTATION_RESUME] Quotation successfully generated: {filename}")
                return success_message
                
            else:
                logger.error(f"[QUOTATION_RESUME] Failed to upload PDF: {upload_result}")
                return f"❌ **Upload Failed**\n\nThe PDF was generated successfully, but there was an issue uploading it to storage. Please try again or contact your administrator.\n\nError: {upload_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"[QUOTATION_RESUME] Error generating quotation: {e}")
            return f"❌ **Generation Failed**\n\nThere was an error generating the quotation PDF. Please try again or contact your administrator.\n\nError: {str(e)}"
    
    elif user_intent == "denial":
        # Approval was denied
        logger.info(f"[QUOTATION_RESUME] Quotation approval denied by user: '{user_response}'")
        return f"""❌ **Quotation Generation Cancelled**

The quotation generation has been cancelled as requested.

**Options:**
- Modify the quotation details and try again
- Generate a new quotation with different requirements
- Contact customer for updated specifications

No quotation document was created or stored."""
    
    else:
        # User provided input instead of approval/denial
        logger.info(f"[QUOTATION_RESUME] User provided input instead of approval: '{user_response}'")
        return f"""📝 **Additional Information Received**

I received: "{user_response}"

However, I need a clear approval or denial for the quotation generation.

**Please respond with:**
- "✅ Generate Quotation" to proceed with PDF creation
- "❌ Cancel" to cancel the quotation

Or let me know if you need to modify any details first."""


async def _build_conversation_state_for_extraction(
    customer_identifier: str,
    vehicle_requirements: str,
    additional_notes: Optional[str],
    conversation_context: str
) -> Dict[str, Any]:
    """
    Build a comprehensive conversation state for extract_fields_from_conversation.
    
    This function creates a rich state object that provides maximum context for
    intelligent field extraction, going beyond the simple mock_state approach.
    
    Args:
        customer_identifier: Customer identifier provided to the tool
        vehicle_requirements: Vehicle requirements provided to the tool
        additional_notes: Additional notes provided to the tool
        conversation_context: Recent conversation context
    
    Returns:
        Enhanced state dict with comprehensive conversation information
    """
    try:
        # Get current conversation ID for additional context
        conversation_id = get_current_conversation_id()
        
        # Build comprehensive message history simulation
        messages = []
        
        # Add conversation context as historical messages if available
        if conversation_context:
            messages.append({
                "type": "system",
                "content": f"Previous conversation context: {conversation_context}"
            })
        
        # Add current tool parameters as user messages for context extraction
        if customer_identifier:
            messages.append({
                "type": "human", 
                "content": f"Customer: {customer_identifier}"
            })
        
        if vehicle_requirements:
            messages.append({
                "type": "human",
                "content": f"Vehicle requirements: {vehicle_requirements}"
            })
        
        if additional_notes:
            messages.append({
                "type": "human",
                "content": f"Additional notes: {additional_notes}"
            })
        
        # Build comprehensive state object
        enhanced_state = {
            "messages": messages,
            "conversation_summary": conversation_context or "",
            "conversation_id": conversation_id,
            "tool_context": {
                "tool_name": "generate_quotation",
                "customer_identifier": customer_identifier,
                "vehicle_requirements": vehicle_requirements,
                "additional_notes": additional_notes
            }
        }
        
        logger.debug(f"[CONVERSATION_STATE] Built enhanced state with {len(messages)} messages")
        return enhanced_state
        
    except Exception as e:
        logger.error(f"[CONVERSATION_STATE] Error building conversation state: {e}")
        # Fallback to simple state if enhancement fails
        return {
            "messages": [{"content": f"Customer: {customer_identifier}, Requirements: {vehicle_requirements}"}],
            "conversation_summary": conversation_context or ""
        }


def _enhance_vehicle_criteria_with_extracted_context(
    vehicle_criteria: Dict[str, Any], 
    extracted_context: Dict[str, Any]
) -> None:
    """
    Enhance vehicle criteria with additional information from extracted conversation context.
    
    This function merges relevant vehicle information from the conversation context
    into the vehicle criteria, filling in gaps that the LLM parsing might have missed.
    
    Args:
        vehicle_criteria: Original criteria from LLM parsing (modified in place)
        extracted_context: Context extracted from conversation
    """
    # Mapping of context fields to vehicle criteria fields
    context_to_criteria_mapping = {
        "vehicle_make": "make",
        "vehicle_model": "model", 
        "vehicle_type": "type",
        "vehicle_year": "year",
        "vehicle_color": "color",
        "vehicle_transmission": "transmission",
        "vehicle_fuel_type": "fuel_type",
        "quantity": "quantity",
        "budget_range": "price_range",
        "special_requirements": "special_features"
    }
    
    # Enhance criteria with extracted context, but don't override existing values
    for context_field, criteria_field in context_to_criteria_mapping.items():
        if (context_field in extracted_context and 
            extracted_context[context_field] and 
            criteria_field not in vehicle_criteria):
            
            vehicle_criteria[criteria_field] = extracted_context[context_field]
            logger.info(f"[VEHICLE_ENHANCEMENT] Added {criteria_field}='{extracted_context[context_field]}' from conversation context")


def _format_vehicle_list_for_hitl(vehicles: List[dict]) -> str:
    """
    Format a list of vehicles for display in HITL prompts.
    
    Args:
        vehicles: List of vehicle dictionaries from database lookup
    
    Returns:
        Formatted string suitable for HITL prompts
    """
    if not vehicles:
        return "No vehicles found."
    
    formatted_list = []
    for i, vehicle in enumerate(vehicles, 1):
        vehicle_info = f"{i}. **{vehicle.get('make', 'Unknown')} {vehicle.get('model', 'Unknown')}**"
        
        details = []
        if vehicle.get('year'):
            details.append(f"Year: {vehicle['year']}")
        if vehicle.get('type'):
            details.append(f"Type: {vehicle['type']}")
        if vehicle.get('color'):
            details.append(f"Color: {vehicle['color']}")
        if vehicle.get('stock_quantity'):
            details.append(f"Available: {vehicle['stock_quantity']} units")
        
        if details:
            vehicle_info += f" ({', '.join(details)})"
        
        formatted_list.append(vehicle_info)
    
    return "\n".join(formatted_list)


def _identify_missing_quotation_information(
    customer_data: dict,
    vehicle_pricing: List[dict],
    extracted_context: Dict[str, Any]
) -> Dict[str, str]:
    """
    Identify missing critical information needed for a complete quotation.
    
    Args:
        customer_data: Customer information from CRM lookup
        vehicle_pricing: Vehicle and pricing information
        extracted_context: Context extracted from conversation
    
    Returns:
        Dictionary of missing information with field names as keys and descriptions as values
    """
    missing_info = {}
    
    # Check critical customer information
    if not customer_data.get('email') and not extracted_context.get('customer_email'):
        missing_info['customer_email'] = "Customer email address for sending the quotation"
    
    if not customer_data.get('phone') and not extracted_context.get('customer_phone'):
        missing_info['customer_phone'] = "Customer phone number for follow-up contact"
    
    # Check if we have delivery/contact address
    if (not customer_data.get('address') and 
        not extracted_context.get('customer_address') and
        not extracted_context.get('delivery_timeline')):
        missing_info['delivery_info'] = "Delivery address or pickup preference"
    
    # Check quantity specification
    if not extracted_context.get('quantity'):
        # Try to infer from vehicle_pricing count, but ask for confirmation if unclear
        if len(vehicle_pricing) > 1:
            missing_info['quantity_confirmation'] = f"Confirmation of quantity needed (showing {len(vehicle_pricing)} vehicle options)"
    
    # Check timeline if not specified
    if not extracted_context.get('delivery_timeline'):
        missing_info['timeline'] = "When you need the vehicle(s) delivered or ready"
    
    # Check financing preferences for business context
    if (not extracted_context.get('financing_preference') and 
        not extracted_context.get('budget_range')):
        missing_info['payment_info'] = "Payment method preference (cash, financing, lease, etc.)"
    
    return missing_info


async def _request_missing_information_via_hitl(
    missing_info: Dict[str, str],
    customer_data: dict,
    vehicle_pricing: List[dict],
    employee_data: dict,
    extracted_context: Dict[str, Any],
    additional_notes: Optional[str],
    quotation_validity_days: int,
    customer_identifier: str = "",
    vehicle_requirements: str = "",
    quotation_state: Dict[str, Any] = None,
    conversation_context: str = "",
    employee_id: str = ""
) -> str:
    """
    Request missing information via HITL flow with intelligent prompting.
    
    Args:
        missing_info: Dictionary of missing information
        customer_data: Customer information
        vehicle_pricing: Vehicle and pricing information  
        employee_data: Employee information
        extracted_context: Extracted conversation context
        additional_notes: Additional notes
        quotation_validity_days: Quotation validity period
    
    Returns:
        HITL request string
    """
    # Build summary of what we have
    summary_parts = []
    
    # Customer summary
    customer_name = customer_data.get('name', 'Customer')
    summary_parts.append(f"**Customer**: {customer_name}")
    
    # Vehicle summary
    if vehicle_pricing:
        vehicle_summary = []
        for item in vehicle_pricing:
            vehicle = item['vehicle']
            pricing = item['pricing']
            vehicle_summary.append(f"• {vehicle.get('make', '')} {vehicle.get('model', '')} - ₱{pricing.get('final_price', 0):,.2f}")
        summary_parts.append(f"**Vehicles**: \n" + "\n".join(vehicle_summary))
    
    # Build missing information request
    missing_items = []
    priority_order = ['customer_email', 'customer_phone', 'quantity_confirmation', 'timeline', 'delivery_info', 'payment_info']
    
    # Sort missing info by priority
    sorted_missing = []
    for priority_field in priority_order:
        if priority_field in missing_info:
            sorted_missing.append((priority_field, missing_info[priority_field]))
    
    # Add any remaining missing items
    for field, description in missing_info.items():
        if field not in [item[0] for item in sorted_missing]:
            sorted_missing.append((field, description))
    
    for i, (field, description) in enumerate(sorted_missing, 1):
        missing_items.append(f"{i}. **{description}**")
    
    return request_input(
        prompt=f"""📋 **Additional Information Needed for Quotation**

I have most of the information needed for your quotation:

{chr(10).join(summary_parts)}

To complete your professional quotation, I need a few more details:

{chr(10).join(missing_items)}

Please provide the missing information above. You can answer all at once or just the most important ones first.""",
        input_type="missing_quotation_info",
        context={
            "source_tool": "generate_quotation",
            "current_step": "missing_information",
            "customer_identifier": customer_identifier,
            "vehicle_requirements": vehicle_requirements,
            "additional_notes": additional_notes,
            "quotation_validity_days": quotation_validity_days,
            "quotation_state": quotation_state or {},
            "conversation_context": conversation_context,
            "customer_data": customer_data,
            "vehicle_pricing": vehicle_pricing,
            "employee_data": employee_data,
            "extracted_context": extracted_context,
            "missing_info": missing_info,
            "employee_id": employee_id
        }
    )


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
# PRICING LOOKUP HELPER (Task 4.3)
# =============================================================================

async def _lookup_current_pricing(
    vehicle_id: str,
    include_inactive: bool = False,
    discounts: float | None = None,
    insurance: float | None = None,
    lto_fees: float | None = None,
    add_ons: Optional[List[Dict[str, Any]]] = None,
) -> Optional[dict]:
    """
    Retrieve current pricing for a vehicle and compute totals with optional components.

    - Reads from `pricing` table (expected columns: id, vehicle_id, base_price, final_price, is_active)
    - If multiple pricing rows exist, prefers active ones; falls back to the most recent
    - Computes a `computed_total` using provided optional components when present

    Args:
        vehicle_id: Vehicle UUID
        include_inactive: If True, allow inactive pricing when no active exists
        discounts: Optional discount amount to subtract from computed total
        insurance: Optional insurance amount to add
        lto_fees: Optional LTO fees to add
        add_ons: Optional list of {name, price} to add

    Returns:
        Dict with base fields and computed totals, or None if pricing not found
    """
    try:
        # Validate inputs
        if not vehicle_id or not isinstance(vehicle_id, str) or not vehicle_id.strip():
            logger.warning("[PRICING_LOOKUP] Missing or invalid vehicle_id")
            return None
        if add_ons is not None and not isinstance(add_ons, list):
            logger.warning("[PRICING_LOOKUP] add_ons must be a list of {name, price}; ignoring provided value")
            add_ons = []
        from core.database import db_client

        client = db_client.client

        def run_query():
            # Select only necessary columns for performance and stability
            q = (
                client.table("pricing")
                .select("id, vehicle_id, base_price, final_price, is_active")
                .eq("vehicle_id", vehicle_id)
            )
            if not include_inactive:
                q = q.eq("is_active", True)
            # Prefer newer rows first if updated_at exists, else by id desc
            try:
                q = q.order("updated_at", desc=True)
            except Exception:
                q = q.order("id", desc=True)
            return q.limit(1).execute()

        result = await asyncio.to_thread(run_query)
        if not result.data:
            # Optionally attempt fallback to inactive rows
            if not include_inactive:
                def fallback_query():
                    q2 = (
                        client.table("pricing")
                        .select("id, vehicle_id, base_price, final_price, is_active")
                        .eq("vehicle_id", vehicle_id)
                    )
                    try:
                        q2 = q2.order("updated_at", desc=True)
                    except Exception:
                        q2 = q2.order("id", desc=True)
                    return q2.limit(1).execute()
                result = await asyncio.to_thread(fallback_query)
            if not result.data:
                logger.info(f"[PRICING_LOOKUP] No pricing found for vehicle_id={vehicle_id}")
                return None

        row = result.data[0]
        base_price = float(row.get("base_price") or 0.0)
        final_price = row.get("final_price")
        try:
            final_price = float(final_price) if final_price is not None else None
        except Exception:
            final_price = None

        # Optional components
        add_ons_list = add_ons or []
        add_on_total = 0.0
        for item in add_ons_list:
            try:
                add_on_total += float(item.get("price") or 0.0)
            except Exception:
                continue

        insurance_val = float(insurance or 0.0)
        lto_val = float(lto_fees or 0.0)
        discount_val = float(discounts or 0.0)

        # Computed total using base price by default when final_price absent
        base_for_calc = final_price if final_price is not None else base_price
        computed_total = max(0.0, base_for_calc + insurance_val + lto_val + add_on_total - discount_val)

        pricing_info = {
            "pricing_id": row.get("id"),
            "vehicle_id": vehicle_id,
            "is_active": row.get("is_active", True),
            "base_price": base_price,
            "final_price": final_price,
            "currency": row.get("currency", "PHP"),
            "effective_date": row.get("effective_date"),
            "computed": {
                "insurance": insurance_val,
                "lto_fees": lto_val,
                "discounts": discount_val,
                "add_ons": add_ons_list,
                "add_on_total": add_on_total,
                "computed_total": computed_total,
            },
        }

        logger.info(f"[PRICING_LOOKUP] vehicle_id={vehicle_id} → base={base_price} final={final_price} total={computed_total}")
        return pricing_info

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[PRICING_LOOKUP] Error fetching pricing for vehicle_id={vehicle_id}: {e}")
        
        # Enhanced error logging with context
        logger.error(f"[PRICING_LOOKUP] Vehicle ID: {vehicle_id}, Include inactive: {include_inactive}, Error: {error_msg}")
        
        # Categorize pricing-specific errors
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"[PRICING_LOOKUP] Database connection issue during pricing fetch: {error_msg}")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            logger.warning(f"[PRICING_LOOKUP] Database access permission issue: {error_msg}")
        elif "constraint" in error_msg.lower() or "foreign key" in error_msg.lower():
            logger.error(f"[PRICING_LOOKUP] Data integrity issue - vehicle_id {vehicle_id} may not exist: {error_msg}")
        elif "type" in error_msg.lower() and "conversion" in error_msg.lower():
            logger.error(f"[PRICING_LOOKUP] Data type conversion error - pricing data may be corrupted: {error_msg}")
        
        return None


# =============================================================================
# EMPLOYEE DETAILS LOOKUP HELPER (Task 4.4)
# =============================================================================

async def _lookup_employee_details(identifier: str) -> Optional[dict]:
    """
    Lookup employee details by id/email/name and include branch info when available.

    Returns fields needed for quotations: name, position, email, phone, branch_name, branch_region.
    """
    try:
        from core.database import db_client

        client = db_client.client
        search = (identifier or "").strip()

        def run_query():
            q = client.table("employees").select("id, name, position, email, phone, branch_id").limit(1)
            # Try UUID exact match first
            if len(search.replace("-", "")) == 32 and search.count("-") == 4:
                res = client.table("employees").select("id, name, position, email, phone, branch_id").eq("id", search).limit(1).execute()
                if res.data:
                    return res
            # Try email exact
            res = client.table("employees").select("id, name, position, email, phone, branch_id").eq("email", search).limit(1).execute()
            if res.data:
                return res
            # Name ilike
            res = client.table("employees").select("id, name, position, email, phone, branch_id").filter("name", "ilike", f"%{search}%").limit(1).execute()
            return res

        emp_res = await asyncio.to_thread(run_query)
        if not emp_res.data:
            return None

        emp = emp_res.data[0]
        branch_name = None
        branch_region = None

        # Fetch branch details if there is a branch_id
        if emp.get("branch_id"):
            def fetch_branch():
                return (
                    client.table("branches")
                    .select("name, region")
                    .eq("id", emp["branch_id"])  # type: ignore[index]
                    .limit(1)
                    .execute()
                )
            br_res = await asyncio.to_thread(fetch_branch)
            if br_res.data:
                branch_name = br_res.data[0].get("name")
                branch_region = br_res.data[0].get("region")

        result = {
            "id": emp.get("id"),
            "name": emp.get("name"),
            "position": emp.get("position"),
            "email": emp.get("email"),
            "phone": emp.get("phone"),
            "branch_name": branch_name,
            "branch_region": branch_region,
        }

        logger.info(f"[EMPLOYEE_LOOKUP] identifier={identifier} → {result.get('name')}")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[EMPLOYEE_LOOKUP] Error looking up employee '{identifier}': {e}")
        
        # Enhanced error logging with context
        logger.error(f"[EMPLOYEE_LOOKUP] Employee identifier: '{identifier}', Error: {error_msg}")
        
        # Categorize employee-specific errors
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"[EMPLOYEE_LOOKUP] Database connection issue during employee lookup: {error_msg}")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            logger.warning(f"[EMPLOYEE_LOOKUP] Database access permission issue: {error_msg}")
        elif "constraint" in error_msg.lower() or "foreign key" in error_msg.lower():
            logger.error(f"[EMPLOYEE_LOOKUP] Data integrity issue - branch_id may not exist for employee: {error_msg}")
        elif "uuid" in error_msg.lower() and "invalid" in error_msg.lower():
            logger.error(f"[EMPLOYEE_LOOKUP] Invalid UUID format for identifier '{identifier}': {error_msg}")
        
        return None


# =============================================================================
# QUOTATION GENERATION TOOL (Task 5.1)
# =============================================================================

class GenerateQuotationParams(BaseModel):
    """Parameters for generating professional PDF quotations (Employee Only)."""
    customer_identifier: str = Field(..., description="Customer identifier for quotation: full name, email address, phone number, or company name")
    vehicle_requirements: str = Field(..., description="Detailed vehicle specifications: make/brand, model, type (sedan/SUV/pickup), year, color, quantity needed, budget range")
    additional_notes: Optional[str] = Field(None, description="Special requirements: financing preferences, trade-in details, delivery timeline, custom features")
    quotation_validity_days: Optional[int] = Field(30, description="Quotation validity period in days (default: 30, maximum: 365)")

@tool(args_schema=GenerateQuotationParams)
@traceable(name="generate_quotation")
async def generate_quotation(
    customer_identifier: str,
    vehicle_requirements: str,
    additional_notes: Optional[str] = None,
    quotation_validity_days: Optional[int] = 30,
    # HITL Resume Parameters (Task 5.3.1.1)
    quotation_state: Dict[str, Any] = None,
    current_step: str = "",
    user_response: str = "",
    conversation_context: str = ""
) -> str:
    """
    Generate official PDF quotations for customers (Employee Only - DO NOT use for simple data lookup).
    
    **Use this tool when customers need:**
    - Official quotation documents for vehicle purchases
    - Professional PDF quotes with pricing and terms
    - Formal proposals they can share or present to decision-makers
    - Documented quotes for their procurement processes
    
    **Do NOT use this tool for:**
    - Simple price inquiries (use simple_query_crm_data instead)
    - Looking up vehicle specifications without creating quotes
    - Checking customer information without generating documents
    - Exploratory pricing research or comparisons
    
    **This tool provides a complete quotation workflow:**
    1. Intelligent context extraction from conversation history
    2. Automated customer and vehicle data lookup from CRM
    3. Interactive HITL flows to gather missing information
    4. Professional quotation preview for employee approval
    5. PDF generation with company branding and legal terms
    6. Secure shareable links with controlled access and expiration
    7. CRM integration for quotation tracking and follow-up
    
    **Access Control:** Employee-only tool with comprehensive access verification.
    
    Args:
        customer_identifier: Customer name, email, phone, or company name
        vehicle_requirements: Detailed vehicle specs (make, model, type, quantity, etc.)
        additional_notes: Special requirements, trade-ins, financing preferences
        quotation_validity_days: Quote validity period (default: 30 days, max: 365)
    
    Returns:
        Professional quotation result with PDF link, or HITL request for missing information
    """
    try:
        # Enhanced input validation (Task 5.6)
        if not customer_identifier or not customer_identifier.strip():
            logger.warning("[GENERATE_QUOTATION] Empty customer_identifier provided")
            return """❌ **Invalid Input**

Customer identifier is required to generate a quotation. Please provide:
- Customer name (e.g., "John Doe")
- Email address (e.g., "john@company.com")  
- Phone number (e.g., "+63 912 345 6789")
- Company name (e.g., "ABC Corporation")

**Example**: `generate_quotation("John Doe", "Toyota Camry sedan")`"""

        if not vehicle_requirements or not vehicle_requirements.strip():
            logger.warning("[GENERATE_QUOTATION] Empty vehicle_requirements provided")
            return """❌ **Invalid Input**

Vehicle requirements are needed to generate a quotation. Please specify:
- Vehicle make/brand (e.g., "Toyota", "Honda", "Ford")
- Model (e.g., "Camry", "Civic", "F-150")
- Type (e.g., "sedan", "SUV", "pickup truck")
- Quantity if multiple vehicles needed

**Example**: `generate_quotation("John Doe", "2 Toyota Camry sedans, 2023 or newer")`"""

        # Validate quotation_validity_days
        if quotation_validity_days is not None and (not isinstance(quotation_validity_days, int) or quotation_validity_days <= 0 or quotation_validity_days > 365):
            logger.warning(f"[GENERATE_QUOTATION] Invalid quotation_validity_days: {quotation_validity_days}")
            quotation_validity_days = 30  # Reset to default
            logger.info("[GENERATE_QUOTATION] Reset quotation_validity_days to default (30 days)")

        # Initialize quotation state for resume logic (Task 5.3.1.2)
        if quotation_state is None:
            quotation_state = {}
            
        logger.info(f"[GENERATE_QUOTATION] Starting quotation generation for customer: {customer_identifier}")
        logger.info(f"[GENERATE_QUOTATION] Vehicle requirements: {vehicle_requirements}")
        logger.info(f"[GENERATE_QUOTATION] Current step: '{current_step}', User response: '{user_response}'")
        logger.info(f"[GENERATE_QUOTATION] State keys: {list(quotation_state.keys())}")
        logger.info(f"[GENERATE_QUOTATION] Validity days: {quotation_validity_days}")
        
        # HITL Resume Detection Logic (Task 5.3.1.2)
        if current_step and user_response:
            logger.info(f"[GENERATE_QUOTATION] 🔄 Resuming from HITL interaction: {current_step}")
            return await _handle_quotation_resume(
                customer_identifier, vehicle_requirements, additional_notes, 
                quotation_validity_days, quotation_state, current_step, user_response, conversation_context
            )
        
        # Enhanced employee access control (Task 5.6)
        employee_id = await get_current_employee_id()
        if not employee_id:
            logger.warning("[GENERATE_QUOTATION] Non-employee user attempted to use quotation generation")
            return """🔒 **Employee Access Required**

Quotation generation is restricted to authorized employees only.

**If you are an employee:**
✅ Ensure you're logged in with your employee account
✅ Check that your employee profile is properly configured
✅ Verify your account has quotation generation permissions
✅ Contact your system administrator if the issue persists

**If you are a customer:**
✅ Contact your sales representative for quotation requests
✅ Use our customer inquiry form for pricing information
✅ Call our sales hotline for immediate assistance

**Need help?** Contact your administrator or IT support team to verify your employee status and permissions."""
        
        logger.info(f"[GENERATE_QUOTATION] Starting quotation generation for employee {employee_id}")
        
        # Step 1: Enhanced conversation context extraction (Task 5.2)
        # Get comprehensive conversation context for intelligent field extraction
        try:
            conversation_context = await get_recent_conversation_context()
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Failed to get conversation context: {e}")
            conversation_context = ""  # Continue without context if extraction fails
        
        # Define comprehensive field definitions for quotation-specific information
        field_definitions = {
            # Customer Information Fields
            "customer_name": "Full name of the customer or contact person",
            "customer_email": "Customer's email address for correspondence",
            "customer_phone": "Customer's phone number for contact",
            "customer_company": "Customer's company name or business",
            "customer_address": "Customer's address for delivery or documentation",
            
            # Vehicle Specification Fields
            "vehicle_make": "Vehicle manufacturer/brand (Toyota, Honda, Ford, etc.)",
            "vehicle_model": "Specific vehicle model name (Camry, Civic, F-150, etc.)",
            "vehicle_type": "Type of vehicle (Sedan, SUV, Pickup, Hatchback, Coupe, etc.)",
            "vehicle_year": "Model year preference or range (2023, 2022-2024, etc.)",
            "vehicle_color": "Preferred color or color options",
            "vehicle_transmission": "Transmission preference (Automatic, Manual, CVT)",
            "vehicle_fuel_type": "Fuel type preference (Gasoline, Diesel, Hybrid, Electric)",
            
            # Purchase Requirements
            "quantity": "Number of vehicles needed",
            "budget_range": "Budget range, maximum budget, or price expectations",
            "financing_preference": "Financing options or payment preferences",
            "trade_in_vehicle": "Details of vehicle to trade in",
            
            # Timeline and Usage
            "delivery_timeline": "When the customer needs the vehicle",
            "primary_use": "How the customer plans to use the vehicle",
            "special_requirements": "Any special features, modifications, or requirements",
            
            # Business Context
            "urgency_level": "How urgent the purchase decision is",
            "decision_makers": "Who else is involved in the purchase decision",
            "previous_interactions": "Reference to previous conversations or quotes"
        }
        
        # Build comprehensive state for context extraction
        # This provides the extract_fields_from_conversation function with rich context
        # to intelligently identify information already provided in the conversation
        enhanced_state = await _build_conversation_state_for_extraction(
            customer_identifier,
            vehicle_requirements,
            additional_notes,
            conversation_context
        )
        
        # Extract all available context from conversation using intelligent analysis
        try:
            extracted_context = await extract_fields_from_conversation(
                enhanced_state,
                field_definitions,
                "generate_quotation"
            )
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Field extraction failed: {e}")
            extracted_context = {}  # Continue with empty context if extraction fails
        
        # Log extracted context for debugging and verification
        if extracted_context:
            logger.info(f"[GENERATE_QUOTATION] ✅ Extracted from conversation: {list(extracted_context.keys())}")
            for field, value in extracted_context.items():
                logger.debug(f"[GENERATE_QUOTATION] {field}: {value}")
        else:
            logger.info("[GENERATE_QUOTATION] ℹ️ No additional context extracted from conversation")
        
        logger.info(f"[GENERATE_QUOTATION] Extracted context: {extracted_context}")
        
        # Step 2: Intelligent customer lookup using extracted context
        customer_data = await _lookup_customer(customer_identifier)
        
        # If primary lookup fails, try using extracted context for alternative searches
        if not customer_data and extracted_context:
            logger.info("[GENERATE_QUOTATION] Primary customer lookup failed, trying extracted context")
            
            # Try alternative customer identifiers from extracted context
            for field in ["customer_email", "customer_phone", "customer_company"]:
                if field in extracted_context and extracted_context[field]:
                    logger.info(f"[GENERATE_QUOTATION] Trying customer lookup with {field}: {extracted_context[field]}")
                    customer_data = await _lookup_customer(extracted_context[field])
                    if customer_data:
                        logger.info(f"[GENERATE_QUOTATION] ✅ Found customer using {field}")
                        break
        
        if not customer_data:
            # Build intelligent prompt using extracted context
            context_info = ""
            if extracted_context:
                context_details = []
                for field, value in extracted_context.items():
                    if field.startswith("customer_") and value:
                        context_details.append(f"- {field.replace('customer_', '').title()}: {value}")
                
                if context_details:
                    context_info = f"\n\n**Information I found from our conversation:**\n" + "\n".join(context_details)
            
            return request_input(
                prompt=f"""📋 **Customer Information Needed**

I couldn't find customer information for "{customer_identifier}" in our CRM system.{context_info}

Please provide the customer's complete details:
- **Full Name**: 
- **Email**: 
- **Phone**: 
- **Company** (if applicable): 
- **Address**: 

Or provide a different customer identifier (name, email, or phone) that I can search for in our system.""",
                input_type="customer_information",
                context={
                    "source_tool": "generate_quotation",
                    "current_step": "customer_lookup",
                    "customer_identifier": customer_identifier,
                    "vehicle_requirements": vehicle_requirements,
                    "additional_notes": additional_notes,
                    "quotation_validity_days": quotation_validity_days,
                    "quotation_state": quotation_state,
                    "conversation_context": conversation_context,
                    "extracted_context": extracted_context,
                    "employee_id": employee_id
                }
            )
        
        # Step 3: Parse vehicle requirements using intelligent LLM-based approach with extracted context
        # This follows the project principle: "Use LLM intelligence rather than keyword matching or complex logic"
        # Enhanced to use comprehensive extracted context for better parsing accuracy
        try:
            vehicle_criteria = await _parse_vehicle_requirements_with_llm(
                vehicle_requirements, 
                extracted_context
            )
            
            # Enhance vehicle criteria with additional context from conversation
            if extracted_context:
                _enhance_vehicle_criteria_with_extracted_context(vehicle_criteria, extracted_context)
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Vehicle parsing failed: {e}")
            # Fallback to basic parsing
            vehicle_criteria = {"make": "", "model": "", "type": ""}
        
        # Get available inventory for dynamic matching and fallback messages
        try:
            available_inventory = await _get_available_makes_and_models()
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Failed to get inventory: {e}")
            available_inventory = []
        
        # Enhance criteria with fuzzy matching to handle variations in brand/model names
        try:
            enhanced_criteria = await _enhance_vehicle_criteria_with_fuzzy_matching(
                vehicle_criteria, 
                available_inventory
            )
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Fuzzy matching failed: {e}")
            enhanced_criteria = vehicle_criteria  # Use original criteria as fallback
        
        # Look up vehicles based on enhanced criteria
        vehicles = await _lookup_vehicle_by_criteria(enhanced_criteria, limit=10)
        
        if not vehicles:
            # Generate dynamic inventory suggestions based on actual available stock
            inventory_suggestions = _generate_inventory_suggestions(available_inventory)
            
            return request_input(
                prompt=f"""🚗 **Vehicle Information Needed**

I couldn't find vehicles matching "{vehicle_requirements}" in our current inventory.

**Available Brands in Stock:**
{inventory_suggestions}

Please provide more specific details:
- **Make/Brand**: Choose from available brands above
- **Model**: Specific model name
- **Type**: (Sedan, SUV, Pickup, Hatchback, Coupe, etc.)
- **Year**: (2023, 2024, or range like 2022-2024)
- **Quantity**: How many vehicles needed?
- **Budget**: Price range if relevant

**Example**: "I need 2 Toyota Camry sedans, 2023 or newer, budget around 1.5M each"

Or let me know if you'd like to see detailed specifications for any specific brand.

**Helpful Tips:**
- Be as specific as possible (e.g., "2023 Toyota Camry Hybrid, white or silver")
- Include quantity if you need multiple vehicles
- Mention your budget range if relevant
- Let me know if you're flexible on certain features""",
                input_type="detailed_vehicle_requirements",
                context={
                    "source_tool": "generate_quotation",
                    "current_step": "vehicle_requirements",
                    "customer_identifier": customer_identifier,
                    "vehicle_requirements": vehicle_requirements,
                    "additional_notes": additional_notes,
                    "quotation_validity_days": quotation_validity_days,
                    "quotation_state": quotation_state,
                    "conversation_context": conversation_context,
                    "customer_data": customer_data,
                    "extracted_context": extracted_context,
                    "available_inventory": available_inventory,
                    "original_criteria": vehicle_criteria,
                    "enhanced_criteria": enhanced_criteria,
                    "employee_id": employee_id
                }
            )
        
        # Step 4: Get employee details with HITL fallback
        employee_data = await _lookup_employee_details(employee_id)
        if not employee_data:
            logger.error(f"[GENERATE_QUOTATION] Could not find employee details for {employee_id}")
            
            # HITL flow for employee data issues
            return request_input(
                prompt=f"""👤 **Employee Information Issue**

I couldn't retrieve your employee information from our system (ID: {employee_id}). This is needed to include your contact details in the quotation.

Please provide your information:
- **Full Name**: 
- **Position/Title**: 
- **Email Address**: 
- **Phone Number**: 
- **Branch/Location**: 

Alternatively, please contact your system administrator to update your employee profile.""",
                input_type="employee_information",
                context={
                    "source_tool": "generate_quotation",
                    "current_step": "employee_data",
                    "customer_identifier": customer_identifier,
                    "vehicle_requirements": vehicle_requirements,
                    "additional_notes": additional_notes,
                    "quotation_validity_days": quotation_validity_days,
                    "quotation_state": quotation_state,
                    "conversation_context": conversation_context,
                    "employee_id": employee_id,
                    "customer_data": customer_data,
                    "extracted_context": extracted_context
                }
            )
        
        # Step 5: For each vehicle, get pricing information
        vehicle_pricing = []
        for vehicle in vehicles[:3]:  # Limit to top 3 matches
            pricing = await _lookup_current_pricing(vehicle["id"])
            if pricing:
                vehicle_pricing.append({
                    "vehicle": vehicle,
                    "pricing": pricing
                })
        
        if not vehicle_pricing:
            # HITL flow for pricing issues - gather more specific vehicle information
            return request_input(
                prompt=f"""⚠️ **Pricing Information Issue**

I found vehicles matching your requirements but couldn't retrieve current pricing information. This might be due to:
- Vehicles temporarily out of stock
- Pricing updates in progress
- Specific configuration not available

**Available vehicles found:**
{_format_vehicle_list_for_hitl(vehicles[:5])}

Please help me by:
1. **Selecting specific vehicles** from the list above, or
2. **Providing alternative preferences** (different model, year, etc.), or
3. **Confirming if you'd like me to check with our sales team** for manual pricing

What would you prefer?""",
                input_type="pricing_resolution",
                context={
                    "source_tool": "generate_quotation",
                    "current_step": "pricing_issues",
                    "customer_identifier": customer_identifier,
                    "vehicle_requirements": vehicle_requirements,
                    "additional_notes": additional_notes,
                    "quotation_validity_days": quotation_validity_days,
                    "quotation_state": quotation_state,
                    "conversation_context": conversation_context,
                    "customer_data": customer_data,
                    "vehicles_found": vehicles,
                    "employee_data": employee_data,
                    "extracted_context": extracted_context,
                    "employee_id": employee_id
                }
            )
        
        # Step 6: Validate completeness and gather missing critical information via HITL
        try:
            missing_info = _identify_missing_quotation_information(
                customer_data, 
                vehicle_pricing, 
                extracted_context
            )
        except Exception as e:
            logger.warning(f"[GENERATE_QUOTATION] Missing info validation failed: {e}")
            missing_info = []  # Continue without additional validation if this fails
        
        if missing_info:
            return await _request_missing_information_via_hitl(
                missing_info,
                customer_data,
                vehicle_pricing,
                employee_data,
                extracted_context,
                additional_notes,
                quotation_validity_days,
                customer_identifier,
                vehicle_requirements,
                quotation_state,
                conversation_context,
                employee_id
            )
        
        # Step 7: Create quotation preview for confirmation
        try:
            quotation_preview = _create_quotation_preview(
                customer_data,
                vehicle_pricing,
                employee_data,
                additional_notes,
                quotation_validity_days
            )
        except Exception as e:
            logger.error(f"[GENERATE_QUOTATION] Failed to create quotation preview: {e}")
            return f"""❌ **Preview Generation Error**

I encountered an issue while creating the quotation preview. This might be due to:
- Invalid customer or vehicle data
- Missing required information
- Template formatting issues

**What you can do:**
✅ Verify all customer and vehicle information is complete
✅ Try generating the quotation again
✅ Contact technical support if the issue persists

**Error Details**: {str(e)}
**Error ID**: {datetime.now().strftime('%Y%m%d-%H%M%S')}"""
        
        # Step 7: Request approval via HITL
        # Ensure quotation state contains all necessary data for PDF generation
        quotation_state.update({
            "customer_data": customer_data,
            "vehicle_pricing": vehicle_pricing,
            "employee_data": employee_data,
            "additional_notes": additional_notes,
            "quotation_validity_days": quotation_validity_days,
            "employee_id": employee_id,
            "quotation_preview": quotation_preview,
            "ready_for_generation": True
        })
        
        context_data = {
            "source_tool": "generate_quotation",
            "current_step": "quotation_approval",
            "customer_identifier": customer_identifier,
            "vehicle_requirements": vehicle_requirements,
            "additional_notes": additional_notes,
            "quotation_validity_days": quotation_validity_days,
            "quotation_state": quotation_state,
            "conversation_context": conversation_context,
            "customer_data": customer_data,
            "vehicle_pricing": vehicle_pricing,
            "employee_data": employee_data,
            "employee_id": employee_id,
            "quotation_preview": quotation_preview
        }
        
        return request_approval(
            prompt=f"""📄 **QUOTATION APPROVAL REQUIRED**

{quotation_preview}

---

🎯 **READY FOR FINAL GENERATION**

I've prepared a comprehensive quotation based on the customer's requirements and our current inventory. Please review the details above carefully.

**Upon approval, the system will:**
✅ Generate a professional PDF quotation document
✅ Store the document securely in our Supabase storage
✅ Create a shareable link valid for 48 hours
✅ Record the quotation in our CRM system
✅ Track quotation status and expiry
✅ Enable customer access via secure link

**Important Notes:**
- Once generated, the quotation cannot be modified
- Customer will receive access to download the PDF
- Quotation will be tracked for follow-up and conversion
- All pricing is based on current inventory and rates

**Please confirm:** Are you ready to generate this quotation?""",
            context=context_data,
            approve_text="✅ Generate Quotation",
            reject_text="❌ Cancel & Revise"
        )
        
    except Exception as e:
        # Enhanced error handling with user-friendly messages and categorization
        error_msg = str(e)
        logger.error(f"[GENERATE_QUOTATION] Error in quotation generation: {e}")
        
        # Categorize errors and provide appropriate user-friendly messages
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return """⚠️ **Database Connection Issue**

I'm having trouble connecting to our database right now. This might be due to:
- Temporary network connectivity issues
- Database maintenance in progress
- High system load

**What you can do:**
✅ Wait a moment and try again
✅ Check if other system functions are working
✅ Contact IT support if the issue persists

**Error ID**: Please reference this timestamp when contacting support: {datetime.now().strftime('%Y%m%d-%H%M%S')}"""
        
        elif "permission" in error_msg.lower() or "access" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return """🔒 **Access Permission Issue**

It looks like there's a permission issue with your account. This could be due to:
- Your employee profile needs updating
- Insufficient privileges for quotation generation
- System role configuration issues

**What you can do:**
✅ Contact your system administrator
✅ Verify your employee status in the system
✅ Request quotation generation permissions

**Note**: Only authorized employees can generate customer quotations."""
        
        elif "pricing" in error_msg.lower() or "vehicle" in error_msg.lower():
            return """💰 **Inventory or Pricing Issue**

I encountered an issue while retrieving vehicle or pricing information. This might be due to:
- Inventory data being updated
- Pricing information temporarily unavailable
- Vehicle specifications being modified

**What you can do:**
✅ Try with different vehicle specifications
✅ Check if the vehicles are currently in stock
✅ Contact the inventory team for current availability

**Tip**: You can also try generating a quotation for alternative vehicle models."""
        
        elif "pdf" in error_msg.lower() or "generation" in error_msg.lower():
            return """📄 **PDF Generation Issue**

I encountered an issue while creating the PDF quotation. This might be due to:
- PDF generation service temporarily unavailable
- Template or formatting issues
- Storage system problems

**What you can do:**
✅ Try generating the quotation again
✅ Ensure all required information is complete
✅ Contact technical support if the issue persists

**Alternative**: You can request a manual quotation from the sales team."""
        
        else:
            # Generic error with helpful guidance
            return f"""❌ **Quotation Generation Error**

I encountered an unexpected issue while generating your quotation. 

**Error Details**: {error_msg}

**What you can do:**
✅ **Try again**: This might be a temporary issue
✅ **Check your inputs**: Ensure customer and vehicle information is correct
✅ **Contact support**: Reference error timestamp {datetime.now().strftime('%Y%m%d-%H%M%S')}

**Alternative Options:**
- Use the CRM query tool to verify customer and vehicle data
- Generate a manual quotation through the sales team
- Try generating quotations for different customers or vehicles

**Need immediate help?** Contact your system administrator or technical support team."""


def _create_quotation_preview(
    customer_data: dict,
    vehicle_pricing: List[dict],
    employee_data: dict,
    additional_notes: Optional[str],
    validity_days: int
) -> str:
    """Create a comprehensive text preview of the quotation for HITL confirmation."""
    
    # Generate quotation number for preview
    from datetime import datetime
    import random
    quotation_number = f"QT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    preview = f"""🏢 **PROFESSIONAL QUOTATION PREVIEW**

📋 **Quotation Details:**
- Quotation Number: {quotation_number}
- Date: {datetime.now().strftime('%B %d, %Y')}
- Valid Until: {(datetime.now().replace(day=datetime.now().day + validity_days)).strftime('%B %d, %Y')}

👤 **Customer Information:**
- **Name**: {customer_data.get('name', 'N/A')}
- **Email**: {customer_data.get('email', 'N/A')}
- **Phone**: {customer_data.get('phone', 'N/A')}
- **Company**: {customer_data.get('company', 'Individual Customer' if not customer_data.get('company') else customer_data.get('company'))}
- **Address**: {customer_data.get('address', 'To be provided')}

🚗 **Vehicle Selections:**"""
    
    total_base_price = 0
    total_final_price = 0
    total_savings = 0
    
    for i, item in enumerate(vehicle_pricing, 1):
        vehicle = item["vehicle"]
        pricing = item["pricing"]
        
        base_price = pricing.get('base_price', 0)
        final_price = pricing.get('final_price', 0)
        discount = base_price - final_price
        
        preview += f"""

**Option {i}: {vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('year', 'N/A')}**
   🔹 **Vehicle Details:**
      • Type: {vehicle.get('type', 'N/A')}
      • Color: {vehicle.get('color', 'Any available color')}
      • Transmission: {vehicle.get('transmission', 'N/A')}
      • Engine: {vehicle.get('engine', 'N/A')}
      • Stock Available: {vehicle.get('stock_quantity', 0)} units
   
   💰 **Pricing:**
      • Base Price: ₱{base_price:,.2f}
      • Final Price: ₱{final_price:,.2f}"""
        
        if discount > 0:
            preview += f"""
      • **You Save**: ₱{discount:,.2f}"""
        
        total_base_price += base_price
        total_final_price += final_price
        total_savings += discount
    
    # Add comprehensive pricing summary
    preview += f"""

💵 **Pricing Summary:**
- **Subtotal**: ₱{total_base_price:,.2f}"""
    
    if total_savings > 0:
        preview += f"""
- **Total Savings**: ₱{total_savings:,.2f}"""
    
    preview += f"""
- **🎯 TOTAL AMOUNT**: ₱{total_final_price:,.2f}
- **Payment Terms**: As per agreement
- **Delivery**: To be arranged upon order confirmation

👨‍💼 **Sales Representative:**
- **Name**: {employee_data.get('name', 'N/A')}
- **Position**: {employee_data.get('position', 'Sales Representative')}
- **Branch**: {employee_data.get('branch_name', 'Main Branch')}
- **Email**: {employee_data.get('email', 'N/A')}
- **Phone**: {employee_data.get('phone', 'N/A')}"""
    
    if additional_notes:
        preview += f"""

📝 **Additional Notes:**
{additional_notes}"""
    
    # Add terms and conditions preview
    preview += f"""

📋 **Terms & Conditions:**
- Quotation valid for {validity_days} days from date of issue
- Prices subject to change without prior notice
- Vehicle specifications may vary based on availability
- Final pricing confirmed upon order placement
- Delivery timeline to be confirmed upon order"""
    
    return preview


# =============================================================================
# TOOL REGISTRY
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
        generate_quotation,  # Professional PDF quotation generation with HITL flows (Task 5.1)
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
            generate_quotation,  # Employee only - professional PDF quotation generation (Task 5.1)
            # gather_further_details ELIMINATED - replaced by business-specific tools using revolutionary 3-field approach
        ]
    else:
        # Unknown users get no tools
        return []


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]
