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
from datetime import datetime
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
from config import get_settings

# LangGraph interrupt functionality for customer messaging confirmation
try:
    from langgraph.types import interrupt
    from langgraph.errors import GraphInterrupt
except ImportError:
    # Fallback for development/testing - create a mock interrupt function and exception
    def interrupt(message: str):
        return f"[MOCK_INTERRUPT] {message}"
    
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass



logger = logging.getLogger(__name__)

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


class UserContext:
    """Context manager for setting user context during tool execution"""


    def __init__(self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, user_type: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_type = user_type
        self.user_id_token = None
        self.conversation_id_token = None
        self.user_type_token = None


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


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context"""
    return current_user_id.get()


def get_current_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context"""
    return current_conversation_id.get()


def get_current_user_type() -> Optional[str]:
    """Get the current user type from context (employee, customer, unknown)"""
    return current_user_type.get()


async def get_current_employee_id() -> Optional[str]:
    """
    Get the current employee ID by resolving user_id to employee_id.

    IMPORTANT: This function handles the mapping from users table (user_id) to employees table (employee_id)
    which is needed for CRM queries that reference employee relationships.

    Returns:
        Optional[str]: Employee ID if user is an employee, None otherwise
    """
    user_id = get_current_user_id()
    if not user_id:
        return None

    try:
        engine = await _get_sql_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT employee_id FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            row = result.fetchone()
            if row and row[0]:
                logger.debug(f"Resolved user_id {user_id} to employee_id {row[0]}")
                return str(row[0])
            else:
                logger.debug(f"User {user_id} is not an employee or employee_id is null")
                return None

    except Exception as e:
        logger.error(f"Error resolving user_id to employee_id: {e}")
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
# DATABASE CONNECTION MANAGEMENT
# =============================================================================

_sql_engine = None
_sql_db = None

async def _get_sql_engine():
    """Get a reusable SQLAlchemy engine with proper connection pooling."""
    global _sql_engine
    if _sql_engine is not None:
        return _sql_engine

    try:
        settings = await get_settings()
        database_url = settings.supabase.postgresql_connection_string

        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        # Create engine with connection pooling to prevent connection exhaustion
        _sql_engine = create_engine(
            database_url,
            # Connection pool settings to prevent "Max client connections reached"
            pool_size=5,          # Number of connections to keep open
            max_overflow=10,      # Additional connections allowed beyond pool_size
            pool_timeout=30,      # Seconds to wait for connection from pool
            pool_recycle=3600,    # Recycle connections every hour
            pool_pre_ping=True,   # Verify connections before use
        )
        logger.info("SQLAlchemy engine created with connection pooling")
        return _sql_engine

    except Exception as e:
        logger.error(f"Failed to create SQL engine: {e}")
        raise


async def _get_sql_database() -> Optional[SQLDatabase]:
    """Get SQL database connection using LangChain's built-in SQLDatabase."""
    global _sql_db
    if _sql_db is not None:
        return _sql_db

    try:
        # Use the centralized engine which already has connection pooling configured
        engine = await _get_sql_engine()
        
        # Create SQLDatabase using the pooled engine
        _sql_db = SQLDatabase(engine=engine)
        logger.info("LangChain SQLDatabase created using pooled engine")
        return _sql_db

    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


async def close_database_connections():
    """Close all database connections to free up connection pool."""
    global _sql_engine, _sql_db
    
    try:
        if _sql_engine:
            _sql_engine.dispose()
            logger.info("SQLAlchemy engine disposed")
            _sql_engine = None
            
        if _sql_db:
            if hasattr(_sql_db, '_engine') and _sql_db._engine:
                _sql_db._engine.dispose()
            logger.info("LangChain SQLDatabase engine disposed")
            _sql_db = None
            
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


async def get_connection_pool_status():
    """Get current connection pool status for monitoring."""
    try:
        engine = await _get_sql_engine()
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
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

async def _lookup_customer(customer_identifier: str) -> Optional[dict]:
    """
    Look up customer information by customer identifier (UUID, name, or email).
    
    Args:
        customer_identifier: The customer UUID, name, or email address to look up
        
    Returns:
        Dict with customer information if found, None otherwise
    """
    try:
        engine = await _get_sql_engine()

        with engine.connect() as conn:
            # Determine lookup strategy based on identifier format
            if "@" in customer_identifier and "." in customer_identifier:
                # Email lookup
                logger.debug(f"Looking up customer by email: {customer_identifier}")
                result = conn.execute(
                    text("""
                        SELECT id, name, email, phone, mobile_number, company, 
                               is_for_business, created_at, updated_at
                        FROM customers 
                        WHERE email = :identifier
                    """),
                    {"identifier": customer_identifier}
                )
            elif len(customer_identifier.replace("-", "")) == 32 and customer_identifier.count("-") == 4:
                # UUID lookup
                logger.debug(f"Looking up customer by UUID: {customer_identifier}")
                result = conn.execute(
                    text("""
                        SELECT id, name, email, phone, mobile_number, company, 
                               is_for_business, created_at, updated_at
                        FROM customers 
                        WHERE id = :identifier
                    """),
                    {"identifier": customer_identifier}
                )
            else:
                # Name lookup (case-insensitive)
                logger.debug(f"Looking up customer by name: {customer_identifier}")
                result = conn.execute(
                    text("""
                        SELECT id, name, email, phone, mobile_number, company, 
                               is_for_business, created_at, updated_at
                        FROM customers 
                        WHERE name ILIKE :identifier
                    """),
                    {"identifier": customer_identifier}
                )
            
            row = result.fetchone()
            
            if row:
                # Split name into first and last name for backward compatibility
                full_name = (row[1] or "").strip()
                name_parts = full_name.split(maxsplit=1) if full_name else ["", ""]
                first_name = name_parts[0] if len(name_parts) > 0 else ""
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                
                customer_info = {
                    "customer_id": str(row[0]),
                    "id": str(row[0]),  # Include both for compatibility
                    "name": full_name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": row[2] or "",
                    "phone": row[3] or row[4] or "",  # Use phone or mobile_number, whichever is available
                    "mobile_number": row[4] or "",
                    "company": row[5] or "",
                    "is_for_business": row[6] or False,
                    "created_at": row[7],
                    "updated_at": row[8]
                }
                
                logger.debug(f"Found customer: {customer_info['customer_id']} - {customer_info['name']}")
                return customer_info
            else:
                logger.warning(f"Customer {customer_identifier} not found")
                return None

    except Exception as e:
        logger.error(f"Error looking up customer {customer_identifier}: {e}")
        return None


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
    Validate and analyze message content for appropriateness and completeness.
    
    Args:
        message_content: The message content to validate
        message_type: The type of message (follow_up, information, promotional, support)
        
    Returns:
        Dict with validation results: {"valid": bool, "errors": list, "warnings": list}
    """
    errors = []
    warnings = []
    
    # Basic content checks
    if not message_content or not message_content.strip():
        errors.append("Message content cannot be empty")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    content = message_content.strip()
    
    # Length validation by message type
    length_limits = {
        "follow_up": 1500,    # Concise follow-ups
        "information": 2000,  # More detailed information
        "promotional": 1200,  # Promotional content should be punchy
        "support": 2000       # Support may need detailed explanations
    }
    
    max_length = length_limits.get(message_type, 1500)
    if len(content) > max_length:
        errors.append(f"Message too long for {message_type} type. Maximum {max_length} characters, got {len(content)}")
    
    if len(content) < 20:
        warnings.append("Message is very short. Consider adding more context for better customer engagement")
    
    # Content quality checks
    if content.isupper():
        warnings.append("Message is in ALL CAPS - consider using normal capitalization for professional communication")
    
    # Check for basic professionalism
    unprofessional_indicators = ['!!!', '???', 'URGENT!!!', 'ASAP!!!']
    for indicator in unprofessional_indicators:
        if indicator in content.upper():
            warnings.append(f"Consider more professional language instead of '{indicator}'")
    
    # Message type specific validation
    if message_type == "promotional":
        if "offer" not in content.lower() and "discount" not in content.lower() and "special" not in content.lower():
            warnings.append("Promotional message might benefit from highlighting special offers or value propositions")
    
    elif message_type == "support":
        if "?" not in content:
            warnings.append("Support message might benefit from including specific questions or next steps")
    
    elif message_type == "follow_up":
        if "thank" not in content.lower() and "follow" not in content.lower():
            warnings.append("Follow-up message might benefit from expressing gratitude or referencing previous interaction")
    
    # Simple inappropriate content check (basic implementation)
    inappropriate_words = ['damn', 'hell', 'crap']  # Basic list - could be expanded
    for word in inappropriate_words:
        if word.lower() in content.lower():
            errors.append(f"Please use professional language. Consider replacing '{word}' with more appropriate terms")
    
    # Check for contact information leakage prevention
    if "@" in content and "email" not in content.lower():
        warnings.append("Message contains @ symbol - ensure you're not accidentally including internal email addresses")
    
    is_valid = len(errors) == 0
    
    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "character_count": len(content),
        "word_count": len(content.split()),
        "max_length": max_length
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
    Send a message to a customer with human-in-the-loop confirmation (Employee Only).
    
    Uses LangGraph's native interrupt functionality to pause execution and wait for
    human confirmation before delivering messages to customers.
    
    Args:
        customer_id: The customer identifier (UUID, name, or email address) to message
        message_content: The content of the message to send
        message_type: Type of message (follow_up, information, promotional, support)
    
    Returns:
        Final delivery result after human confirmation
    """
    try:
        # Check user type - only employees can use this tool
        user_type = get_current_user_type()
        if user_type not in ["employee", "admin"]:
            logger.warning(f"[CUSTOMER_MESSAGE] Non-employee user ({user_type}) attempted to use customer messaging tool")
            return "I apologize, but customer messaging is only available to employees. Please contact your administrator if you need assistance."
        
        # Validate inputs
        if not customer_id or not customer_id.strip():
            return "Error: Customer identifier is required to send a message."
        
        # Validate message type
        valid_types = ["follow_up", "information", "promotional", "support"]
        if message_type not in valid_types:
            message_type = "follow_up"  # Default fallback
            
        # Comprehensive message content validation
        validation_result = _validate_message_content(message_content, message_type)
        if not validation_result["valid"]:
            error_msg = "❌ **Message Validation Failed:**\n"
            for error in validation_result["errors"]:
                error_msg += f"• {error}\n"
            if validation_result["warnings"]:
                error_msg += "\n⚠️ **Additional Suggestions:**\n"
                for warning in validation_result["warnings"]:
                    error_msg += f"• {warning}\n"
            return error_msg.strip()
        
        logger.info(f"[CUSTOMER_MESSAGE] Employee {get_current_user_id()} initiated message to customer {customer_id}")
        
        # Validate customer exists and get customer information
        customer_info = await _lookup_customer(customer_id)
        if not customer_info:
            return f"Error: Customer '{customer_id}' not found. Please verify the customer identifier (name, email, or UUID) and try again."
        
        logger.info(f"[CUSTOMER_MESSAGE] Validated customer: {customer_info['first_name']} {customer_info['last_name']} ({customer_info['email']})")
        
        # Format the message with professional templates
        formatted_message = _format_message_by_type(message_content, message_type, customer_info)
        
        # Create comprehensive confirmation prompt for the interrupt
        interrupt_message = f"""🚨 **CUSTOMER MESSAGE CONFIRMATION REQUIRED**

📧 **Target Customer:**
  • Name: {customer_info['first_name']} {customer_info['last_name']}
  • Email: {customer_info['email']}
  • Phone: {customer_info.get('phone', 'N/A')}
  • Customer ID: {customer_id}

📝 **Message Details:**
  • Type: {message_type.replace('_', ' ').title()}
  • Character Count: {validation_result['character_count']}/{validation_result['max_length']}
  • Word Count: {validation_result['word_count']} words"""

        # Add warnings if any
        if validation_result["warnings"]:
            interrupt_message += f"\n\n⚠️ **Suggestions for Improvement:**"
            for warning in validation_result["warnings"]:
                interrupt_message += f"\n  • {warning}"

        interrupt_message += f"""

📄 **Formatted Message Preview:**
{'-' * 60}
{formatted_message}
{'-' * 60}

**Please respond with one of the following:**
• "APPROVE" - Send the message as shown above
• "CANCEL" - Cancel the message and do not send
• "MODIFY: [your changes]" - Suggest modifications to the message

What is your decision?"""

        logger.info(f"[CUSTOMER_MESSAGE] Triggering interrupt for confirmation - Customer: {customer_id}")
        
        # Use LangGraph's native interrupt to pause execution and wait for human input
        confirmation_response = interrupt(interrupt_message)
        
        # Process the confirmation response (execution resumes here after human input)
        if not confirmation_response:
            return "❌ **Message Cancelled:** No confirmation response received. The message was not sent."
        
        response_text = str(confirmation_response).upper().strip()
        customer_name = f"{customer_info['first_name']} {customer_info['last_name']}"
        
        logger.info(f"[CUSTOMER_MESSAGE] Processing confirmation response: {response_text[:50]}...")
        
        if response_text == "APPROVE":
            # Execute message delivery
            logger.info(f"[CUSTOMER_MESSAGE] Message approved - executing delivery to customer {customer_id}")
            
            # Simulate message delivery (Phase 2 implementation - will be replaced with actual delivery)
            import asyncio
            import random
            
            await asyncio.sleep(1)  # Simulate delivery delay
            delivery_successful = random.random() < 0.95  # 95% success rate
            
            if delivery_successful:
                logger.info(f"[CUSTOMER_MESSAGE] Message delivered successfully to customer {customer_id}")
                return f"""✅ **Message Delivered Successfully!**

The message has been delivered to {customer_name} ({customer_info['email']}).

**Message Type:** {message_type.replace('_', ' ').title()}
**Delivered at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The customer has been successfully contacted with your {message_type.replace('_', ' ')} message."""
            else:
                logger.warning(f"[CUSTOMER_MESSAGE] Message delivery failed to customer {customer_id}")
                return f"""❌ **Message Delivery Failed**

Failed to deliver message to {customer_name} - delivery service unavailable.

**Failed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please try sending the message again or contact the customer through alternative means."""

        elif response_text == "CANCEL":
            logger.info(f"[CUSTOMER_MESSAGE] Message cancelled by employee for customer {customer_id}")
            return f"""⭕ **Message Cancelled**

The message to {customer_name} has been cancelled and was not sent.

You can create a new message anytime using the customer messaging tool."""

        elif response_text.startswith("MODIFY:"):
            # Extract modification suggestion
            modification = str(confirmation_response)[7:].strip()  # Remove "MODIFY:" prefix
            logger.info(f"[CUSTOMER_MESSAGE] Employee requested modifications for customer {customer_id}: {modification}")
            return f"""🔄 **Modification Requested**

Your suggested changes: "{modification}"

Please use the customer messaging tool again with your revised message content. The system will validate and format your updated message for another confirmation cycle."""

        else:
            logger.warning(f"[CUSTOMER_MESSAGE] Invalid confirmation response: {confirmation_response}")
            return f"""❓ **Invalid Response**

Your response "{confirmation_response}" was not recognized.

Please respond with:
• "APPROVE" to send the message
• "CANCEL" to cancel the message  
• "MODIFY: [your changes]" to suggest modifications

The message was not sent. Please try the customer messaging tool again."""

    except GraphInterrupt:
        # Interrupt is expected behavior for human-in-the-loop - re-raise it
        logger.info(f"[CUSTOMER_MESSAGE] Triggering human-in-the-loop interrupt for customer messaging confirmation")
        raise
    except Exception as e:
        logger.error(f"Error in trigger_customer_message: {e}")
        return f"Sorry, I encountered an error while processing your customer message request. Please try again or contact support if the issue persists."


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
        trigger_customer_message
    ]


def get_tools_for_user_type(user_type: str = "employee"):
    """Get tools filtered by user type for access control."""
    if user_type == "customer":
        # Customers get full RAG access but restricted CRM access
        return [
            simple_query_crm_data,  # Will be internally filtered for table access
            simple_rag  # Full access
        ]
    elif user_type in ["employee", "admin"]:
        # Employees get full access to all tools including customer messaging
        return [
            simple_query_crm_data,  # Full access
            simple_rag,  # Full access  
            trigger_customer_message  # Employee only - customer outreach tool
        ]
    else:
        # Unknown users get no tools
        return []


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]
