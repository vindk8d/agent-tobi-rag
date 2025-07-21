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
from enum import Enum
from typing import Optional
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


class UserContext:
    """Context manager for setting user context during tool execution"""


    def __init__(self, user_id: Optional[str] = None, conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_id_token = None
        self.conversation_id_token = None


    def __enter__(self):
        # Set context variables
        if self.user_id:
            self.user_id_token = current_user_id.set(self.user_id)
            logger.debug(f"Set user context: user_id={self.user_id}")

        if self.conversation_id:
            self.conversation_id_token = current_conversation_id.set(self.conversation_id)
            logger.debug(f"Set user context: conversation_id={self.conversation_id}")

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        if self.user_id_token:
            current_user_id.reset(self.user_id_token)
            logger.debug("Reset user_id context")

        if self.conversation_id_token:
            current_conversation_id.reset(self.conversation_id_token)
            logger.debug("Reset conversation_id context")


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context"""
    return current_user_id.get()


def get_current_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context"""
    return current_conversation_id.get()


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
        settings = await get_settings()
        database_url = settings.supabase.postgresql_connection_string

        # Handle postgres:// to postgresql:// URL format
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        engine = create_engine(database_url)

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
        "conversation_id": current_conversation_id.get()
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
# SIMPLIFIED SQL DATABASE CONNECTION
# =============================================================================

_sql_db = None


async def _get_sql_database() -> Optional[SQLDatabase]:
    """Get SQL database connection using LangChain's built-in SQLDatabase."""
    global _sql_db
    if _sql_db is not None:
        return _sql_db

    try:
        settings = await get_settings()
        database_url = settings.supabase.postgresql_connection_string

        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        _sql_db = SQLDatabase.from_uri(database_url)
        return _sql_db

    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
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

        settings = await get_settings()
        database_url = settings.supabase.postgresql_connection_string

        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        engine = create_engine(database_url)

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

async def _generate_sql_query_simple(question: str, db: SQLDatabase) -> str:
    """
    Generate SQL query using LangChain best practice approach.
    No hard-coded keywords - let the LLM use natural language understanding.
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

        # Simple, clear prompt - following LangChain tutorial style
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

        # Include conversation and employee context if available
        context_section = ""
        if conversation_context:
            context_section = f"\nRecent conversation context:\n{conversation_context}\n"

        employee_section = ""
        if employee_context:
            employee_section = employee_context

        prompt = ChatPromptTemplate.from_template(system_template)

        # Simple LCEL chain - LangChain best practice
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
    Query CRM database using simplified, natural language approach.

    This tool follows LangChain best practices:
    - Relies on LLM natural language understanding
    - No hard-coded keyword matching
    - Simple, graceful error handling
    - Context-aware through conversation history
    - User-aware through employee context

    Args:
        question: Natural language business question
        time_period: Optional time period filter

    Returns:
        Natural language response based on data
    """
    try:
        logger.info(f"Processing CRM query: {question}")

        # Add time period to question if provided
        if time_period:
            question = f"{question} for {time_period}"

        # Get database connection
        db = await _get_sql_database()
        if not db:
            return "Sorry, I cannot access the database right now. Please try again later."

        # Generate SQL query using simplified approach
        sql_query = await _generate_sql_query_simple(question, db)
        logger.info(f"Generated query: {sql_query}")

        # Execute query
        raw_result = await _execute_sql_query_simple(sql_query, db)

        # Format result
        formatted_result = await _format_sql_result_simple(question, sql_query, raw_result)

        return formatted_result

    except Exception as e:
        logger.error(f"Error in simple_query_crm_data: {e}")
        return f"I encountered an issue while processing your question. Please try rephrasing it or ask for help."

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
        simple_rag
    ]


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]
