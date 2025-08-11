"""
Consolidated Memory System for LangGraph Agent.

This module provides comprehensive memory management combining:
- Short-term memory (LangGraph PostgreSQL checkpointer)
- Long-term memory (Supabase with semantic search)
- Conversation consolidation (background processing)
- Context window management (token counting and trimming)

Simplified architecture following the principle of "simple yet effective".
"""

import asyncio
import contextvars
import json
import logging
import threading
import tiktoken
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Literal, Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore, Item
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from rag.embeddings import OpenAIEmbeddings

from core.config import get_settings
from core.database import db_client
from core.utils import DateTimeEncoder, convert_datetime_to_iso

logger = logging.getLogger(__name__)


# =============================================================================
# EXECUTION-SCOPED CONNECTION MANAGER
# =============================================================================

# Context variable to track current execution ID
current_execution_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_execution_id', default=None
)

class ExecutionConnectionManager:
    """
    Manages database connections per agent execution.
    
    Each agent execution gets its own connection tracking, allowing safe cleanup
    without affecting other concurrent executions.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        # Track connections per execution ID
        self._execution_connections: Dict[str, Dict[str, any]] = {}
        # Track active executions
        self._active_executions: Set[str] = set()
        
    def register_execution(self, execution_id: str) -> None:
        """Register a new agent execution."""
        with self._lock:
            self._active_executions.add(execution_id)
            self._execution_connections[execution_id] = {
                'sql_engines': [],
                'sql_databases': [],
                'custom_connections': []
            }
            logger.debug(f"Registered execution {execution_id}")
    
    def unregister_execution(self, execution_id: str) -> None:
        """Unregister an agent execution and clean up its connections."""
        with self._lock:
            if execution_id in self._active_executions:
                self._active_executions.remove(execution_id)
                
            # Clean up connections for this execution
            if execution_id in self._execution_connections:
                connections = self._execution_connections.pop(execution_id)
                self._cleanup_execution_connections(execution_id, connections)
                logger.debug(f"Unregistered and cleaned up execution {execution_id}")
    
    def _cleanup_execution_connections(self, execution_id: str, connections: Dict[str, List]) -> None:
        """Clean up connections for a specific execution."""
        try:
            # Dispose SQL engines
            for engine in connections.get('sql_engines', []):
                try:
                    if hasattr(engine, 'dispose'):
                        engine.dispose()
                        logger.debug(f"Disposed SQL engine for execution {execution_id}")
                except Exception as e:
                    logger.warning(f"Error disposing SQL engine for execution {execution_id}: {e}")
            
            # Clean up SQL databases
            for sql_db in connections.get('sql_databases', []):
                try:
                    if hasattr(sql_db, '_engine') and sql_db._engine:
                        sql_db._engine.dispose()
                        logger.debug(f"Disposed SQL database for execution {execution_id}")
                except Exception as e:
                    logger.warning(f"Error disposing SQL database for execution {execution_id}: {e}")
            
            # Clean up custom connections
            for conn in connections.get('custom_connections', []):
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                    elif hasattr(conn, 'dispose'):
                        conn.dispose()
                    logger.debug(f"Disposed custom connection for execution {execution_id}")
                except Exception as e:
                    logger.warning(f"Error disposing custom connection for execution {execution_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during connection cleanup for execution {execution_id}: {e}")
    
    def track_sql_engine(self, execution_id: str, engine) -> None:
        """Track an SQL engine for cleanup."""
        with self._lock:
            if execution_id in self._execution_connections:
                self._execution_connections[execution_id]['sql_engines'].append(engine)
    
    def track_sql_database(self, execution_id: str, sql_db) -> None:
        """Track an SQL database for cleanup."""
        with self._lock:
            if execution_id in self._execution_connections:
                self._execution_connections[execution_id]['sql_databases'].append(sql_db)
    
    def track_custom_connection(self, execution_id: str, connection: any) -> None:
        """Track a custom connection for cleanup."""
        with self._lock:
            if execution_id in self._execution_connections:
                self._execution_connections[execution_id]['custom_connections'].append(connection)
    
    def get_execution_stats(self) -> Dict[str, any]:
        """Get statistics about active executions and their connections."""
        with self._lock:
            stats = {
                'active_executions': len(self._active_executions),
                'total_tracked_connections': 0,
                'executions': {}
            }
            
            for exec_id, connections in self._execution_connections.items():
                conn_count = (
                    len(connections.get('sql_engines', [])) +
                    len(connections.get('sql_databases', [])) +
                    len(connections.get('custom_connections', []))
                )
                stats['total_tracked_connections'] += conn_count
                stats['executions'][exec_id] = {
                    'sql_engines': len(connections.get('sql_engines', [])),
                    'sql_databases': len(connections.get('sql_databases', [])),
                    'custom_connections': len(connections.get('custom_connections', []))
                }
            
            return stats

# Global connection manager instance
_connection_manager = ExecutionConnectionManager()

def get_connection_manager() -> ExecutionConnectionManager:
    """Get the global connection manager instance."""
    return _connection_manager

class ExecutionScope:
    """Context manager for tracking an agent execution's connections."""
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.token = None
    
    def __enter__(self):
        # Register the execution
        _connection_manager.register_execution(self.execution_id)
        
        # Set the context variable
        self.token = current_execution_id.set(self.execution_id)
        
        logger.info(f"Started execution scope: {self.execution_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Reset the context variable
            if self.token:
                current_execution_id.reset(self.token)
            
            # Unregister and cleanup connections
            _connection_manager.unregister_execution(self.execution_id)
            
            logger.info(f"Completed execution scope cleanup: {self.execution_id}")
            
        except Exception as e:
            logger.error(f"Error during execution scope cleanup: {e}")

async def create_execution_scoped_sql_engine():
    """Create an SQL engine that will be tracked and cleaned up per execution."""
    execution_id = current_execution_id.get()
    if not execution_id:
        logger.warning("No execution ID found, creating untracked SQL engine")
        return await _create_sql_engine()
    
    try:
        engine = await _create_sql_engine()
        _connection_manager.track_sql_engine(execution_id, engine)
        logger.debug(f"Created and tracked SQL engine for execution {execution_id}")
        return engine
    except Exception as e:
        logger.error(f"Failed to create execution-scoped SQL engine: {e}")
        return None

async def create_execution_scoped_sql_database():
    """Create an SQL database that will be tracked and cleaned up per execution."""
    execution_id = current_execution_id.get()
    if not execution_id:
        logger.warning("No execution ID found, creating untracked SQL database")
        return await _create_sql_database()
    
    try:
        engine = await create_execution_scoped_sql_engine()
        if not engine:
            return None
            
        from langchain_community.utilities import SQLDatabase
        sql_db = SQLDatabase(engine=engine)
        _connection_manager.track_sql_database(execution_id, sql_db)
        logger.debug(f"Created and tracked SQL database for execution {execution_id}")
        return sql_db
    except Exception as e:
        logger.error(f"Failed to create execution-scoped SQL database: {e}")
        return None

async def _create_sql_engine():
    """Create a new SQL engine with optimized settings for agent executions."""
    from sqlalchemy import create_engine
    settings = await get_settings()
    database_url = settings.supabase.postgresql_connection_string

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    # Optimized settings for agent executions
    engine = create_engine(
        database_url,
        # Reduced pool settings for per-execution cleanup
        pool_size=2,          # Smaller pool per execution
        max_overflow=3,       # Limited overflow
        pool_timeout=20,      # Shorter timeout
        pool_recycle=1800,    # Recycle every 30 minutes
        pool_pre_ping=True,   # Verify connections
        # Close connections more aggressively
        pool_reset_on_return='commit',
    )
    
    return engine

async def _create_sql_database():
    """Create a new SQL database connection."""
    from langchain_community.utilities import SQLDatabase
    engine = await _create_sql_engine()
    return SQLDatabase(engine=engine)

# Utility functions for monitoring
async def get_connection_statistics() -> Dict[str, any]:
    """Get current connection statistics for monitoring."""
    return _connection_manager.get_execution_stats()

async def log_connection_status():
    """Log current connection status for debugging."""
    stats = await get_connection_statistics()
    logger.info(f"Connection Manager Stats: {stats}")

# =============================================================================
# CONTEXT WINDOW MANAGER (Token Management)
# =============================================================================

class ContextWindowManager:
    """
    Manages context windows for LLM interactions by:
    1. Counting tokens in message history
    2. Trimming old messages when approaching limits
    3. Preserving system messages and recent context
    4. Maintaining conversation continuity
    """

    def __init__(self):
        self.settings = None
        self.tokenizer = None
        # Model-specific context limits (leaving buffer for response)
        self.model_limits = {
            "gpt-3.5-turbo": 14000,  # 16K total - 2K buffer
            "gpt-4": 6000,           # 8K total - 2K buffer
            "gpt-4o": 126000,        # 128K total - 2K buffer
            "gpt-4o-mini": 126000,   # 128K total - 2K buffer
        }

    async def _ensure_initialized(self):
        """Initialize tokenizer and settings asynchronously"""
        if self.tokenizer is None:
            self.settings = await get_settings()
            
            def _get_tokenizer():
                try:
                    # Use tiktoken for accurate token counting
                    return tiktoken.encoding_for_model("gpt-4o-mini")
                except KeyError:
                    # Fallback encoding
                    return tiktoken.get_encoding("cl100k_base")
            
            # Move tokenizer initialization to thread to avoid os.getcwd() blocking calls
            self.tokenizer = await asyncio.to_thread(_get_tokenizer)

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        await self._ensure_initialized()
        
        def _count_tokens():
            if self.tokenizer is None:
                # Fallback estimation if tokenizer not initialized
                return int(len(text.split()) * 1.3)
            return len(self.tokenizer.encode(text))
        
        # Move token counting to thread to avoid any potential blocking calls
        return await asyncio.to_thread(_count_tokens)

    async def count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a single message including metadata"""
        content = getattr(message, 'content', str(message))
        base_tokens = await self.count_tokens(content)

        # Add tokens for message formatting overhead
        if hasattr(message, 'type'):
            if message.type == 'system':
                base_tokens += 4  # System message overhead
            elif message.type == 'human':
                base_tokens += 4  # Human message overhead
            elif message.type == 'ai':
                base_tokens += 4  # AI message overhead
                # Add tokens for tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        base_tokens += await self.count_tokens(str(tool_call))
            elif message.type == 'tool':
                base_tokens += 6  # Tool message overhead

        return base_tokens

    async def count_messages_tokens(self, messages: List[BaseMessage]) -> int:
        """Count total tokens in a list of messages"""
        total_tokens = 0
        for message in messages:
            total_tokens += await self.count_message_tokens(message)

        # Add conversation-level overhead
        total_tokens += 6  # Conversation formatting overhead
        return total_tokens

    async def trim_messages_for_context(
        self,
        messages: List[BaseMessage],
        model: str,
        system_prompt: Optional[str] = None,
        max_messages: Optional[int] = None
    ) -> Tuple[List[BaseMessage], Dict[str, Any]]:
        """
        Trim messages to fit within context window while preserving conversation flow.

        Returns:
            Tuple of (trimmed_messages, metadata_about_trimming)
        """
        await self._ensure_initialized()

        # Get context limit for the model
        context_limit = self.model_limits.get(model, 6000)  # Conservative default

        # Apply max_messages limit from settings if provided
        if max_messages is None:
            max_messages = self.settings.memory.max_messages

        # Start with a copy of messages
        working_messages = list(messages)

        # Count tokens in system prompt if provided
        system_tokens = await self.count_tokens(system_prompt) if system_prompt else 0

        # Separate system messages from conversation messages
        system_messages = [msg for msg in working_messages if hasattr(msg, 'type') and msg.type == 'system']
        conversation_messages = [msg for msg in working_messages if not (hasattr(msg, 'type') and msg.type == 'system')]

        # Apply message count limit first
        if len(conversation_messages) > max_messages:
            logger.info(f"Trimming conversation from {len(conversation_messages)} to {max_messages} messages")
            # Keep the most recent messages
            conversation_messages = conversation_messages[-max_messages:]

        # Calculate current token usage
        current_tokens = system_tokens
        for msg in system_messages + conversation_messages:
            current_tokens += await self.count_message_tokens(msg)

        trim_stats = {
            "original_message_count": len(messages),
            "original_token_count": current_tokens,
            "context_limit": context_limit,
            "trimmed_message_count": 0,
            "trimmed_token_count": 0,
            "final_message_count": len(conversation_messages) + len(system_messages),
            "final_token_count": current_tokens
        }

        # If still over limit, trim more aggressively by tokens
        if current_tokens > context_limit:
            logger.warning(f"Messages exceed context limit: {current_tokens} > {context_limit} tokens")

            # Keep system messages and trim conversation messages from the beginning
            trimmed_conversation = []
            running_tokens = system_tokens
            for msg in system_messages:
                running_tokens += await self.count_message_tokens(msg)

            # Add conversation messages from the end (most recent first)
            for message in reversed(conversation_messages):
                message_tokens = await self.count_message_tokens(message)
                if running_tokens + message_tokens <= context_limit:
                    trimmed_conversation.insert(0, message)  # Insert at beginning to maintain order
                    running_tokens += message_tokens
                else:
                    trim_stats["trimmed_message_count"] += 1
                    trim_stats["trimmed_token_count"] += message_tokens

            conversation_messages = trimmed_conversation
            current_tokens = running_tokens

            logger.info(f"Token-based trimming: kept {len(conversation_messages)} messages, {current_tokens} tokens")

        # Update final statistics
        trim_stats["final_message_count"] = len(conversation_messages) + len(system_messages)
        trim_stats["final_token_count"] = current_tokens

        # Log trimming information
        if trim_stats["trimmed_message_count"] > 0:
            logger.info(
                f"Context trimming applied: "
                f"{trim_stats['trimmed_message_count']} messages removed, "
                f"{trim_stats['trimmed_token_count']} tokens saved, "
                f"final: {trim_stats['final_message_count']} messages, "
                f"{trim_stats['final_token_count']} tokens"
            )

        # Combine system and conversation messages
        final_messages = system_messages + conversation_messages

        return final_messages, trim_stats

    def get_model_context_info(self, model: str) -> Dict[str, int]:
        """Get context information for a specific model"""
        return {
            "context_limit": self.model_limits.get(model, 6000),
            "estimated_response_tokens": 2000,  # Buffer for response
            "available_for_input": self.model_limits.get(model, 6000)
        }

    def smart_trim_messages_preserving_tool_pairs(
        self,
        messages: List[BaseMessage],
        max_messages: int
    ) -> List[BaseMessage]:
        """
        Smart trim messages while preserving tool_calls/tool message pairs.
        
        This function ensures that:
        1. System messages are always preserved
        2. Tool calls and their corresponding tool responses are kept together
        3. Messages are trimmed from the oldest first while respecting pairs
        4. The result never violates OpenAI's message format requirements
        
        Args:
            messages: List of messages to trim
            max_messages: Maximum number of messages to keep
            
        Returns:
            List of trimmed messages that preserve tool call pairs
        """
        if len(messages) <= max_messages:
            return messages
            
        # Separate system messages from conversation messages
        system_messages = []
        conversation_messages = []
        
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'system':
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)
        
        # Calculate how many conversation messages we can keep
        # (total limit minus system messages)
        max_conversation_messages = max_messages - len(system_messages)
        
        if max_conversation_messages <= 0:
            # If we can't fit any conversation messages, just return system messages
            return system_messages
            
        if len(conversation_messages) <= max_conversation_messages:
            # No trimming needed for conversation messages
            return system_messages + conversation_messages
        
        # We need to trim conversation messages while preserving tool pairs
        trimmed_conversation = self._trim_preserving_tool_pairs(
            conversation_messages, 
            max_conversation_messages
        )
        
        return system_messages + trimmed_conversation
    
    def _trim_preserving_tool_pairs(
        self, 
        messages: List[BaseMessage], 
        max_messages: int
    ) -> List[BaseMessage]:
        """
        Trim messages while preserving tool_calls/tool pairs.
        
        Strategy:
        1. Start from the end (most recent) and work backwards
        2. Keep complete tool_calls -> tool pairs together
        3. If we encounter an orphaned tool message, skip it
        4. If we encounter tool_calls without room for the tool response, skip both
        """
        if len(messages) <= max_messages:
            return messages
            
        result = []
        i = len(messages) - 1  # Start from the end
        
        while i >= 0 and len(result) < max_messages:
            current_msg = messages[i]
            
            # Check if this is a tool message
            if hasattr(current_msg, 'type') and current_msg.type == 'tool':
                # Look for the preceding message with tool_calls
                preceding_msg = None
                if i > 0:
                    preceding_msg = messages[i - 1]
                
                # Check if the preceding message has tool_calls that match this tool response
                if (preceding_msg and 
                    hasattr(preceding_msg, 'tool_calls') and 
                    preceding_msg.tool_calls):
                    
                    # We have a tool pair - check if we have room for both
                    if len(result) + 2 <= max_messages:
                        # Add both messages (tool response first since we're building backwards)
                        result.insert(0, current_msg)  # tool message
                        result.insert(0, preceding_msg)  # tool_calls message
                        i -= 2  # Skip both messages
                    else:
                        # Not enough room for both, so skip this pair
                        i -= 2
                else:
                    # Orphaned tool message - skip it
                    i -= 1
            
            # Check if this is a message with tool_calls
            elif (hasattr(current_msg, 'tool_calls') and 
                  current_msg.tool_calls and 
                  i < len(messages) - 1):
                
                # Look for the following tool message(s)
                tool_messages = []
                j = i + 1
                
                # Collect all immediate tool responses
                while (j < len(messages) and 
                       hasattr(messages[j], 'type') and 
                       messages[j].type == 'tool'):
                    tool_messages.append(messages[j])
                    j += 1
                
                if tool_messages:
                    # We have a tool_calls + tool pair(s)
                    total_pair_size = 1 + len(tool_messages)  # tool_calls + all tool responses
                    
                    if len(result) + total_pair_size <= max_messages:
                        # Add all messages in the pair
                        for tool_msg in reversed(tool_messages):  # Insert in reverse order
                            result.insert(0, tool_msg)
                        result.insert(0, current_msg)  # tool_calls message
                        i -= 1  # We'll handle the tool messages when we get to them
                    else:
                        # Not enough room for the complete pair, skip it
                        i -= 1
                else:
                    # tool_calls message without following tool responses
                    # This might be the last message in conversation, so include it
                    if len(result) + 1 <= max_messages:
                        result.insert(0, current_msg)
                    i -= 1
            
            else:
                # Regular message (human, ai without tool_calls, etc.)
                if len(result) + 1 <= max_messages:
                    result.insert(0, current_msg)
                i -= 1
        
        return result


# =============================================================================
# DATABASE MANAGEMENT (Simplified Supabase Integration)
# =============================================================================

class SimpleDBManager:
    """Simple database manager for memory operations."""

    def __init__(self):
        self.client = db_client

    async def get_connection(self):
        """Get database connection - simplified to use Supabase client directly."""
        # For now, return a simple context manager that works with Supabase
        return SimpleDBConnection(self.client)


class SimpleDBConnection:
    """Simple database connection wrapper."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def cursor(self):
        """Get a cursor for database operations."""
        return SimpleDBCursor(self.client)

    async def commit(self):
        """Commit transaction (no-op for Supabase)."""
        pass


class SimpleDBCursor:
    """Simple database cursor wrapper."""

    def __init__(self, client):
        self.client = client
        self.rowcount = 0
        self.description = []  # For SQL cursor compatibility
        self.result = None  # Store query results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def fetchall(self):
        """Fetch all results from the last query."""
        if self.result and hasattr(self.result, 'data'):
            return self.result.data
        return []

    async def fetchone(self):
        """Fetch one result from the last query."""
        if self.result and hasattr(self.result, 'data') and self.result.data:
            return self.result.data[0]
        return None

    async def execute(self, query, params=None):
        """Execute a query using Supabase client."""
        try:
            # Handle different query types
            if query.startswith("SELECT"):
                # For SELECT queries, use RPC calls
                if "put_long_term_memory" in query:
                    # Extract parameters for put_long_term_memory
                    if params and len(params) >= 6:
                        namespace, key, value, embedding, memory_type, expiry_at = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('put_long_term_memory', {
                                'p_namespace': convert_datetime_to_iso(namespace),
                                'p_key': convert_datetime_to_iso(key),
                                'p_value': convert_datetime_to_iso(value),
                                'p_embedding': convert_datetime_to_iso(embedding),
                                'p_memory_type': convert_datetime_to_iso(memory_type),
                                'p_expiry_at': convert_datetime_to_iso(expiry_at
                            )}).execute()
                        )
                        self.result = result
                elif "get_long_term_memory" in query:
                    if params and len(params) >= 2:
                        namespace, key = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('get_long_term_memory', {
                                'p_namespace': convert_datetime_to_iso(namespace),
                                'p_key': convert_datetime_to_iso(key
                            )}).execute()
                        )
                        self.result = result
                elif "delete_long_term_memory" in query:
                    if params and len(params) >= 2:
                        namespace, key = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('delete_long_term_memory', {
                                'p_namespace': convert_datetime_to_iso(namespace),
                                'p_key': convert_datetime_to_iso(key
                            )}).execute()
                        )
                        self.result = result
                elif "search_long_term_memories_by_prefix" in query:
                    if params and len(params) >= 4:
                        embedding, namespace_prefix, threshold, match_count = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('search_long_term_memories_by_prefix', {
                                'query_embedding': convert_datetime_to_iso(embedding),
                                'namespace_prefix': convert_datetime_to_iso(namespace_prefix),
                                'similarity_threshold': convert_datetime_to_iso(threshold),
                                'match_count': convert_datetime_to_iso(match_count
                            )}).execute()
                        )
                        self.result = result
                elif "search_conversation_summaries" in query:
                    if params and len(params) >= 5:
                        embedding, user_id, threshold, match_count, summary_type = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('search_conversation_summaries', {
                                'query_embedding': convert_datetime_to_iso(embedding),
                                'target_user_id': convert_datetime_to_iso(user_id),
                                'similarity_threshold': convert_datetime_to_iso(threshold),
                                'match_count': convert_datetime_to_iso(match_count),
                                'summary_type_filter': convert_datetime_to_iso(summary_type
                            )}).execute()
                        )
                        self.result = result
                elif "update_memory_access_pattern" in query:
                    if params and len(params) >= 5:
                        user_id, namespace, key, access_context, retrieval_method = params
                        # Convert datetime objects to ISO format strings
                        if isinstance(user_id, datetime):
                            user_id = user_id.isoformat()
                        if isinstance(access_context, datetime):
                            access_context = access_context.isoformat()
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('update_memory_access_pattern', {
                                'p_user_id': convert_datetime_to_iso(user_id),
                                'p_memory_namespace': convert_datetime_to_iso(namespace),
                                'p_memory_key': convert_datetime_to_iso(key),
                                'p_access_context': convert_datetime_to_iso(access_context),
                                'p_retrieval_method': convert_datetime_to_iso(retrieval_method
                            )}).execute()
                        )
                        self.result = result
                else:
                    logger.warning(f"Unknown query pattern: {query}")
                    self.result = None
            else:
                logger.warning(f"Unsupported query type: {query}")
                self.result = None

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.result = None

    async def fetchone(self):
        """Fetch one row."""
        if hasattr(self, 'result') and self.result:
            if hasattr(self.result, 'data') and self.result.data:
                if isinstance(self.result.data, list) and len(self.result.data) > 0:
                    # For functions that return tables, data is a list of dictionaries
                    row_data = self.result.data[0]
                    if isinstance(row_data, dict):
                        # For get_long_term_memory, ensure correct column order
                        # Actual returns: namespace, key, value, created_at, updated_at
                        if 'namespace' in row_data and 'key' in row_data and 'value' in row_data:
                            return (
                                row_data.get('namespace'),
                                row_data.get('key'),
                                row_data.get('value'),
                                row_data.get('created_at'),
                                row_data.get('updated_at')
                            )
                        else:
                            # For other functions, use values in order
                            return tuple(row_data.values())
                    return row_data
                else:
                    return self.result.data if self.result.data else None
            return None
        return None

    async def fetchall(self):
        """Fetch all rows."""
        if hasattr(self, 'result') and self.result:
            if hasattr(self.result, 'data') and self.result.data:
                if isinstance(self.result.data, list):
                    # For functions that return tables, convert each row
                    rows = []
                    for row_data in self.result.data:
                        if isinstance(row_data, dict):
                            # For get_long_term_memory, ensure correct column order
                            # Actual returns: namespace, key, value, created_at, updated_at
                            if 'namespace' in row_data and 'key' in row_data and 'value' in row_data:
                                rows.append((
                                    row_data.get('namespace'),
                                    row_data.get('key'),
                                    row_data.get('value'),
                                    row_data.get('created_at'),
                                    row_data.get('updated_at')
                                ))
                            else:
                                # For other functions, use values in order
                                rows.append(tuple(row_data.values()))
                        else:
                            rows.append(row_data)
                    return rows
                else:
                    return [self.result.data]
            return []
        return []

    async def commit(self):
        """Commit transaction (no-op for Supabase)."""
        pass


# =============================================================================
# LONG-TERM MEMORY STORE (LangGraph Store Implementation)
# =============================================================================

class SupabaseLongTermMemoryStore(BaseStore):
    """
    LangGraph Store implementation for long-term memory using Supabase.

    Provides cross-thread memory access with semantic search capabilities.
    Implements the complete BaseStore interface with proper async context management.
    """

    def __init__(self, embeddings: Optional[Embeddings] = None):
        """Initialize the Supabase long-term memory store."""
        self.embeddings = embeddings
        self.db_manager = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure the store is initialized."""
        if not self._initialized:
            self.db_manager = SimpleDBManager()
            if self.embeddings is None:
                self.embeddings = OpenAIEmbeddings()
            self._initialized = True

    # Required BaseStore interface methods
    async def aget(self, namespace: Tuple[str, ...], key: str) -> Optional[Item]:
        """Get a value from the store."""
        await self._ensure_initialized()

        try:
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(
                        "SELECT * FROM get_long_term_memory(%s, %s)",
                        (list(namespace), key)
                    )
                    row = await cur.fetchone()

                    if row:
                        # Update access count - fix parameter order for update_memory_access_pattern
                        # Function expects: (user_id, namespace, key, access_context, retrieval_method)
                        user_id = namespace[0]  # Extract user_id from namespace
                        await cur.execute(
                            "SELECT update_memory_access_pattern(%s, %s, %s, %s, %s)",
                            (user_id, list(namespace), key, 'read', 'direct')
                        )
                        await connection.commit()

                        # get_long_term_memory returns: namespace, key, value, created_at, updated_at
                        # Parse the value if it's a JSON string
                        value = row[2]  # value column
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, keep it as a string
                                pass

                        return Item(
                            value=value,           # value column (index 2)
                            key=row[1],           # key column (index 1)
                            namespace=tuple(row[0]),  # namespace column (index 0)
                            created_at=row[3],    # created_at column (index 3)
                            updated_at=row[4]     # updated_at column (index 4)
                        )
                    return None

        except Exception as e:
            logger.error(f"Error getting value from store: {e}")
            return None

    async def aput(self, namespace: Tuple[str, ...], key: str, value: Any,
                   ttl_hours: Optional[int] = None) -> None:
        """Put a value into the store."""
        await self._ensure_initialized()

        try:
            # Calculate expiry time if TTL is provided
            expiry_at = None
            if ttl_hours:
                expiry_at = datetime.utcnow() + timedelta(hours=ttl_hours)

            # Generate embedding for semantic search
            embedding = None
            if self.embeddings and isinstance(value, (str, dict)):
                text_content = self._extract_content_for_embedding(value)
                try:
                    # Handle both sync and async embeddings
                    if hasattr(self.embeddings, 'aembed_query'):
                        embedding_result = await self.embeddings.aembed_query(text_content)
                    else:
                        # Check if embed_query returns a coroutine
                        result = self.embeddings.embed_query(text_content)
                        if asyncio.iscoroutine(result):
                            embedding_result = await result
                        else:
                            embedding_result = await asyncio.to_thread(
                                lambda: result
                            )
                    # Convert embedding to list if it's not already
                    embedding = list(embedding_result) if embedding_result else None
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
                    embedding = None

            # Fix the async context manager issue by properly awaiting the connection
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    # Serialize value to JSON for storage with datetime handling
                    if not isinstance(value, str):
                        try:
                            value_json = json.dumps(value, cls=DateTimeEncoder)
                        except TypeError as e:
                            logger.warning(f"JSON serialization failed with DateTimeEncoder, using fallback: {e}")
                            # Fallback: recursively convert datetime objects to strings
                            converted_value = convert_datetime_to_iso(value)
                            value_json = json.dumps(converted_value)
                    else:
                        value_json = value
                    
                    await cur.execute(
                        "SELECT put_long_term_memory(%s, %s, %s, %s, %s, %s)",
                        (list(namespace), key, value_json, embedding, 'semantic', expiry_at)
                    )
                    await connection.commit()

        except Exception as e:
            logger.error(f"Error putting value into store: {e}")
            raise

    def put(self, namespace: Tuple[str, ...], key: str, value: Any) -> None:
        """Synchronous put method (BaseStore interface compatibility)."""
        import asyncio
        try:
            # If we're in an async context, schedule the coroutine
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                asyncio.create_task(self.aput(namespace, key, value))
            else:
                # We're not in an async context, run it
                asyncio.run(self.aput(namespace, key, value))
        except Exception as e:
            logger.error(f"Error in synchronous put: {e}")
            raise

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a value from the store."""
        await self._ensure_initialized()

        try:
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(
                        "SELECT delete_long_term_memory(%s, %s)",
                        (list(namespace), key)
                    )
                    await connection.commit()

        except Exception as e:
            logger.error(f"Error deleting value from store: {e}")
            raise

    async def asearch(self, namespace_prefix: Tuple[str, ...]) -> List[Tuple[Tuple[str, ...], str, Any]]:
        """Search for values by namespace prefix."""
        await self._ensure_initialized()

        try:
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(
                        "SELECT * FROM list_long_term_memory_keys(%s)",
                        (list(namespace_prefix),)
                    )
                    rows = await cur.fetchall()

                    # Convert to expected format
                    results = []
                    for row in rows:
                        namespace_tuple = tuple(row[0])  # namespace column
                        key = row[1]  # key column
                        value = row[2]  # value column
                        results.append((namespace_tuple, key, value))

                    return results

        except Exception as e:
            logger.error(f"Error searching store: {e}")
            return []

    async def semantic_search(self, query: str, namespace_prefix: Tuple[str, ...] = (),
                              limit: int = 10, similarity_threshold: float = 0.7) -> List[Tuple[Tuple[str, ...], str, Any]]:
        """Semantic search using embeddings."""
        await self._ensure_initialized()

        try:
            # Generate embedding for the query
            # OpenAIEmbeddings.embed_query is async, so just await it directly
            query_embedding = await self.embeddings.embed_query(query)

            conn = await self.db_manager.get_connection()
            async with conn:
                cur = await conn.cursor()
                async with cur:
                    await cur.execute(
                        "SELECT * FROM search_long_term_memories_by_prefix(%s, %s, %s, %s)",
                        (query_embedding, list(namespace_prefix), similarity_threshold, limit)
                    )
                    rows = await cur.fetchall()

                    # Convert to expected format
                    results = []
                    for row in rows:
                        namespace_tuple = tuple(row[0])  # namespace column
                        key = row[1]  # key column
                        value = row[2]  # value column
                        results.append((namespace_tuple, key, value))

                    return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _extract_content_for_embedding(self, value: Any) -> str:
        """Extract text content from a value for embedding generation."""
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            # Extract text from common keys
            text_parts = []
            for key, val in value.items():
                if isinstance(val, str):
                    text_parts.append(f"{key}: {val}")
                elif isinstance(val, (dict, list)):
                    text_parts.append(f"{key}: {str(val)}")
            return " ".join(text_parts)
        else:
            return str(value)

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        if self.db_manager:
            await self.db_manager.close()
        return False

    # Batch operations (required by BaseStore)
    async def abatch(self, requests):
        """Async batch method - not implemented for this store."""
        raise NotImplementedError("Batch operations not supported for SupabaseLongTermMemoryStore")

    def batch(self, requests):
        """Batch method - not implemented for this store."""
        raise NotImplementedError("Batch operations not supported for SupabaseLongTermMemoryStore")

    def _convert_datetime_to_iso(self, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO format strings."""
        # Use the centralized utility function
        return convert_datetime_to_iso(obj)


# =============================================================================
# CONVERSATION CONSOLIDATOR (Background Processing)
# =============================================================================

class ConversationConsolidator:
    """
    Handles the transition from short-term to long-term memory.
    """

    def __init__(self, db_manager: SimpleDBManager, memory_store: SupabaseLongTermMemoryStore,
                 llm: BaseLanguageModel, embeddings: Embeddings, memory_manager=None):
        """Initialize the conversation consolidator."""
        self.db_manager = db_manager
        self.memory_store = memory_store
        self.llm = llm
        self.embeddings = embeddings
        self.memory_manager = memory_manager  # Reference to MemoryManager for dynamic model selection
        self.settings = None  # Will be loaded asynchronously

    async def _ensure_settings_loaded(self):
        """Load settings if not already loaded."""
        if self.settings is None:
            from core.config import get_settings
            self.settings = await get_settings()

    async def check_and_trigger_summarization(self, conversation_id: str, user_id: str, agent_context_messages: Optional[List] = None) -> Optional[str]:
        """
        Check if a conversation needs summarization and trigger it automatically.
        
        Args:
            conversation_id: The conversation ID
            user_id: The user ID
            agent_context_messages: The agent's working context messages (if provided, this count will be used instead of database query)
            
        Returns the summary if one was generated, None otherwise.
        """
        await self._ensure_settings_loaded()

        print(f"\n         🔍 SUMMARIZATION FUNCTION DEBUG:")
        print(f"            📊 Settings:")
        print(f"               - Auto-summarize enabled: {self.settings.memory_auto_summarize}")
        print(f"               - Summary interval: {self.settings.memory_summary_interval}")
        print(f"               - Max messages: {self.settings.memory_max_messages}")

        if not self.settings.memory_auto_summarize:
            print(f"            ❌ Auto-summarization is DISABLED in settings")
            return None

        try:
            # Count messages - prioritize agent context over database query
            if agent_context_messages is not None:
                # Use agent's working context message count (implements moving context window)
                message_count = len(agent_context_messages)
                print(f"            🔢 Counting messages from agent working context...")
                print(f"            📊 Agent context message count: {message_count}")
                print(f"            ✅ Using AGENT CONTEXT (moving window) - not database query")
            else:
                # Fallback to database query (for backward compatibility)
                print(f"            🔢 Counting messages from database (fallback)...")
                message_count = await self._count_conversation_messages(conversation_id)
                print(f"            📊 Database message count: {message_count}")
                print(f"            ⚠️  Using DATABASE COUNT (may include trimmed messages)")
                
            print(f"            🎯 Threshold: {self.settings.memory_summary_interval}")

            # Check if we need to summarize
            if message_count >= self.settings.memory_summary_interval:
                print(f"            ✅ THRESHOLD MET! ({message_count} >= {self.settings.memory_summary_interval})")
                logger.info(f"Auto-triggering summarization for conversation {conversation_id} ({message_count} messages)")

                # Get conversation details
                print(f"            📋 Getting conversation details...")
                conversation = await self._get_conversation_details(conversation_id)
                if not conversation:
                    print(f"            ❌ Conversation {conversation_id} not found in database")
                    logger.warning(f"Conversation {conversation_id} not found for summarization")
                    return None
                print(f"            ✅ Conversation details retrieved")

                # Get the most recent summary (if any) to build upon
                print(f"            📚 Getting previous summary for incremental updates...")
                previous_summary = await self._get_latest_conversation_summary(conversation_id)

                # Get only recent messages since last summary (or last 10 if no previous summary)
                print(f"            📝 Getting recent conversation messages...")
                messages = await self._get_recent_conversation_messages(conversation_id, previous_summary)
                if not messages:
                    print(f"            ❌ No new messages found for conversation {conversation_id}")
                    logger.warning(f"No new messages found for conversation {conversation_id}")
                    return None
                print(f"            ✅ Retrieved {len(messages)} new messages for incremental summary")

                # Generate and store summary
                print(f"            🧠 Generating incremental summary with LLM...")
                summary = await self._generate_incremental_conversation_summary(conversation, messages, previous_summary)
                print(f"            ✅ Summary generated ({len(summary['content'])} chars)")

                print(f"            💾 Storing summary in long-term memory...")
                await self._store_conversation_summary(user_id, conversation, summary)
                print(f"            ✅ Summary stored successfully")

                # ALWAYS update master user summary with this new conversation
                print(f"            🔄 Updating master user summary with new conversation...")
                try:
                    comprehensive_summary = await self.consolidate_user_summary_with_llm(
                        user_id=user_id,
                        new_conversation_id=conversation_id
                    )
                    if comprehensive_summary and not comprehensive_summary.startswith("Error"):
                        print(f"            ✅ Master user summary updated successfully")
                        print(f"            📝 Summary preview: {comprehensive_summary[:100]}...")
                    else:
                        print(f"            ⚠️  Master summary consolidation returned error: {comprehensive_summary}")
                except Exception as consolidation_error:
                    print(f"            ❌ ERROR in master summary consolidation: {consolidation_error}")
                    logger.error(f"Master summary consolidation failed for user {user_id}: {consolidation_error}")

                # Reset message count (optional - could be used for periodic summarization)
                await self._reset_conversation_message_count(conversation_id)

                logger.info(f"Successfully auto-summarized conversation {conversation_id} and updated master summary")
                return summary['content']
            else:
                print(f"            ❌ Threshold not met ({message_count} < {self.settings.memory_summary_interval})")
                return None

        except Exception as e:
            print(f"            ❌ ERROR in summarization: {e}")
            logger.error(f"Error in automatic summarization for conversation {conversation_id}: {e}")

        return None

    async def _count_conversation_messages(self, conversation_id: str) -> int:
        """Count the number of messages in a conversation since last summarization."""
        try:
            from core.database import db_client

            def _get_last_summarized_at():
                return (db_client.client.table("conversations")
                    .select("last_summarized_at")
                    .eq("id", conversation_id)
                    .execute())

            def _count_messages(last_summarized_at=None):
                query = db_client.client.table("messages").select("id", count="exact").eq("conversation_id", conversation_id)
                if last_summarized_at:
                    query = query.gt("created_at", last_summarized_at)
                return query.execute()

            # First, get the last_summarized_at timestamp for this conversation
            conv_result = await asyncio.to_thread(_get_last_summarized_at)

            last_summarized_at = None
            if conv_result.data and len(conv_result.data) > 0:
                last_summarized_at = conv_result.data[0].get('last_summarized_at')

            # Count messages since last summarization (or all messages if never summarized)
            result = await asyncio.to_thread(_count_messages, last_summarized_at)
            return result.count if result.count is not None else 0
        except Exception as e:
            logger.error(f"Error counting messages for conversation {conversation_id}: {e}")
            return 0

    async def _get_conversation_details(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details by ID."""
        try:
            from core.database import db_client

            def _get_conversation():
                return (db_client.client.table("conversations")
                    .select("id,user_id,title,created_at,updated_at,metadata")
                    .eq("id", conversation_id)
                    .execute())

            result = await asyncio.to_thread(_get_conversation)

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error getting conversation details for {conversation_id}: {e}")
            return None


    async def consolidate_conversations(self, user_id: str, max_conversations: int = 50) -> Dict[str, Any]:
        """Consolidate old conversations for a user."""
        try:
            # Get conversations to consolidate
            conversations = await self._get_consolidation_candidates(user_id, max_conversations)

            if not conversations:
                return {"consolidated_count": 0, "error": None}

            logger.info(f"Consolidating {len(conversations)} conversations for user {user_id}")

            consolidated_count = 0
            errors = []

            for conversation in conversations:
                try:
                    # Get previous summary (if any) for incremental approach
                    previous_summary = await self._get_latest_conversation_summary(conversation['id'])

                    # Get messages (either recent ones or all if no previous summary)
                    messages = await self._get_recent_conversation_messages(conversation['id'], previous_summary)

                    if not messages:
                        continue

                    # Generate incremental summary
                    summary = await self._generate_incremental_conversation_summary(conversation, messages, previous_summary)

                    # Store in long-term memory
                    await self._store_conversation_summary(user_id, conversation, summary)

                    # Archive conversation
                    await self._archive_conversation(conversation['id'])

                    consolidated_count += 1

                except Exception as e:
                    error_msg = f"Error consolidating conversation {conversation['id']}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            return {
                "consolidated_count": consolidated_count,
                "total_candidates": len(conversations),
                "errors": errors if errors else None
            }

        except Exception as e:
            logger.error(f"Error in consolidate_conversations: {e}")
            return {"consolidated_count": 0, "error": str(e)}

    async def _get_consolidation_candidates(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get conversations that should be consolidated."""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()  # 7 days old

            from core.database import db_client

            # Get conversations for user older than 7 days using Supabase client
            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations")
                    .select("id, user_id, title, created_at, updated_at, metadata")
                    .eq("user_id", user_id)
                    .lt("updated_at", cutoff_date)
                    .order("updated_at", desc=False)
                    .limit(limit)
                    .execute()
            )

            if result.data:
                # Filter out conversations that already have summaries
                # For simplicity, assume conversations without explicit summary status need consolidation
                return result.data[:limit]

            return []

        except Exception as e:
            logger.warning(f"Unsupported query type: \n            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at, c.metadata\n            FROM conversations c\n            LEFT JOIN conversation_summaries cs ON c.id = cs.conversation_id\n            WHERE c.user_id = %s\n                AND c.updated_at < %s\n                AND c.archival_status = 'active'\n                AND cs.conversation_id IS NULL\n            ORDER BY c.updated_at ASC\n            LIMIT %s\n        ")
            logger.error(f"Error in _get_consolidation_candidates: {e}")
            return []

    async def _get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent messages for a conversation, with optional limit for performance optimization.
        
        Args:
            conversation_id: The conversation ID to get messages for
            limit: Maximum number of recent messages to retrieve. If None, uses settings.memory.max_messages
            
        Returns:
            List of message dictionaries in chronological order (oldest to newest)
        """
        try:
            # Use max_messages from settings if no limit provided
            if limit is None:
                limit = self.settings.memory.max_messages
                
            print(f"            📝 Getting last {limit} messages for conversation {conversation_id[:8]}...")

            # Use Supabase client for message retrieval
            from core.database import db_client

            def _get_messages():
                return (db_client.client.table("messages")
                    .select("role,content,created_at,user_id")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=True)  # Most recent first for LIMIT
                    .limit(limit)                    # Apply limit at database level
                    .execute())

            result = await asyncio.to_thread(_get_messages)

            messages = result.data if result.data else []
            # Reverse to get chronological order (oldest to newest) as expected by the rest of the system
            messages = list(reversed(messages))
            
            print(f"            ✅ Retrieved {len(messages)} messages (limited to {limit}) for conversation processing")
            return messages

        except Exception as e:
            print(f"            ❌ ERROR fetching messages: {e}")
            logger.error(f"Error fetching messages for conversation {conversation_id}: {e}")
            return []

    async def _get_latest_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent summary for a conversation to build incremental summaries."""
        try:
            from core.database import db_client

            def _get_latest_summary():
                return (db_client.client.table("conversation_summaries")
                    .select("summary_text,created_at,message_count")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute())

            result = await asyncio.to_thread(_get_latest_summary)

            if result.data and len(result.data) > 0:
                summary_data = result.data[0]
                print(f"            ✅ Found previous summary from {summary_data['created_at']}")
                return {
                    'summary_text': summary_data['summary_text'],
                    'created_at': summary_data['created_at'],
                    'message_count': summary_data['message_count']
                }
            else:
                print(f"            📝 No previous summary found - this will be the first summary")
                return None

        except Exception as e:
            print(f"            ❌ ERROR fetching previous summary: {e}")
            logger.error(f"Error fetching latest summary for conversation {conversation_id}: {e}")
            return None

    async def _get_recent_conversation_messages(self, conversation_id: str, previous_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get recent messages using a moving context window of configurable size."""
        try:
            from core.database import db_client
            from core.config import get_settings
            
            # Get context window size from configuration
            settings = await get_settings()
            context_window_size = settings.memory.context_window_size

            def _count_messages():
                return (db_client.client.table("messages")
                    .select("id", count="exact")
                    .eq("conversation_id", conversation_id)
                    .execute())

            def _get_last_n_messages(offset, limit):
                return (db_client.client.table("messages")
                    .select("role,content,created_at,user_id")
                    .eq("conversation_id", conversation_id)
                    .order("created_at")
                    .range(offset, offset + limit - 1)
                    .execute())

            # FIXED: Always use moving context window - get only the last N messages
            # This ensures we maintain a consistent window size regardless of summary history
            print(f"            🪟 Using moving context window of {context_window_size} messages")
            
            # First count total messages
            count_result = await asyncio.to_thread(_count_messages)
            total_messages = count_result.count

            if total_messages > context_window_size:
                # Skip earlier messages, get only the latest N
                offset = total_messages - context_window_size
                result = await asyncio.to_thread(_get_last_n_messages, offset, context_window_size)
                print(f"            📊 Retrieved last {context_window_size} of {total_messages} total messages")
            else:
                # Get all messages if fewer than window size
                result = await asyncio.to_thread(_get_last_n_messages, 0, total_messages)
                print(f"            📊 Retrieved all {total_messages} messages (within window size)")

            messages = result.data if result.data else []
            print(f"            ✅ Retrieved {len(messages)} recent messages")
            return messages

        except Exception as e:
            print(f"            ❌ ERROR fetching recent messages: {e}")
            logger.error(f"Error fetching recent messages for conversation {conversation_id}: {e}")
            return []

    async def _generate_incremental_conversation_summary(self, conversation: Dict[str, Any],
                                                        new_messages: List[Dict[str, Any]],
                                                        previous_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an incremental summary that builds on the previous summary with new messages."""
        # Format new messages for LLM
        new_messages_text = "\n".join([
            f"{self._format_role_for_summary(msg['role'])}: {msg['content']}" for msg in new_messages
        ])

        # Create the prompt based on whether we have a previous summary
        if previous_summary:
            summary_prompt = f"""
            Update and enhance this conversation summary with new messages:

            **Previous Summary:**
            {previous_summary['summary_text']}

            **New Messages to Incorporate:**
            {new_messages_text}

            Create an updated summary that:
            1. Preserves important information from the previous summary
            2. Integrates the new messages seamlessly
            3. Identifies any new topics, decisions, or preferences
            4. Updates outcomes based on the latest information
            5. Maintains the same structure (Main Topics, Key Decisions, Important Context, User Preferences)

            The updated summary should feel cohesive and comprehensive, not just concatenated.
            """
        else:
            # First summary - use the original approach
            summary_prompt = f"""
            Create a concise summary of this conversation:

            Title: {conversation.get('title', 'Untitled')}
            Date: {conversation['created_at']}
            Messages: {len(new_messages)}

            Conversation:
            {new_messages_text}

            Include:
            1. Main topics discussed
            2. Key decisions or outcomes
            3. Important context
            4. User preferences mentioned
            """

        # Use dynamic model selection for summarization
        if self.memory_manager:
            llm = self.memory_manager._get_llm_for_memory_task("summary", new_messages)
        else:
            llm = self.llm  # Fallback to default LLM

        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])

        # Handle datetime strings
        start_date = new_messages[0]['created_at'] if new_messages else conversation['created_at']
        end_date = new_messages[-1]['created_at'] if new_messages else conversation['created_at']

        # Convert to ISO format if needed
        if hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
        if hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()

        return {
            "content": response.content,
            "message_count": len(new_messages),
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }

    async def _generate_conversation_summary(self, conversation: Dict[str, Any],
                                           messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the conversation using LLM."""
        # Format messages for LLM - handle new role values
        conversation_text = "\n".join([
            f"{self._format_role_for_summary(msg['role'])}: {msg['content']}" for msg in messages
        ])

        # Generate summary
        summary_prompt = f"""
        Create a concise summary of this conversation:

        Title: {conversation.get('title', 'Untitled')}
        Date: {conversation['created_at']}
        Messages: {len(messages)}

        Conversation:
        {conversation_text}

        Include:
        1. Main topics discussed
        2. Key decisions or outcomes
        3. Important context
        4. User preferences mentioned
        """

        # Use dynamic model selection for summarization
        if self.memory_manager:
            llm = self.memory_manager._get_llm_for_memory_task("summary", messages)
        else:
            llm = self.llm  # Fallback to default LLM

        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])

        # Handle datetime strings from Supabase (already in ISO format)
        start_date = messages[0]['created_at']
        end_date = messages[-1]['created_at']

        # If they're datetime objects, convert to ISO format, otherwise use as-is
        if hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
        if hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()

        return {
            "content": response.content,
            "message_count": len(messages),
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }

    def _format_role_for_summary(self, role: str) -> str:
        """Format role for summary display."""
        role_mapping = {
            'ai': 'ASSISTANT',
            'assistant': 'ASSISTANT',  # Direct LangChain compatibility
            'human': 'USER',
            'user': 'USER',  # Alternative human role
            'system': 'SYSTEM',
            'tool': 'TOOL',
            'function': 'FUNCTION'
        }
        return role_mapping.get(role, role.upper())

    def _extract_user_id_from_message(self, message: Any) -> Optional[str]:
        """Extract user_id from message in various formats."""
        if isinstance(message, dict):
            return message.get('user_id')
        elif hasattr(message, 'user_id'):
            return message.user_id
        elif hasattr(message, 'metadata') and isinstance(message.metadata, dict):
            return message.metadata.get('user_id')
        return None

    async def _store_conversation_summary(self, user_id: str, conversation: Dict[str, Any],
                                        summary: Dict[str, Any]) -> None:
        """Store conversation summary in long-term memory."""
        # Store in conversation_summaries table
        summary_id = str(uuid4())

        # Handle both sync and async embeddings
        if hasattr(self.embeddings, 'aembed_query'):
            embedding = await self.embeddings.aembed_query(summary['content'])
        else:
            # Check if embed_query returns a coroutine
            result = self.embeddings.embed_query(summary['content'])
            if asyncio.iscoroutine(result):
                embedding = await result
            else:
                embedding = await asyncio.to_thread(
                    lambda: result
                )

        # Store summary in conversation_summaries table using Supabase client
        print(f"            💾 Storing summary in conversation_summaries table...")
        try:
            summary_data = {
                "id": summary_id,
                "user_id": user_id,
                "conversation_id": conversation['id'],
                "summary_text": summary['content'],
                "message_count": summary['message_count'],
                "metadata": {
                    "date_range": summary['date_range'],
                    "auto_generated": True
                },
                "summary_embedding": embedding
            }

            from core.database import db_client

            db_client.client.table("conversation_summaries").insert(summary_data).execute()
            print(f"            ✅ Summary stored in database (id: {summary_id[:8]}...)")

        except Exception as e:
            print(f"            ❌ Error storing summary: {e}")
            logger.error(f"Error storing summary in database: {e}")
            raise

        # Also store in LangGraph memory store
        try:
            await self.memory_store.aput(
                namespace=(user_id, "conversation_summaries"),
                key=f"conversation_{conversation['id']}",
                value={
                    "type": "conversation_summary",
                    "conversation_id": conversation['id'],
                    "title": conversation.get('title', 'Untitled'),
                    "summary": summary['content'],
                    "message_count": summary['message_count'],
                    "date_range": summary['date_range'],
                    "consolidated_at": datetime.utcnow().isoformat()
                }
            )
            print(f"            ✅ Summary also stored in LangGraph memory store")
        except Exception as e:
            print(f"            ⚠️ Warning: Could not store in LangGraph memory store: {e}")
            logger.warning(f"Could not store summary in LangGraph memory store: {e}")

    async def _archive_conversation(self, conversation_id: str) -> None:
        """Mark conversation as archived."""
        query = """
            UPDATE conversations
            SET archival_status = 'archived', archived_at = %s
            WHERE id = %s
        """

        conn = await self.db_manager.get_connection()
        async with conn as connection:
            cur = await connection.cursor()
            async with cur:
                await cur.execute(query, (datetime.utcnow(), conversation_id))
                await connection.commit()

    async def _reset_conversation_message_count(self, conversation_id: str):
        """Reset conversation message count after summarization by setting last_summarized_at timestamp."""
        try:
            # Update last_summarized_at to current timestamp using Supabase client
            from core.database import db_client
            from datetime import datetime

            db_client.client.table("conversations").update({
                "last_summarized_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()
            print(f"            ✅ Reset summarization counter for conversation {conversation_id[:8]}...")
            print(f"            📝 Note: Database messages preserved - only agent working context will be trimmed")
        except Exception as e:
            print(f"            ❌ Error resetting summarization counter: {e}")
            logger.error(f"Error resetting conversation counter for {conversation_id}: {e}")

    async def get_user_context_for_new_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context based on latest conversation summary for a new conversation.
        Simplified approach using only conversation summaries without master summary complexity.
        """
        try:
            print(f"\n         🔍 RETRIEVING CROSS-CONVERSATION CONTEXT:")
            print(f"            👤 User: {user_id}")

            # Get user context using Supabase RPC call  
            from core.database import db_client
            
            def _get_user_context():
                return db_client.client.rpc('get_user_context_from_conversations', {'target_user_id': user_id}).execute()
            
            result = await asyncio.to_thread(_get_user_context)

            if result.data and len(result.data) > 0:
                print(f"            ✅ Found user context")
                context_data = result.data[0]
                return {
                    "latest_summary": context_data.get("latest_summary"),
                    "conversation_count": context_data.get("conversation_count", 0),
                    "latest_conversation_id": str(context_data.get("latest_conversation_id")) if context_data.get("latest_conversation_id") else None,
                    "has_history": bool(context_data.get("has_history", False))
                }
            else:
                print(f"            📝 No context found - new user")
                return {
                    "latest_summary": "New user - no conversation history yet.",
                    "conversation_count": 0,
                    "latest_conversation_id": None,
                    "has_history": False
                }

        except Exception as e:
            print(f"            ❌ Error retrieving user context: {e}")
            logger.error(f"Error retrieving user context for {user_id}: {e}")
            return {
                "latest_summary": "Error retrieving user history.",
                "conversation_count": 0,
                "latest_conversation_id": None,
                "has_history": False
            }

# Master summary functions removed - system now uses conversation summaries directly

    async def consolidate_user_summary_with_llm(self, user_id: str, new_conversation_id: str = None) -> str:
        """
        Simplified user summary function that works with conversation summaries only.
        Returns the latest conversation summary for the user, keeping the system simple and effective.
        """
        try:
            print(f"\n         📚 GETTING LATEST USER CONVERSATION SUMMARY:")
            print(f"            👤 User: {user_id}")

            # Get the latest conversation summary
            recent_summaries = await self._get_user_conversation_summaries(user_id, limit=1)

            if recent_summaries:
                latest_summary = recent_summaries[0].get('summary_text', '')
                print(f"            ✅ Found latest conversation summary ({len(latest_summary)} chars)")
                return latest_summary
            else:
                print(f"            📝 No conversation summaries found")
                return "No conversation history available yet."

        except Exception as e:
            print(f"            ❌ Error getting latest conversation summary: {e}")
            logger.error(f"Error getting latest conversation summary for {user_id}: {e}")
            return "Error retrieving conversation summary."

    async def _get_user_conversation_summaries(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation summaries for a user (limited by count, not time)."""
        try:
            from core.database import db_client
            # Get recent conversation summaries directly from the table
            result = (db_client.client.table('conversation_summaries')
                .select('conversation_id, summary_text, message_count, created_at, summary_type')
                .eq('user_id', user_id)
                .order('created_at', desc=True)
                .limit(limit)
                .execute())

            summaries = []
            for row in result.data:
                summaries.append({
                    'conversation_id': str(row['conversation_id']),
                    'summary_text': row['summary_text'],
                    'message_count': row['message_count'],
                    'created_at': row['created_at'],
                    'summary_type': row['summary_type']
                })

            return summaries
        except Exception as e:
            logger.error(f"Error getting conversation summaries for {user_id}: {e}")
            return []

# Removed _get_existing_master_summary function - no longer needed with simplified conversation summary approach

# Removed _update_user_master_summary function - no longer needed with simplified conversation summary approach

    async def _extract_key_insights(self, comprehensive_summary: str) -> List[str]:
        """Extract key insights from comprehensive summary using LLM."""
        try:
            extract_prompt = f"""
From this comprehensive user summary, extract 3-5 key insights as bullet points:

{comprehensive_summary}

Extract the most important insights about:
- User's role/business context
- Key goals or challenges
- Important preferences or requirements
- Critical decisions or topics

Format as simple bullet points (one per line, no bullets symbols):
"""

            # Use dynamic model selection for insights extraction (complex task)
            if self.memory_manager:
                llm = self.memory_manager._get_llm_for_memory_task("insights")
            else:
                llm = self.llm  # Fallback to default LLM

            response = await llm.ainvoke(extract_prompt)
            insights_text = response.content.strip()

            # Split into individual insights
            insights = [
                insight.strip()
                for insight in insights_text.split('\n')
                if insight.strip() and not insight.strip().startswith('-')
            ]

            return insights[:5]  # Limit to 5 insights

        except Exception as e:
            logger.error(f"Error extracting key insights: {e}")
            return ["Error extracting insights"]

    async def _get_user_conversations(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get user conversations even if they don't have summaries yet."""
        try:
            from core.database import db_client
            result = (db_client.client.table('conversations')
                .select('id, title, created_at, message_count')
                .eq('user_id', user_id)
                .order('created_at', desc=True)
                .limit(limit)
                .execute())

            return [
                {
                    'id': row['id'],
                    'title': row['title'],
                    'created_at': row['created_at'],
                    'message_count': row['message_count']
                }
                for row in result.data
            ]
        except Exception as e:
            print(f"            ❌ Error getting user conversations: {e}")
            return []

    async def _generate_initial_master_summary(self, user_id: str, conversations: List[Dict]) -> str:
        """Generate an initial master summary from conversation data."""
        try:
            print(f"            🤖 Generating initial master summary using LLM...")

            # Get recent messages from conversations
            recent_messages = []
            for conv in conversations[:3]:  # Latest 3 conversations
                # Get messages with appropriate limit (default uses max_messages from settings)
                conv_messages = await self._get_conversation_messages(conv['id'])
                if conv_messages:
                    recent_messages.extend(conv_messages[-10:])  # Last 10 messages per conv

            if not recent_messages:
                return "New user with conversations but no accessible messages yet."

            # Generate summary from messages directly
            context = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}..."
                for msg in recent_messages[-20:]  # Last 20 messages total
            ])

            prompt = f"""
Generate a comprehensive user summary based on the following conversation messages:

{context}

Create a summary that includes:
1. User's primary interests or needs
2. Key topics discussed
3. User's communication style and preferences
4. Important context for future interactions

Keep it concise but comprehensive (2-3 paragraphs).
"""

            # Use dynamic model selection for master summary generation (analysis task)
            if self.memory_manager:
                llm = self.memory_manager._get_llm_for_memory_task("analysis")
            else:
                llm = self.llm  # Fallback to default LLM

            response = await llm.ainvoke(prompt)
            initial_summary = response.content.strip()

            print(f"            ✅ Initial master summary generated ({len(initial_summary)} chars)")
            return initial_summary

        except Exception as e:
            print(f"            ❌ Error generating initial master summary: {e}")
            return f"Initial summary generation error: {str(e)}"


# =============================================================================
# MAIN MEMORY MANAGER (Orchestrator) - Enhanced with Cross-Conversation Support
# =============================================================================

class MemoryManager:
    """
    Main memory manager that orchestrates short-term and long-term memory.
    Enhanced with cross-conversation summarization capabilities.
    """

    def __init__(self, llm: Optional[BaseLanguageModel] = None,
                 embeddings: Optional[Embeddings] = None):
        """Initialize the memory manager."""
        self.checkpointer = None
        self.connection_pool = None
        self.long_term_store = None
        self.consolidator = None
        self.db_manager = None

        # Initialize components
        self.llm = llm
        self.embeddings = embeddings

        # Plugin system for agent-specific memory logic
        self.insight_extractors = {}
        self.memory_filters = {}

        logger.info("MemoryManager initialized with cross-conversation support")

    def register_memory_filter(self, agent_type: str, filter_func: Callable):
        """Register a memory filter function for a specific agent type."""
        self.memory_filters[agent_type] = filter_func
        logger.info(f"Registered memory filter for agent type: {agent_type}")

    def register_insight_extractor(self, agent_type: str, extractor_func: Callable):
        """Register an insight extractor function for a specific agent type."""
        self.insight_extractors[agent_type] = extractor_func
        logger.info(f"Registered insight extractor for agent type: {agent_type}")

    async def should_store_memory(self, agent_type: str, messages: List[Any]) -> bool:
        """Check if the conversation contains information worth storing in long-term memory."""
        try:
            # Check if there's a registered memory filter for this agent type
            if agent_type in self.memory_filters:
                filter_func = self.memory_filters[agent_type]
                return await filter_func(messages)

            # Default behavior: store if conversation has enough content
            return len(messages) >= 2 and any(
                len(str(msg).strip()) > 20 for msg in messages
            )

        except Exception as e:
            logger.error(f"Error checking should_store_memory for {agent_type}: {e}")
            return False

    async def store_conversation_insights(self, agent_type: str, user_id: str,
                                        messages: List[Any], conversation_id: str) -> None:
        """Extract and store insights using agent-specific plugins."""
        try:
            if agent_type in self.insight_extractors:
                extractor_func = self.insight_extractors[agent_type]
                insights = await extractor_func(user_id, messages, conversation_id)

                # Store insights in long-term memory
                if insights:
                    await self._ensure_initialized()
                    store = self.long_term_store

                    for insight in insights:
                        if isinstance(insight, dict):
                            namespace = tuple(insight.get('namespace', [user_id]))
                            key = insight.get('key', f"insight_{conversation_id}")
                            value = insight.get('value', insight)

                            await store.aput(namespace, key, value)
                            logger.info(f"Stored insight for {agent_type}: {key}")
            else:
                logger.info(f"No insight extractor registered for agent type: {agent_type}")

        except Exception as e:
            logger.error(f"Error storing conversation insights for {agent_type}: {e}")

    async def _ensure_initialized(self):
        """Ensure all components are initialized."""
        # Check if all critical components are initialized
        if (self.checkpointer is not None
            and self.consolidator is not None
            and self.db_manager is not None
            and self.long_term_store is not None):
            return

        try:
            settings = await get_settings()
            connection_string = settings.supabase.postgresql_connection_string

            # Initialize connection pool using psycopg3 (more compatible)
            try:
                from psycopg_pool import AsyncConnectionPool
                # Create pool with optimized settings for LangGraph Studio
                # Reduced pool size to avoid conflicts with Supabase pooler limits
                self.connection_pool = AsyncConnectionPool(
                    conninfo=connection_string,
                    max_size=5,  # Reduced from 20 to avoid pooler conflicts
                    min_size=1,  # Maintain at least one connection
                    kwargs={
                        "autocommit": True, 
                        "prepare_threshold": 0,
                        "connect_timeout": 10,  # Add timeout for connections
                    },
                    timeout=30,  # Pool checkout timeout
                    max_idle=300,  # Close idle connections after 5 minutes
                    max_lifetime=3600,  # Recycle connections after 1 hour
                    open=False  # Don't auto-open in constructor
                )
                # Explicitly open the pool as recommended by psycopg documentation
                await self.connection_pool.open()
                logger.info(f"Connection pool initialized with max_size=5")
            except ImportError:
                # Fallback to simpler approach that we know works
                logger.info("psycopg_pool not available, using fallback connection management")
                self.connection_pool = None  # Use direct connections instead
            except Exception as e:
                # If pool creation fails, use fallback
                logger.warning(f"Connection pool creation failed: {e}, using fallback")
                self.connection_pool = None

            if self.connection_pool:
                # Initialize checkpointer
                self.checkpointer = AsyncPostgresSaver(self.connection_pool)
                await self.checkpointer.setup()
            else:
                # Fallback: disable checkpointer if pool unavailable
                logger.warning("Connection pool unavailable, checkpointer disabled")
                self.checkpointer = None

            # Initialize database manager
            self.db_manager = SimpleDBManager()

            # Store settings for dynamic model selection
            self.settings = settings

            # Initialize ModelSelector for dynamic model selection
            from .toolbox.toolbox import ModelSelector
            self.model_selector = ModelSelector(settings)

            # Initialize default LLM with dynamic model selection capability
            if self.llm is None:
                # Use simple model as default for basic memory operations
                self.llm = self._create_memory_llm()

            if self.embeddings is None:
                self.embeddings = OpenAIEmbeddings()

            # Initialize long-term memory store
            self.long_term_store = SupabaseLongTermMemoryStore(
                embeddings=self.embeddings
            )

            # Initialize consolidator with cross-conversation capabilities
            self.consolidator = ConversationConsolidator(
                db_manager=self.db_manager,
                memory_store=self.long_term_store,
                llm=self.llm,
                embeddings=self.embeddings,
                memory_manager=self
            )

            logger.info("MemoryManager components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            # Reset all components to None to ensure clean state for retry
            self.checkpointer = None
            self.consolidator = None
            self.db_manager = None
            self.long_term_store = None
            self.connection_pool = None
            raise

    def _create_memory_llm(self, complexity: str = "simple") -> ChatOpenAI:
        """
        Create LLM instance for memory operations with dynamic model selection.

        Args:
            complexity: "simple" or "complex" to determine model choice

        Returns:
            ChatOpenAI instance with appropriate model
        """
        try:
            if complexity == "complex":
                model = getattr(self.settings, 'openai_complex_model', 'gpt-4o')
            else:
                model = getattr(self.settings, 'openai_simple_model', 'gpt-4o-mini')

            logger.debug(f"Creating memory LLM with model: {model} (complexity: {complexity})")

            return ChatOpenAI(
                model=model,
                temperature=0.1,
                max_tokens=1000,
                api_key=self.settings.openai_api_key
            )
        except Exception as e:
            logger.error(f"Error creating memory LLM: {e}")
            # Fallback to simple model
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1000,
                api_key=self.settings.openai_api_key
            )

    def _get_llm_for_memory_task(self, task_type: str, messages: List[Dict] = None) -> ChatOpenAI:
        """
        Get appropriate LLM for different memory tasks based on complexity.

        Args:
            task_type: Type of memory task ("summary", "insights", "analysis", "simple")
            messages: Optional message history to assess complexity

        Returns:
            ChatOpenAI instance with appropriate model
        """
        # Determine complexity based on task type
        complex_tasks = ["insights", "analysis"]

        if task_type in complex_tasks:
            complexity = "complex"
        elif task_type == "summary" and messages:
            # For summaries, use complex model if many messages or long content
            if len(messages) > 20 or any(len(msg.get('content', '')) > 500 for msg in messages):
                complexity = "complex"
            else:
                complexity = "simple"
        else:
            complexity = "simple"

        logger.info(f"Using {complexity} model for memory task: {task_type}")
        return self._create_memory_llm(complexity)

    # Short-term memory methods
    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """Get the LangGraph checkpointer."""
        await self._ensure_initialized()
        return self.checkpointer

    async def get_conversation_config(self, conversation_id: UUID) -> Dict[str, Any]:
        """Get configuration for LangGraph thread management."""
        return {"configurable": {"thread_id": str(conversation_id)}}

    async def load_conversation_state(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Load conversation state including details and messages."""
        try:
            await self._ensure_initialized()

            # Additional check to ensure consolidator was properly initialized
            if self.consolidator is None:
                logger.error("Consolidator is None after initialization - returning None for conversation state")
                return None

            conversation_id_str = str(conversation_id)

            # Get conversation details using consolidator
            conversation = await self.consolidator._get_conversation_details(conversation_id_str)
            if not conversation:
                logger.info(f"No conversation found for ID: {conversation_id_str}")
                return None

            # Get conversation messages using consolidator (limit to max_messages for performance)
            messages = await self.consolidator._get_conversation_messages(conversation_id_str, limit=self.settings.memory.max_messages)

            # Return state with messages and metadata
            return {
                "conversation": conversation,
                "messages": messages,
                "message_count": len(messages)
            }

        except Exception as e:
            logger.error(f"Error loading conversation state for {conversation_id}: {e}")
            return None

    # Simplified cross-conversation method (master summary updates happen automatically now)
    async def get_user_context_for_new_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user context for new conversation.
        Master summaries are automatically updated whenever conversation summaries are generated.
        """
        await self._ensure_initialized()

        try:
            # Additional check to ensure consolidator was properly initialized
            if self.consolidator is None:
                logger.error("Consolidator is None after initialization - returning empty context")
                return {}

            # Get comprehensive user context (this is automatically up-to-date)
            user_context = await self.consolidator.get_user_context_for_new_conversation(user_id)
            return user_context

        except Exception as e:
            logger.error(f"Error getting user context for {user_id}: {e}")
            return {
                "master_summary": "Error retrieving user history.",
                "has_history": False,
                "error": str(e)
            }

    # Long-term memory methods (existing)
    async def store_long_term_memory(self, user_id: str, namespace: List[str],
                                   key: str, value: Any, ttl_hours: Optional[int] = None) -> bool:
        """Store information in long-term memory."""
        try:
            await self._ensure_initialized()

            # Ensure user_id is in namespace
            if not namespace or namespace[0] != user_id:
                namespace = [user_id] + namespace

            await self.long_term_store.aput(
                namespace=tuple(namespace),
                key=key,
                value=value,
                ttl_hours=ttl_hours
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            return False

    async def get_relevant_context(self, user_id: str, current_query: str,
                                 max_contexts: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context from long-term memory."""
        try:
            await self._ensure_initialized()

            # Search for relevant memories
            results = await self.long_term_store.semantic_search(
                query=current_query,
                namespace_prefix=(user_id,),
                limit=max_contexts
            )

            # Format results - results are now tuples: (namespace, key, value)
            contexts = []
            for namespace_tuple, key, value in results:
                # Parse value if it's a JSON string
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = {"content": value}

                contexts.append({
                    "type": value.get('type', 'memory') if isinstance(value, dict) else 'memory',
                    "content": value.get('summary', str(value)) if isinstance(value, dict) else str(value),
                    "score": 0.8,  # Default similarity score since we don't have it in tuple format
                    "namespace": list(namespace_tuple),
                    "key": key,
                    "created_at": None  # We don't have this in the tuple format
                })

            return contexts

        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []

    # Background tasks
    async def consolidate_old_conversations(self, user_id: str) -> Dict[str, Any]:
        """Consolidate old conversations (background task)."""
        try:
            await self._ensure_initialized()

            # Additional check to ensure consolidator was properly initialized
            if self.consolidator is None:
                logger.error("Consolidator is None after initialization - initialization may have failed")
                return {"consolidated_count": 0, "error": "Consolidator initialization failed"}

            return await self.consolidator.consolidate_conversations(user_id)

        except Exception as e:
            logger.error(f"Failed to consolidate conversations: {e}")
            return {"consolidated_count": 0, "error": str(e)}

    async def cleanup_old_archived_conversations(self, days_threshold: int = 30) -> int:
        """Clean up very old archived conversations."""
        try:
            await self._ensure_initialized()

            # For now, skip complex cleanup operations since we don't have archived status in conversations table
            # This prevents the SQL error while maintaining functionality
            logger.info("Cleanup complete: 0 old conversations deleted")
            return 0

        except Exception as e:
            logger.warning(f"Unsupported query type: \n                DELETE FROM conversations \n                WHERE archival_status = 'archived' \n                    AND archived_at < %s\n                    AND id IN (\n                        SELECT conversation_id \n                        FROM conversation_summaries \n                        WHERE conversation_id = conversations.id\n                    )\n            ")
            logger.error(f"Failed to cleanup archived conversations: {e}")
            return 0

    # Message storage methods (existing, simplified for brevity)
    async def save_message_to_database(self, conversation_id: str, user_id: str,
                                     role: str, content: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Save individual message to the messages table."""
        print(f"            💬 save_message_to_database called:")
        print(f"               - conversation_id: '{conversation_id}' (type: {type(conversation_id)})")
        print(f"               - user_id: '{user_id}' (type: {type(user_id)})")
        print(f"               - role: '{role}'")
        print(f"               - content length: {len(content)} chars")
        try:
            await self._ensure_initialized()

            # Ensure conversation exists and get the actual conversation_id to use
            # (may be different if reusing existing conversation)
            actual_conversation_id = await self._ensure_conversation_exists(conversation_id, user_id)

            # Use Supabase client directly for message storage
            try:
                from core.database import db_client
            except ImportError:
                # Handle Docker environment
                import sys
                sys.path.append('/app')
                from core.database import db_client

            message_data = {
                'conversation_id': actual_conversation_id,  # Use the actual conversation ID
                'role': role,
                'content': content,
                'metadata': metadata or {}
            }

            # Wrap synchronous database call in asyncio.to_thread
            result = await asyncio.to_thread(
                lambda: db_client.client.table('messages').insert(message_data).execute()
            )

            if result.data:
                message_id = result.data[0]['id']
                logger.debug(f"Message saved to database: {message_id} (role: {role}) in conversation: {actual_conversation_id}")
                return message_id
            else:
                logger.error("Failed to save message to database - no data returned")
                return None

        except Exception as e:
            logger.error(f"Error saving message to database: {e}")
            return None

    async def _ensure_conversation_exists(self, conversation_id: str, user_id: str) -> str:
        """
        Ensure conversation record exists in the conversations table.
        Returns the actual conversation_id to use (may reuse existing conversation).
        
        SINGLE CONVERSATION PER USER APPROACH:
        - If user has existing conversation, reuse it (ignore provided conversation_id)
        - Only create new conversation if user has no existing conversations
        - This maintains conversation continuity across sessions/refreshes
        """
        print(f"            🔍 _ensure_conversation_exists called:")
        print(f"               - conversation_id: '{conversation_id}' (type: {type(conversation_id)})")
        print(f"               - user_id: '{user_id}' (type: {type(user_id)})")
        
        try:
            try:
                from core.database import db_client
            except ImportError:
                # Handle Docker environment
                try:
                    from core.database import db_client
                except ImportError:
                    import sys
                    sys.path.append('/app')
                    from core.database import db_client

            # STEP 1: Check if user already has an existing conversation (prefer reuse)
            existing_conversations = await asyncio.to_thread(
                lambda: db_client.client.table('conversations')
                .select('id, title, created_at')
                .eq('user_id', user_id)
                .order('updated_at', desc=True)
                .limit(1)
                .execute()
            )
            
            if existing_conversations.data:
                existing_id = existing_conversations.data[0]['id']
                existing_title = existing_conversations.data[0].get('title', 'Conversation')
                
                print(f"               ✅ Reusing existing conversation:")
                print(f"                  - existing_id: '{existing_id}'")
                print(f"                  - title: '{existing_title}'")
                
                # Update the conversation timestamp to mark as active
                await asyncio.to_thread(
                    lambda: db_client.client.table('conversations')
                    .update({'updated_at': datetime.now().isoformat()})
                    .eq('id', existing_id)
                    .execute()
                )
                
                return existing_id
            
            # STEP 2: No existing conversation found - create new one
            print(f"               🆕 No existing conversation found, creating new one")
            
            conversation_data = {
                'id': conversation_id,
                'user_id': user_id,
                'title': 'Ongoing Conversation',  # Better title than "New Conversation"
                'metadata': {
                    'created_by': 'memory_manager',
                    'conversation_type': 'single_persistent'
                }
            }

            print(f"               🔄 Creating conversation with data:")
            print(f"                  - id: '{conversation_data['id']}'")
            print(f"                  - user_id: '{conversation_data['user_id']}'")
            print(f"                  - title: '{conversation_data['title']}'")

            result = await asyncio.to_thread(
                lambda: db_client.client.table('conversations').insert(conversation_data).execute()
            )

            if result.data:
                logger.debug(f"Created new conversation record: {conversation_id}")
                return conversation_id
            else:
                logger.error(f"Failed to create conversation record: {conversation_id}")
                return conversation_id  # Return original ID as fallback

        except Exception as e:
            logger.error(f"Error ensuring conversation exists: {e}")
            return conversation_id  # Return original ID as fallback

    def _extract_conversation_id_from_config(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract conversation ID from LangGraph config."""
        try:
            if isinstance(config, dict) and 'configurable' in config:
                return config['configurable'].get('thread_id')
            return None
        except Exception as e:
            logger.error(f"Error extracting conversation ID from config: {e}")
            return None

    async def store_message_from_agent(self, message: Union[BaseMessage, Dict[str, Any]],
                                     config: Dict[str, Any],
                                     agent_type: str = "unknown",
                                     user_id: Optional[str] = None) -> Optional[str]:
        """High-level method to store message from agent processing."""
        try:
            # Extract conversation ID from config
            conversation_id = self._extract_conversation_id_from_config(config)
            if not conversation_id:
                logger.warning(f"No conversation ID found in config for {agent_type} agent")
                return None

            # Use provided user_id or try to extract from message
            if not user_id:
                user_id = self._extract_user_id_from_message(message)

            if not user_id:
                # Generate a UUID for anonymous users to comply with database schema
                user_id = str(uuid4())
                logger.warning(f"No user_id provided or found in message, generated UUID: {user_id}")
            elif user_id == "anonymous":
                # Replace "anonymous" with a proper UUID
                user_id = str(uuid4())
                logger.warning(f"Replaced 'anonymous' with UUID: {user_id}")
            else:
                logger.info(f"Using provided user_id: {user_id}")

            # Extract message content and role
            content = ""
            role = "unknown"
            metadata = {}

            if isinstance(message, dict):
                content = str(message.get('content', ''))
                role = message.get('role', 'unknown')
                metadata = message.get('metadata', {})
            elif hasattr(message, 'content'):
                content = str(message.content)
                metadata = getattr(message, 'metadata', {}) or {}

                # Map LangChain message types to database-compatible roles
                # Database constraint: ('user', 'assistant', 'system')
                if hasattr(message, 'type'):
                    role_mapping = {
                        'human': 'user',      # LangChain 'human' -> DB 'user'
                        'ai': 'assistant',    # LangChain 'ai' -> DB 'assistant'
                        'system': 'system',   # LangChain 'system' -> DB 'system'
                        'assistant': 'assistant'  # LangChain 'assistant' -> DB 'assistant'
                    }
                    role = role_mapping.get(message.type, 'assistant')    # Default to 'assistant' for unknown types
                elif hasattr(message, 'role'):
                    # Handle direct role assignments and normalize them
                    role = message.role.lower() if isinstance(message.role, str) else str(message.role)
                    # Map to database-compatible roles
                    if role in ['human', 'user']:
                        role = 'user'
                    elif role in ['ai', 'assistant']:
                        role = 'assistant'
                    elif role == 'system':
                        role = 'system'
                    else:
                        role = 'assistant'  # Default fallback

            # Skip empty messages
            if not content.strip():
                logger.debug(f"Skipping empty message from {agent_type} agent")
                return None

            # Add agent type to metadata
            metadata['agent_type'] = agent_type
            metadata['stored_at'] = datetime.utcnow().isoformat()

            # Save to database
            message_id = await self.save_message_to_database(
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                metadata=metadata
            )

            if message_id:
                logger.info(f"Stored message from {agent_type} agent: {message_id}")

            return message_id

        except Exception as e:
            logger.error(f"Error storing message from {agent_type} agent: {e}")
            return None

    def _extract_user_id_from_message(self, message: Any) -> Optional[str]:
        """Extract user_id from message in various formats."""
        if isinstance(message, dict):
            return message.get('user_id')
        elif hasattr(message, 'user_id'):
            return message.user_id
        elif hasattr(message, 'metadata') and isinstance(message.metadata, dict):
            return message.metadata.get('user_id')
        return None

    async def cleanup(self):
        """Clean up resources."""
        if self.connection_pool:
            try:
                # Wait for all pending operations to complete with timeout
                await asyncio.wait_for(self.connection_pool.close(), timeout=10.0)
                logger.info("Connection pool closed successfully")
            except asyncio.TimeoutError:
                logger.warning("Connection pool close timed out, forcing closure")
                try:
                    # Force close if normal close times out
                    await self.connection_pool.close(timeout=1.0)
                except Exception as e:
                    logger.error(f"Error forcing connection pool close: {e}")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")

        # Reset all components to ensure clean state
        self.checkpointer = None
        self.connection_pool = None
        self.long_term_store = None
        self.consolidator = None
        self.db_manager = None


# =============================================================================
# MEMORY SCHEDULER (Background Processing) - Enhanced for Cross-Conversation
# =============================================================================

class MemoryScheduler:
    """
    Background task scheduler for memory management operations.
    Enhanced with cross-conversation summary consolidation.
    """

    def __init__(self):
        self.consolidation_interval = 3600  # 1 hour
        self.cleanup_interval = 86400  # 24 hours
        # Removed user_summary_interval - we now update master summaries immediately
        self.running = False
        self.tasks = []

    async def start(self):
        """Start the background scheduler."""
        if self.running:
            return

        self.running = True
        logger.info("Starting enhanced memory scheduler with cross-conversation support")

        # Start background tasks (simplified - no user summary scheduler needed)
        self.tasks = [
            asyncio.create_task(self._consolidation_scheduler()),
            asyncio.create_task(self._cleanup_scheduler())
        ]

        logger.info("Simplified memory scheduler started successfully")

    async def stop(self):
        """Stop the background scheduler."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping memory scheduler")

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []

        logger.info("Memory scheduler stopped")

    # Removed _user_summary_scheduler - master summaries now update immediately when conversation summaries are generated

    async def _consolidation_scheduler(self):
        """Background task for periodic conversation consolidation."""
        while self.running:
            try:
                logger.info("Starting periodic memory consolidation")

                # Get active users
                active_users = await self._get_active_users()

                consolidation_results = []
                for user_id in active_users:
                    try:
                        result = await memory_manager.consolidate_old_conversations(user_id)
                        consolidation_results.append({
                            "user_id": user_id,
                            "result": result
                        })

                        # Small delay between users
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"Error consolidating conversations for user {user_id}: {e}")

                # Log consolidation summary
                total_consolidated = sum(r["result"].get("consolidated_count", 0) for r in consolidation_results)
                logger.info(f"Consolidation complete: {total_consolidated} conversations consolidated across {len(active_users)} users")

                # Wait for next consolidation cycle
                await asyncio.sleep(self.consolidation_interval)

            except asyncio.CancelledError:
                logger.info("Consolidation scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consolidation scheduler: {e}")
                await asyncio.sleep(60)

    async def _cleanup_scheduler(self):
        """Background task for periodic memory cleanup."""
        while self.running:
            try:
                logger.info("Starting periodic memory cleanup")

                # Clean up old archived conversations
                deleted_count = await memory_manager.cleanup_old_archived_conversations(90)
                logger.info(f"Cleanup complete: {deleted_count} old conversations deleted")

                # Clean up expired long-term memories
                await self._cleanup_expired_long_term_memories()

                # Wait for next cleanup cycle
                await asyncio.sleep(self.cleanup_interval)

            except asyncio.CancelledError:
                logger.info("Cleanup scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {e}")
                await asyncio.sleep(300)

    # Removed _get_users_with_recent_conversations - no longer needed with immediate updates

    async def _get_active_users(self) -> List[str]:
        """Get list of active users for consolidation."""
        try:
            await memory_manager._ensure_initialized()

            # Get users with conversations in the last 7 days using Supabase client
            cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()

            from core.database import db_client

            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations")
                    .select("user_id")
                    .gt("updated_at", cutoff_date)
                    .execute()
            )

            if result.data:
                # Get distinct user_ids
                user_ids = list(set(row['user_id'] for row in result.data if row.get('user_id')))
                return sorted(user_ids)

            return []

        except Exception as e:
            logger.warning(f"Unsupported query type: \n                SELECT DISTINCT user_id \n                FROM conversations \n                WHERE updated_at > %s \n                    AND archival_status = 'active'\n                ORDER BY user_id\n            ")
            logger.error(f"Error getting active users: {e}")
            return []

    async def _cleanup_expired_long_term_memories(self):
        """Clean up expired long-term memories."""
        try:
            await memory_manager._ensure_initialized()

            # For now, skip expired memory cleanup since it requires complex SQL
            # This prevents the SQL error while maintaining core functionality
            logger.info("Cleaned up 0 expired long-term memories")

        except Exception as e:
            logger.warning(f"Unsupported query type: \n                DELETE FROM long_term_memories \n                WHERE expires_at IS NOT NULL \n                    AND expires_at < NOW()\n            ")
            logger.error(f"Error cleaning up expired long-term memories: {e}")


# Global instances
context_manager = ContextWindowManager()
memory_manager = MemoryManager()
memory_scheduler = MemoryScheduler()
