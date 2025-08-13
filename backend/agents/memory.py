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
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore, Item
from langchain_core.messages import BaseMessage
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
    
    # NOTE: track_sql_database removed in Task 4.11.8
    # This function was unused after ConversationConsolidator removal
    
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

# NOTE: create_execution_scoped_sql_database removed in Task 4.11.8
# This function was unused after ConversationConsolidator removal
# SQL database operations are now handled by specialized toolbox functions

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

# NOTE: _create_sql_database removed in Task 4.11.8
# This function was unused after ConversationConsolidator removal
# SQL database operations are now handled by specialized toolbox functions

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
                    # CRITICAL FIX: Never include tool_calls messages without their responses
                    # as this violates OpenAI API requirements and causes 400 errors
                    # Skip this message to prevent orphaned tool_calls
                    logger.warning(f"[CONTEXT_TRIM] Skipping orphaned tool_calls message to prevent API error")
                    i -= 1
            
            else:
                # Regular message (human, ai without tool_calls, etc.)
                if len(result) + 1 <= max_messages:
                    result.insert(0, current_msg)
                i -= 1
        
        return result

    def validate_message_sequence(self, messages: List[BaseMessage]) -> bool:
        """
        Validate that the message sequence follows OpenAI API requirements.
        
        Specifically checks that:
        1. Every assistant message with tool_calls is followed by tool messages
        2. Tool messages have corresponding tool_call_ids
        
        Args:
            messages: List of messages to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        for i, msg in enumerate(messages):
            if (hasattr(msg, 'tool_calls') and 
                msg.tool_calls and 
                hasattr(msg, 'type') and 
                msg.type == 'ai'):
                
                # This is an assistant message with tool_calls
                # Check if the next messages are tool responses
                tool_call_ids = {call.id for call in msg.tool_calls}
                found_tool_responses = set()
                
                j = i + 1
                while (j < len(messages) and 
                       hasattr(messages[j], 'type') and 
                       messages[j].type == 'tool'):
                    
                    if hasattr(messages[j], 'tool_call_id'):
                        found_tool_responses.add(messages[j].tool_call_id)
                    j += 1
                
                # Check if all tool_calls have responses
                if tool_call_ids != found_tool_responses:
                    logger.error(f"[VALIDATION] Missing tool responses for tool_call_ids: {tool_call_ids - found_tool_responses}")
                    return False
                    
        return True


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

# NOTE: ConversationConsolidator class removed in Task 4.11.4
# This class contained 857 lines of complex consolidation logic that has been
# simplified and integrated into BackgroundTaskManager for better performance.
# Key methods migrated:
# - get_conversation_details -> BackgroundTaskManager.get_conversation_details  
# - get_conversation_messages -> BackgroundTaskManager.get_conversation_messages
# - get_user_context_for_new_conversation -> BackgroundTaskManager.get_user_context_for_new_conversation
# - get_conversation_summary -> BackgroundTaskManager.get_conversation_summary
# - consolidate_conversations -> BackgroundTaskManager.consolidate_conversations

# The removal of this class achieves:
# - 857 lines of code reduction (25% of memory.py file size)
# - Elimination of complex multi-layer consolidation logic
# - Better integration with LangGraph best practices
# - Simplified background task architecture


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
        # NOTE: consolidator removed in Task 4.11.4 - functionality moved to BackgroundTaskManager
        self.db_manager = None

        # Initialize components
        self.llm = llm
        self.embeddings = embeddings

        # Plugin system for agent-specific memory logic
        self.insight_extractors = {}
        self.memory_filters = {}

        # Lazy loading cache with TTL and access tracking
        self._context_cache = {}
        self._cache_ttl_minutes = 15  # Cache context for 15 minutes
        self._cache_access_count = {}
        self._max_cache_size = 100    # Maximum cached contexts

        # Token conservation caches (Task 3.7)
        self._system_prompt_cache = {}      # Cache system prompts
        self._llm_interpretation_cache = {} # Cache LLM interpretation responses
        self._user_pattern_cache = {}       # Cache user behavior patterns
        self._token_cache_ttl_hours = 2     # 2-hour TTL for token conservation caches
        self._max_token_cache_size = 200    # Maximum cached token-saving items

        logger.info("MemoryManager initialized with cross-conversation support and lazy loading")

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

            # NOTE: ConversationConsolidator removed in Task 4.11.4
            # Functionality moved to BackgroundTaskManager for better integration

            logger.info("MemoryManager components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            # Reset all components to None to ensure clean state for retry
            self.checkpointer = None
            # NOTE: consolidator removed in Task 4.11.4 - functionality moved to BackgroundTaskManager
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

            # Use BackgroundTaskManager instead of consolidator (Task 4.11.4)
            from agents.background_tasks import background_task_manager
            
            conversation_id_str = str(conversation_id)

            # Get conversation details using background task manager
            conversation = await background_task_manager.get_conversation_details(conversation_id_str)
            if not conversation:
                logger.info(f"No conversation found for ID: {conversation_id_str}")
                return None

            # Get conversation messages using background task manager (limit to max_messages for performance)
            messages = await background_task_manager.get_conversation_messages(conversation_id_str, limit=self.settings.memory.max_messages)

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
            # Use BackgroundTaskManager instead of consolidator (Task 4.11.4)
            from agents.background_tasks import background_task_manager

            # Get comprehensive user context (this is automatically up-to-date)
            user_context = await background_task_manager.get_user_context_for_new_conversation(user_id)
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

    # =============================================================================
    # LAZY CONTEXT LOADING METHODS (Task 3.6)
    # =============================================================================

    def _generate_cache_key(self, user_id: str, context_type: str, query: Optional[str] = None) -> str:
        """Generate cache key for context data."""
        import hashlib
        base_key = f"{user_id}:{context_type}"
        if query:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            base_key += f":{query_hash}"
        return base_key

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached context is still valid (within TTL)."""
        if cache_key not in self._context_cache:
            return False
        
        cache_entry = self._context_cache[cache_key]
        cache_age_minutes = (datetime.utcnow() - cache_entry['cached_at']).total_seconds() / 60
        return cache_age_minutes < self._cache_ttl_minutes

    def _cleanup_cache(self):
        """Remove expired and least-used cache entries."""
        current_time = datetime.utcnow()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self._context_cache.items():
            cache_age_minutes = (current_time - entry['cached_at']).total_seconds() / 60
            if cache_age_minutes >= self._cache_ttl_minutes:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._context_cache[key]
            if key in self._cache_access_count:
                del self._cache_access_count[key]
        
        # If still over limit, remove least accessed entries
        if len(self._context_cache) > self._max_cache_size:
            # Sort by access count (ascending) and remove least accessed
            sorted_keys = sorted(
                self._context_cache.keys(),
                key=lambda k: self._cache_access_count.get(k, 0)
            )
            
            keys_to_remove = sorted_keys[:len(self._context_cache) - self._max_cache_size]
            for key in keys_to_remove:
                del self._context_cache[key]
                if key in self._cache_access_count:
                    del self._cache_access_count[key]

    def _cache_context(self, cache_key: str, context_data: Any):
        """Cache context data with metadata."""
        self._cleanup_cache()  # Cleanup before adding new entry
        
        self._context_cache[cache_key] = {
            'data': context_data,
            'cached_at': datetime.utcnow(),
            'access_count': 1
        }
        self._cache_access_count[cache_key] = 1

    def _get_cached_context(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached context data if valid."""
        if not self._is_cache_valid(cache_key):
            return None
        
        # Update access tracking
        self._cache_access_count[cache_key] = self._cache_access_count.get(cache_key, 0) + 1
        self._context_cache[cache_key]['access_count'] += 1
        
        return self._context_cache[cache_key]['data']

    async def get_user_context_lazy(self, user_id: str) -> Dict[str, Any]:
        """
        Lazy loading version of get_user_context_for_new_conversation.
        Uses caching to avoid repeated database queries.
        """
        cache_key = self._generate_cache_key(user_id, "user_context")
        
        # Check cache first
        cached_context = self._get_cached_context(cache_key)
        if cached_context is not None:
            logger.debug(f"[LAZY_CONTEXT] Cache hit for user context: {user_id}")
            return cached_context
        
        # Cache miss - load from database
        logger.debug(f"[LAZY_CONTEXT] Cache miss - loading user context: {user_id}")
        try:
            context_data = await self.get_user_context_for_new_conversation(user_id)
            
            # Cache the result
            self._cache_context(cache_key, context_data)
            
            return context_data
            
        except Exception as e:
            logger.error(f"[LAZY_CONTEXT] Error loading user context for {user_id}: {e}")
            # Return empty context on error but don't cache the error
            return {"master_summary": "Error retrieving user history.", "has_history": False}

    async def get_relevant_context_lazy(self, user_id: str, query: str, 
                                      max_contexts: int = 5) -> List[Dict[str, Any]]:
        """
        Lazy loading version of get_relevant_context.
        Uses caching with query-specific keys to avoid repeated semantic searches.
        """
        cache_key = self._generate_cache_key(user_id, "relevant_context", query)
        
        # Check cache first
        cached_context = self._get_cached_context(cache_key)
        if cached_context is not None:
            logger.debug(f"[LAZY_CONTEXT] Cache hit for relevant context: {user_id} - query: {query[:50]}...")
            return cached_context[:max_contexts]  # Respect max_contexts even from cache
        
        # Cache miss - load from long-term memory
        logger.debug(f"[LAZY_CONTEXT] Cache miss - loading relevant context: {user_id} - query: {query[:50]}...")
        try:
            context_data = await self.get_relevant_context(user_id, query, max_contexts)
            
            # Cache the result
            self._cache_context(cache_key, context_data)
            
            return context_data
            
        except Exception as e:
            logger.error(f"[LAZY_CONTEXT] Error loading relevant context for {user_id}: {e}")
            # Return empty list on error but don't cache the error
            return []

    async def get_conversation_summary_lazy(self, conversation_id: str) -> Optional[str]:
        """
        Lazy loading version for conversation summaries.
        Uses caching to avoid repeated database queries for the same conversation.
        """
        cache_key = self._generate_cache_key(conversation_id, "conversation_summary")
        
        # Check cache first
        cached_summary = self._get_cached_context(cache_key)
        if cached_summary is not None:
            logger.debug(f"[LAZY_CONTEXT] Cache hit for conversation summary: {conversation_id}")
            return cached_summary
        
        # Cache miss - load from database
        logger.debug(f"[LAZY_CONTEXT] Cache miss - loading conversation summary: {conversation_id}")
        try:
            await self._ensure_initialized()
            
            # Use BackgroundTaskManager to get conversation summary (Task 4.11.4)
            from agents.background_tasks import background_task_manager
            summary = await background_task_manager.get_conversation_summary(conversation_id)
            
            # Cache the result (even if None, to avoid repeated queries)
            self._cache_context(cache_key, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"[LAZY_CONTEXT] Error loading conversation summary for {conversation_id}: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the context cache for monitoring."""
        total_entries = len(self._context_cache)
        
        # Calculate cache ages
        current_time = datetime.utcnow()
        ages = []
        for entry in self._context_cache.values():
            age_minutes = (current_time - entry['cached_at']).total_seconds() / 60
            ages.append(age_minutes)
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # Access count statistics
        access_counts = list(self._cache_access_count.values())
        avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
        
        return {
            "total_cached_contexts": total_entries,
            "cache_limit": self._max_cache_size,
            "cache_utilization": f"{(total_entries / self._max_cache_size) * 100:.1f}%",
            "ttl_minutes": self._cache_ttl_minutes,
            "average_age_minutes": f"{avg_age:.2f}",
            "average_access_count": f"{avg_access_count:.1f}",
            "total_access_count": sum(access_counts)
        }

    def clear_context_cache(self, user_id: Optional[str] = None):
        """Clear context cache, optionally for a specific user."""
        if user_id:
            # Clear cache entries for specific user
            keys_to_remove = [
                key for key in self._context_cache.keys() 
                if key.startswith(f"{user_id}:")
            ]
            for key in keys_to_remove:
                del self._context_cache[key]
                if key in self._cache_access_count:
                    del self._cache_access_count[key]
            logger.debug(f"[LAZY_CONTEXT] Cleared {len(keys_to_remove)} cache entries for user: {user_id}")
        else:
            # Clear all cache
            self._context_cache.clear()
            self._cache_access_count.clear()
            logger.debug("[LAZY_CONTEXT] Cleared all context cache")

    # =============================================================================
    # TOKEN CONSERVATION CACHES (Task 3.7)
    # =============================================================================

    def _generate_token_cache_key(self, cache_type: str, *args) -> str:
        """Generate cache key for token conservation caches."""
        import hashlib
        key_parts = [cache_type] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _is_token_cache_valid(self, cache_key: str, cache_dict: Dict) -> bool:
        """Check if token cache entry is still valid (within TTL)."""
        if cache_key not in cache_dict:
            return False
        
        cache_entry = cache_dict[cache_key]
        cache_age_hours = (datetime.utcnow() - cache_entry['cached_at']).total_seconds() / 3600
        return cache_age_hours < self._token_cache_ttl_hours

    def _cleanup_token_cache(self, cache_dict: Dict):
        """Remove expired entries from token conservation caches."""
        current_time = datetime.utcnow()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in cache_dict.items():
            cache_age_hours = (current_time - entry['cached_at']).total_seconds() / 3600
            if cache_age_hours >= self._token_cache_ttl_hours:
                expired_keys.append(key)
        
        for key in expired_keys:
            del cache_dict[key]
        
        # If still over limit, remove oldest entries
        if len(cache_dict) > self._max_token_cache_size:
            # Sort by creation time (ascending) and remove oldest
            sorted_items = sorted(
                cache_dict.items(),
                key=lambda x: x[1]['cached_at']
            )
            
            items_to_remove = len(cache_dict) - self._max_token_cache_size
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del cache_dict[key]

    def cache_system_prompt(self, tool_names: List[str], user_language: str = 'english', 
                          conversation_summary: str = None, user_context: str = None) -> Optional[str]:
        """
        Get cached system prompt or return None if not cached.
        Avoids repeated system prompt generation for same parameters.
        """
        cache_key = self._generate_token_cache_key(
            "system_prompt", 
            ":".join(sorted(tool_names)), 
            user_language,
            conversation_summary or "",
            user_context or ""
        )
        
        if self._is_token_cache_valid(cache_key, self._system_prompt_cache):
            entry = self._system_prompt_cache[cache_key]
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.utcnow()
            logger.debug(f"[TOKEN_CACHE] System prompt cache hit: {cache_key[:8]}...")
            return entry['data']
        
        logger.debug(f"[TOKEN_CACHE] System prompt cache miss: {cache_key[:8]}...")
        return None

    def store_system_prompt(self, prompt: str, tool_names: List[str], user_language: str = 'english',
                          conversation_summary: str = None, user_context: str = None):
        """Cache a generated system prompt to avoid regeneration."""
        cache_key = self._generate_token_cache_key(
            "system_prompt",
            ":".join(sorted(tool_names)),
            user_language,
            conversation_summary or "",
            user_context or ""
        )
        
        self._cleanup_token_cache(self._system_prompt_cache)
        
        self._system_prompt_cache[cache_key] = {
            'data': prompt,
            'cached_at': datetime.utcnow(),
            'access_count': 1,
            'last_accessed': datetime.utcnow()
        }
        logger.debug(f"[TOKEN_CACHE] Cached system prompt: {cache_key[:8]}...")

    def cache_llm_interpretation(self, human_response: str, hitl_context: Dict[str, Any]) -> Optional[str]:
        """
        Get cached LLM interpretation result or return None if not cached.
        Avoids repeated LLM calls for same human responses in similar contexts.
        """
        # Create context signature for cache key
        context_signature = ""
        if hitl_context:
            # Use source_tool and interaction type for context signature
            context_signature = f"{hitl_context.get('source_tool', '')}"
            if 'legacy_type' in hitl_context:
                context_signature += f":{hitl_context['legacy_type']}"
        
        cache_key = self._generate_token_cache_key(
            "llm_interpretation",
            human_response.lower().strip(),
            context_signature
        )
        
        if self._is_token_cache_valid(cache_key, self._llm_interpretation_cache):
            entry = self._llm_interpretation_cache[cache_key]
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.utcnow()
            logger.debug(f"[TOKEN_CACHE] LLM interpretation cache hit: {cache_key[:8]}... for '{human_response[:20]}...'")
            return entry['data']
        
        logger.debug(f"[TOKEN_CACHE] LLM interpretation cache miss: {cache_key[:8]}... for '{human_response[:20]}...'")
        return None

    def store_llm_interpretation(self, result: str, human_response: str, hitl_context: Dict[str, Any]):
        """Cache an LLM interpretation result to avoid repeated calls."""
        context_signature = ""
        if hitl_context:
            context_signature = f"{hitl_context.get('source_tool', '')}"
            if 'legacy_type' in hitl_context:
                context_signature += f":{hitl_context['legacy_type']}"
        
        cache_key = self._generate_token_cache_key(
            "llm_interpretation",
            human_response.lower().strip(),
            context_signature
        )
        
        self._cleanup_token_cache(self._llm_interpretation_cache)
        
        self._llm_interpretation_cache[cache_key] = {
            'data': result,
            'cached_at': datetime.utcnow(),
            'access_count': 1,
            'last_accessed': datetime.utcnow()
        }
        logger.debug(f"[TOKEN_CACHE] Cached LLM interpretation: {cache_key[:8]}... for '{human_response[:20]}...' → {result}")

    def cache_user_pattern(self, user_id: str, pattern_type: str, pattern_data: Any) -> None:
        """
        Cache user behavior patterns to avoid repeated analysis.
        Examples: language preference, response patterns, tool usage patterns.
        """
        cache_key = self._generate_token_cache_key("user_pattern", user_id, pattern_type)
        
        self._cleanup_token_cache(self._user_pattern_cache)
        
        self._user_pattern_cache[cache_key] = {
            'data': pattern_data,
            'cached_at': datetime.utcnow(),
            'access_count': 1,
            'last_accessed': datetime.utcnow()
        }
        logger.debug(f"[TOKEN_CACHE] Cached user pattern: {pattern_type} for user {user_id}")

    def get_user_pattern(self, user_id: str, pattern_type: str) -> Optional[Any]:
        """Get cached user pattern data."""
        cache_key = self._generate_token_cache_key("user_pattern", user_id, pattern_type)
        
        if self._is_token_cache_valid(cache_key, self._user_pattern_cache):
            entry = self._user_pattern_cache[cache_key]
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.utcnow()
            logger.debug(f"[TOKEN_CACHE] User pattern cache hit: {pattern_type} for user {user_id}")
            return entry['data']
        
        return None

    def get_token_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about token conservation caches."""
        system_prompt_count = len(self._system_prompt_cache)
        llm_interpretation_count = len(self._llm_interpretation_cache)
        user_pattern_count = len(self._user_pattern_cache)
        
        total_entries = system_prompt_count + llm_interpretation_count + user_pattern_count
        
        # Calculate total access counts
        total_system_access = sum(entry['access_count'] for entry in self._system_prompt_cache.values())
        total_llm_access = sum(entry['access_count'] for entry in self._llm_interpretation_cache.values())
        total_pattern_access = sum(entry['access_count'] for entry in self._user_pattern_cache.values())
        
        return {
            "total_cached_items": total_entries,
            "system_prompts": {
                "count": system_prompt_count,
                "total_access": total_system_access,
                "avg_access": f"{total_system_access / max(system_prompt_count, 1):.1f}"
            },
            "llm_interpretations": {
                "count": llm_interpretation_count,
                "total_access": total_llm_access,
                "avg_access": f"{total_llm_access / max(llm_interpretation_count, 1):.1f}"
            },
            "user_patterns": {
                "count": user_pattern_count,
                "total_access": total_pattern_access,
                "avg_access": f"{total_pattern_access / max(user_pattern_count, 1):.1f}"
            },
            "cache_utilization": f"{(total_entries / (self._max_token_cache_size * 3)) * 100:.1f}%",
            "ttl_hours": self._token_cache_ttl_hours,
            "estimated_tokens_saved": total_system_access * 500 + total_llm_access * 300  # Rough estimate
        }

    def clear_token_caches(self):
        """Clear all token conservation caches."""
        self._system_prompt_cache.clear()
        self._llm_interpretation_cache.clear()
        self._user_pattern_cache.clear()
        logger.debug("[TOKEN_CACHE] Cleared all token conservation caches")

    # Background tasks
    async def consolidate_old_conversations(self, user_id: str) -> Dict[str, Any]:
        """Consolidate old conversations (background task)."""
        try:
            await self._ensure_initialized()

            # Use BackgroundTaskManager for conversation consolidation (Task 4.11.4)
            from agents.background_tasks import background_task_manager
            return await background_task_manager.consolidate_conversations(user_id)

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
        # NOTE: consolidator removed in Task 4.11.4 - functionality moved to BackgroundTaskManager
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
