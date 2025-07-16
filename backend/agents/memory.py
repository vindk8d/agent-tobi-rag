"""
Consolidated Memory System for LangGraph Agent.

This module provides comprehensive memory management combining:
- Short-term memory (LangGraph PostgreSQL checkpointer)
- Long-term memory (Supabase with semantic search)
- Conversation consolidation (background processing)

Simplified architecture following the principle of "simple yet effective".
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore, Item
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import get_settings
from backend.database import db_client

logger = logging.getLogger(__name__)


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
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
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
                                'p_namespace': namespace,
                                'p_key': key,
                                'p_value': value,
                                'p_embedding': embedding,
                                'p_memory_type': memory_type,
                                'p_expiry_at': expiry_at
                            }).execute()
                        )
                        self.result = result
                elif "get_long_term_memory" in query:
                    if params and len(params) >= 2:
                        namespace, key = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('get_long_term_memory', {
                                'p_namespace': namespace,
                                'p_key': key
                            }).execute()
                        )
                        self.result = result
                elif "delete_long_term_memory" in query:
                    if params and len(params) >= 2:
                        namespace, key = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('delete_long_term_memory', {
                                'p_namespace': namespace,
                                'p_key': key
                            }).execute()
                        )
                        self.result = result
                elif "search_long_term_memories_by_prefix" in query:
                    if params and len(params) >= 4:
                        embedding, namespace_prefix, threshold, match_count = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('search_long_term_memories_by_prefix', {
                                'query_embedding': embedding,
                                'namespace_prefix': namespace_prefix,
                                'similarity_threshold': threshold,
                                'match_count': match_count
                            }).execute()
                        )
                        self.result = result
                elif "search_conversation_summaries" in query:
                    if params and len(params) >= 5:
                        embedding, user_id, threshold, match_count, summary_type = params
                        result = await asyncio.to_thread(
                            lambda: self.client.client.rpc('search_conversation_summaries', {
                                'query_embedding': embedding,
                                'target_user_id': user_id,
                                'similarity_threshold': threshold,
                                'match_count': match_count,
                                'summary_type_filter': summary_type
                            }).execute()
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
                                'p_user_id': user_id,
                                'p_memory_namespace': namespace,
                                'p_memory_key': key,
                                'p_access_context': access_context,
                                'p_retrieval_method': retrieval_method
                            }).execute()
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
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._initialized = True
    
    # Required BaseStore interface methods
    async def aget(self, namespace: Tuple[str, ...], key: str) -> Optional[Item]:
        """Get a value from the store."""
        await self._ensure_initialized()
        
        try:
            conn = await self.db_manager.get_connection()
            async with conn:
                cur = await conn.cursor()
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
                        await conn.commit()
                        
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
            
            conn = await self.db_manager.get_connection()
            async with conn:
                cur = await conn.cursor()
                async with cur:
                    # Serialize value to JSON for storage
                    value_json = json.dumps(value) if not isinstance(value, str) else value
                    await cur.execute(
                        "SELECT put_long_term_memory(%s, %s, %s, %s, %s, %s)",
                        (list(namespace), key, value_json, embedding, 'semantic', expiry_at)
                    )
                    await conn.commit()
        
        except Exception as e:
            logger.error(f"Error putting value into store: {e}")
            raise
    
    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a value from the store."""
        await self._ensure_initialized()
        
        try:
            conn = await self.db_manager.get_connection()
            async with conn:
                cur = await conn.cursor()
                async with cur:
                    await cur.execute(
                        "SELECT delete_long_term_memory(%s, %s)",
                        (list(namespace), key)
                    )
                    await conn.commit()
        
        except Exception as e:
            logger.error(f"Error deleting value from store: {e}")
            raise
    
    async def asearch(self, namespace_prefix: Tuple[str, ...]) -> List[Tuple[Tuple[str, ...], str, Any]]:
        """Search for values by namespace prefix."""
        await self._ensure_initialized()
        
        try:
            conn = await self.db_manager.get_connection()
            async with conn:
                cur = await conn.cursor()
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
            # Handle both sync and async embeddings
            if hasattr(self.embeddings, 'aembed_query'):
                query_embedding = await self.embeddings.aembed_query(query)
            else:
                # Check if embed_query returns a coroutine
                result = self.embeddings.embed_query(query)
                if asyncio.iscoroutine(result):
                    query_embedding = await result
                else:
                    query_embedding = await asyncio.to_thread(
                        lambda: result
                    )
            
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


# =============================================================================
# CONVERSATION CONSOLIDATOR (Background Processing)
# =============================================================================

class ConversationConsolidator:
    """
    Handles the transition from short-term to long-term memory.
    """
    
    def __init__(self, db_manager: SimpleDBManager, memory_store: SupabaseLongTermMemoryStore,
                 llm: BaseLanguageModel, embeddings: Embeddings):
        """Initialize the conversation consolidator."""
        self.db_manager = db_manager
        self.memory_store = memory_store
        self.llm = llm
        self.embeddings = embeddings
        
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
                    # Get messages for this conversation
                    messages = await self._get_conversation_messages(conversation['id'])
                    
                    if not messages:
                        continue
                    
                    # Generate summary
                    summary = await self._generate_conversation_summary(conversation, messages)
                    
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
        cutoff_date = datetime.utcnow() - timedelta(days=7)  # 7 days old
        
        query = """
            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at, c.metadata
            FROM conversations c
            LEFT JOIN conversation_summaries cs ON c.id = cs.conversation_id
            WHERE c.user_id = %s
                AND c.updated_at < %s
                AND c.archival_status = 'active'
                AND cs.conversation_id IS NULL
            ORDER BY c.updated_at ASC
            LIMIT %s
        """
        
        async with self.db_manager.get_connection() as conn:
            cur = await conn.cursor()
            async with cur:
                await cur.execute(query, (user_id, cutoff_date, limit))
                rows = await cur.fetchall()
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    async def _get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        query = """
            SELECT role, content, created_at, user_id
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """
        
        async with self.db_manager.get_connection() as conn:
            cur = await conn.cursor()
            async with cur:
                await cur.execute(query, (conversation_id,))
                rows = await cur.fetchall()
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
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
        
        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        
        return {
            "content": response.content,
            "message_count": len(messages),
            "date_range": {
                "start": messages[0]['created_at'].isoformat(),
                "end": messages[-1]['created_at'].isoformat()
            }
        }
    
    def _format_role_for_summary(self, role: str) -> str:
        """Format role for summary display."""
        role_mapping = {
            'bot': 'ASSISTANT',
            'human': 'USER', 
            'HITL': 'HUMAN_SUPPORT'
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
        
        query = """
            INSERT INTO conversation_summaries (
                id, user_id, conversation_id, summary, message_count, 
                date_range, embedding, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        async with self.db_manager.get_connection() as conn:
            cur = await conn.cursor()
            async with cur:
                await cur.execute(query, (
                    summary_id, user_id, conversation['id'],
                    summary['content'], summary['message_count'],
                    summary['date_range'], embedding, datetime.utcnow()
                ))
                await conn.commit()
        
        # Also store in LangGraph memory store
        await self.memory_store.put(
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
    
    async def _archive_conversation(self, conversation_id: str) -> None:
        """Mark conversation as archived."""
        query = """
            UPDATE conversations 
            SET archival_status = 'archived', archived_at = %s
            WHERE id = %s
        """
        
        async with self.db_manager.get_connection() as conn:
            cur = await conn.cursor()
            async with cur:
                await cur.execute(query, (datetime.utcnow(), conversation_id))
                await conn.commit()


# =============================================================================
# MAIN MEMORY MANAGER (Orchestrator)
# =============================================================================

class MemoryManager:
    """
    Main memory manager that orchestrates short-term and long-term memory.
    Designed to be reusable across multiple agents.
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
        
        logger.info("MemoryManager initialized")
    
    async def _ensure_initialized(self):
        """Ensure all components are initialized."""
        if self.checkpointer is not None:
            return
        
        try:
            settings = await get_settings()
            connection_string = settings.supabase.postgresql_connection_string
            
            # Initialize connection pool
            from psycopg_pool import AsyncConnectionPool
            
            self.connection_pool = AsyncConnectionPool(
                conninfo=connection_string,
                max_size=20,
                kwargs={"autocommit": True, "prepare_threshold": 0}
            )
            
            await self.connection_pool.open()
            
            # Initialize checkpointer
            self.checkpointer = AsyncPostgresSaver(self.connection_pool)
            await self.checkpointer.setup()
            
            # Initialize database manager
            self.db_manager = SimpleDBManager()
            
            # Initialize default LLM and embeddings
            if self.llm is None:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1000)
            
            if self.embeddings is None:
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Initialize long-term memory store
            self.long_term_store = SupabaseLongTermMemoryStore(
                embeddings=self.embeddings
            )
            
            # Initialize consolidator
            self.consolidator = ConversationConsolidator(
                db_manager=self.db_manager,
                memory_store=self.long_term_store,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            logger.info("MemoryManager components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            raise
    
    # Short-term memory methods
    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """Get the LangGraph checkpointer."""
        await self._ensure_initialized()
        return self.checkpointer
    
    async def get_conversation_config(self, conversation_id: UUID) -> Dict[str, Any]:
        """Get configuration for LangGraph thread management."""
        return {"configurable": {"thread_id": str(conversation_id)}}
    
    # Long-term memory methods
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
            return await self.consolidator.consolidate_conversations(user_id)
            
        except Exception as e:
            logger.error(f"Failed to consolidate conversations: {e}")
            return {"consolidated_count": 0, "error": str(e)}
    
    async def cleanup_old_archived_conversations(self, days_threshold: int = 30) -> int:
        """Clean up very old archived conversations."""
        try:
            await self._ensure_initialized()
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            query = """
                DELETE FROM conversations 
                WHERE archival_status = 'archived' 
                    AND archived_at < %s
                    AND id IN (
                        SELECT conversation_id 
                        FROM conversation_summaries 
                        WHERE conversation_id = conversations.id
                    )
            """
            
            async with self.db_manager.get_connection() as conn:
                cur = await conn.cursor()
                async with cur:
                    await cur.execute(query, (cutoff_date,))
                    deleted_count = cur.rowcount
                    await conn.commit()
                    
                    return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup archived conversations: {e}")
            return 0
    
    # =============================================================================
    # MESSAGE STORAGE METHODS (Row-by-row storage in messages table)
    # =============================================================================
    
    async def save_message_to_database(self, conversation_id: str, user_id: str, 
                                     role: str, content: str, 
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save individual message to the messages table.
        
        Args:
            conversation_id: UUID of the conversation
            user_id: UUID of the user
            role: Message role ('human', 'bot', 'system')
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Message ID if successful, None if failed
        """
        try:
            await self._ensure_initialized()
            
            # Ensure conversation exists
            await self._ensure_conversation_exists(conversation_id, user_id)
            
            # Use Supabase client directly for message storage
            from backend.database import db_client
            
            message_data = {
                'conversation_id': conversation_id,
                'user_id': user_id,
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
                logger.debug(f"Message saved to database: {message_id} (role: {role})")
                return message_id
            else:
                logger.error("Failed to save message to database - no data returned")
                return None
                
        except Exception as e:
            logger.error(f"Error saving message to database: {e}")
            return None
    
    async def _ensure_conversation_exists(self, conversation_id: str, user_id: str) -> None:
        """Ensure conversation record exists in the conversations table."""
        try:
            from backend.database import db_client
            
            # Check if conversation exists - wrap in asyncio.to_thread
            result = await asyncio.to_thread(
                lambda: db_client.client.table('conversations').select('id').eq('id', conversation_id).execute()
            )
            
            if not result.data:
                # Create conversation record
                conversation_data = {
                    'id': conversation_id,
                    'user_id': user_id,
                    'title': 'New Conversation',
                    'metadata': {'created_by': 'memory_manager'}
                }
                
                # Wrap synchronous database call in asyncio.to_thread
                result = await asyncio.to_thread(
                    lambda: db_client.client.table('conversations').insert(conversation_data).execute()
                )
                
                if result.data:
                    logger.debug(f"Created conversation record: {conversation_id}")
                else:
                    logger.error(f"Failed to create conversation record: {conversation_id}")
                    
        except Exception as e:
            logger.error(f"Error ensuring conversation exists: {e}")
    
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
                                     agent_type: str = "unknown") -> Optional[str]:
        """
        High-level method to store message from agent processing.
        Designed to be reusable across different agents.
        
        Args:
            message: LangChain message object or dict
            config: LangGraph configuration containing conversation info
            agent_type: Type of agent for logging purposes
            
        Returns:
            Message ID if successful, None if failed
        """
        try:
            # Extract conversation ID from config
            conversation_id = self._extract_conversation_id_from_config(config)
            if not conversation_id:
                logger.warning(f"No conversation ID found in config for {agent_type} agent")
                return None
            
            # Extract user ID from message
            user_id = self._extract_user_id_from_message(message)
            anonymous_user_created = False
            if not user_id:
                # Generate a UUID for anonymous users to comply with database schema
                user_id = str(uuid4())
                logger.warning(f"No user_id found in message, generated UUID: {user_id}")
                anonymous_user_created = True
            elif user_id == "anonymous":
                # Replace "anonymous" with a proper UUID
                user_id = str(uuid4())
                logger.warning(f"Replaced 'anonymous' with UUID: {user_id}")
                anonymous_user_created = True
            
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
                
                # Map LangChain message types to our role system
                if hasattr(message, 'type'):
                    role_mapping = {
                        'human': 'human',
                        'ai': 'bot',
                        'system': 'system'
                    }
                    role = role_mapping.get(message.type, 'unknown')
                elif hasattr(message, 'role'):
                    role = message.role
            
            # Skip empty messages
            if not content.strip():
                logger.debug(f"Skipping empty message from {agent_type} agent")
                return None
            
            # Add agent type to metadata
            metadata['agent_type'] = agent_type
            metadata['stored_at'] = datetime.utcnow().isoformat()
            
            # Create anonymous user record if needed
            if anonymous_user_created:
                try:
                    from backend.database import db_client
                    user_data = {
                        'id': user_id,
                        'email': f'anonymous_{user_id[:8]}@system.generated',
                        'display_name': f'Anonymous User {user_id[:8]}',
                        'user_type': 'customer',
                        'metadata': {'auto_generated': True, 'agent_type': agent_type}
                    }
                    # Wrap synchronous database call in asyncio.to_thread
                    await asyncio.to_thread(
                        lambda: db_client.client.table('users').insert(user_data).execute()
                    )
                    logger.info(f"Created anonymous user record: {user_id}")
                except Exception as e:
                    if 'duplicate key' not in str(e):
                        logger.error(f"Failed to create anonymous user: {e}")
            
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
    
    # =============================================================================
    # END MESSAGE STORAGE METHODS
    # =============================================================================
    
    async def load_conversation_state(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Load conversation state from checkpointer and long-term memory."""
        try:
            await self._ensure_initialized()
            
            # Get conversation config
            config = await self.get_conversation_config(conversation_id)
            
            # Load from checkpointer (short-term memory)
            checkpointer_state = await self.checkpointer.aget_tuple(config)
            
            # Load from long-term memory if available
            long_term_context = await self.get_relevant_context(
                user_id="anonymous",  # We'll get user_id from state if available
                current_query="conversation_context",
                max_contexts=3
            )
            
            # Combine both sources
            state = {}
            
            if checkpointer_state:
                state.update(checkpointer_state.values or {})
            
            if long_term_context:
                state['long_term_context'] = long_term_context
            
            return state if state else None
            
        except Exception as e:
            logger.error(f"Failed to load conversation state: {e}")
            return None
    

    
    # Plugin system for agent-specific memory logic
    def register_insight_extractor(self, agent_type: str, extractor_func):
        """Register an insight extractor for a specific agent type."""
        self.insight_extractors[agent_type] = extractor_func
        logger.info(f"Registered insight extractor for agent type: {agent_type}")
    
    def register_memory_filter(self, agent_type: str, filter_func):
        """Register a memory filter for a specific agent type."""
        self.memory_filters[agent_type] = filter_func
        logger.info(f"Registered memory filter for agent type: {agent_type}")
    
    async def should_store_memory(self, agent_type: str, messages: List[BaseMessage]) -> bool:
        """
        Generic memory storage decision logic with agent-specific filtering.
        """
        # Use agent-specific filter if available
        if agent_type in self.memory_filters:
            return await self.memory_filters[agent_type](messages)
        
        # Default logic for any agent
        if len(messages) < 2:
            return False
        
        # Check if recent messages contain storable information
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        
        for message in recent_messages:
            # Handle both new message format and LangChain message objects
            content = None
            
            if hasattr(message, 'content'):
                content = str(message.content).lower()
            elif isinstance(message, dict) and message.get('content'):
                content = str(message.get('content', '')).lower()
            
            if content:
                # Look for preference or fact indicators
                if any(phrase in content for phrase in [
                    "i prefer", "i like", "i want", "i need",
                    "my name is", "i am", "i work at", "i live in",
                    "remember", "next time", "always", "never"
                ]):
                    return True
        
        return False
    
    async def extract_conversation_insights(self, agent_type: str, messages: List[BaseMessage], 
                                          conversation_id: UUID) -> List[Dict[str, Any]]:
        """
        Generic insight extraction with agent-specific extractors.
        """
        # Use agent-specific extractor if available
        if agent_type in self.insight_extractors:
            return await self.insight_extractors[agent_type](messages, conversation_id)
        
        # Default extraction logic
        insights = []
        for message in messages:
            # Handle both new message format and LangChain message objects
            is_human_message = False
            content = None
            
            if hasattr(message, 'type') and message.type == 'human':
                is_human_message = True
                content = message.content
            elif hasattr(message, 'role') and message.role == 'human':
                is_human_message = True
                content = message.content
            elif isinstance(message, dict) and message.get('role') == 'human':
                is_human_message = True
                content = message.get('content', '')
            
            if is_human_message and content:
                content_lower = content.lower()
                
                # Extract preferences
                if any(phrase in content_lower for phrase in ["i prefer", "i like", "i want"]):
                    insights.append({
                        "type": "preference",
                        "content": content,
                        "extracted_at": datetime.now().isoformat()
                    })
                
                # Extract facts
                elif any(phrase in content_lower for phrase in ["my name is", "i am", "i work at"]):
                    insights.append({
                        "type": "personal_fact",
                        "content": content,
                        "extracted_at": datetime.now().isoformat()
                    })
        
        return insights
    
    async def store_conversation_insights(self, agent_type: str, user_id: str, 
                                        messages: List[BaseMessage], conversation_id: UUID) -> None:
        """
        Store insights using agent-specific extraction logic.
        """
        try:
            # If user_id is not provided, try to extract from messages
            if not user_id:
                for message in messages:
                    extracted_user_id = self._extract_user_id_from_message(message)
                    if extracted_user_id:
                        user_id = extracted_user_id
                        break
                
                if not user_id:
                    logger.warning(f"No user_id found for conversation {conversation_id}, skipping insights storage")
                    return
            
            # Extract insights using agent-specific or default logic
            insights = await self.extract_conversation_insights(agent_type, messages, conversation_id)
            
            # Store insights in long-term memory
            if insights:
                for i, insight in enumerate(insights):
                    await self.store_long_term_memory(
                        user_id=user_id,
                        namespace=["insights", agent_type],
                        key=f"conversation_{conversation_id}_{i}",
                        value=insight,
                        ttl_hours=24*30*6  # 6 months
                    )
                
                logger.info(f"Stored {len(insights)} insights for {agent_type} agent, user {user_id}")
        
        except Exception as e:
            logger.error(f"Error storing conversation insights: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.connection_pool:
            try:
                await self.connection_pool.close()
                logger.info("Connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
        
        self.checkpointer = None
        self.connection_pool = None
        self.long_term_store = None
        self.consolidator = None
        self.db_manager = None


# =============================================================================
# MEMORY SCHEDULER (Background Processing)
# =============================================================================

class MemoryScheduler:
    """
    Background task scheduler for memory management operations.
    Handles consolidation and cleanup without blocking the main agent flow.
    """
    
    def __init__(self):
        self.consolidation_interval = 3600  # 1 hour
        self.cleanup_interval = 86400  # 24 hours
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the background scheduler."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting memory scheduler")
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._consolidation_scheduler()),
            asyncio.create_task(self._cleanup_scheduler())
        ]
        
        logger.info("Memory scheduler started successfully")
    
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
    
    async def _consolidation_scheduler(self):
        """Background task for periodic conversation consolidation."""
        while self.running:
            try:
                logger.info("Starting periodic memory consolidation")
                
                # Get active users (this would need to be implemented based on your user system)
                active_users = await self._get_active_users()
                
                consolidation_results = []
                for user_id in active_users:
                    try:
                        result = await memory_manager.consolidate_old_conversations(user_id)
                        consolidation_results.append({
                            "user_id": user_id,
                            "result": result
                        })
                        
                        # Small delay between users to avoid overwhelming the system
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
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
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
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _cleanup_expired_long_term_memories(self):
        """Clean up expired long-term memories."""
        try:
            await memory_manager._ensure_initialized()
            
            query = """
                DELETE FROM long_term_memories 
                WHERE expires_at IS NOT NULL 
                    AND expires_at < NOW()
            """
            
            async with memory_manager.db_manager.get_connection() as conn:
                cur = await conn.cursor()
                async with cur:
                    await cur.execute(query)
                    deleted_count = cur.rowcount
                    await conn.commit()
                    
                    logger.info(f"Cleaned up {deleted_count} expired long-term memories")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired long-term memories: {e}")
    
    async def _get_active_users(self) -> list[str]:
        """Get list of active users for consolidation."""
        try:
            await memory_manager._ensure_initialized()
            
            # Get users with conversations in the last 7 days
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            query = """
                SELECT DISTINCT user_id 
                FROM conversations 
                WHERE updated_at > %s 
                    AND archival_status = 'active'
                ORDER BY user_id
            """
            
            async with memory_manager.db_manager.get_connection() as conn:
                cur = await conn.cursor()
                async with cur:
                    await cur.execute(query, (cutoff_date,))
                    rows = await cur.fetchall()
                    
                    return [row[0] for row in rows if row[0]]
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    async def trigger_consolidation(self, user_id: str) -> Dict[str, Any]:
        """Manually trigger consolidation for a specific user."""
        try:
            logger.info(f"Manually triggering consolidation for user {user_id}")
            result = await memory_manager.consolidate_old_conversations(user_id)
            logger.info(f"Manual consolidation complete for user {user_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in manual consolidation for user {user_id}: {e}")
            return {"consolidated_count": 0, "error": str(e)}
    
    async def trigger_cleanup(self) -> Dict[str, Any]:
        """Manually trigger cleanup."""
        try:
            logger.info("Manually triggering cleanup")
            deleted_count = await memory_manager.cleanup_old_archived_conversations(90)
            await self._cleanup_expired_long_term_memories()
            
            result = {"deleted_conversations": deleted_count}
            logger.info(f"Manual cleanup complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in manual cleanup: {e}")
            return {"deleted_conversations": 0, "error": str(e)}


# Global instances
memory_manager = MemoryManager()
memory_scheduler = MemoryScheduler() 