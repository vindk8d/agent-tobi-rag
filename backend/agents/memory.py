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

from config import get_settings
from database import db_client

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
                    # Serialize value to JSON for storage
                    value_json = json.dumps(value) if not isinstance(value, str) else value
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
        self.settings = None  # Will be loaded asynchronously
    
    async def _ensure_settings_loaded(self):
        """Load settings if not already loaded."""
        if self.settings is None:
            from config import get_settings
            self.settings = await get_settings()
    
    async def check_and_trigger_summarization(self, conversation_id: str, user_id: str) -> Optional[str]:
        """
        Check if a conversation needs summarization and trigger it automatically.
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
            # Count messages in the conversation
            print(f"            🔢 Counting messages for conversation {conversation_id[:8]}...")
            message_count = await self._count_conversation_messages(conversation_id)
            print(f"            📊 Message count: {message_count}")
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
            from database import db_client
            
            # First, get the last_summarized_at timestamp for this conversation
            conv_result = (db_client.client.table("conversations")
                          .select("last_summarized_at")
                          .eq("id", conversation_id)
                          .execute())
            
            last_summarized_at = None
            if conv_result.data and len(conv_result.data) > 0:
                last_summarized_at = conv_result.data[0].get('last_summarized_at')
            
            # Count messages since last summarization (or all messages if never summarized)
            query = db_client.client.table("messages").select("id", count="exact").eq("conversation_id", conversation_id)
            
            if last_summarized_at:
                # Only count messages created after last summarization
                query = query.gt("created_at", last_summarized_at)
            
            result = query.execute()
            return result.count if result.count is not None else 0
        except Exception as e:
            logger.error(f"Error counting messages for conversation {conversation_id}: {e}")
            return 0
    
    async def _get_conversation_details(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details by ID."""
        try:
            from database import db_client
            
            result = (db_client.client.table("conversations")
                     .select("id,user_id,title,created_at,updated_at,metadata")
                     .eq("id", conversation_id)
                     .execute())
            
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
        
        conn = await self.db_manager.get_connection()
        async with conn as connection:
            cur = await connection.cursor()
            async with cur:
                await cur.execute(query, (user_id, cutoff_date, limit))
                rows = await cur.fetchall()
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    async def _get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        try:
            print(f"            📝 Getting messages for conversation {conversation_id[:8]}...")
            
            # Use Supabase client for message retrieval
            from database import db_client
            
            result = (db_client.client.table("messages")
                     .select("role,content,created_at,user_id")
                     .eq("conversation_id", conversation_id)
                     .order("created_at")
                     .execute())
            
            messages = result.data if result.data else []
            print(f"            ✅ Retrieved {len(messages)} messages for summarization")
            return messages
            
        except Exception as e:
            print(f"            ❌ ERROR fetching messages: {e}")
            logger.error(f"Error fetching messages for conversation {conversation_id}: {e}")
            return []

    async def _get_latest_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent summary for a conversation to build incremental summaries."""
        try:
            from database import db_client
            
            result = (db_client.client.table("conversation_summaries")
                     .select("summary_text,created_at,message_count")
                     .eq("conversation_id", conversation_id)
                     .order("created_at", desc=True)
                     .limit(1)
                     .execute())
            
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
        """Get only recent messages since the last summary, or last 10 messages if no previous summary."""
        try:
            from database import db_client
            
            query = (db_client.client.table("messages")
                    .select("role,content,created_at,user_id")
                    .eq("conversation_id", conversation_id)
                    .order("created_at"))
            
            if previous_summary:
                # Get messages since the last summary
                last_summary_time = previous_summary['created_at']
                query = query.gt("created_at", last_summary_time)
                print(f"            📅 Getting messages since last summary: {last_summary_time}")
            else:
                # No previous summary - get the last 10 messages for first summarization
                # First count total messages
                count_result = (db_client.client.table("messages")
                               .select("id", count="exact")
                               .eq("conversation_id", conversation_id)
                               .execute())
                total_messages = count_result.count
                
                if total_messages > 10:
                    # Skip earlier messages, get only the latest 10
                    offset = total_messages - 10
                    query = query.range(offset, offset + 9)  # range is inclusive
                
                print(f"            🆕 No previous summary - getting last 10 of {total_messages} total messages")
            
            result = query.execute()
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
        
        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        
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
        
        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        
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
            
            from database import db_client
            
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
            from database import db_client
            from datetime import datetime
            
            db_client.client.table("conversations").update({
                "last_summarized_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()
            print(f"            ✅ Reset summarization counter for conversation {conversation_id[:8]}...")
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
            
            # Get user context using the simplified database function
            query = "SELECT * FROM get_user_context_from_conversations(%s)"
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(query, (user_id,))
                    result = await cur.fetchone()
                    
                    if result:
                        print(f"            ✅ Found user context")
                        latest_summary, conversation_count, latest_conversation_id, has_history = result
                        return {
                            "latest_summary": latest_summary,
                            "conversation_count": conversation_count or 0,
                            "latest_conversation_id": str(latest_conversation_id) if latest_conversation_id else None,
                            "has_history": bool(has_history)
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
            from database import db_client
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
            
            response = await self.llm.ainvoke(extract_prompt)
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
            from database import db_client
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
            
            response = await self.llm.ainvoke(prompt)
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
        if self.checkpointer is not None:
            return
        
        try:
            settings = await get_settings()
            connection_string = settings.supabase.postgresql_connection_string
            
            # Initialize connection pool using psycopg2 (more compatible)
            try:
                from psycopg_pool import AsyncConnectionPool
                self.connection_pool = AsyncConnectionPool(
                    conninfo=connection_string,
                    max_size=20,
                    kwargs={"autocommit": True, "prepare_threshold": 0}
                )
            except ImportError:
                # Fallback to simpler approach with psycopg2 that we know works
                logger.info("psycopg_pool not available, using fallback connection management")
                import psycopg2
                self.connection_pool = None  # Use direct connections instead
            
            if self.connection_pool:
                await self.connection_pool.open()
                # Initialize checkpointer
                self.checkpointer = AsyncPostgresSaver(self.connection_pool)
                await self.checkpointer.setup()
            else:
                # Fallback: disable checkpointer if pool unavailable
                logger.warning("Connection pool unavailable, checkpointer disabled")
                self.checkpointer = None
            
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
            
            # Initialize consolidator with cross-conversation capabilities
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
    
    async def load_conversation_state(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Load conversation state including details and messages."""
        try:
            await self._ensure_initialized()
            
            conversation_id_str = str(conversation_id)
            
            # Get conversation details
            conversation = await self._get_conversation_details(conversation_id_str)
            if not conversation:
                logger.info(f"No conversation found for ID: {conversation_id_str}")
                return None
            
            # Get conversation messages
            messages = await self._get_conversation_messages(conversation_id_str)
            
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
            
            conn = await self.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(query, (cutoff_date,))
                    deleted_count = cur.rowcount
                    await connection.commit()
                    
                    return deleted_count
            
        except Exception as e:
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
            
            # Ensure conversation exists
            await self._ensure_conversation_exists(conversation_id, user_id)
            
            # Use Supabase client directly for message storage
            try:
                from database import db_client
            except ImportError:
                # Handle Docker environment
                import sys
                sys.path.append('/app')
                from database import db_client
            
            message_data = {
                'conversation_id': conversation_id,
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
        print(f"            🔍 _ensure_conversation_exists called:")
        print(f"               - conversation_id: '{conversation_id}' (type: {type(conversation_id)})")
        print(f"               - user_id: '{user_id}' (type: {type(user_id)})")
        try:
            try:
                from backend.database import db_client
            except ImportError:
                # Handle Docker environment
                try:
                    from database import db_client
                except ImportError:
                    import sys
                    sys.path.append('/app')
                    from database import db_client
            
            # Check if conversation exists
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
                
                print(f"               🔄 Creating conversation with data:")
                print(f"                  - id: '{conversation_data['id']}'")
                print(f"                  - user_id: '{conversation_data['user_id']}'")
                print(f"                  - title: '{conversation_data['title']}'")
                
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
            
            anonymous_user_created = False
            if not user_id:
                # Generate a UUID for anonymous users to comply with database schema
                user_id = str(uuid4())
                logger.warning(f"No user_id provided or found in message, generated UUID: {user_id}")
                anonymous_user_created = True
            elif user_id == "anonymous":
                # Replace "anonymous" with a proper UUID
                user_id = str(uuid4())
                logger.warning(f"Replaced 'anonymous' with UUID: {user_id}")
                anonymous_user_created = True
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
            
            # Get users with conversations in the last 7 days
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            query = """
                SELECT DISTINCT user_id 
                FROM conversations 
                WHERE updated_at > %s 
                    AND archival_status = 'active'
                ORDER BY user_id
            """
            
            conn = await memory_manager.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(query, (cutoff_date,))
                    rows = await cur.fetchall()
                    
                    return [row[0] for row in rows if row[0]]
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    async def _cleanup_expired_long_term_memories(self):
        """Clean up expired long-term memories."""
        try:
            await memory_manager._ensure_initialized()
            
            query = """
                DELETE FROM long_term_memories 
                WHERE expires_at IS NOT NULL 
                    AND expires_at < NOW()
            """
            
            conn = await memory_manager.db_manager.get_connection()
            async with conn as connection:
                cur = await connection.cursor()
                async with cur:
                    await cur.execute(query)
                    deleted_count = cur.rowcount
                    await connection.commit()
                    
                    logger.info(f"Cleaned up {deleted_count} expired long-term memories")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired long-term memories: {e}")


# Global instances
memory_manager = MemoryManager()
memory_scheduler = MemoryScheduler() 