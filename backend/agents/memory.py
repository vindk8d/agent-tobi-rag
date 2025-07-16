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
from langgraph.store import BaseStore, BaseItem
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import get_settings
from backend.database import db_client, DatabaseManager

logger = logging.getLogger(__name__)


# =============================================================================
# LONG-TERM MEMORY STORE (LangGraph Store Implementation)
# =============================================================================

class SupabaseLongTermMemoryStore(BaseStore):
    """
    LangGraph Store implementation for long-term memory using Supabase.
    
    Provides cross-thread memory access with semantic search capabilities.
    """
    
    def __init__(self, db_manager: DatabaseManager, embeddings: Embeddings):
        """Initialize the Supabase long-term memory store."""
        self.db_manager = db_manager
        self.embeddings = embeddings
        
    async def put(self, namespace: Tuple[str, ...], key: str, value: Any, 
                  ttl_hours: Optional[int] = None) -> None:
        """Store a value in long-term memory."""
        try:
            # Generate embedding for semantic search
            content_for_embedding = self._extract_content_for_embedding(value)
            embedding = await self.embeddings.aembed_query(content_for_embedding)
            
            # Calculate expiration
            expires_at = None
            if ttl_hours:
                expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            # Store in database
            memory_id = str(uuid4())
            query = """
                INSERT INTO long_term_memories (
                    id, namespace, key, value, embedding, expires_at, 
                    created_at, access_count, last_accessed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (namespace, key) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    expires_at = EXCLUDED.expires_at,
                    last_accessed = EXCLUDED.last_accessed
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (
                        memory_id, list(namespace), key, json.dumps(value),
                        embedding, expires_at, datetime.utcnow(), 0, datetime.utcnow()
                    ))
                    await conn.commit()
            
            logger.debug(f"Stored memory: {'/'.join(namespace)}/{key}")
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def get(self, namespace: Tuple[str, ...], key: str) -> Optional[BaseItem]:
        """Retrieve a value from long-term memory."""
        try:
            query = """
                SELECT id, namespace, key, value, created_at, access_count
                FROM long_term_memories 
                WHERE namespace = %s AND key = %s
                    AND (expires_at IS NULL OR expires_at > NOW())
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (list(namespace), key))
                    row = await cur.fetchone()
                    
                    if row:
                        # Update access count
                        await cur.execute(
                            "UPDATE long_term_memories SET access_count = access_count + 1, last_accessed = NOW() WHERE id = %s",
                            (row[0],)
                        )
                        await conn.commit()
                        
                        return BaseItem(
                            value=json.loads(row[3]),
                            key=row[2],
                            namespace=tuple(row[1]),
                            created_at=row[4],
                            updated_at=row[4]
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
    

    
    async def search(self, query: str, namespace_prefix: Optional[Tuple[str, ...]] = None,
                    limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Build SQL query
            sql_query = """
                SELECT id, namespace, key, value, created_at, access_count,
                       (embedding <=> %s) as similarity_score
                FROM long_term_memories 
                WHERE (expires_at IS NULL OR expires_at > NOW())
                    AND (embedding <=> %s) < %s
            """
            params = [query_embedding, query_embedding, 1 - similarity_threshold]
            
            if namespace_prefix:
                sql_query += " AND namespace[1:%s] = %s"
                params.extend([len(namespace_prefix), list(namespace_prefix)])
            
            sql_query += " ORDER BY embedding <=> %s LIMIT %s"
            params.extend([query_embedding, limit])
            
            async with self.db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql_query, params)
                    rows = await cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            "id": row[0],
                            "namespace": row[1],
                            "key": row[2],
                            "value": json.loads(row[3]),
                            "created_at": row[4],
                            "access_count": row[5],
                            "similarity_score": float(row[6])
                        })
                    
                    return results
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def _extract_content_for_embedding(self, value: Any) -> str:
        """Extract content from value for embedding generation."""
        if isinstance(value, dict):
            if "content" in value:
                return str(value["content"])
            elif "summary" in value:
                return str(value["summary"])
            else:
                return str(value)
        else:
            return str(value)


# =============================================================================
# CONVERSATION CONSOLIDATOR (Background Processing)
# =============================================================================

class ConversationConsolidator:
    """
    Handles the transition from short-term to long-term memory.
    """
    
    def __init__(self, db_manager: DatabaseManager, memory_store: SupabaseLongTermMemoryStore,
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
                AND c.is_archived = false
                AND cs.conversation_id IS NULL
            ORDER BY c.updated_at ASC
            LIMIT %s
        """
        
        async with self.db_manager.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (user_id, cutoff_date, limit))
                rows = await cur.fetchall()
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    async def _get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        query = """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """
        
        async with self.db_manager.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (conversation_id,))
                rows = await cur.fetchall()
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    async def _generate_conversation_summary(self, conversation: Dict[str, Any], 
                                           messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the conversation using LLM."""
        # Format messages for LLM
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
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
    
    async def _store_conversation_summary(self, user_id: str, conversation: Dict[str, Any], 
                                        summary: Dict[str, Any]) -> None:
        """Store conversation summary in long-term memory."""
        # Store in conversation_summaries table
        summary_id = str(uuid4())
        embedding = await self.embeddings.aembed_query(summary['content'])
        
        query = """
            INSERT INTO conversation_summaries (
                id, user_id, conversation_id, summary, message_count, 
                date_range, embedding, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        async with self.db_manager.get_connection() as conn:
            async with conn.cursor() as cur:
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
            SET is_archived = true, archived_at = %s
            WHERE id = %s
        """
        
        async with self.db_manager.get_connection() as conn:
            async with conn.cursor() as cur:
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
            self.db_manager = DatabaseManager()
            
            # Initialize default LLM and embeddings
            if self.llm is None:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1000)
            
            if self.embeddings is None:
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Initialize long-term memory store
            self.long_term_store = SupabaseLongTermMemoryStore(
                db_manager=self.db_manager,
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
            
            await self.long_term_store.put(
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
            results = await self.long_term_store.search(
                query=current_query,
                namespace_prefix=(user_id,),
                limit=max_contexts
            )
            
            # Format results
            contexts = []
            for result in results:
                if result.get('similarity_score', 0) > 0.7:  # Relevance threshold
                    contexts.append({
                        "type": result.get('value', {}).get('type', 'memory'),
                        "content": result.get('value', {}).get('summary', str(result.get('value', {}))),
                        "score": result.get('similarity_score', 0),
                        "namespace": result.get('namespace', []),
                        "created_at": result.get('created_at')
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
                WHERE is_archived = true 
                    AND archived_at < %s
                    AND id IN (
                        SELECT conversation_id 
                        FROM conversation_summaries 
                        WHERE conversation_id = conversations.id
                    )
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (cutoff_date,))
                    deleted_count = cur.rowcount
                    await conn.commit()
                    
                    return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup archived conversations: {e}")
            return 0
    

    
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
            content = str(message.content).lower()
            
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
            if hasattr(message, 'type') and message.type == 'human':
                content = message.content.lower()
                
                # Extract preferences
                if any(phrase in content for phrase in ["i prefer", "i like", "i want"]):
                    insights.append({
                        "type": "preference",
                        "content": message.content,
                        "extracted_at": datetime.now().isoformat()
                    })
                
                # Extract facts
                elif any(phrase in content for phrase in ["my name is", "i am", "i work at"]):
                    insights.append({
                        "type": "personal_fact",
                        "content": message.content,
                        "extracted_at": datetime.now().isoformat()
                    })
        
        return insights
    
    async def store_conversation_insights(self, agent_type: str, user_id: str, 
                                        messages: List[BaseMessage], conversation_id: UUID) -> None:
        """
        Store insights using agent-specific extraction logic.
        """
        try:
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
                async with conn.cursor() as cur:
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
                    AND is_archived = false
                ORDER BY user_id
            """
            
            async with memory_manager.db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
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