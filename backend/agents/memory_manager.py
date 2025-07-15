"""
Conversation Memory Manager using LangGraph PostgreSQL persistence with Supabase.

This module provides proper LangGraph persistence following best practices:
- PostgreSQL checkpointer for true persistence
- Automatic message history management
- No manual message appending needed
"""

import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import psycopg

from backend.config import get_settings
from backend.agents.state import AgentState, ConversationMemory
from backend.database import db_client

import logging
logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Memory manager for conversation persistence using LangGraph PostgreSQL checkpointer.
    
    This class provides LangGraph-compatible conversation memory management:
    - PostgreSQL checkpointer for persistent conversation storage
    - Automatic message history through LangGraph's add_messages annotation
    - No manual message handling required
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.checkpointer = None
        self.connection_pool = None
        logger.info("ConversationMemoryManager initialized")
        
    async def _ensure_initialized(self):
        """Ensure the memory manager is initialized."""
        if self.checkpointer is not None:
            return
        
        try:
            settings = await get_settings()
            connection_string = settings.supabase.postgresql_connection_string
            
            logger.info(f"Initializing AsyncPostgresSaver with async connection pool")
            
            # Create async connection pool for AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
            }
            
            self.connection_pool = AsyncConnectionPool(
                conninfo=connection_string,
                max_size=20,
                kwargs=connection_kwargs
            )
            
            # Open the connection pool
            await self.connection_pool.open()
            
            # Create AsyncPostgresSaver with async connection pool
            self.checkpointer = AsyncPostgresSaver(self.connection_pool)
            
            # Setup the checkpointer tables
            await self.checkpointer.setup()
            
            logger.info("AsyncPostgresSaver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationMemoryManager: {e}")
            raise
    
    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """Get the initialized PostgreSQL checkpointer."""
        await self._ensure_initialized()
        return self.checkpointer
    
    async def save_conversation_to_database(self, conversation_id: UUID, user_id: str, 
                                         messages: List[BaseMessage], summary: Optional[str] = None):
        """
        Save conversation data to Supabase for additional context.
        This complements the LangGraph checkpointing.
        """
        try:
            # Save conversation record
            conversation_data = {
                "id": str(conversation_id),
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": {
                    "summary": summary,
                    "last_message_at": datetime.now().isoformat(),
                    "message_count": len(messages)
                }
            }
            
            # Use upsert to update existing conversation
            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations").upsert(conversation_data).execute()
            )
            
            logger.info(f"Saved conversation {conversation_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to save conversation to database: {e}")
    
    async def load_conversation_context(self, conversation_id: UUID, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation context from Supabase.
        This provides additional context that complements LangGraph persistence.
        """
        try:
            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations")
                .select("*")
                .eq("id", str(conversation_id))
                .eq("user_id", user_id)
                .execute()
            )
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to load conversation context: {e}")
            return None
    
    async def get_conversation_config(self, conversation_id: UUID) -> Dict[str, Any]:
        """
        Get the configuration for LangGraph thread management.
        """
        return {
            "configurable": {
                "thread_id": str(conversation_id)
            }
        }
    
    async def cleanup_old_conversations(self, days_old: int = 30):
        """
        Clean up old conversations from both LangGraph checkpoints and database.
        """
        try:
            # Clean up database conversations older than specified days
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations")
                .delete()
                .lt("created_at", cutoff_date.isoformat())
                .execute()
            )
            
            logger.info(f"Cleaned up old conversations: {len(result.data)} removed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")

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
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.connection_pool:
            try:
                # For async connection pool, we need to close it properly
                # This is just a best-effort cleanup in destructor
                if hasattr(self.connection_pool, 'close'):
                    # If it's still open, we can't await in __del__, so we'll just try to close
                    try:
                        self.connection_pool.close()
                    except Exception:
                        pass
            except Exception:
                pass  # Ignore errors during cleanup


# Global instance
memory_manager = ConversationMemoryManager() 