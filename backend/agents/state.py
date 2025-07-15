"""
LangGraph state schema for the RAG agent following best practices.
Uses proper message history management with add_messages annotation.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from uuid import UUID
from datetime import datetime
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph state schema for conversational RAG agent with proper message persistence.
    
    Follows LangGraph best practices:
    - Uses add_messages annotation for automatic message history management
    - Supports conversation persistence via PostgreSQL checkpointer
    - Maintains context through LangGraph's built-in mechanisms
    """
    
    # Core conversation state with proper message history management
    # Using add_messages annotation for automatic message appending
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Persistent memory identifiers
    conversation_id: Optional[UUID]
    user_id: Optional[str]
    
    # RAG context and results
    retrieved_docs: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    
    # Optional conversation summary (for very long conversations)
    conversation_summary: Optional[str]


class ConversationMemory(TypedDict):
    """
    Conversation memory structure for persistent storage.
    Maps to database tables: conversations and messages.
    """
    conversation_id: UUID
    user_id: str
    messages: List[BaseMessage]
    conversation_summary: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class RetrievalContext(TypedDict):
    """
    Context structure for document retrieval results.
    Used internally by RAG tools.
    """
    query: str
    documents: List[Dict[str, Any]]
    similarity_threshold: float
    top_k: int
    search_metadata: Dict[str, Any] 