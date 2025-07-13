"""
LangGraph state schema for the RAG agent.
Simple, minimal state for effective tracing and conversation management.
"""

from typing import List, Dict, Any, Optional, TypedDict
from uuid import UUID
from datetime import datetime
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Minimal LangGraph state schema for conversational RAG agent.
    
    This state passes through the entire agent graph, maintaining:
    - User interaction context
    - Retrieved documents from vector search
    - Conversation history for context
    - Generated response and metadata
    """
    
    # User input
    user_query: str
    conversation_id: Optional[UUID]
    user_id: Optional[str]
    
    # Retrieved documents from RAG
    retrieved_docs: List[Dict[str, Any]]
    retrieval_metadata: Dict[str, Any]
    
    # Conversation context
    conversation_history: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    
    # Generated response
    response: str
    response_metadata: Dict[str, Any]
    
    # Agent workflow metadata
    current_step: str
    error_message: Optional[str]
    timestamp: datetime
    
    # Sources and attribution
    sources: List[Dict[str, Any]]
    confidence_score: Optional[float]
    
    # Messages for tool-calling (LangGraph requirement)
    messages: List[BaseMessage]


class ConversationMemory(TypedDict):
    """
    Simple conversation memory structure for maintaining context.
    """
    conversation_id: UUID
    messages: List[Dict[str, str]]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class RetrievalContext(TypedDict):
    """
    Context structure for document retrieval results.
    """
    query: str
    documents: List[Dict[str, Any]]
    similarity_threshold: float
    top_k: int
    search_metadata: Dict[str, Any] 