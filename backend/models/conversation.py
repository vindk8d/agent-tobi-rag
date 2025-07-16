"""
Conversation models for chat interactions
"""

from typing import List, Optional, Dict, Any
from pydantic import Field
from enum import Enum
from uuid import UUID

from .base import BaseModel, IdentifiedModel


class ConversationType(str, Enum):
    """Type of conversation"""
    CHAT = "chat"
    QUERY = "query"
    FEEDBACK = "feedback"


class MessageRole(str, Enum):
    """Role of the message sender"""
    HUMAN = "human"
    BOT = "bot"
    HITL = "HITL"
    
    # Keep deprecated aliases for backward compatibility
    USER = "human"  # Deprecated: use HUMAN
    ASSISTANT = "bot"  # Deprecated: use BOT
    SYSTEM = "human"  # Deprecated: system messages treated as human


class Message(BaseModel):
    """Individual message in a conversation"""
    
    role: MessageRole
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationRequest(BaseModel):
    """Request model for conversation endpoints"""
    
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[UUID] = None
    conversation_type: ConversationType = ConversationType.CHAT
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    
    
class ConversationResponse(BaseModel):
    """Response model for conversation endpoints"""
    
    message: str
    conversation_id: UUID
    response_metadata: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    

class ConversationHistory(IdentifiedModel):
    """Complete conversation history model"""
    
    conversation_id: UUID
    user_id: Optional[str] = None
    messages: List[Message] = []
    conversation_type: ConversationType = ConversationType.CHAT
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True 