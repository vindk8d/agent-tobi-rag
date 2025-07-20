"""
Chat API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field

from models.base import APIResponse
from models.conversation import ConversationRequest, ConversationResponse, MessageRole
from database import db_client
from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize the agent (will be lazily loaded)
_agent_instance = None

async def get_agent():
    """Get or create the agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = UnifiedToolCallingRAGAgent()
    return _agent_instance


class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuity")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    include_sources: bool = Field(True, description="Whether to include sources in response")

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    message: str = Field(..., description="Agent response message")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used")
    suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    metadata: Dict[str, Any] = Field(default={}, description="Additional response metadata")

class ChatMessage(BaseModel):
    """Chat message model for history"""
    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (human/user/ai/assistant/bot)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    conversation_id: str = Field(..., description="Conversation ID")

@router.get("/recent-messages/{user_id}", response_model=APIResponse[List[ChatMessage]])
async def get_recent_messages(user_id: str, limit: int = 20):
    """
    Get recent messages for a user across all their conversations
    """
    try:
        logger.info(f"Fetching recent messages for user {user_id}")
        
        # Get recent conversations for the user
        conversations_result = db_client.client.table('conversations')\
            .select('id')\
            .eq('user_id', user_id)\
            .order('updated_at', desc=True)\
            .limit(5)\
            .execute()
        
        if not conversations_result.data:
            return APIResponse.success_response(
                data=[],
                message="No conversations found for user"
            )
        
        conversation_ids = [conv['id'] for conv in conversations_result.data]
        
        # Get recent messages from these conversations
        messages_result = db_client.client.table('messages')\
            .select('id, role, content, created_at, conversation_id')\
            .in_('conversation_id', conversation_ids)\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        
        # Convert to ChatMessage format
        chat_messages = []
        for msg in messages_result.data:
            chat_messages.append(ChatMessage(
                id=msg['id'],
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['created_at'],
                conversation_id=msg['conversation_id']
            ))
        
        # Reverse to show chronologically (oldest first)
        chat_messages.reverse()
        
        return APIResponse.success_response(
            data=chat_messages,
            message=f"Retrieved {len(chat_messages)} recent messages"
        )
        
    except Exception as e:
        logger.error(f"Error fetching recent messages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch recent messages: {str(e)}"
        )

@router.post("/message", response_model=APIResponse[ChatResponse])
async def send_message(request: ChatRequest):
    """
    Send a message to the sales copilot agent and get a response
    """
    try:
        # Get the agent instance
        agent = await get_agent()
        
        # Generate conversation_id if not provided
        conversation_id = request.conversation_id or str(uuid4())
        
        logger.info(f"Processing chat message for conversation {conversation_id}")
        
        # Invoke the agent
        result = await agent.invoke(
            user_query=request.message,
            conversation_id=conversation_id,
            user_id=request.user_id
        )
        
        # Extract the final AI message from the result
        final_message = ""
        sources = []
        suggestions = []
        
        if result and 'messages' in result:
            # Get the last AI message
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and msg.content and not msg.content.startswith('['):
                    final_message = msg.content
                    break
        
        # Extract sources if available
        if result and 'sources' in result:
            sources = result.get('sources', [])
        
        # Extract retrieved docs as additional context
        if result and 'retrieved_docs' in result:
            retrieved_docs = result.get('retrieved_docs', [])
            for doc in retrieved_docs[:3]:  # Limit to top 3
                if isinstance(doc, dict):
                    sources.append({
                        'content': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                        'metadata': doc.get('metadata', {})
                    })
        
        # Create response
        chat_response = ChatResponse(
            message=final_message or "I apologize, but I couldn't generate a proper response. Please try again.",
            conversation_id=conversation_id,
            sources=sources if request.include_sources else [],
            suggestions=suggestions,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": "N/A",
                "model_used": "gpt-4o-mini"
            }
        )
        
        return APIResponse.success_response(
            data=chat_response,
            message="Message processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process message: {str(e)}"
        )

@router.get("/conversations/{conversation_id}/history", response_model=APIResponse)
async def get_conversation_history(conversation_id: str):
    """
    Get the history of a conversation
    """
    try:
        # This would typically fetch from your conversation storage
        # For now, return a placeholder response
        return APIResponse.success_response(
            data={
                "conversation_id": conversation_id,
                "messages": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            },
            message="Conversation history retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}", response_model=APIResponse)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages
    """
    try:
        logger.info(f"Deleting conversation: {conversation_id}")
        
        # First, delete all messages in the conversation (due to foreign key constraints)
        messages_result = db_client.client.table('messages')\
            .delete()\
            .eq('conversation_id', conversation_id)\
            .execute()
        
        deleted_messages = len(messages_result.data) if messages_result.data else 0
        logger.info(f"Deleted {deleted_messages} messages from conversation {conversation_id}")
        
        # Then delete the conversation itself
        conversation_result = db_client.client.table('conversations')\
            .delete()\
            .eq('id', conversation_id)\
            .execute()
        
        deleted_conversations = len(conversation_result.data) if conversation_result.data else 0
        
        if deleted_conversations == 0:
            logger.warning(f"No conversation found with ID: {conversation_id}")
            return APIResponse.success_response(
                data={"conversation_id": conversation_id, "deleted_messages": deleted_messages},
                message="No conversation found to delete, but any associated messages were removed"
            )
        
        logger.info(f"Successfully deleted conversation {conversation_id} and {deleted_messages} messages")
        
        return APIResponse.success_response(
            data={
                "conversation_id": conversation_id,
                "deleted_messages": deleted_messages,
                "deleted_conversations": deleted_conversations
            },
            message=f"Conversation and {deleted_messages} messages deleted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {str(e)}"
        ) 