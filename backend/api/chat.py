"""
Chat API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from uuid import uuid4
import uuid
from pydantic import BaseModel, Field
import re

try:
    from langgraph.errors import GraphInterrupt
    from langgraph.types import Command
except ImportError:
    # Fallback for development/testing
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass
    
    class Command:
        """Mock Command for development/testing"""
        def __init__(self, resume=None):
            self.resume = resume

from models.base import APIResponse, ConfirmationRequest, ConfirmationResponse, DeliveryResult, ConfirmationStatus
from database import db_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Models for chat functionality
class MessageRequest(BaseModel):
    """Request model for chat messages"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    user_id: Optional[str] = Field(None, description="User ID") 
    include_sources: bool = Field(True, description="Whether to include sources in response")

class ChatResponse(BaseModel):
    """Response model for chat messages"""
    message: str = Field(..., description="Agent response")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    is_interrupted: bool = Field(default=False, description="Whether processing was interrupted")
    conversation_id: str = Field(..., description="Conversation ID")
    confirmation_id: Optional[str] = Field(None, description="Confirmation ID if interrupted")

class PendingConfirmation(BaseModel):
    """Model for pending confirmations"""
    confirmation_id: str
    conversation_id: str
    message: str
    created_at: datetime
    status: ConfirmationStatus

# Initialize the agent (will be lazily loaded)
_agent_instance = None

async def get_agent():
    """Get or create the agent instance"""
    global _agent_instance
    if _agent_instance is None:
        from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
        _agent_instance = UnifiedToolCallingRAGAgent()
        await _agent_instance._ensure_initialized()  # Use the correct method name
    return _agent_instance

# In-memory storage for active confirmations (replace with Redis/database in production)
_active_confirmations: Dict[str, PendingConfirmation] = {}

# Global tracking for preventing infinite confirmation loops
_confirmation_creation_tracker: Dict[str, int] = {}  # conversation_id -> creation_count
MAX_CONFIRMATIONS_PER_CONVERSATION = 3  # Circuit breaker limit

@router.post("/message", response_model=ChatResponse)
async def post_chat_message(request: MessageRequest, background_tasks: BackgroundTasks):
    """Process a chat message with the agent."""
    # Generate defaults for optional fields
    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"
    
    logger.info(f"Processing chat message for conversation {conversation_id}")

    # Circuit Breaker: Check for excessive confirmation creation
    confirmation_count = _confirmation_creation_tracker.get(conversation_id, 0)
    if confirmation_count >= MAX_CONFIRMATIONS_PER_CONVERSATION:
        logger.error(f"Circuit breaker triggered: too many confirmations for conversation {conversation_id}")
        # Clear all confirmations for this conversation to reset the cycle
        for conf_id, confirmation in list(_active_confirmations.items()):
            if confirmation.conversation_id == conversation_id:
                logger.info(f"Circuit breaker clearing confirmation {conf_id}")
                del _active_confirmations[conf_id]
        # Reset the counter
        _confirmation_creation_tracker[conversation_id] = 0
        
        return ChatResponse(
            message="I detected a system issue with confirmations. The conversation has been reset. Please try your request again.",
            sources=[],
            is_interrupted=False,
            conversation_id=conversation_id
        )

    # Get the agent instance
    agent = await get_agent()

    # Check if there are pending confirmations (interrupted state) for this conversation
    pending_confirmations = [
        req for req in _active_confirmations.values() 
        if req.conversation_id == conversation_id and req.status == ConfirmationStatus.PENDING
    ]
    
    if pending_confirmations:
        logger.info(f"Found {len(pending_confirmations)} pending confirmations, resuming LangGraph execution with user message: '{request.message}'")
        
        # Mark confirmations as handled (the agent will determine the actual action)
        for confirmation in pending_confirmations:
            confirmation.status = ConfirmationStatus.APPROVED  # Generic status, agent handles the logic
            _active_confirmations[confirmation.confirmation_id] = confirmation
        
        # Resume LangGraph execution using proper Command pattern following best practices
        try:
            logger.info(f"Resuming LangGraph execution with Command(resume='{request.message}')")
            
            # CRITICAL FIX: Use agent.graph.ainvoke (not agent.invoke) for Command objects
            # This fixes the AsyncPostgresSaver threading issue that causes resume failures
            result = await agent.graph.ainvoke(
                Command(resume=request.message),  # Direct Command object
                config={
                    "configurable": {"thread_id": conversation_id}
                }
            )
            
        except Exception as e:
            logger.error(f"Error resuming LangGraph execution with Command: {e}")
            
            # Clear any stale confirmations to prevent infinite loops
            logger.info("Cleaning up stale confirmations due to resume failure")
            for conf_id, confirmation in list(_active_confirmations.items()):
                if confirmation.conversation_id == conversation_id:
                    logger.info(f"Removing stale confirmation {conf_id}")
                    del _active_confirmations[conf_id]
            
            # If Command pattern fails, DO NOT fall back to regular processing
            # This prevents the infinite loop of creating new interrupts
            logger.warning("Command resume failed - returning error instead of fallback processing")
            return ChatResponse(
                message="I encountered an issue processing your confirmation. Please try again.",
                sources=[],
                is_interrupted=False,
                conversation_id=conversation_id
            )
    else:
        # No pending confirmations, process as regular message
        logger.info("No pending confirmations found, processing as regular message")
        result = await agent.invoke(
            user_query=request.message,
            conversation_id=conversation_id,
            user_id=user_id
        )

    # Check for LangGraph interrupt state following best practices
    logger.info(f"Agent result keys: {list(result.keys()) if result else 'None'}")
    
    # LangGraph best practice: Check for interrupted state using '__interrupt__' field
    is_interrupted = False
    interrupt_message = ""
    
    if result:
        # Method 1: Check for LangGraph's actual interrupt indicator
        if isinstance(result, dict) and '__interrupt__' in result:
            logger.info(f"LangGraph interrupt detected: __interrupt__ field found")
            is_interrupted = True
        elif hasattr(result, '__interrupt__'):
            logger.info(f"LangGraph interrupt detected: __interrupt__ attribute found")
            is_interrupted = True
    
    # If interrupted, extract the interrupt message
    if is_interrupted and result:
        logger.info(f"Extracting interrupt message from result...")
        
        if isinstance(result, dict) and '__interrupt__' in result:
            interrupt_data = result['__interrupt__']
            logger.info(f"__interrupt__ field type: {type(interrupt_data)}")
            logger.info(f"__interrupt__ field content: {str(interrupt_data)[:300]}...")
            
            if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                # Handle list of Interrupt objects (LangGraph's actual format)
                first_interrupt = interrupt_data[0]
                logger.info(f"First interrupt object type: {type(first_interrupt)}")
                if hasattr(first_interrupt, 'value'):
                    interrupt_message = str(first_interrupt.value)
                    logger.info(f"Found interrupt message in __interrupt__[0].value field")

    # Handle LangGraph interrupt
    if is_interrupted and interrupt_message:
        logger.info(f"Processing LangGraph interrupt for conversation {conversation_id}")

        # Circuit Breaker: Track confirmation creation
        if conversation_id not in _confirmation_creation_tracker:
            _confirmation_creation_tracker[conversation_id] = 0
        _confirmation_creation_tracker[conversation_id] += 1
        
        logger.info(f"Confirmation creation count for {conversation_id}: {_confirmation_creation_tracker[conversation_id]}")

        # Create confirmation object
        confirmation = PendingConfirmation(
            confirmation_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            message=interrupt_message,
            created_at=datetime.now(),
            status=ConfirmationStatus.PENDING
        )

        # Store in active confirmations
        _active_confirmations[confirmation.confirmation_id] = confirmation
        logger.info(f"Created confirmation {confirmation.confirmation_id}")

        return ChatResponse(
            message=interrupt_message,
            sources=[],
            is_interrupted=True,
            confirmation_id=confirmation.confirmation_id,
            conversation_id=conversation_id
        )

    # Check if confirmation_data has been cleared (HITL completed successfully)
    # Reset confirmation creation counter on successful completion
    confirmation_data = result.get("confirmation_data")
    confirmation_result = result.get("confirmation_result")

    if confirmation_result and not confirmation_data:
        logger.info(f"HITL completed with result: {confirmation_result} - resetting confirmation counter")
        # Reset the circuit breaker counter on successful completion
        _confirmation_creation_tracker[conversation_id] = 0
        
        # Clean up any pending confirmations for this conversation
        for conf_id, confirmation in list(_active_confirmations.items()):
            if confirmation.conversation_id == conversation_id:
                logger.info(f"Cleaning up confirmation {conf_id} after HITL completion")
                del _active_confirmations[conf_id]

    # Normal processing - extract final message
    final_message = "I apologize, but I encountered an issue processing your request."
    sources = []
    
    if result and result.get('messages'):
        logger.info(f"Found {len(result['messages'])} messages in result")
        # Log message details for debugging
        for i, msg in enumerate(result['messages']):
            msg_type = getattr(msg, 'type', 'no_type')
            msg_role = getattr(msg, 'role', 'no_role')
            content_preview = str(getattr(msg, 'content', 'no_content'))[:50] + "..."
            logger.info(f"Message {i}: type={msg_type}, role={msg_role}, content={content_preview}")
        
        # Get the last AI message
        for msg in reversed(result['messages']):
            # Check for LangChain message types (ai, assistant)
            if hasattr(msg, 'type') and msg.type in ['ai', 'assistant'] and hasattr(msg, 'content'):
                final_message = msg.content
                break
            elif hasattr(msg, 'role') and msg.role in ['ai', 'assistant'] and hasattr(msg, 'content'):
                final_message = msg.content
                break
            elif hasattr(msg, 'content') and not hasattr(msg, 'type') and not hasattr(msg, 'role'):
                # Fallback for messages without type or role
                final_message = str(msg.content)
                break
    
    logger.info(f"Final message extracted: {final_message[:100]}...")
    logger.info(f"Final message type: {type(final_message)}")
    
    # Extract sources if available
    if result and result.get('retrieved_docs'):
        sources = [
            {
                "content": doc.get('content', '')[:500],
                "metadata": doc.get('metadata', {})
            }
            for doc in result['retrieved_docs'][:3]  # Limit to top 3
        ]

    return ChatResponse(
        message=final_message,
        sources=sources,
        is_interrupted=False,
        conversation_id=conversation_id
    )

@router.get("/confirmation/pending/{conversation_id}")
async def get_pending_confirmations(conversation_id: str):
    """Get pending confirmations for a conversation"""
    logger.info(f"Fetching pending confirmations for conversation {conversation_id}")
    
    # Check if conversation still exists in database
    try:
        conversation_check = db_client.client.table('conversations').select('id').eq('id', conversation_id).execute()
        if not conversation_check.data:
            logger.info(f"Conversation {conversation_id} not found - may have been deleted")
            return {"success": True, "data": []}
    except Exception as e:
        logger.warning(f"Error checking conversation {conversation_id}: {e}")
        # Continue anyway - return empty confirmations
    
    pending = [
        {
            "confirmation_id": confirmation.confirmation_id,
            "message": confirmation.message,
            "created_at": confirmation.created_at.isoformat(),
            "status": confirmation.status.value
        }
        for confirmation in _active_confirmations.values()
        if (confirmation.conversation_id == conversation_id and 
            confirmation.status == ConfirmationStatus.PENDING)
    ]
    
    return {"success": True, "data": pending}

@router.post("/confirmation/{confirmation_id}/respond")
async def respond_to_confirmation(confirmation_id: str, response: Dict[str, Any]):
    """Respond to a pending confirmation"""
    logger.info(f"Processing confirmation response: {confirmation_id}")
    
    if confirmation_id not in _active_confirmations:
        raise HTTPException(status_code=404, detail="Confirmation not found")
    
    confirmation = _active_confirmations[confirmation_id]
    
    # Update confirmation status
    action = response.get('action', 'deny')
    if action == 'approve':
        confirmation.status = ConfirmationStatus.APPROVED
    else:
        confirmation.status = ConfirmationStatus.DENIED
    
    _active_confirmations[confirmation_id] = confirmation
    
    # Reset confirmation creation counter on response
    conversation_id = confirmation.conversation_id
    if conversation_id in _confirmation_creation_tracker:
        _confirmation_creation_tracker[conversation_id] = 0
    
    logger.info(f"Confirmation {confirmation_id} {action}ed")
    
    return {"success": True, "message": f"Confirmation {action}ed"}

@router.delete("/conversation/{conversation_id}", response_model=APIResponse)
async def clear_conversation(conversation_id: str):
    """Clear a conversation and all its messages from the database"""
    try:
        logger.info(f"Clearing conversation {conversation_id}")
        
        # First check if conversation exists
        conversation_check = db_client.client.table('conversations').select('id').eq('id', conversation_id).execute()
        if not conversation_check.data:
            logger.warning(f"Conversation {conversation_id} not found")
            return APIResponse.success_response(
                data={"conversation_id": conversation_id},
                message="Conversation not found (may have been already deleted)"
            )
        
        # Check how many messages exist
        messages_check = db_client.client.table('messages').select('id').eq('conversation_id', conversation_id).execute()
        message_count = len(messages_check.data or [])
        logger.info(f"Found {message_count} messages to delete for conversation {conversation_id}")
        
        # Delete messages first (due to foreign key constraints)
        messages_result = db_client.client.table('messages').delete().eq('conversation_id', conversation_id).execute()
        deleted_messages = len(messages_result.data or [])
        logger.info(f"Deleted {deleted_messages} messages from conversation {conversation_id}")
        
        # Delete conversation
        conversation_result = db_client.client.table('conversations').delete().eq('id', conversation_id).execute()
        deleted_conversations = len(conversation_result.data or [])
        logger.info(f"Deleted {deleted_conversations} conversations with ID {conversation_id}")
        
        return APIResponse.success_response(
            data={
                "conversation_id": conversation_id,
                "deleted_messages": deleted_messages,
                "deleted_conversations": deleted_conversations
            },
            message=f"Conversation cleared successfully - deleted {deleted_messages} messages and {deleted_conversations} conversations"
        )
        
    except Exception as e:
        logger.error(f"Error clearing conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

@router.delete("/user/{user_id}/all-conversations", response_model=APIResponse)
async def clear_all_user_conversations(user_id: str):
    """Clear ALL conversations and messages for a specific user"""
    try:
        logger.info(f"Clearing ALL conversations for user {user_id}")
        
        # First check if user exists
        user_check = db_client.client.table('users').select('id').eq('id', user_id).execute()
        if not user_check.data:
            logger.warning(f"User {user_id} not found")
            return APIResponse.success_response(
                data={"user_id": user_id},
                message="User not found"
            )
        
        # Get all conversations for this user
        conversations_result = db_client.client.table('conversations').select('id').eq('user_id', user_id).execute()
        conversation_ids = [conv['id'] for conv in conversations_result.data or []]
        conversation_count = len(conversation_ids)
        
        if conversation_count == 0:
            logger.info(f"No conversations found for user {user_id}")
            return APIResponse.success_response(
                data={
                    "user_id": user_id,
                    "deleted_messages": 0,
                    "deleted_conversations": 0
                },
                message="No conversations found for user"
            )
        
        logger.info(f"Found {conversation_count} conversations to delete for user {user_id}")
        
        # Delete all messages from all conversations for this user
        total_deleted_messages = 0
        for conv_id in conversation_ids:
            messages_result = db_client.client.table('messages').delete().eq('conversation_id', conv_id).execute()
            deleted_messages = len(messages_result.data or [])
            total_deleted_messages += deleted_messages
            logger.info(f"Deleted {deleted_messages} messages from conversation {conv_id}")
        
        # Delete all conversations for this user
        conversations_delete_result = db_client.client.table('conversations').delete().eq('user_id', user_id).execute()
        deleted_conversations = len(conversations_delete_result.data or [])
        logger.info(f"Deleted {deleted_conversations} conversations for user {user_id}")
        
        return APIResponse.success_response(
            data={
                "user_id": user_id,
                "deleted_messages": total_deleted_messages,
                "deleted_conversations": deleted_conversations,
                "conversation_ids": conversation_ids
            },
            message=f"All user conversations cleared successfully - deleted {total_deleted_messages} messages and {deleted_conversations} conversations"
        )
        
    except Exception as e:
        logger.error(f"Error clearing all conversations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear user conversations: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chat-api"}
