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
except ImportError:
    # Fallback for development/testing
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass

from models.base import APIResponse, ConfirmationStatus
from core.database import db_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Simplified models for ephemeral confirmation flow
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
    error: bool = Field(default=False, description="Whether an error occurred during processing")
    error_type: Optional[str] = Field(None, description="Type of error for client-side handling")
    # REVOLUTIONARY 3-FIELD HITL ARCHITECTURE
    hitl_phase: Optional[str] = Field(None, description="HITL interaction phase (needs_prompt, awaiting_response, approved, denied)")
    hitl_prompt: Optional[str] = Field(None, description="HITL prompt text for user interaction")
    hitl_context: Optional[Dict[str, Any]] = Field(None, description="HITL context data for interaction")

class PendingConfirmation(BaseModel):
    """Simplified ephemeral confirmation model"""
    confirmation_id: str = Field(..., description="Unique confirmation ID")
    conversation_id: str = Field(..., description="Associated conversation ID")
    message: str = Field(..., description="Message content requiring confirmation")
    created_at: str = Field(..., description="Creation timestamp")
    status: str = Field(default="pending", description="Current status")

# Simple in-memory confirmation tracking (ephemeral)
# ❌ LEGACY CONFIRMATION SYSTEM - DISABLED
# The new HITL system handles all confirmations internally.
# This legacy system is disabled to prevent duplicate confirmations.
_pending_confirmations: Dict[str, PendingConfirmation] = {}

def clear_conversation_confirmations(conversation_id: str):
    """Clear all legacy confirmations for a specific conversation."""
    to_remove = [
        conf_id for conf_id, conf in _pending_confirmations.items() 
        if conf.conversation_id == conversation_id
    ]
    for conf_id in to_remove:
        del _pending_confirmations[conf_id]
    
    if to_remove:
        logger.info(f"🔍 [LEGACY_CLEANUP] Cleared {len(to_remove)} stale legacy confirmations for conversation {conversation_id}")
    else:
        logger.debug(f"🔍 [LEGACY_CLEANUP] No legacy confirmations to clear for conversation {conversation_id}")

# Initialize the agent (will be lazily loaded)
_agent_instance = None

async def get_agent():
    """Get or create the agent instance"""
    global _agent_instance
    if _agent_instance is None:
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        _agent_instance = UnifiedToolCallingRAGAgent()
        await _agent_instance._ensure_initialized()  # Use the correct method name
    return _agent_instance

@router.post("/message", response_model=ChatResponse)
async def post_chat_message(request: MessageRequest, background_tasks: BackgroundTasks):
    """Process a chat message with the agent."""
    
    # EMERGENCY LOGGING - Check if function is being called at all
    print(f"🚨 [EMERGENCY] Function post_chat_message CALLED with message: '{request.message}'")
    logger.info(f"🚨 [EMERGENCY] Function post_chat_message CALLED with message: '{request.message}'")
    
    # Generate defaults for optional fields
    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"
    
    logger.info(f"🔍 [CHAT_DEBUG] Processing message: '{request.message}' for conversation {conversation_id}")

    # STATE-BASED APPROVAL DETECTION: Check if agent is waiting for HITL approval
    agent = await get_agent()
    
    # Check the agent's current state for HITL data using the graph
    try:
        # Ensure agent is initialized 
        await agent._ensure_initialized()
        
        # Use the agent's graph to get current state (ASYNC method)
        current_state = await agent.graph.aget_state(config={
            "configurable": {"thread_id": conversation_id}
        })
        
        # Extract HITL state information using REVOLUTIONARY 3-field architecture
        hitl_phase = None
        hitl_prompt = None
        hitl_context = None
        is_awaiting_hitl = False
        
        if current_state and hasattr(current_state, 'values'):
            state_values = current_state.values or {}
            hitl_phase = state_values.get('hitl_phase')
            hitl_prompt = state_values.get('hitl_prompt')
            hitl_context = state_values.get('hitl_context')
            # Agent is awaiting HITL response when hitl_phase indicates waiting for user input
            is_awaiting_hitl = hitl_phase in ["awaiting_response", "needs_confirmation", "needs_prompt"]
            
        logger.info(f"🔍 [CHAT_DEBUG] HITL 3-field state check: phase={hitl_phase}, awaiting_hitl={is_awaiting_hitl}")
        if hitl_phase:
            logger.info(f"🔍 [CHAT_DEBUG] HITL phase: {hitl_phase}")
            logger.info(f"🔍 [CHAT_DEBUG] HITL prompt: {hitl_prompt[:50] if hitl_prompt else None}...")
            logger.info(f"🔍 [CHAT_DEBUG] HITL context: {hitl_context}")
            
    except Exception as e:
        logger.warning(f"🔍 [CHAT_DEBUG] Could not check agent HITL state: {e}")
        hitl_phase = None
        hitl_prompt = None
        hitl_context = None
        is_awaiting_hitl = False

    # Agent will handle all approval logic - API just routes messages
    is_approval_message = False  # Default to False, let agent interpret
    
    logger.info(f"🔍 [CHAT_DEBUG] Routing message to agent: message='{request.message}', awaiting_hitl={is_awaiting_hitl}")

    if is_awaiting_hitl and is_approval_message:
        logger.info(f"🔍 [CHAT_DEBUG] STATE-BASED APPROVAL DETECTED: Agent is awaiting HITL response, passing approval to agent")
        
        # STATE-BASED APPROACH: Resume interrupted conversation with approval response
        logger.info(f"🔍 [CHAT_DEBUG] ✅ RESUMING HITL CONVERSATION with approval: '{request.message}'")
        
        processed_result = await agent.resume_interrupted_conversation(
            conversation_id=conversation_id,
            user_response=request.message
        )
        
        # Convert agent result to API format
        processed_result = await agent._process_agent_result_for_api(processed_result, conversation_id)
        
        logger.info(f"🔍 [CHAT_DEBUG] Agent processed approval. Interrupted: {processed_result.get('is_interrupted', False)}")
        
        # Return the processed approval response
        agent_response_message = processed_result.get("message", "I've processed your approval.")
        return ChatResponse(
            message=agent_response_message,
            sources=processed_result.get("sources", []),
            is_interrupted=processed_result.get("is_interrupted", False),
            conversation_id=processed_result.get("conversation_id", conversation_id)
        )

    # Circuit Breaker: Check for excessive confirmation creation
    # confirmation_count = _confirmation_creation_tracker.get(conversation_id, 0) # This line is removed
    # if confirmation_count >= MAX_CONFIRMATIONS_PER_CONVERSATION: # This line is removed
    #     logger.error(f"Circuit breaker triggered: too many confirmations for conversation {conversation_id}") # This line is removed
    #     # Clear all confirmations for this conversation to reset the cycle # This line is removed
    #     storage = get_confirmation_storage() # This line is removed
    #     conv_confirmations = storage.get_confirmations_by_conversation(conversation_id) # This line is removed
    #     for confirmation in conv_confirmations: # This line is removed
    #         logger.info(f"Circuit breaker clearing confirmation {confirmation.confirmation_id}") # This line is removed
    #         storage.delete_confirmation(confirmation.confirmation_id) # This line is removed
    #     # Reset the counter # This line is removed
    #     _confirmation_creation_tracker[conversation_id] = 0 # This line is removed
        
    #     return ChatResponse( # This line is removed
    #         message="I detected a system issue with confirmations. The conversation has been reset. Please try your request again.", # This line is removed
    #         sources=[], # This line is removed
    #         is_interrupted=False, # This line is removed
    #         conversation_id=conversation_id # This line is removed
    #     ) # This line is removed

    # Regular message processing - no approval detected
    logger.info(f"🔍 [CHAT_DEBUG] REGULAR: Processing as regular message")
    
    # Use clean semantic interface - no LangGraph details in API layer
    processed_result = await agent.process_user_message(
        user_query=request.message,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    logger.info(f"🔍 [CHAT_DEBUG] Agent processing completed. Interrupted: {processed_result.get('is_interrupted', False)}")
    
    # ✅ NEW HITL SYSTEM: Clean semantic interface handling
    # The new HITL system is self-contained and handles all confirmation logic internally.
    # No need to create additional API-layer confirmations - just return the agent's response.
    
    if processed_result.get("is_interrupted", False):
        logger.info(f"🔍 [CHAT_DEBUG] ✅ HITL INTERRUPT: Agent requires human interaction, returning prompt directly")
        
        # Return HITL prompt directly with full 3-field architecture - no additional confirmation creation needed
        hitl_prompt = processed_result.get("message", "Confirmation required")
        return ChatResponse(
            message=hitl_prompt,
            sources=processed_result.get("sources", []),
            is_interrupted=True,
            conversation_id=processed_result.get("conversation_id", conversation_id),
            error=False,  # HITL interactions are not errors
            error_type=None,
            # REVOLUTIONARY 3-FIELD HITL ARCHITECTURE - pass through to frontend
            hitl_phase=processed_result.get("hitl_phase"),
            hitl_prompt=processed_result.get("hitl_prompt"),
            hitl_context=processed_result.get("hitl_context")
        )
    
    # Normal conversation response (no interruption)
    logger.info(f"🔍 [CHAT_DEBUG] ✅ NORMAL RESPONSE: Returning agent response")
    
    # Clear any stale legacy confirmations for completed conversations
    clear_conversation_confirmations(conversation_id)
    
    agent_response = processed_result.get("message", "I apologize, but I encountered an issue processing your request.")
    
    # Check if there was an error in the agent processing
    has_error = processed_result.get("error", False)
    error_type = None
    if has_error:
        logger.warning(f"🔍 [CHAT_DEBUG] ⚠️ ERROR STATE: Agent returned error state")
        error_details = processed_result.get("error_details", "")
        
        # Determine error type for client-side handling
        if "Failed to resume conversation" in error_details:
            error_type = "resume_failed"
        elif "conversation session has expired" in agent_response.lower():
            error_type = "session_expired"
        elif "technical issue" in agent_response.lower():
            error_type = "technical_error"
        else:
            error_type = "general_error"
    
    return ChatResponse(
        message=agent_response,
        sources=processed_result.get("sources", []),
        is_interrupted=False,
        conversation_id=processed_result.get("conversation_id", conversation_id),
        error=has_error,
        error_type=error_type
    )

@router.get("/confirmation/pending/{conversation_id}")
async def get_pending_confirmations(conversation_id: str):
    """Get pending confirmations for a conversation"""
    logger.info(f"Fetching pending confirmations for conversation {conversation_id}")
    
    # Log current confirmation state before filtering
    logger.debug(f"🔍 [CONFIRMATION_STATE] BEFORE_PENDING_FETCH for conversation {conversation_id}")
    
    # Check if conversation still exists in database
    try:
        conversation_check = db_client.client.table('conversations').select('id').eq('id', conversation_id).execute()
        if not conversation_check.data:
            logger.info(f"Conversation {conversation_id} not found - may have been deleted")
            return {"success": True, "data": []}
    except Exception as e:
        logger.warning(f"Error checking conversation {conversation_id}: {e}")
        # Continue anyway - return empty confirmations
    
    # Get all confirmations for this conversation from Redis
    # storage = get_confirmation_storage() # This line is removed
    all_conv_confirmations = [
        conf for conf in _pending_confirmations.values()
        if conf.conversation_id == conversation_id
    ]
    
    logger.info(f"🔍 [PENDING_CONFIRMATIONS] Found {len(all_conv_confirmations)} total confirmations for conversation {conversation_id}")
    for conf in all_conv_confirmations:
        status_value = conf.status.value if hasattr(conf.status, 'value') else conf.status
        logger.info(f"🔍 [PENDING_CONFIRMATIONS]   - {conf.confirmation_id}: {status_value} (requested: {conf.created_at})")
    
    # Filter for pending only
    pending_confirmations = [
        {
            "confirmation_id": confirmation.confirmation_id,
            "message": confirmation.message,
            "created_at": confirmation.created_at,
            "status": confirmation.status
        }
        for confirmation in all_conv_confirmations
    ]
    
    logger.info(f"🔍 [PENDING_CONFIRMATIONS] Returning {len(pending_confirmations)} pending confirmations to frontend")
    if pending_confirmations:
        for p in pending_confirmations:
            logger.info(f"🔍 [PENDING_CONFIRMATIONS] Returning: {p['confirmation_id']} - {p['status']}")
    else:
        logger.info(f"🔍 [PENDING_CONFIRMATIONS] No pending confirmations found for conversation {conversation_id}")
    
    return {"success": True, "data": pending_confirmations}

@router.post("/confirmation/{confirmation_id}/respond")
async def respond_to_confirmation(confirmation_id: str, response: Dict[str, Any]):
    """Respond to a pending confirmation and resume the agent conversation"""
    logger.info(f"Processing confirmation response: {confirmation_id}")
    
    confirmation = _pending_confirmations.get(confirmation_id)
    if not confirmation:
        raise HTTPException(status_code=404, detail="Confirmation not found")
    
    # Update confirmation status
    action = response.get('action', 'deny')
    new_status = ConfirmationStatus.APPROVED if action == 'approve' else ConfirmationStatus.DENIED
    
    # Add to in-memory tracking
    confirmation.status = new_status
    logger.info(f"Confirmation {confirmation_id} {action}ed")
    
    # CRITICAL FIX: Resume the interrupted agent conversation using clean semantic interface
    try:
        agent = await get_agent()
        conversation_id = confirmation.conversation_id
        
        logger.info(f"🔄 Resuming agent conversation {conversation_id} with response: {action}")
        
        # Use clean semantic interface - no LangGraph details in API layer
        result = await agent.resume_interrupted_conversation(
            conversation_id=conversation_id,
            user_response=action
        )
        
        logger.info(f"✅ Agent conversation resumed successfully. Keys: {list(result.keys()) if result else 'None'}")
        
        # Remove the confirmation from pending list since it's now processed
        if confirmation_id in _pending_confirmations:
            del _pending_confirmations[confirmation_id]
            logger.info(f"🧹 Removed processed confirmation {confirmation_id} from pending list")
        
        return {"success": True, "message": f"Confirmation {action}ed and conversation resumed"}
        
    except Exception as e:
        logger.error(f"❌ Failed to resume agent conversation after confirmation: {e}")
        return {"success": True, "message": f"Confirmation {action}ed (warning: conversation resumption failed)"}

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
        
        # Clear in-memory confirmations for this conversation
        clear_conversation_confirmations(conversation_id)

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
        
        # Clear in-memory confirmations for all conversations of this user
        for conv_id in conversation_ids:
            clear_conversation_confirmations(conv_id)

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
