"""
Chat API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import BaseModel, Field
import re

try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    # Fallback for development/testing
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass

from models.base import APIResponse, ConfirmationRequest, ConfirmationResponse, DeliveryResult, ConfirmationStatus
from database import db_client
from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize the agent (will be lazily loaded)
_agent_instance = None

# In-memory storage for active confirmations (replace with Redis/database in production)
_active_confirmations: Dict[str, ConfirmationRequest] = {}

# Background task to handle expired confirmations
async def cleanup_expired_confirmations():
    """Background task to mark expired confirmations as TIMEOUT"""
    current_time = datetime.utcnow()
    expired_confirmations = []
    
    for confirmation_id, request in _active_confirmations.items():
        if (request.status == ConfirmationStatus.PENDING and 
            current_time > request.expires_at):
            request.status = ConfirmationStatus.TIMEOUT
            expired_confirmations.append(confirmation_id)
            logger.info(f"Marking confirmation {confirmation_id} as expired")
    
    # Update expired confirmations
    for confirmation_id in expired_confirmations:
        if confirmation_id in _active_confirmations:
            _active_confirmations[confirmation_id].status = ConfirmationStatus.TIMEOUT

# Initialize cleanup scheduler (would use APScheduler or Celery in production)
import asyncio
from threading import Thread

def start_background_cleanup():
    """Start background task for cleanup in a separate thread"""
    def run_cleanup():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def periodic_cleanup():
            while True:
                try:
                    await cleanup_expired_confirmations()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in background cleanup: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        loop.run_until_complete(periodic_cleanup())
    
    cleanup_thread = Thread(target=run_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Started background confirmation cleanup task")

# Start cleanup when module is loaded
start_background_cleanup()


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
    role: str = Field(..., description="Message role (human/user/ai/assistant)")
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

        # Check for LangGraph interrupt state following best practices
        logger.info(f"Agent result keys: {list(result.keys()) if result else 'None'}")
        
        # LangGraph best practice: Check for interrupted state using 'next' field
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
            
            # Method 2: Fallback - check other interrupt indicators
            elif hasattr(result, '__dict__'):
                result_dict = result.__dict__
                logger.info(f"Result attributes: {list(result_dict.keys())}")
                if '__interrupt__' in result_dict or 'next' in result_dict:
                    is_interrupted = True
                    logger.info(f"Interrupt indicators found in result attributes")
        
        # If interrupted, extract the interrupt message from all possible locations
        if is_interrupted and result:
            logger.info(f"Extracting interrupt message from result...")
            
            # Method 1: Check if interrupt message is directly in __interrupt__ field
            if isinstance(result, dict) and '__interrupt__' in result:
                interrupt_data = result['__interrupt__']
                logger.info(f"__interrupt__ field type: {type(interrupt_data)}")
                logger.info(f"__interrupt__ field content: {str(interrupt_data)[:300]}...")
                
                if isinstance(interrupt_data, dict) and 'value' in interrupt_data:
                    interrupt_message = str(interrupt_data['value'])
                    logger.info(f"Found interrupt message in __interrupt__.value field")
                elif isinstance(interrupt_data, dict):
                    # Log all keys in the dict to see what's available
                    logger.info(f"__interrupt__ dict keys: {list(interrupt_data.keys())}")
                    # Try common keys
                    for key in ['message', 'content', 'text', 'data', 'interrupt']:
                        if key in interrupt_data:
                            interrupt_message = str(interrupt_data[key])
                            logger.info(f"Found interrupt message in __interrupt__.{key} field")
                            break
                elif isinstance(interrupt_data, str):
                    interrupt_message = interrupt_data
                    logger.info(f"Found interrupt message in __interrupt__ field")
                elif isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                    # Handle list of Interrupt objects (LangGraph's actual format)
                    first_interrupt = interrupt_data[0]
                    logger.info(f"First interrupt object type: {type(first_interrupt)}")
                    if hasattr(first_interrupt, 'value'):
                        interrupt_message = str(first_interrupt.value)
                        logger.info(f"Found interrupt message in __interrupt__[0].value field")
                    elif hasattr(first_interrupt, '__dict__'):
                        # Log all attributes to see what's available
                        attrs = list(first_interrupt.__dict__.keys()) if hasattr(first_interrupt, '__dict__') else []
                        logger.info(f"Interrupt object attributes: {attrs}")
                        interrupt_message = str(first_interrupt)
                        logger.info(f"Using string representation of interrupt object")
                    else:
                        interrupt_message = str(first_interrupt)
                        logger.info(f"Using string representation of interrupt object")
                else:
                    logger.warning(f"Unknown __interrupt__ data type: {type(interrupt_data)}")
            
            # Method 2: Search through messages (fallback)
            if not interrupt_message and result.get('messages'):
                logger.info(f"Searching through {len(result['messages'])} messages for interrupt content")
                for i, msg in enumerate(reversed(result['messages'])):
                    if hasattr(msg, 'content') and msg.content:
                        content = str(msg.content)
                        logger.info(f"Message {i} preview: {content[:100]}...")
                        if ("CUSTOMER MESSAGE CONFIRMATION REQUIRED" in content or 
                            "What is your decision?" in content or
                            "APPROVE" in content):
                            interrupt_message = content
                            logger.info(f"Found interrupt message in messages[{len(result['messages'])-1-i}]")
                            break
            
            # Method 3: Log what we have if still no message found
            if not interrupt_message:
                logger.warning(f"No interrupt message found despite __interrupt__ detection")
                logger.warning(f"Result keys: {list(result.keys())}")
                if result.get('messages'):
                    logger.warning(f"Messages count: {len(result['messages'])}")
                    for i, msg in enumerate(result['messages'][-3:]):  # Last 3 messages
                        if hasattr(msg, 'content'):
                            logger.warning(f"Recent message {i}: {str(msg.content)[:200]}...")
                else:
                    logger.warning(f"No messages found in result")
        
        # Handle interrupt state following LangGraph best practices
        if is_interrupted and interrupt_message:
            logger.info(f"Processing LangGraph interrupt for conversation {conversation_id}")
            
            # Generate confirmation ID
            confirmation_id = str(uuid4())
            
            # Extract customer information from interrupt message for required fields
            customer_name = "Unknown Customer"
            customer_email = None
            
            # Try to extract customer info from the interrupt message
            if "Name:" in interrupt_message:
                name_match = re.search(r'Name:\s*([^\n•]+)', interrupt_message)
                if name_match:
                    customer_name = name_match.group(1).strip()
            
            if "Email:" in interrupt_message:
                email_match = re.search(r'Email:\s*([^\n•]+)', interrupt_message)
                if email_match:
                    customer_email = email_match.group(1).strip()
            
            # Create confirmation request with correct field names
            confirmation_request = ConfirmationRequest(
                confirmation_id=confirmation_id,
                customer_id="unknown",  # Will extract from interrupt message if possible
                customer_name=customer_name,
                customer_email=customer_email,
                message_content=interrupt_message,  # Full interrupt message as content
                message_type="follow_up",
                requested_by=request.user_id,
                conversation_id=conversation_id,
                expires_at=datetime.utcnow() + timedelta(minutes=15)  # 15 minute timeout
            )
            
            # Store confirmation for tracking
            _active_confirmations[confirmation_id] = confirmation_request
            
            # Return confirmation as response
            chat_response = ChatResponse(
                message=interrupt_message,
                conversation_id=conversation_id,
                sources=[],
                suggestions=["APPROVE", "CANCEL", "MODIFY: [your changes]"],
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "confirmation_id": confirmation_id,
                    "requires_confirmation": True,
                    "message_type": "customer_message_confirmation"
                }
            )
            
            return APIResponse.success_response(
                data=chat_response,
                message="Confirmation required for customer message"
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

    except GraphInterrupt as e:
        # Handle human-in-the-loop confirmation requests
        logger.info(f"GraphInterrupt triggered for conversation {conversation_id}: {str(e)}")
        
        # Extract interrupt details
        interrupt_value = str(e.value) if hasattr(e, 'value') else str(e)
        interrupt_ns = e.ns if hasattr(e, 'ns') else []
        
        # Generate confirmation ID
        confirmation_id = str(uuid4())
        
        # Create confirmation request
        confirmation_request = ConfirmationRequest(
            id=confirmation_id,
            conversation_id=conversation_id,
            user_id=request.user_id,
            message=interrupt_value,
            status=ConfirmationStatus.PENDING,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow().replace(minute=datetime.utcnow().minute + 15),  # 15 minute timeout
            metadata={
                "interrupt_ns": interrupt_ns,
                "message_type": "customer_message_confirmation"
            }
        )
        
        # Store confirmation for tracking
        _active_confirmations[confirmation_id] = confirmation_request
        
        # Return confirmation as response
        chat_response = ChatResponse(
            message=interrupt_value,  # Return the confirmation message
            conversation_id=conversation_id,
            sources=[],
            suggestions=["APPROVE", "CANCEL", "MODIFY: [your changes]"],
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "confirmation_id": confirmation_id,
                "requires_confirmation": True,
                "message_type": "customer_message_confirmation"
            }
        )
        
        return APIResponse.success_response(
            data=chat_response,
            message="Confirmation required for customer message"
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

@router.post("/confirmation/request", response_model=APIResponse[ConfirmationRequest])
async def create_confirmation_request(request: ConfirmationRequest):
    """
    Create a new confirmation request for customer messaging
    """
    try:
        logger.info(f"Creating confirmation request: {request.confirmation_id}")
        
        # Store the confirmation request
        _active_confirmations[request.confirmation_id] = request
        
        return APIResponse.success_response(
            data=request,
            message="Confirmation request created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating confirmation request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create confirmation request: {str(e)}"
        )

@router.get("/confirmation/pending/{conversation_id}", response_model=APIResponse[List[ConfirmationRequest]])
async def get_pending_confirmations(conversation_id: str):
    """
    Get all pending confirmation requests for a conversation
    """
    try:
        logger.info(f"Fetching pending confirmations for conversation {conversation_id}")
        
        # Filter pending confirmations for this conversation
        pending = [
            req for req in _active_confirmations.values() 
            if req.conversation_id == conversation_id and req.status == ConfirmationStatus.PENDING
        ]
        
        # Check for expired confirmations
        now = datetime.utcnow()
        for req in pending:
            if now > req.expires_at:
                req.status = ConfirmationStatus.TIMEOUT
                _active_confirmations[req.confirmation_id] = req
        
        # Filter out timed-out confirmations
        active_pending = [req for req in pending if req.status == ConfirmationStatus.PENDING]
        
        return APIResponse.success_response(
            data=active_pending,
            message=f"Retrieved {len(active_pending)} pending confirmations"
        )
        
    except Exception as e:
        logger.error(f"Error fetching pending confirmations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch pending confirmations: {str(e)}"
        )

@router.post("/confirmation/respond", response_model=APIResponse[ConfirmationRequest])
async def respond_to_confirmation(response: ConfirmationResponse):
    """
    Respond to a confirmation request (approve/cancel/modify)
    """
    try:
        logger.info(f"Processing confirmation response: {response.confirmation_id}")
        
        # Get the confirmation request
        if response.confirmation_id not in _active_confirmations:
            raise HTTPException(
                status_code=404,
                detail="Confirmation request not found"
            )
        
        confirmation_req = _active_confirmations[response.confirmation_id]
        
        # Check if already responded or expired
        if confirmation_req.status != ConfirmationStatus.PENDING:
            raise HTTPException(
                status_code=400,
                detail=f"Confirmation request is already {confirmation_req.status.value}"
            )
        
        # Check expiration
        if datetime.utcnow() > confirmation_req.expires_at:
            confirmation_req.status = ConfirmationStatus.TIMEOUT
            _active_confirmations[response.confirmation_id] = confirmation_req
            raise HTTPException(
                status_code=400,
                detail="Confirmation request has expired"
            )
        
        # Update the confirmation request
        confirmation_req.status = response.action
        
        # Handle modified message
        if response.action == ConfirmationStatus.MODIFIED and response.modified_message:
            confirmation_req.message_content = response.modified_message
        
        _active_confirmations[response.confirmation_id] = confirmation_req
        
        return APIResponse.success_response(
            data=confirmation_req,
            message=f"Confirmation {response.action.value} successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing confirmation response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process confirmation response: {str(e)}"
        )

@router.get("/confirmation/{confirmation_id}", response_model=APIResponse[ConfirmationRequest])
async def get_confirmation_status(confirmation_id: str):
    """
    Get the status of a specific confirmation request
    """
    try:
        logger.info(f"Getting status for confirmation: {confirmation_id}")
        
        if confirmation_id not in _active_confirmations:
            raise HTTPException(
                status_code=404,
                detail="Confirmation request not found"
            )
        
        confirmation_req = _active_confirmations[confirmation_id]
        
        # Check expiration
        if confirmation_req.status == ConfirmationStatus.PENDING and datetime.utcnow() > confirmation_req.expires_at:
            confirmation_req.status = ConfirmationStatus.TIMEOUT
            _active_confirmations[confirmation_id] = confirmation_req
        
        return APIResponse.success_response(
            data=confirmation_req,
            message="Confirmation status retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting confirmation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get confirmation status: {str(e)}"
        )

@router.post("/confirmation/delivery", response_model=APIResponse[DeliveryResult])
async def record_delivery_result(delivery_result: DeliveryResult):
    """
    Record the result of message delivery attempt
    """
    try:
        logger.info(f"Recording delivery result for confirmation: {delivery_result.confirmation_id}")
        
        # Verify confirmation exists
        if delivery_result.confirmation_id not in _active_confirmations:
            raise HTTPException(
                status_code=404,
                detail="Confirmation request not found"
            )
        
        # In production, store delivery results in database
        # For now, just log the result
        logger.info(f"Message delivery result: {delivery_result.delivery_status} - {delivery_result.delivery_message}")
        
        return APIResponse.success_response(
            data=delivery_result,
            message="Delivery result recorded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording delivery result: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record delivery result: {str(e)}"
        )

@router.delete("/confirmation/{confirmation_id}", response_model=APIResponse[str])
async def cleanup_confirmation(confirmation_id: str):
    """
    Clean up a completed confirmation request
    """
    try:
        logger.info(f"Cleaning up confirmation: {confirmation_id}")
        
        if confirmation_id in _active_confirmations:
            del _active_confirmations[confirmation_id]
        
        return APIResponse.success_response(
            data=confirmation_id,
            message="Confirmation cleaned up successfully"
        )
        
    except Exception as e:
        logger.error(f"Error cleaning up confirmation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup confirmation: {str(e)}"
        )
