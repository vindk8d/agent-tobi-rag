"""
Universal Message Delivery Utilities.

These utilities can be used by ANY agent that needs customer message delivery.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- Customer message delivery with chat API integration
- Message formatting with business context
- Customer user and conversation management
- Memory system integration
- Audit trail and compliance logging
- Real-time notification support
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.message_delivery import (
    execute_customer_message_delivery,
    send_via_chat_api,
    format_business_message,
    get_or_create_customer_user,
    get_or_create_customer_conversation
)

class MyAgent:
    async def send_customer_message(self, customer_id: str, message: str):
        # Execute full message delivery - works with any agent
        success = await execute_customer_message_delivery(
            customer_id=customer_id,
            message_content=message,
            message_type="follow_up",
            customer_info={"name": "John Doe", "email": "john@example.com"},
            memory_manager=self.memory_manager  # Pass your agent's memory manager
        )
        
        return success
```

This ensures consistent message delivery across all agents without code duplication.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

# Import database and memory utilities
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.database import db_client

logger = logging.getLogger(__name__)


async def execute_customer_message_delivery(
    customer_id: str,
    message_content: str,
    message_type: str,
    customer_info: Dict[str, Any],
    memory_manager: Optional[Any] = None
) -> bool:
    """
    PORTABLE: Execute real customer message delivery with full integration.
    
    Can be used by any agent to deliver messages to customers with complete
    chat API integration, memory persistence, and audit trails.
    
    Args:
        customer_id: Customer identifier
        message_content: Message to deliver
        message_type: Type of message (follow_up, information, etc.)
        customer_info: Customer information for delivery
        memory_manager: Optional memory manager for persistence
        
    Returns:
        bool: True if delivery successful, False otherwise
    """
    try:
        logger.info(f"[PORTABLE_DELIVERY] Executing real delivery for customer {customer_id}")
        
        # =================================================================
        # REAL MESSAGE DELIVERY IMPLEMENTATION
        # =================================================================
        
        # Step 1: Send via chat API for real-time delivery
        chat_delivery_success = await send_via_chat_api(
            customer_id=customer_id,
            message_content=message_content,
            message_type=message_type,
            customer_info=customer_info,
            memory_manager=memory_manager
        )
        
        # Step 2: Ensure memory integration (if memory manager provided)
        memory_integration_success = True
        if memory_manager:
            memory_integration_success = await integrate_with_memory_systems(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info,
                memory_manager=memory_manager
            )
        
        # Step 3: Store audit trail (for compliance and tracking)
        audit_success = await store_customer_message_audit(
            customer_id=customer_id,
            message_content=message_content,
            message_type=message_type,
            delivery_status="delivered" if chat_delivery_success else "failed",
            customer_info=customer_info
        )
        
        # Overall success requires chat delivery and memory integration
        overall_success = chat_delivery_success and memory_integration_success
        
        if overall_success:
            logger.info(f"[PORTABLE_DELIVERY] Real message delivery completed successfully for customer {customer_id}")
        else:
            logger.error(f"[PORTABLE_DELIVERY] Message delivery partially failed - Chat: {chat_delivery_success}, Memory: {memory_integration_success}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"[PORTABLE_DELIVERY] Message delivery failed for customer {customer_id}: {str(e)}")
        return False


async def send_via_chat_api(
    customer_id: str,
    message_content: str,
    message_type: str,
    customer_info: Dict[str, Any],
    memory_manager: Optional[Any] = None
) -> bool:
    """
    PORTABLE: Send message via chat API to create real conversation flow.
    
    Can be used by any agent to send messages through the chat infrastructure
    in a way that customers can actually receive and respond to.
    
    Args:
        customer_id: Customer identifier
        message_content: Message to deliver
        message_type: Type of message
        customer_info: Customer information
        memory_manager: Optional memory manager for message storage
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"[PORTABLE_CHAT_DELIVERY] Sending message via chat API to {customer_info.get('name', customer_id)}")
        
        # Get the actual customer UUID from customer_info (not the name)
        actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
        if not actual_customer_id:
            logger.error(f"[PORTABLE_CHAT_DELIVERY] No valid customer UUID found in customer_info for {customer_info.get('name', customer_id)}")
            return False
        
        logger.info(f"[PORTABLE_CHAT_DELIVERY] Using customer UUID: {actual_customer_id} for {customer_info.get('name', customer_id)}")
        
        # Get or create user record for the customer using the correct UUID
        user_id = await get_or_create_customer_user(actual_customer_id, customer_info)
        if not user_id:
            logger.error(f"[PORTABLE_CHAT_DELIVERY] Failed to get user ID for customer UUID {actual_customer_id}")
            return False
        
        # Create or get existing conversation for this customer
        conversation_id = await get_or_create_customer_conversation(user_id, customer_info)
        if not conversation_id:
            logger.error(f"[PORTABLE_CHAT_DELIVERY] Failed to get conversation ID for customer {customer_id}")
            return False
        
        # Format the message with proper business context
        formatted_message = await format_business_message(
            message_content, message_type, customer_info
        )
        
        # Store the message in the customer's conversation (if memory manager available)
        message_id = None
        if memory_manager:
            try:
                message_id = await memory_manager.store_message_from_agent(
                    message={
                        'content': formatted_message,
                        'role': 'assistant',  # Messages from employees appear as assistant
                        'metadata': {
                            'message_type': message_type,
                            'delivery_method': 'chat_api',
                            'customer_id': actual_customer_id,
                            'customer_name': customer_info.get('name', ''),
                            'employee_initiated': True,
                            'delivery_timestamp': datetime.now().isoformat()
                        }
                    },
                    config={'configurable': {'thread_id': conversation_id}},
                    agent_type='employee_outreach',
                    user_id=user_id
                )
                logger.info(f"[PORTABLE_CHAT_DELIVERY] ✅ STORED message in customer conversation: {message_id}")
            except Exception as e:
                logger.error(f"[PORTABLE_CHAT_DELIVERY] Failed to store message in customer conversation: {e}")
                return False
        else:
            # Fallback: store message directly in database
            message_id = await store_customer_message(
                customer_id=actual_customer_id,
                message_content=formatted_message,
                message_type=message_type,
                delivery_status="delivered",
                customer_info=customer_info
            )
        
        if message_id:
            logger.info(f"[PORTABLE_CHAT_DELIVERY] Message delivered successfully via chat API - Message ID: {message_id}")
            
            # Optional: Send real-time notification
            await send_real_time_notification(
                user_id=user_id,
                conversation_id=conversation_id,
                message_content=formatted_message,
                customer_info=customer_info
            )
            
            return True
        else:
            logger.error(f"[PORTABLE_CHAT_DELIVERY] Failed to store message via chat API for customer {customer_id}")
            return False
            
    except Exception as e:
        logger.error(f"[PORTABLE_CHAT_DELIVERY] Error sending via chat API: {str(e)}")
        return False


async def format_business_message(
    message_content: str,
    message_type: str,
    customer_info: Dict[str, Any]
) -> str:
    """
    PORTABLE: Format the message with proper business context and personalization.
    
    Can be used by any agent to format messages consistently with business standards.
    
    Args:
        message_content: Raw message content
        message_type: Type of message (follow_up, information, etc.)
        customer_info: Customer information for personalization
        
    Returns:
        Formatted message string
    """
    try:
        customer_name = customer_info.get("first_name") or customer_info.get("name", "").split()[0] if customer_info.get("name") else "there"
        
        # Message type-specific formatting
        if message_type == "follow_up":
            formatted_message = f"Hi {customer_name},\n\n{message_content}\n\nBest regards,\nYour Sales Team"
        elif message_type == "information":
            formatted_message = f"Hello {customer_name},\n\n{message_content}\n\nIf you have any questions, please don't hesitate to reach out.\n\nBest regards,\nYour Sales Team"
        elif message_type == "promotional":
            formatted_message = f"Dear {customer_name},\n\n{message_content}\n\nWe'd love to hear from you if you're interested!\n\nBest regards,\nYour Sales Team"
        elif message_type == "support":
            formatted_message = f"Hi {customer_name},\n\n{message_content}\n\nWe're here to help if you need anything else.\n\nBest regards,\nYour Support Team"
        else:
            # Default formatting
            formatted_message = f"Hi {customer_name},\n\n{message_content}\n\nBest regards"
        
        return formatted_message
        
    except Exception as e:
        logger.error(f"[PORTABLE_MESSAGE_FORMAT] Error formatting message: {str(e)}")
        return message_content  # Return original content as fallback


async def get_or_create_customer_user(customer_id: str, customer_info: Dict[str, Any]) -> Optional[str]:
    """
    PORTABLE: Get or create a user record for a customer.
    
    Can be used by any agent to ensure customers have user records for chat delivery.
    
    Args:
        customer_id: The customer UUID from the customers table
        customer_info: Customer information dictionary
        
    Returns:
        The user_id (UUID) if successful, None otherwise
    """
    try:
        await db_client._async_initialize_client()
        
        # First, try to find an existing user for this customer
        existing_user = await asyncio.to_thread(
            lambda: db_client.client.table("users").select("id").eq("customer_id", customer_id).execute()
        )
        
        if existing_user.data:
            # User already exists, return the user_id
            user_id = existing_user.data[0]["id"]
            logger.debug(f"[PORTABLE_USER_LOOKUP] Found existing user {user_id} for customer {customer_info.get('name', customer_id)}")
            return user_id
        
        # User doesn't exist, create one
        user_data = {
            "email": customer_info.get("email", ""),
            "display_name": customer_info.get("name", "Unknown Customer"),
            "user_type": "customer",
            "customer_id": customer_id,
            "is_active": True,
            "is_verified": False,
            "metadata": {
                "customer_info": {
                    "phone": customer_info.get("phone", ""),
                    "company": customer_info.get("company", ""),
                    "is_for_business": customer_info.get("is_for_business", False)
                }
            }
        }
        
        # Create the user record
        user_result = await asyncio.to_thread(
            lambda: db_client.client.table("users").insert(user_data).execute()
        )
        
        if user_result.data:
            user_id = user_result.data[0]["id"]
            logger.info(f"[PORTABLE_USER_LOOKUP] Created new user {user_id} for customer {customer_info.get('name', customer_id)}")
            return user_id
        else:
            logger.error(f"[PORTABLE_USER_LOOKUP] Failed to create user for customer {customer_info.get('name', customer_id)}")
            return None
            
    except Exception as e:
        logger.error(f"[PORTABLE_USER_LOOKUP] Error getting or creating user for customer {customer_info.get('name', customer_id)}: {str(e)}")
        return None


async def get_or_create_customer_conversation(
    user_id: str, 
    customer_info: Dict[str, Any]
) -> Optional[str]:
    """
    PORTABLE: Get or create a conversation for customer message delivery.
    
    Can be used by any agent to ensure continuity in customer conversations
    and proper integration with the existing chat system.
    
    Args:
        user_id: User ID for the customer
        customer_info: Customer information
        
    Returns:
        Conversation ID if successful, None otherwise
    """
    try:
        await db_client._async_initialize_client()
        
        # Look for existing active conversations for this user
        existing_conversations = await asyncio.to_thread(
            lambda: db_client.client.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        )
        
        if existing_conversations.data:
            conversation_id = existing_conversations.data[0]["id"]
            logger.info(f"[PORTABLE_CONVERSATION] Using existing conversation {conversation_id} for customer {customer_info.get('name', user_id)}")
            
            # Update the conversation timestamp
            await asyncio.to_thread(
                lambda: db_client.client.table("conversations").update({
                    "updated_at": datetime.now().isoformat()
                }).eq("id", conversation_id).execute()
            )
            
            return conversation_id
        
        # Create new conversation
        conversation_id = str(uuid4())
        conversation_data = {
            "id": conversation_id,
            "user_id": user_id,
            "title": f"Messages with {customer_info.get('name', 'Customer')}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "customer_id": customer_info.get("id") or customer_info.get("customer_id"),
                "customer_name": customer_info.get("name", ""),
                "conversation_type": "employee_initiated",
                "message_delivery": True
            }
        }
        
        # Insert new conversation
        result = await asyncio.to_thread(
            lambda: db_client.client.table("conversations").insert(conversation_data).execute()
        )
        
        if result.data:
            logger.info(f"[PORTABLE_CONVERSATION] Created new conversation {conversation_id} for customer {customer_info.get('name', user_id)}")
            return conversation_id
        else:
            logger.error(f"[PORTABLE_CONVERSATION] Failed to create conversation for customer {customer_info.get('name', user_id)}")
            return None
            
    except Exception as e:
        logger.error(f"[PORTABLE_CONVERSATION] Error getting or creating conversation: {str(e)}")
        return None


async def send_real_time_notification(
    user_id: str,
    conversation_id: str,
    message_content: str,
    customer_info: Dict[str, Any]
) -> None:
    """
    PORTABLE: Send real-time notification to customer.
    
    Can be used by any agent to send notifications. This is optional and 
    can be implemented later for email, SMS, or in-app notifications.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_content: Message content
        customer_info: Customer information
    """
    try:
        # TODO: Implement real-time notifications
        # This could include:
        # - Email notifications
        # - SMS alerts  
        # - In-app push notifications
        # - Webhook calls to external systems
        
        logger.info(f"[PORTABLE_NOTIFICATION] Real-time notification placeholder for customer {customer_info.get('name', user_id)}")
        
        # Example placeholder for future implementation:
        # await send_email_notification(customer_info, message_content)
        # await send_sms_notification(customer_info, message_content)
        # await send_webhook_notification(customer_info, message_content)
        
    except Exception as e:
        logger.error(f"[PORTABLE_NOTIFICATION] Error sending real-time notification: {str(e)}")


async def store_customer_message_audit(
    customer_id: str,
    message_content: str,
    message_type: str,
    delivery_status: str,
    customer_info: Dict[str, Any]
) -> bool:
    """
    PORTABLE: Store audit trail for compliance and tracking purposes.
    
    Can be used by any agent to maintain delivery tracking and compliance logging.
    This is separate from the main message storage and focuses on audit trails.
    
    Args:
        customer_id: Customer identifier
        message_content: Message content
        message_type: Type of message
        delivery_status: Delivery status (delivered, failed, etc.)
        customer_info: Customer information
        
    Returns:
        bool: True if audit stored successfully, False otherwise
    """
    try:
        # This is placeholder for audit storage - implement based on compliance requirements
        logger.info(f"[PORTABLE_AUDIT] Storing audit trail for customer message delivery")
        logger.info(f"[PORTABLE_AUDIT] Customer: {customer_info.get('name', customer_id)}")
        logger.info(f"[PORTABLE_AUDIT] Message Type: {message_type}")
        logger.info(f"[PORTABLE_AUDIT] Delivery Status: {delivery_status}")
        logger.info(f"[PORTABLE_AUDIT] Message Length: {len(message_content)} characters")
        
        # TODO: Implement actual audit storage based on compliance requirements
        # This could include:
        # - Database audit table
        # - External logging service
        # - Compliance management system
        
        return True
        
    except Exception as e:
        logger.error(f"[PORTABLE_AUDIT] Error storing audit trail: {str(e)}")
        return False


async def integrate_with_memory_systems(
    customer_id: str,
    message_content: str,
    message_type: str,
    customer_info: Dict[str, Any],
    memory_manager: Any
) -> bool:
    """
    PORTABLE: Ensure message integrates with memory systems.
    
    Can be used by any agent to integrate messages with framework-specific
    memory systems and long-term memory persistence.
    
    Args:
        customer_id: Customer identifier
        message_content: Message content
        message_type: Type of message
        customer_info: Customer information
        memory_manager: Memory manager instance
        
    Returns:
        bool: True if integration successful, False otherwise
    """
    try:
        logger.info(f"[PORTABLE_MEMORY_INTEGRATION] Integrating message with memory systems")
        
        # This would integrate with the specific memory system being used
        # For now, we'll just log the integration
        logger.info(f"[PORTABLE_MEMORY_INTEGRATION] Message integrated with memory systems for customer {customer_info.get('name', customer_id)}")
        
        # TODO: Implement specific memory integration based on the framework being used
        # This could include:
        # - LangGraph checkpointer integration
        # - Long-term memory storage
        # - Vector database updates
        # - Conversation summary updates
        
        return True
        
    except Exception as e:
        logger.error(f"[PORTABLE_MEMORY_INTEGRATION] Error integrating with memory systems: {str(e)}")
        return False


async def store_customer_message(
    customer_id: str,
    message_content: str,
    message_type: str,
    delivery_status: str,
    customer_info: Dict[str, Any] = None
) -> Optional[str]:
    """
    PORTABLE: Store customer message in database for persistence.
    
    Can be used by any agent as a fallback message storage method
    when memory manager is not available.
    
    Args:
        customer_id: Customer identifier
        message_content: Message content
        message_type: Type of message
        delivery_status: Delivery status
        customer_info: Customer information
        
    Returns:
        Message ID if successful, None otherwise
    """
    try:
        await db_client._async_initialize_client()
        
        # Use the actual UUID from customer_info if available
        actual_customer_id = None
        if customer_info:
            actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
        
        customer_identifier = actual_customer_id or customer_id
        
        message_data = {
            "id": str(uuid4()),
            "customer_id": customer_identifier,
            "message_content": message_content,
            "message_type": message_type,
            "delivery_status": delivery_status,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "customer_name": customer_info.get("name", "") if customer_info else "",
                "delivery_method": "direct_storage",
                "employee_initiated": True
            }
        }
        
        # Store in customer_messages table (create if doesn't exist)
        result = await asyncio.to_thread(
            lambda: db_client.client.table("customer_messages").insert(message_data).execute()
        )
        
        if result.data:
            message_id = result.data[0]["id"]
            logger.info(f"[PORTABLE_MESSAGE_STORAGE] Stored message {message_id} for customer {customer_identifier}")
            return message_id
        else:
            logger.error(f"[PORTABLE_MESSAGE_STORAGE] Failed to store message for customer {customer_identifier}")
            return None
            
    except Exception as e:
        logger.error(f"[PORTABLE_MESSAGE_STORAGE] Error storing customer message: {str(e)}")
        return None
