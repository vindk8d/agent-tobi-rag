"""
Customer Message Tools

Tools for sending messages and communications to customers with proper
validation, formatting, and human-in-the-loop confirmation workflows.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

# Core imports
from langchain_core.tools import tool
from langsmith import traceable

# Toolbox imports
from .toolbox import (
    get_current_employee_id,
    get_current_user_id,
    lookup_customer,
    validate_employee_access
)

# HITL imports - using relative imports for toolbox
try:
    from backend.agents.hitl import request_approval, request_input
except ImportError:
    # Fallback for when running from backend directory
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.hitl import request_approval, request_input

logger = logging.getLogger(__name__)

# =============================================================================
# MESSAGE VALIDATION AND FORMATTING
# =============================================================================

def _validate_message_content(message_content: str, message_type: str) -> Dict[str, Any]:
    """
    Validate message content for appropriateness and completeness.
    
    Args:
        message_content: The message text to validate
        message_type: Type of message (follow_up, information, promotional, support)
        
    Returns:
        Dictionary with validation results, errors, and warnings
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check for empty or very short messages
    if not message_content or not message_content.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("Message content cannot be empty")
        return validation_result
    
    content = message_content.strip()
    
    # Check minimum length
    if len(content) < 10:
        validation_result["valid"] = False
        validation_result["errors"].append("Message is too short - please provide more detail")
    
    # Check for professional tone (basic checks)
    unprofessional_words = ["stupid", "dumb", "idiotic", "hate", "terrible"]
    if any(word in content.lower() for word in unprofessional_words):
        validation_result["warnings"].append("Consider using more professional language")
    
    # Type-specific validation
    if message_type == "promotional":
        if "discount" not in content.lower() and "offer" not in content.lower() and "special" not in content.lower():
            validation_result["warnings"].append("Promotional messages typically include offers or special deals")
    
    elif message_type == "support":
        if "help" not in content.lower() and "assist" not in content.lower() and "support" not in content.lower():
            validation_result["warnings"].append("Support messages should clearly offer assistance")
    
    return validation_result

def _format_message_by_type(message_content: str, message_type: str, customer_info: Dict[str, Any]) -> str:
    """
    Format message content with appropriate templates based on message type.
    
    Args:
        message_content: Raw message content
        message_type: Type of message for template selection
        customer_info: Customer information for personalization
        
    Returns:
        Professionally formatted message
    """
    customer_name = customer_info.get('name', 'Valued Customer')
    company_name = "AutoDealer Pro"  # Could be configurable
    
    if message_type == "follow_up":
        return f"""Dear {customer_name},

I hope this message finds you well. I wanted to follow up with you regarding your recent inquiry.

{message_content}

Please don't hesitate to reach out if you have any questions or would like to schedule a time to discuss your needs further.

Best regards,
{company_name} Sales Team"""

    elif message_type == "information":
        return f"""Hello {customer_name},

I have some information that might be helpful for your vehicle search:

{message_content}

If you need any clarification or have additional questions, please feel free to contact me.

Best regards,
{company_name} Sales Team"""

    elif message_type == "promotional":
        return f"""Dear {customer_name},

We have an exciting opportunity that might interest you:

{message_content}

This offer is available for a limited time. Contact us today to learn more!

Best regards,
{company_name} Sales Team"""

    elif message_type == "support":
        return f"""Hello {customer_name},

I'm reaching out to provide assistance with your recent inquiry:

{message_content}

Our team is here to help ensure you have the best possible experience. Please let me know if there's anything else I can do for you.

Best regards,
{company_name} Customer Support"""

    else:
        # Default formatting
        return f"""Dear {customer_name},

{message_content}

Thank you for your business.

Best regards,
{company_name} Team"""

# =============================================================================
# MESSAGE DELIVERY FUNCTIONS
# =============================================================================

async def _deliver_customer_message(
    customer_info: Dict[str, Any],
    message_content: str,
    formatted_message: str,
    message_type: str,
    sender_employee_id: str
) -> Dict[str, Any]:
    """
    Deliver the message to the customer through available channels.
    
    Args:
        customer_info: Customer information including contact details
        message_content: Original message content
        formatted_message: Professionally formatted message
        message_type: Type of message
        sender_employee_id: ID of the employee sending the message
        
    Returns:
        Dictionary with delivery results and status
    """
    try:
        # For now, we'll simulate message delivery
        # In a real implementation, this would integrate with email/SMS services
        
        customer_name = customer_info.get('name', 'Unknown Customer')
        customer_email = customer_info.get('email', 'no-email')
        
        logger.info(f"[MESSAGE_DELIVERY] Delivering {message_type} message to {customer_name} ({customer_email})")
        
        # Create message record in database
        message_id = await _create_customer_message_record(
            customer_info, message_content, message_type, sender_employee_id
        )
        
        # Simulate successful delivery
        delivery_result = {
            "status": "delivered",
            "message_id": message_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "delivery_method": "email",  # Would be determined by customer preferences
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[MESSAGE_DELIVERY] ✅ Message delivered successfully to {customer_name}")
        return delivery_result
        
    except Exception as e:
        logger.error(f"[MESSAGE_DELIVERY] Error delivering message: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def _create_customer_message_record(
    customer_info: Dict[str, Any],
    message_content: str,
    message_type: str,
    sender_employee_id: str
) -> Optional[str]:
    """
    Create a record of the customer message in the database.
    
    Args:
        customer_info: Customer information
        message_content: Message content
        message_type: Type of message
        sender_employee_id: Employee who sent the message
        
    Returns:
        Message ID if successful, None otherwise
    """
    try:
        from .toolbox import get_sql_database
        
        # Get database connection
        db = await get_sql_database()
        if not db:
            logger.warning("[MESSAGE_RECORD] No database connection available")
            return None
        
        # Generate a simple message ID
        import uuid
        message_id = str(uuid.uuid4())
        
        # Insert message record (simplified - would use proper ORM in real implementation)
        customer_id = customer_info.get('id', customer_info.get('customer_id', 'unknown'))
        
        insert_query = """
        INSERT INTO customer_messages (id, customer_id, sender_employee_id, message_type, 
                                     message_content, status, created_at)
        VALUES (%s, %s, %s, %s, %s, 'sent', NOW())
        """
        
        # Note: This is a simplified approach. In production, you'd use proper ORM
        logger.info(f"[MESSAGE_RECORD] Recording message {message_id} for customer {customer_id}")
        
        return message_id
        
    except Exception as e:
        logger.error(f"[MESSAGE_RECORD] Error creating message record: {e}")
        return None

# =============================================================================
# HITL CONFIRMATION HANDLERS
# =============================================================================

async def _handle_confirmation_approved(context_data: Dict[str, Any]) -> str:
    """Handle approved customer message confirmation."""
    try:
        logger.info("[CUSTOMER_MESSAGE] Processing approved message confirmation")
        
        # Extract data from context
        customer_info = context_data.get("customer_info", {})
        message_content = context_data.get("message_content", "")
        formatted_message = context_data.get("formatted_message", "")
        message_type = context_data.get("message_type", "follow_up")
        sender_employee_id = context_data.get("sender_employee_id", "")
        
        # Deliver the message
        delivery_result = await _deliver_customer_message(
            customer_info=customer_info,
            message_content=message_content,
            formatted_message=formatted_message,
            message_type=message_type,
            sender_employee_id=sender_employee_id
        )
        
        if delivery_result["status"] == "delivered":
            customer_name = customer_info.get('name', 'Customer')
            return f"""✅ **Message Sent Successfully**

**To:** {customer_name}
**Type:** {message_type.replace('_', ' ').title()}
**Delivered:** {delivery_result.get('delivery_method', 'email')}

Your message has been delivered to the customer. They should receive it shortly."""

        else:
            return f"""❌ **Message Delivery Failed**

There was an issue delivering your message: {delivery_result.get('error', 'Unknown error')}

Please try again or contact technical support if the issue persists."""
            
    except Exception as e:
        logger.error(f"[CUSTOMER_MESSAGE] Error handling approved confirmation: {e}")
        return f"❌ Error processing your message confirmation. Please try again."

async def _handle_confirmation_denied(context_data: Dict[str, Any]) -> str:
    """Handle denied customer message confirmation."""
    try:
        logger.info("[CUSTOMER_MESSAGE] Processing denied message confirmation")
        
        customer_info = context_data.get("customer_info", {})
        customer_name = customer_info.get('name', 'Customer')
        
        return f"""🚫 **Message Cancelled**

The message to {customer_name} has been cancelled and will not be sent.

You can create a new message anytime using the customer messaging tool."""
        
    except Exception as e:
        logger.error(f"[CUSTOMER_MESSAGE] Error handling denied confirmation: {e}")
        return "Message cancelled."

# =============================================================================
# MAIN CUSTOMER MESSAGE TOOL
# =============================================================================

class TriggerCustomerMessageParams(BaseModel):
    """Parameters for triggering customer outreach messages."""
    customer_id: str = Field(..., description="Customer identifier: UUID, name, or email address")
    message_content: str = Field(..., description="Content of the message to send")
    message_type: str = Field(default="follow_up", description="Type of message: follow_up, information, promotional, support")

@tool(args_schema=TriggerCustomerMessageParams)
@traceable(name="trigger_customer_message")
async def trigger_customer_message(customer_id: str, message_content: str, message_type: str = "follow_up") -> str:
    """
    Prepare a customer message for human-in-the-loop confirmation (Employee Only).
    
    This tool validates inputs, prepares the message, and returns confirmation data
    for human approval before sending messages to customers.
    
    **Use this tool for:**
    - Following up with potential customers
    - Sending informational updates about products or services
    - Delivering promotional offers and special deals
    - Providing customer support and assistance
    - Professional business communication with customers
    
    **Do NOT use this tool for:**
    - Internal employee communications (use internal tools)
    - Automated system notifications (use appropriate automation)
    - Bulk marketing campaigns (use marketing automation tools)
    - Urgent communications (use direct contact methods)
    
    **Message Types:**
    - follow_up: General follow-up on inquiries or previous interactions
    - information: Sharing helpful information, updates, or details
    - promotional: Special offers, discounts, or promotional content
    - support: Customer service, assistance, or problem resolution
    
    **Access Control:** Employee-only tool with comprehensive validation.
    
    Args:
        customer_id: The customer identifier (UUID, name, or email address) to message
        message_content: The content of the message to send
        message_type: Type of message (follow_up, information, promotional, support)
    
    Returns:
        Confirmation request for human approval before sending the message
    """
    try:
        # Enhanced employee access control
        is_employee, access_message = validate_employee_access()
        if not is_employee:
            return access_message
        
        sender_employee_id = await get_current_employee_id()
        
        logger.info(f"[CUSTOMER_MESSAGE] Processing message request from employee {sender_employee_id}")
        logger.info(f"[CUSTOMER_MESSAGE] Target customer: {customer_id}")
        logger.info(f"[CUSTOMER_MESSAGE] Message type: {message_type}")
        
        # Validate inputs
        if not customer_id or not customer_id.strip():
            return """❌ **Customer Identifier Required**

Please provide a customer identifier to send a message:
• Customer name (e.g., "John Smith")
• Email address (e.g., "john@email.com")  
• Customer ID or UUID

This helps me find the right customer to message."""
        
        # Validate message type
        valid_types = ["follow_up", "information", "promotional", "support"]
        if message_type not in valid_types:
            message_type = "follow_up"  # Default fallback
            logger.info(f"[CUSTOMER_MESSAGE] Invalid message type provided, defaulting to: {message_type}")
            
        # Validate message content
        validation_result = _validate_message_content(message_content, message_type)
        if not validation_result["valid"]:
            error_msg = "❌ **Message Validation Failed:**\n"
            for error in validation_result["errors"]:
                error_msg += f"• {error}\n"
            if validation_result["warnings"]:
                error_msg += "\n⚠️ **Suggestions:**\n"
                for warning in validation_result["warnings"]:
                    error_msg += f"• {warning}\n"
            return error_msg.strip()
        
        # Look up customer information
        logger.info(f"[CUSTOMER_MESSAGE] Looking up customer: {customer_id}")
        customer_info = await lookup_customer(customer_id)
        
        if not customer_info:
            # Customer not found - request additional information
            not_found_prompt = f"""🔍 **Customer Not Found**

I couldn't find a customer matching: "{customer_id}"

**Please provide additional information to help me locate the customer:**
• Full name or partial name
• Email address or domain
• Phone number
• Company name
• Any other identifying information

**Your message:** "{message_content}"
**Message type:** {message_type.replace('_', ' ').title()}

*I'll search again once you provide more details.*"""

            # Prepare context for retry
            context_data = {
                "tool": "trigger_customer_message", 
                "action": "customer_lookup_retry",
                "original_customer_id": customer_id,
                "message_content": message_content,
                "message_type": message_type,
                "sender_employee_id": sender_employee_id,
                "requested_by": get_current_user_id(),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[CUSTOMER_MESSAGE] Customer '{customer_id}' not found - requesting additional information via HITL")
            
            # Request additional customer information
            return request_input(
                prompt=not_found_prompt,
                input_type="customer_identifier", 
                context=context_data,
                validation_hints=[
                    "Any additional detail helps - even partial information is useful",
                    "Don't worry if you only remember some details",
                    "I'll search based on whatever you can provide"
                ]
            )
        
        logger.info(f"[CUSTOMER_MESSAGE] Validated customer: {customer_info.get('name', 'Unknown')} ({customer_info.get('email', 'no-email')})")
        
        # Format the message with professional templates
        formatted_message = _format_message_by_type(message_content, message_type, customer_info)
        
        # Create confirmation prompt for the user
        customer_name = customer_info.get('name', 'Unknown Customer')
        customer_email = customer_info.get('email', 'no-email')
        message_type_display = message_type.replace('_', ' ').title()
        
        confirmation_prompt = f"""🔄 **Customer Message Confirmation**

**To:** {customer_name} ({customer_email})
**Type:** {message_type_display}
**Your Message:** {message_content}

**Formatted Preview:**
{formatted_message}

Do you want to send this message to the customer?"""
        
        # Prepare context data for post-confirmation processing
        context_data = {
            "tool": "trigger_customer_message",
            "customer_info": customer_info,
            "message_content": message_content,
            "formatted_message": formatted_message,
            "message_type": message_type,
            "validation_result": validation_result,
            "customer_id": customer_id,
            "sender_employee_id": sender_employee_id,
            "requested_by": get_current_user_id(),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[CUSTOMER_MESSAGE] Message prepared for HITL confirmation - Customer: {customer_name}")
        
        # Request approval for sending the message
        return request_approval(
            prompt=confirmation_prompt,
            context=context_data,
            approve_text="send",
            deny_text="cancel"
        )

    except Exception as e:
        logger.error(f"Error in trigger_customer_message: {e}")
        return f"""❌ **Message Preparation Error**

Sorry, I encountered an error while preparing your customer message request: {str(e)}

**Please try:**
- Checking your customer identifier and message content
- Trying again in a few moments
- Contacting support if the issue persists"""

# =============================================================================
# CUSTOMER MESSAGE ANALYTICS
# =============================================================================

async def get_message_history(customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get message history for a customer (internal use)."""
    try:
        from .toolbox import get_sql_database
        
        db = await get_sql_database()
        if not db:
            return []
        
        # This would query the customer_messages table
        # Simplified implementation for now
        logger.info(f"[MESSAGE_HISTORY] Getting message history for customer: {customer_id}")
        return []
        
    except Exception as e:
        logger.error(f"[MESSAGE_HISTORY] Error getting message history: {e}")
        return []

async def get_message_stats(employee_id: Optional[str] = None) -> Dict[str, Any]:
    """Get message statistics for reporting (internal use)."""
    try:
        from .toolbox import get_sql_database
        
        db = await get_sql_database()
        if not db:
            return {"status": "unavailable"}
        
        # This would query message statistics
        # Simplified implementation for now
        logger.info(f"[MESSAGE_STATS] Getting message statistics for employee: {employee_id}")
        return {
            "status": "available",
            "total_messages": 0,
            "messages_today": 0,
            "response_rate": "N/A"
        }
        
    except Exception as e:
        logger.error(f"[MESSAGE_STATS] Error getting message stats: {e}")
        return {"status": "error", "error": str(e)}
