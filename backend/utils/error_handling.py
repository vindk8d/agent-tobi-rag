"""
Universal Error Handling Utilities.

These utilities can be used by ANY agent that needs consistent error handling.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- Processing error handling with proper error responses
- Tool recall error state creation
- Consistent error messaging and state management
- Framework-agnostic error handling patterns
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.error_handling import handle_processing_error, create_tool_recall_error_state

class MyAgent:
    async def process_request(self, state: dict):
        try:
            # Process request logic
            return result
        except Exception as e:
            # Handle processing errors consistently - works with any agent
            return await handle_processing_error(state, e)
    
    async def handle_tool_recall(self, state: dict):
        try:
            # Tool recall logic
            return result
        except Exception as e:
            # Create error state for tool failures - works with any agent
            return create_tool_recall_error_state(state, str(e))
```

This ensures consistent error handling across all agents without code duplication.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


async def handle_processing_error(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """
    PORTABLE: Handle processing errors with proper error response.
    
    Can be used by any agent to provide consistent error handling and recovery.
    Creates user-friendly error messages while preserving state integrity.
    
    Args:
        state: Agent state (any agent's state format)
        error: The exception that occurred
        
    Returns:
        Updated state with error message and preserved context
    """
    try:
        logger.error(f"[PORTABLE_ERROR_HANDLING] Processing error: {error}")
        
        # Create user-friendly error message
        error_message = AIMessage(
            content="I apologize, but I encountered an error while processing your request. Please try again, and if the problem persists, let me know."
        )
        
        # Preserve existing messages and add error message
        messages = list(state.get("messages", []))
        messages.append(error_message)
        
        # Return updated state with error handling
        return {
            "messages": messages,
            "sources": state.get("sources", []),
            "retrieved_docs": state.get("retrieved_docs", []),
            "conversation_id": state.get("conversation_id"),
            "user_id": state.get("user_id"),
            "conversation_summary": state.get("conversation_summary"),
            "long_term_context": state.get("long_term_context", []),
            "employee_id": state.get("employee_id"),
            "customer_id": state.get("customer_id"),
            # Clear any HITL state on error
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None,
            # Add error metadata for debugging
            "error_occurred": True,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
    except Exception as nested_error:
        logger.error(f"[PORTABLE_ERROR_HANDLING] Error in error handler: {nested_error}")
        # Fallback error state
        return {
            **state,
            "messages": state.get("messages", []) + [
                AIMessage(content="I encountered a system error. Please try again.")
            ],
            "error_occurred": True,
            "error_type": "nested_error",
            "error_message": "Error in error handler"
        }


def create_tool_recall_error_state(state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """
    PORTABLE: Create error state for tool recall failures.
    
    Can be used by any agent to handle tool execution failures consistently.
    Helper function to create a clean error state when tool re-calling fails.
    
    Args:
        state: Agent state (any agent's state format)
        error_message: Specific error message for the tool failure
        
    Returns:
        Updated state with tool recall error handling
    """
    try:
        logger.error(f"[PORTABLE_TOOL_ERROR] Tool recall error: {error_message}")
        
        # Create user-friendly error response
        error_response = f"I encountered an error while processing your request: {error_message}. Please try again."
        
        # Preserve existing messages and add error message
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": error_response,
            "type": "ai"
        })
        
        # Return updated state with error handling
        return {
            **state,
            "messages": messages,
            # Clear HITL state on tool error
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None,
            # Add tool error metadata
            "tool_error_occurred": True,
            "tool_error_message": error_message
        }
        
    except Exception as nested_error:
        logger.error(f"[PORTABLE_TOOL_ERROR] Error in tool error handler: {nested_error}")
        # Fallback error state
        return {
            **state,
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": "I encountered a system error with tool processing. Please try again.",
                "type": "ai"
            }],
            "tool_error_occurred": True,
            "tool_error_message": "Error in tool error handler"
        }


def create_user_friendly_error_message(error: Exception, context: str = "") -> str:
    """
    PORTABLE: Create user-friendly error messages based on error type.
    
    Can be used by any agent to convert technical errors into user-friendly messages.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Returns:
        User-friendly error message string
    """
    try:
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Map common errors to user-friendly messages
        if "timeout" in error_str or "connectiontimeout" in error_type.lower():
            return "The request timed out. Please try again in a moment."
        
        elif "connection" in error_str or "network" in error_str:
            return "I'm having trouble connecting to our services. Please try again shortly."
        
        elif "permission" in error_str or "unauthorized" in error_str:
            return "I don't have permission to perform that action. Please contact support if you need assistance."
        
        elif "not found" in error_str or "404" in error_str:
            return "The requested information could not be found. Please check your request and try again."
        
        elif "database" in error_str or "sql" in error_str:
            return "I'm experiencing a temporary issue accessing our database. Please try again in a moment."
        
        elif "rate limit" in error_str or "too many requests" in error_str:
            return "I'm receiving too many requests right now. Please wait a moment before trying again."
        
        elif "validation" in error_str or "invalid" in error_str:
            return "There seems to be an issue with the information provided. Please check your input and try again."
        
        else:
            # Generic error message with context if provided
            base_message = "I encountered an unexpected error while processing your request."
            if context:
                return f"{base_message} (Context: {context}) Please try again, and if the problem persists, contact support."
            else:
                return f"{base_message} Please try again, and if the problem persists, contact support."
                
    except Exception as nested_error:
        logger.error(f"[PORTABLE_ERROR_MESSAGE] Error creating user-friendly message: {nested_error}")
        return "I encountered a system error. Please try again or contact support."


def log_error_with_context(error: Exception, context: Dict[str, Any], operation: str = "") -> None:
    """
    PORTABLE: Log errors with structured context information.
    
    Can be used by any agent to provide consistent error logging with context.
    
    Args:
        error: The exception that occurred
        context: Context information (state, user_id, conversation_id, etc.)
        operation: Description of the operation that failed
    """
    try:
        logger.error(f"[PORTABLE_ERROR_LOG] {operation} failed: {type(error).__name__}: {error}")
        
        # Log relevant context information
        if context.get("user_id"):
            logger.error(f"[PORTABLE_ERROR_LOG] User ID: {context['user_id']}")
        
        if context.get("conversation_id"):
            logger.error(f"[PORTABLE_ERROR_LOG] Conversation ID: {context['conversation_id']}")
        
        if context.get("customer_id"):
            logger.error(f"[PORTABLE_ERROR_LOG] Customer ID: {context['customer_id']}")
        
        if context.get("employee_id"):
            logger.error(f"[PORTABLE_ERROR_LOG] Employee ID: {context['employee_id']}")
        
        # Log current operation context
        if context.get("current_operation"):
            logger.error(f"[PORTABLE_ERROR_LOG] Current Operation: {context['current_operation']}")
        
        # Log message count for context
        messages = context.get("messages", [])
        if messages:
            logger.error(f"[PORTABLE_ERROR_LOG] Message Count: {len(messages)}")
            
    except Exception as nested_error:
        logger.error(f"[PORTABLE_ERROR_LOG] Error in error logging: {nested_error}")


def is_recoverable_error(error: Exception) -> bool:
    """
    PORTABLE: Determine if an error is recoverable and should trigger a retry.
    
    Can be used by any agent to implement intelligent retry logic.
    
    Args:
        error: The exception to evaluate
        
    Returns:
        bool: True if the error is likely recoverable, False otherwise
    """
    try:
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Recoverable error patterns
        recoverable_patterns = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "rate limit",
            "busy",
            "unavailable",
            "502",  # Bad Gateway
            "503",  # Service Unavailable
            "504",  # Gateway Timeout
        ]
        
        # Non-recoverable error patterns
        non_recoverable_patterns = [
            "permission",
            "unauthorized",
            "forbidden",
            "not found",
            "validation",
            "invalid",
            "syntax",
            "parse",
            "401",  # Unauthorized
            "403",  # Forbidden
            "404",  # Not Found
            "400",  # Bad Request
        ]
        
        # Check for non-recoverable patterns first
        for pattern in non_recoverable_patterns:
            if pattern in error_str or pattern in error_type.lower():
                return False
        
        # Check for recoverable patterns
        for pattern in recoverable_patterns:
            if pattern in error_str or pattern in error_type.lower():
                return True
        
        # Default to non-recoverable for unknown errors to avoid infinite loops
        return False
        
    except Exception as nested_error:
        logger.error(f"[PORTABLE_ERROR_RECOVERY] Error evaluating recoverability: {nested_error}")
        return False
