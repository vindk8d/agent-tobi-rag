"""
Universal API Processing Utilities.

These utilities can be used by ANY agent that needs clean API response processing.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- Agent result processing for clean API responses
- HITL (Human-in-the-Loop) state detection and handling
- Error message formatting with user-friendly messages
- Message filtering and extraction
- Source processing and extraction
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.api_processing import process_agent_result_for_api, process_user_message

class MyAgent:
    async def handle_request(self, user_query: str, conversation_id: str, user_id: str):
        # Process user message with clean API interface - works with any agent
        result = await process_user_message(
            agent_invoke_func=self.invoke,  # Pass your agent's invoke method
            user_query=user_query,
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        # Result contains clean API response:
        # - message: Final response content
        # - sources: Document sources
        # - is_interrupted: HITL interaction needed
        # - hitl_phase/hitl_prompt/hitl_context: HITL data
        
        return result
```

This ensures consistent API processing across all agents without code duplication.
"""

import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


async def process_agent_result_for_api(result: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
    """
    PORTABLE: Process agent result and return clean semantic data for API layer.
    
    Can be used by any agent to provide consistent API response formatting.
    Encapsulates all LangGraph-specific result processing logic with clean interface.
    
    Args:
        result: Raw result from any agent execution
        conversation_id: Conversation ID for context
        
    Returns:
        Dict containing clean, semantic data for API response:
        - message: Final message content
        - sources: Document sources if any
        - is_interrupted: Boolean indicating if human interaction is needed
        - hitl_phase/hitl_prompt/hitl_context: HITL interaction data if interrupted
    """
    try:
        logger.info(f"🔍 [PORTABLE_API_PROCESSING] Processing agent result for API response")
        logger.info(f"🔍 [PORTABLE_API_PROCESSING] Result keys: {list(result.keys()) if result else 'None'}")
        
        # Initialize clean response structure
        api_response = {
            "message": "I apologize, but I encountered an issue processing your request.",
            "sources": [],
            "is_interrupted": False
        }
        
        if not result:
            return api_response
        
        # =================================================================
        # STEP 1: Check for HITL (Human-in-the-Loop) interactions
        # =================================================================
        hitl_phase = result.get('hitl_phase')
        hitl_prompt = result.get('hitl_prompt')
        hitl_context = result.get('hitl_context')
        
        if hitl_phase and hitl_prompt:
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] 🎯 HITL STATE DETECTED! Phase: {hitl_phase}")
            
            # Return HITL prompt with clean interface - no framework-specific details
            api_response.update({
                "message": hitl_prompt,
                "is_interrupted": True,
                "hitl_phase": hitl_phase,
                "hitl_prompt": hitl_prompt,
                "hitl_context": hitl_context or {}
            })
            return api_response
        
        # =================================================================
        # STEP 2: Check for legacy framework interrupts (for backward compatibility)
        # =================================================================
        is_interrupted = False
        interrupt_message = ""
        
        # Check for framework-specific interrupt indicators
        if isinstance(result, dict) and '__interrupt__' in result:
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] Legacy framework interrupt detected")
            is_interrupted = True
            
            # Extract interrupt message from framework's internal format
            interrupt_data = result['__interrupt__']
            if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                first_interrupt = interrupt_data[0]
                if hasattr(first_interrupt, 'value'):
                    interrupt_message = str(first_interrupt.value)
                    logger.info(f"🔍 [PORTABLE_API_PROCESSING] Extracted legacy interrupt message")
        elif hasattr(result, '__interrupt__'):
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] Legacy framework interrupt attribute detected")
            is_interrupted = True
        
        # Handle legacy interrupts
        if is_interrupted and interrupt_message:
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] Processing legacy interrupt for conversation {conversation_id}")
            
            api_response.update({
                "message": interrupt_message,
                "is_interrupted": True,
                "hitl_phase": "awaiting_response",
                "hitl_prompt": interrupt_message,
                "hitl_context": {"legacy": True}
            })
            return api_response
        
        # =================================================================
        # STEP 3: Check for error states first
        # =================================================================
        if result.get('error'):
            logger.error(f"🔍 [PORTABLE_API_PROCESSING] Agent returned error: {result['error']}")
            error_msg = result['error']
            
            # Provide user-friendly error messages based on error type
            if "Failed to resume conversation" in error_msg:
                user_friendly_msg = "I'm having trouble processing your approval right now. Please try again, or contact support if the issue persists."
            elif "Synchronous calls to AsyncPostgresSaver" in error_msg:
                user_friendly_msg = "I encountered a technical issue while processing your request. Please try again in a moment."
            elif "not found in database" in error_msg:
                user_friendly_msg = "This conversation session has expired. Please start a new conversation."
            else:
                user_friendly_msg = "I encountered an unexpected error. Please try your request again or contact support if the problem continues."
            
            api_response.update({
                "message": user_friendly_msg,
                "sources": [],
                "is_interrupted": False,
                "error": True,
                "error_details": error_msg  # For debugging/logging
            })
            return api_response
        
        # =================================================================
        # STEP 4: Extract normal conversation response
        # =================================================================
        logger.info(f"🔍 [PORTABLE_API_PROCESSING] Processing normal conversation response")
        
        # Extract final message content (excluding tool messages to prevent duplicates)
        final_message_content = ""
        messages = result.get('messages', [])
        
        if messages:
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] Found {len(messages)} messages in result")
            
            # Filter out framework-specific tool messages
            # Tool messages are internal constructs and shouldn't appear in user interface
            filtered_messages = []
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'tool':
                    logger.debug(f"🔍 [PORTABLE_API_PROCESSING] Filtering out tool message: {getattr(msg, 'name', 'unknown')}")
                    continue
                elif isinstance(msg, dict) and msg.get('type') == 'tool':
                    logger.debug(f"🔍 [PORTABLE_API_PROCESSING] Filtering out tool message dict: {msg.get('name', 'unknown')}")
                    continue
                else:
                    filtered_messages.append(msg)
            
            logger.info(f"🔍 [PORTABLE_API_PROCESSING] Filtered {len(messages) - len(filtered_messages)} tool messages, {len(filtered_messages)} remaining")
            
            # Get the final assistant message (after filtering tool messages)
            for msg in reversed(filtered_messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    final_message_content = msg.content
                    logger.info(f"🔍 [PORTABLE_API_PROCESSING] Found final AI message: '{final_message_content[:100]}...'")
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                    final_message_content = msg.get('content', '')
                    logger.info(f"🔍 [PORTABLE_API_PROCESSING] Found final assistant message: '{final_message_content[:100]}...'")
                    break
                elif hasattr(msg, 'content') and not hasattr(msg, 'type'):
                    # Fallback for generic message objects
                    final_message_content = msg.content
                    logger.info(f"🔍 [PORTABLE_API_PROCESSING] Found final generic message: '{final_message_content[:100]}...'")
                    break
            
            if not final_message_content:
                logger.warning(f"🔍 [PORTABLE_API_PROCESSING] No suitable final message found in {len(filtered_messages)} filtered messages")
                # Use the last non-tool message as fallback
                if filtered_messages:
                    last_msg = filtered_messages[-1]
                    if hasattr(last_msg, 'content'):
                        final_message_content = last_msg.content
                    elif isinstance(last_msg, dict):
                        final_message_content = last_msg.get('content', '')
                    logger.info(f"🔍 [PORTABLE_API_PROCESSING] Using fallback message: '{final_message_content[:100]}...'")
        
        # Fallback if no content found - provide helpful message instead of false success
        if not final_message_content.strip():
            logger.warning(f"🔍 [PORTABLE_API_PROCESSING] No message content found in agent result")
            final_message_content = "I processed your request, but didn't generate a response message. If you were expecting a specific action or response, please try rephrasing your request."
        
        # Extract sources if any
        sources = result.get('sources', [])
        logger.info(f"🔍 [PORTABLE_API_PROCESSING] Found {len(sources)} sources")
        
        # Return clean response
        api_response.update({
            "message": final_message_content,
            "sources": sources,
            "is_interrupted": False
        })
        
        logger.info(f"🔍 [PORTABLE_API_PROCESSING] Constructed final API response")
        return api_response
        
    except Exception as e:
        logger.error(f"🔍 [PORTABLE_API_PROCESSING] Error processing agent result: {e}")
        return {
            "message": "I apologize, but I encountered an unexpected error while processing your request. Please try again, and if the problem persists, contact support.",
            "sources": [],
            "is_interrupted": False,
            "error": True,
            "error_details": str(e)
        }


async def process_user_message(
    agent_invoke_func: Callable,
    user_query: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    PORTABLE: High-level method for processing user messages with clean API interface.
    
    Can be used by any agent to provide consistent user message processing.
    Provides a semantic interface for the API layer, encapsulating all framework-specific logic.
    
    Args:
        agent_invoke_func: The agent's invoke method (any agent can pass their invoke method)
        user_query: User's message/query
        conversation_id: Conversation ID for persistence
        user_id: User ID for context
        **kwargs: Additional arguments passed to invoke
        
    Returns:
        Dict containing clean API response structure:
        - message: Final response message
        - sources: Document sources if any
        - is_interrupted: Boolean indicating if human interaction needed
        - hitl_phase/hitl_prompt/hitl_context: HITL interaction data if interrupted
        - conversation_id: Conversation ID for persistence
    """
    try:
        logger.info(f"🚀 [PORTABLE_API_INTERFACE] Processing user message for conversation {conversation_id}")
        
        # Invoke the agent with the user query (works with any agent's invoke method)
        raw_result = await agent_invoke_func(
            user_query=user_query,
            conversation_id=conversation_id,
            user_id=user_id,
            **kwargs
        )
        
        # Process the result for clean API response using portable processing
        processed_result = await process_agent_result_for_api(raw_result, conversation_id)
        
        # Add conversation ID to response
        processed_result["conversation_id"] = conversation_id or raw_result.get("conversation_id")
        
        logger.info(f"✅ [PORTABLE_API_INTERFACE] User message processed successfully. Interrupted: {processed_result['is_interrupted']}")
        
        return processed_result
        
    except Exception as e:
        logger.error(f"❌ [PORTABLE_API_INTERFACE] Error processing user message: {e}")
        return {
            "message": "I apologize, but I encountered an error while processing your request.",
            "sources": [],
            "is_interrupted": False,
            "conversation_id": conversation_id,
            "error": True,
            "error_details": str(e)
        }
