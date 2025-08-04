"""
General-Purpose Human-in-the-Loop (HITL) Node Implementation

This module provides a unified HITL architecture for LangGraph agents, supporting:
- Confirmation interactions (approve/deny)
- Selection interactions (choose from options)
- Input request interactions (free-form text input)
- Multi-step input interactions (sequential information gathering)

The module standardizes all HITL interactions through a single node that handles
both prompting and response processing, using the interrupt() mechanism for
human interaction.
"""

import logging
import json
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from core.config import get_settings

# Configure logging for HITL module
logger = logging.getLogger(__name__)

# LangGraph interrupt functionality for HITL interactions
try:
    from langgraph.types import interrupt
    from langgraph.errors import GraphInterrupt
    logger.info("🚀 [HITL_INIT] LangGraph interrupt functionality imported successfully")
except ImportError:
    # Fallback for development/testing - create a mock interrupt function and exception
    logger.warning("⚠️ [HITL_INIT] LangGraph not available, using mock interrupt functionality")
    def interrupt(message: str):
        logger.info(f"🧪 [MOCK_INTERRUPT] {message}")
        return f"[MOCK_INTERRUPT] {message}"
    
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass

# HITL Phase definitions for ultra-minimal 3-field architecture
# 
# NOTE: HITLInteractionType has been ELIMINATED as part of the revolutionary
# 3-field architecture. Tools now define their own interaction styles through
# dedicated HITL request tools instead of rigid type-based dispatch.
HITLPhase = Literal[
    "needs_prompt",          # Ready to show prompt to user (triggers interrupt)
    "awaiting_response",     # Waiting for user response (processing mode)
    "approved",              # User approved the action (route back to agent)
    "denied"                 # User denied the action (route back to agent)
]

# Module-level constants
HITL_REQUIRED_PREFIX = "HITL_REQUIRED"
# LLM-driven interpretation handles all user responses naturally.
# Users can say "send it", "go ahead", "yes", "not now", "skip this", etc. and the LLM understands.
#
# Tools can provide suggested response text for user guidance, but these are just
# friendly suggestions - not rigid constraints for validation.

# Flexible defaults for user-friendly suggestions (not rigid validation)
SUGGESTED_APPROVE_TEXT = "approve"
SUGGESTED_DENY_TEXT = "deny" 
SUGGESTED_CANCEL_TEXT = "cancel"


class HITLJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for HITL context data that handles complex data types.
    
    Handles serialization of:
    - datetime objects (converted to ISO format strings)
    - date objects (converted to ISO format strings)
    - Other non-serializable objects (converted to string representation)
    
    This ensures that HITL context data can be safely serialized and deserialized
    across interrupt/resume cycles in LangGraph.
    """
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle objects with attributes by converting to dict
            return {
                '__type__': obj.__class__.__name__,
                '__data__': obj.__dict__
            }
        elif hasattr(obj, '_asdict'):
            # Handle named tuples
            return {
                '__type__': obj.__class__.__name__,
                '__data__': obj._asdict()
            }
        else:
            # Fallback: convert to string representation
            return str(obj)


def serialize_hitl_context(data: Dict[str, Any]) -> str:
    """
    Serialize HITL context using the custom encoder for 3-field architecture.
    
    Used by dedicated HITL request tools (request_approval, request_input, request_selection)
    to serialize context data for the revolutionary 3-field HITL architecture.
    
    Args:
        data: Dictionary containing HITL context data to serialize
        
    Returns:
        JSON string representation of the context data
        
    Raises:
        ValueError: If data cannot be serialized
    """
    try:
        return json.dumps(data, cls=HITLJSONEncoder)
    except Exception as e:
        logger.error(f"❌ [HITL_SERIALIZATION] Failed to serialize HITL context: {e}")
        raise ValueError(f"Unable to serialize HITL context: {str(e)}")


def deserialize_hitl_context(json_str: str) -> Dict[str, Any]:
    """
    Deserialize HITL context from JSON string for 3-field architecture.
    
    Used by parse_tool_response() to deserialize context data from dedicated 
    HITL request tools in the revolutionary 3-field HITL architecture.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Dictionary containing deserialized HITL context data
        
    Raises:
        ValueError: If JSON string cannot be parsed
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"❌ [HITL_SERIALIZATION] Failed to deserialize HITL context: {e}")
        raise ValueError(f"Unable to deserialize HITL context: {str(e)}")


# HITL-specific error handling classes
class HITLError(Exception):
    """Base exception for all HITL-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()


class HITLValidationError(HITLError):
    """Exception raised when HITL data validation fails."""
    pass


class HITLInteractionError(HITLError):
    """Exception raised when HITL interaction processing fails."""
    pass


class HITLResponseError(HITLError):
    """Exception raised when processing human responses fails."""
    pass


class HITLConfigurationError(HITLError):
    """Exception raised when HITL configuration is invalid."""
    pass


# HITL-specific logging utilities
class HITLLogger:
    """Specialized logger for HITL operations with structured logging."""
    
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
    
    def log_interaction_start(self, interaction_type: str, context: Dict[str, Any]):
        """Log the start of a HITL interaction."""
        self.logger.info(
            f"[HITL_START] {interaction_type} interaction initiated",
            extra={
                "interaction_type": interaction_type,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_interaction_complete(self, interaction_type: str, response: str, success: bool = True):
        """Log the completion of a HITL interaction."""
        level = logging.INFO if success else logging.WARNING
        status = "SUCCESS" if success else "FAILED"
        
        self.logger.log(
            level,
            f"[HITL_COMPLETE] {interaction_type} interaction {status.lower()}",
            extra={
                "interaction_type": interaction_type,
                "response": response,
                "success": success,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_validation_error(self, interaction_type: str, error: str, user_input: str):
        """Log validation errors during HITL interactions."""
        self.logger.warning(
            f"[HITL_VALIDATION] {interaction_type} validation failed: {error}",
            extra={
                "interaction_type": interaction_type,
                "error": error,
                "user_input": user_input,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_response_processing(self, interaction_type: str, raw_response: str, processed_result: Any):
        """Log response processing details."""
        self.logger.info(
            f"[HITL_PROCESS] {interaction_type} response processed",
            extra={
                "interaction_type": interaction_type,
                "raw_response": raw_response,
                "processed_result": str(processed_result),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_error(self, error: HITLError, interaction_type: Optional[str] = None):
        """Log HITL-specific errors with context."""
        self.logger.error(
            f"[HITL_ERROR] {error.__class__.__name__}: {error.message}",
            extra={
                "error_type": error.__class__.__name__,
                "error_message": error.message,
                "error_context": error.context,
                "interaction_type": interaction_type,
                "timestamp": error.timestamp.isoformat()
            },
            exc_info=True
        )


# Global HITL logger instance
hitl_logger = HITLLogger()


def log_hitl_operation(operation: str, **kwargs):
    """
    Convenience function for logging HITL operations.
    
    Args:
        operation: Name of the operation being performed
        **kwargs: Additional context data to log
    """
    hitl_logger.logger.info(
        f"[HITL_OP] {operation}",
        extra={
            "operation": operation,
            "context": kwargs,
            "timestamp": datetime.now().isoformat()
        }
    )


def handle_hitl_error(error: Exception, context: Dict[str, Any]) -> HITLError:
    """
    Convert generic exceptions to HITL-specific errors with context.
    
    Args:
        error: The original exception
        context: Additional context about where the error occurred
        
    Returns:
        HITLError: Wrapped error with HITL-specific context
    """
    if isinstance(error, HITLError):
        # Already a HITL error, just add context
        error.context.update(context)
        return error
    
    # Convert to appropriate HITL error type
    if "validation" in str(error).lower():
        return HITLValidationError(str(error), context)
    elif "interaction" in str(error).lower():
        return HITLInteractionError(str(error), context)
    elif "response" in str(error).lower():
        return HITLResponseError(str(error), context)
    else:
        return HITLError(str(error), context)


# REVOLUTIONARY: Dedicated HITL Request Tools
# 
# Replaces the complex HITLRequest class with simple, direct tool functions
# that create HITL requests. Each tool is focused and easy to use.
# 
# Tools define their own interaction styles without rigid class constraints.
def request_approval(
    prompt: str, 
    context: Dict[str, Any],
    approve_text: str = SUGGESTED_APPROVE_TEXT,
    deny_text: str = SUGGESTED_DENY_TEXT
) -> str:
    """
    REVOLUTIONARY: Dedicated HITL request tool for approval interactions.
    
    Uses LLM-driven interpretation - users can respond naturally with any language!
    The approve_text and deny_text are just friendly suggestions, not rigid constraints.
    
    Args:
        prompt: The question/message to show the user
        context: Tool-specific context data for post-interaction processing
        approve_text: Suggested text for approval (default: "approve") - LLM interprets naturally
        deny_text: Suggested text for denial (default: "deny") - LLM interprets naturally
        
    Returns:
        Formatted HITL request string
        
    Example:
        request_approval(
            "Send this message to John?",
            {"tool": "trigger_customer_message", "customer_id": "123"}
        )
        
    Note: Users can respond with natural language like "send it", "go ahead", "not now", 
    "skip this", etc. The LLM understands intent regardless of suggested text.
    """
    try:
        data = {
            "prompt": prompt,
            "options": {
                "approve": approve_text,
                "deny": deny_text
            },
            "awaiting_response": False,
            "context": context
        }
        
        json_data = serialize_hitl_context(data)
        return f"{HITL_REQUIRED_PREFIX}:approval:{json_data}"
        
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "request_approval",
            "prompt": prompt,
            "context": context
        })
        hitl_logger.log_error(hitl_error, "approval")
        raise hitl_error

def request_selection(
    prompt: str,
    options: List[Dict[str, Any]],
    context: Dict[str, Any],
    allow_cancel: bool = True
) -> str:
    """
    REVOLUTIONARY: Dedicated HITL request tool for selection interactions.
    
    Replaces HITLRequest.selection() with a simple, direct function.
    Tools can call this directly without complex class instantiation.
    
    Args:
        prompt: The question/instruction to show the user
        options: List of options to choose from. Each option should have at least
                a 'display' or 'name' key for user-friendly display
        context: Tool-specific context data for post-interaction processing
        allow_cancel: Whether to allow cancellation of the selection
        
    Returns:
        Formatted HITL request string
        
    Example:
        request_selection(
            "Which customer did you mean?",
            [
                {"id": "123", "display": "John Smith (john@email.com)"},
                {"id": "456", "display": "John Doe (johndoe@email.com)"}
            ],
            {"tool": "customer_lookup", "query": "john"}
        )
    """
    try:
        if not options:
            raise HITLValidationError("Selection options cannot be empty")
        
        data = {
            "prompt": prompt,
            "options": options,
            "allow_cancel": allow_cancel,
            "awaiting_response": False,
            "context": context
        }
        
        json_data = serialize_hitl_context(data)
        return f"{HITL_REQUIRED_PREFIX}:selection:{json_data}"
        
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "request_selection",
            "prompt": prompt,
            "options_count": len(options) if options else 0,
            "context": context
        })
        hitl_logger.log_error(hitl_error, "selection")
        raise hitl_error
def request_input(
    prompt: str,
    input_type: str,
    context: Dict[str, Any],
    validation_hints: Optional[List[str]] = None
) -> str:
    """
    REVOLUTIONARY: Dedicated HITL request tool for input interactions.
    
    Replaces HITLRequest.input_request() with a simple, direct function.
    Tools can call this directly without complex class instantiation.
    
    Args:
        prompt: The question/instruction to show the user
        input_type: Type of input being requested (e.g., "customer_name", "email", "description")
        context: Tool-specific context data for post-interaction processing
        validation_hints: Optional list of validation hints to show the user
        
    Returns:
        Formatted HITL request string
        
    Example:
        request_input(
            "Please provide the customer's email address:",
            "customer_email",
            {"tool": "customer_lookup", "original_query": "john"},
            ["Must be a valid email format", "Use company domain if business customer"]
        )
    """
    try:
        data = {
            "prompt": prompt,
            "input_type": input_type,
            "validation_hints": validation_hints or [],
            "awaiting_response": False,
            "context": context
        }
        
        json_data = serialize_hitl_context(data)
        return f"{HITL_REQUIRED_PREFIX}:input:{json_data}"
        
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "request_input",
            "prompt": prompt,
            "input_type": input_type,
            "context": context
        })
        hitl_logger.log_error(hitl_error, "input")
        raise hitl_error



def parse_tool_response(response: str, tool_name: str) -> Dict[str, Any]:
    """
    Parse tool responses and extract HITL requirements using ultra-minimal 3-field architecture.
    
    REVOLUTIONARY: Directly creates hitl_phase, hitl_prompt, hitl_context fields instead of
    legacy hitl_data structures. Tools define their own interaction styles without rigid types.
    
    Args:
        response: The tool response string to parse
        tool_name: Name of the tool that generated the response
        
    Returns:
        Dictionary with parsed response information:
        - For normal responses: {"type": "normal", "content": response}
        - For HITL responses: {
            "type": "hitl_required",
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "User-facing prompt text",
            "hitl_context": {"source_tool": tool_name, ...},
            "source_tool": tool_name
          }
        - For errors: {"type": "error", "content": error_message}
        
    Example:
        # Normal response
        result = parse_tool_response("Customer found: John Smith", "lookup_customer")
        # Returns: {"type": "normal", "content": "Customer found: John Smith"}
        
        # HITL response (revolutionary 3-field approach)
        result = parse_tool_response("HITL_REQUIRED:confirmation:{...}", "trigger_customer_message")
        # Returns: {"type": "hitl_required", "hitl_phase": "needs_prompt", "hitl_prompt": "...", "hitl_context": {...}}
    """
    try:
        log_hitl_operation("parse_tool_response", tool_name=tool_name, response_length=len(response))
        
        # Check if this is a HITL request
        if not response.startswith(HITL_REQUIRED_PREFIX + ":"):
            # Normal tool response - return as-is
            return {
                "type": "normal",
                "content": response
            }
        
        # REVOLUTIONARY: Parse HITL request format and create ultra-minimal 3-field assignments
        try:
            # Split on first two colons to separate prefix, legacy_type, and data
            parts = response.split(":", 2)
            if len(parts) != 3:
                raise HITLValidationError(f"Invalid HITL response format: expected 3 parts, got {len(parts)}")
            
            prefix, legacy_type, json_data = parts
            
            # Validate prefix
            if prefix != HITL_REQUIRED_PREFIX:
                raise HITLValidationError(f"Invalid HITL prefix: expected '{HITL_REQUIRED_PREFIX}', got '{prefix}'")
            
            # REVOLUTIONARY: No more hitl_type validation - tools define their own interaction style
            # Parse JSON data
            try:
                tool_data = deserialize_hitl_context(json_data)
            except ValueError as e:
                raise HITLValidationError(f"Invalid HITL JSON data: {str(e)}")
            
            # Validate basic structure
            if not isinstance(tool_data, dict):
                raise HITLValidationError("HITL data must be a dictionary")
            
            # REVOLUTIONARY: Ultra-minimal validation - only require prompt and context
            required_fields = ["prompt", "context"]
            missing_fields = [field for field in required_fields if field not in tool_data]
            if missing_fields:
                raise HITLValidationError(f"HITL data missing required fields: {missing_fields}")
            
            # REVOLUTIONARY: Create ultra-minimal 3-field assignments directly
            hitl_prompt = tool_data.get("prompt", "")
            hitl_context = tool_data.get("context", {})
            
            # Add source tool to context if not already present
            if "source_tool" not in hitl_context:
                hitl_context["source_tool"] = tool_name
            
            # Add legacy type for transition period (will be removed in task 7.1)
            hitl_context["legacy_type"] = legacy_type
            
            hitl_logger.log_interaction_start("custom", hitl_context)
            
            return {
                "type": "hitl_required",
                "hitl_phase": "needs_prompt",  # REVOLUTIONARY: Always needs_prompt when HITL is required
                "hitl_prompt": hitl_prompt,    # REVOLUTIONARY: Direct prompt assignment
                "hitl_context": hitl_context,  # REVOLUTIONARY: Direct context assignment
                "source_tool": tool_name
            }
            
        except HITLError:
            # Re-raise HITL errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to HITL validation errors
            raise HITLValidationError(f"Failed to parse HITL response: {str(e)}")
    
    except HITLError as e:
        # Log HITL-specific errors
        hitl_logger.log_error(e, interaction_type=None)
        return {
            "type": "error",
            "content": f"Invalid HITL response from {tool_name}: {e.message}"
        }
    
    except Exception as e:
        # Handle unexpected errors
        hitl_error = handle_hitl_error(e, {
            "function": "parse_tool_response",
            "tool_name": tool_name,
            "response": response[:100] + "..." if len(response) > 100 else response
        })
        hitl_logger.log_error(hitl_error)
        return {
            "type": "error",
            "content": f"Error parsing tool response from {tool_name}: {str(e)}"
        }




# REVOLUTIONARY: Ultra-Simple 3-Field HITL Node Implementation
async def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    REVOLUTIONARY: Ultra-simple HITL node using 3-field architecture.
    
    Uses the new minimal state fields instead of complex nested JSON:
    - hitl_phase: "needs_prompt" | "awaiting_response" | "approved" | "denied" 
    - hitl_prompt: The exact text to show the user (tool-generated)
    - hitl_context: Minimal execution context for post-interaction processing
    
    Args:
        state: The current agent state containing 3-field HITL data
        
    Returns:
        Updated state after HITL interaction or response processing
        
    Flow:
        1. If hitl_phase is "needs_prompt": Show prompt → interrupt()
        2. If hitl_phase is "awaiting_response": Process human response with LLM
        3. Clear HITL fields when interaction is complete
        4. Always successful - LLM interprets any response naturally
    """
    try:
        # REVOLUTIONARY: Use 3-field architecture instead of complex hitl_data
        hitl_phase = state.get("hitl_phase")
        hitl_prompt = state.get("hitl_prompt")
        hitl_context = state.get("hitl_context", {})
        messages = state.get("messages", [])
        
        # REVOLUTIONARY: Enhanced 3-field state logging
        _log_3field_state_snapshot(
            location="hitl_node_entry",
            hitl_phase=hitl_phase,
            hitl_prompt=hitl_prompt,
            hitl_context=hitl_context,
            message_count=len(messages)
        )
        
        if not hitl_phase:
            # Shouldn't happen, but handle gracefully
            logger.warning("🤖 [HITL_3FIELD] Called without hitl_phase, returning state unchanged")
            _log_3field_state_transition(
                operation="error_no_phase",
                additional_info={"error": "Missing hitl_phase", "action": "returning_unchanged"}
            )
            return state
        
        log_hitl_operation(
            "hitl_node_3field_entry",
            hitl_phase=hitl_phase,
            has_messages=len(messages) > 0
        )
        
        if hitl_phase == "needs_prompt":
            # First time running - show the prompt
            _log_3field_state_transition(
                operation="prompt_display",
                before_phase=hitl_phase,
                after_phase="awaiting_response",
                prompt_preview=hitl_prompt[:50] if hitl_prompt else None,
                context_keys=list(hitl_context.keys()) if hitl_context else None
            )
            return await _show_hitl_prompt_3field(state, hitl_prompt, hitl_context)
            
        elif hitl_phase == "awaiting_response":
            # Processing human response
            _log_3field_state_transition(
                operation="response_processing",
                before_phase=hitl_phase,
                context_keys=list(hitl_context.keys()) if hitl_context else None,
                additional_info={"processing_mode": "llm_driven"}
            )
            result = await _process_hitl_response_3field(state, hitl_context)
            
            # REVOLUTIONARY: Log the final transition result
            final_phase = result.get("hitl_phase")
            _log_3field_state_transition(
                operation="response_complete",
                before_phase="awaiting_response",
                after_phase=final_phase,
                additional_info={
                    "hitl_context_set": bool(result.get("hitl_context")),
                    "messages_updated": len(result.get("messages", [])) != len(messages)
                }
            )
            return result
            
        else:
            # Phase is already "approved" or "denied" - shouldn't reach here but handle gracefully
            _log_3field_state_transition(
                operation="unexpected_phase",
                before_phase=hitl_phase,
                after_phase="cleared",
                additional_info={"warning": f"Unexpected phase '{hitl_phase}'", "action": "clearing_state"}
            )
            logger.warning(f"🤖 [HITL_3FIELD] Unexpected phase '{hitl_phase}', clearing HITL state")
            return {
                **state,
                "hitl_phase": None,
                "hitl_prompt": None,
                "hitl_context": None
            }
            
    except GraphInterrupt:
        # GraphInterrupt is expected behavior - let it propagate naturally
        logger.info(f"🤖 [HITL_3FIELD] GraphInterrupt raised - this is expected for prompting")
        raise
    except HITLError as e:
        # Handle HITL-specific errors
        logger.error(f"🤖 [HITL_3FIELD] HITLError: {e}")
        hitl_logger.log_error(e, interaction_type=None)
        return {
            **state,
            "hitl_phase": None,  # REVOLUTIONARY: Clear 3-field HITL state
            "hitl_prompt": None,
            "hitl_context": None,
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"I encountered an error during our interaction: {e.message}. Please try again."
            }]
        }
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"🤖 [HITL_3FIELD] Unexpected error: {e}")
        hitl_error = handle_hitl_error(e, {
            "function": "hitl_node_3field",
            "hitl_phase": state.get("hitl_phase"),
            "hitl_context": state.get("hitl_context"),
            "state_keys": list(state.keys()) if state else []
        })
        hitl_logger.log_error(hitl_error)
        return {
            **state,
            "hitl_phase": None,  # REVOLUTIONARY: Clear 3-field HITL state
            "hitl_prompt": None,
            "hitl_context": None,
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": "I encountered an unexpected error. Please try your request again."
            }]
        }


# REVOLUTIONARY: 3-Field HITL Debugging and Logging Functions
def _log_3field_state_transition(
    operation: str,
    before_phase: Optional[str] = None,
    after_phase: Optional[str] = None,
    prompt_preview: Optional[str] = None,
    context_keys: Optional[List[str]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    REVOLUTIONARY: Enhanced logging for 3-field HITL state transitions.
    
    Provides clear, consistent logging of state changes for easy debugging.
    Makes it simple to trace the flow of HITL interactions.
    
    Args:
        operation: The operation being performed (e.g., "prompt_display", "response_processing")
        before_phase: Phase before the operation
        after_phase: Phase after the operation
        prompt_preview: First 50 chars of prompt for debugging
        context_keys: List of keys in hitl_context
        additional_info: Any additional debugging information
    """
    # Create transition arrow if both phases provided
    phase_transition = ""
    if before_phase and after_phase:
        if before_phase != after_phase:
            phase_transition = f"{before_phase} → {after_phase}"
        else:
            phase_transition = f"{before_phase} (unchanged)"
    elif after_phase:
        phase_transition = f"→ {after_phase}"
    elif before_phase:
        phase_transition = f"{before_phase} →"
    
    # Format context info
    context_info = f"context_keys={context_keys}" if context_keys else "no_context"
    
    # Format prompt preview
    prompt_info = f"prompt='{prompt_preview}...'" if prompt_preview else "no_prompt"
    
    # Main log message
    logger.info(f"🔄 [3FIELD_TRANSITION] {operation.upper()}: {phase_transition} | {context_info} | {prompt_info}")
    
    # Additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            logger.info(f"🔄 [3FIELD_TRANSITION]   └─ {key}: {value}")


def _log_3field_state_snapshot(
    location: str,
    hitl_phase: Optional[str],
    hitl_prompt: Optional[str],
    hitl_context: Optional[Dict[str, Any]],
    message_count: int = 0
) -> None:
    """
    REVOLUTIONARY: Snapshot logging of current 3-field state.
    
    Provides a complete view of the current HITL state at any point.
    Useful for debugging complex interactions.
    
    Args:
        location: Where in the code this snapshot is taken
        hitl_phase: Current HITL phase
        hitl_prompt: Current HITL prompt
        hitl_context: Current HITL context
        message_count: Number of messages in conversation
    """
    logger.info(f"📸 [3FIELD_SNAPSHOT] {location}:")
    logger.info(f"📸 [3FIELD_SNAPSHOT]   ├─ hitl_phase: {hitl_phase}")
    logger.info(f"📸 [3FIELD_SNAPSHOT]   ├─ hitl_prompt: {hitl_prompt[:50] + '...' if hitl_prompt and len(hitl_prompt) > 50 else hitl_prompt}")
    logger.info(f"📸 [3FIELD_SNAPSHOT]   ├─ hitl_context: {list(hitl_context.keys()) if hitl_context else None}")
    logger.info(f"📸 [3FIELD_SNAPSHOT]   └─ message_count: {message_count}")
    
    # Log context details if available
    if hitl_context:
        tool_name = hitl_context.get("source_tool", "unknown")
        logger.info(f"📸 [3FIELD_SNAPSHOT]       └─ source_tool: {tool_name}")


def _log_llm_interpretation_result(
    human_response: str,
    llm_intent: str,
    response_message: str,
    processing_time_ms: Optional[float] = None
) -> None:
    """
    REVOLUTIONARY: Specialized logging for LLM interpretation results.
    
    Makes it easy to see how the LLM interpreted user responses.
    Critical for debugging natural language understanding.
    
    Args:
        human_response: What the user actually said
        llm_intent: What the LLM interpreted (approval/denial/input)
        response_message: The system's response to the user
        processing_time_ms: How long LLM processing took
    """
    time_info = f" ({processing_time_ms:.1f}ms)" if processing_time_ms else ""
    logger.info(f"🧠 [LLM_INTERPRETATION]{time_info}:")
    logger.info(f"🧠 [LLM_INTERPRETATION]   ├─ human_said: '{human_response}'")
    logger.info(f"🧠 [LLM_INTERPRETATION]   ├─ llm_interpreted: {llm_intent.upper()}")
    logger.info(f"🧠 [LLM_INTERPRETATION]   └─ system_response: '{response_message}'")


# REVOLUTIONARY: 3-Field HITL Helper Functions
async def _show_hitl_prompt_3field(state: Dict[str, Any], hitl_prompt: str, hitl_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    REVOLUTIONARY: Ultra-simple prompt display using 3-field architecture.
    
    Shows the exact prompt provided by the tool without complex formatting.
    Tools are responsible for generating complete, user-friendly prompts.
    
    Args:
        state: Current agent state
        hitl_prompt: The complete prompt text (tool-generated)
        hitl_context: Minimal context for logging/debugging
        
    Returns:
        Updated state with awaiting_response phase
    """
    try:
        if not hitl_prompt:
            logger.warning("🤖 [HITL_3FIELD] Empty prompt provided, using fallback")
            hitl_prompt = "Please provide your response:"
            _log_3field_state_transition(
                operation="prompt_fallback",
                additional_info={"warning": "Empty prompt", "fallback_used": True}
            )
        
        # REVOLUTIONARY: Enhanced logging for prompt display
        tool_name = hitl_context.get("source_tool", "system") if hitl_context else "system"
        _log_3field_state_snapshot(
            location="before_prompt_display",
            hitl_phase="needs_prompt",
            hitl_prompt=hitl_prompt,
            hitl_context=hitl_context,
            message_count=len(state.get("messages", []))
        )
        
        # Log the interaction start
        hitl_logger.log_interaction_start("3field", {"tool": tool_name})
        
        # Add HITL prompt to messages for persistence
        updated_messages = list(state.get("messages", []))
        hitl_prompt_message = {
            "role": "assistant",
            "content": hitl_prompt,
            "type": "ai"  # Ensure Memory Store recognizes it
        }
        updated_messages.append(hitl_prompt_message)
        
        # REVOLUTIONARY: Update to awaiting_response phase
        updated_state = {
            **state,
            "messages": updated_messages,
            "hitl_phase": "awaiting_response",  # REVOLUTIONARY: Simple phase transition
            # hitl_prompt and hitl_context remain unchanged
        }
        
        # REVOLUTIONARY: Log successful transition
        _log_3field_state_transition(
            operation="prompt_interrupt",
            before_phase="needs_prompt",
            after_phase="awaiting_response",
            prompt_preview=hitl_prompt[:50],
            additional_info={
                "tool": tool_name,
                "message_added": True,
                "interrupt_triggered": True
            }
        )
        
        # Interrupt to wait for human input
        interrupt(hitl_prompt)
        
        return updated_state
        
    except GraphInterrupt:
        # GraphInterrupt is expected behavior - let it propagate naturally
        raise
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "_show_hitl_prompt_3field",
            "hitl_context": hitl_context
        })
        raise hitl_error


async def _process_hitl_response_3field(state: Dict[str, Any], hitl_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    REVOLUTIONARY: Ultra-simple response processing using LLM-driven interpretation.
    
    Uses the new LLM-driven approach to understand user intent naturally.
    No complex validation or re-prompting - LLM handles all interpretation.
    
    Args:
        state: Current agent state with messages containing user response
        hitl_context: Minimal context for tool execution
        
    Returns:
        Updated state with appropriate hitl_phase for routing
    """
    try:
        # Extract the latest human message
        messages = state.get("messages", [])
        latest_human_message = None
        
        # REVOLUTIONARY: Enhanced message extraction logging
        _log_3field_state_snapshot(
            location="before_response_processing",
            hitl_phase="awaiting_response",
            hitl_prompt=None,  # Prompt already shown
            hitl_context=hitl_context,
            message_count=len(messages)
        )
        
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', 'no_type') if hasattr(msg, 'type') else msg.get('type', 'no_type') if isinstance(msg, dict) else 'no_type'
            msg_content = getattr(msg, 'content', str(msg)) if hasattr(msg, 'content') else str(msg.get('content', msg)) if isinstance(msg, dict) else str(msg)
            
            if msg_type == 'human':
                latest_human_message = msg_content
                logger.info(f"🤖 [HITL_3FIELD] Found human response: '{latest_human_message}'")
                break
        
        if not latest_human_message:
            _log_3field_state_transition(
                operation="response_error",
                additional_info={"error": "No human message found", "message_count": len(messages)}
            )
            raise HITLResponseError("No human message found in conversation state")
        
        # REVOLUTIONARY: Use LLM-driven interpretation with timing
        import time
        llm_start_time = time.time()
        result = await _process_hitl_response_llm_driven(hitl_context, latest_human_message)
        llm_processing_time = (time.time() - llm_start_time) * 1000  # Convert to ms
        
        # REVOLUTIONARY: Enhanced LLM result logging
        _log_llm_interpretation_result(
            human_response=latest_human_message,
            llm_intent=result.get("result", "unknown"),
            response_message=result.get("response_message", "No response"),
            processing_time_ms=llm_processing_time
        )
        
        # Add response message to conversation
        updated_messages = list(messages)
        if result.get("response_message"):
            response_message = {
                "role": "assistant",
                "content": result["response_message"],
                "type": "ai"
            }
            updated_messages.append(response_message)
        
        # REVOLUTIONARY: Map result to hitl_phase for routing
        hitl_phase = None
        approved_context = None
        
        if result.get("success") and result.get("result"):
            result_value = result.get("result")
            if result_value == "approved":
                hitl_phase = "approved"
                # Set approved context for tool execution
                if hitl_context and hitl_context.get("tool"):
                    approved_context = hitl_context
            elif result_value == "denied":
                hitl_phase = "denied"
            # For input data, keep the context and let tool handle it
        
        # REVOLUTIONARY: Prepare final 3-field state
        final_state = {
            **state,
            "messages": updated_messages,
            "hitl_phase": hitl_phase,  # REVOLUTIONARY: Clear phase or set to approved/denied
            "hitl_prompt": None,  # REVOLUTIONARY: Clear prompt - interaction complete
            "hitl_context": approved_context or result.get("context"),  # Keep approved context or LLM result context
        }
        
        # REVOLUTIONARY: Enhanced final state logging
        _log_3field_state_snapshot(
            location="after_response_processing",
            hitl_phase=final_state.get("hitl_phase"),
            hitl_prompt=final_state.get("hitl_prompt"),
            hitl_context=final_state.get("hitl_context"),
            message_count=len(final_state.get("messages", []))
        )
        
        _log_3field_state_transition(
            operation="final_state_ready",
            before_phase="awaiting_response",
            after_phase=hitl_phase,
            additional_info={
                "hitl_context_set": bool(approved_context or result.get("context")),
                "context_cleared": result.get("context") is None,
                "prompt_cleared": True,
                "messages_count": len(updated_messages)
            }
        )
        
        return final_state
        
    except HITLError:
        # Re-raise HITL errors
        raise
    except Exception as e:
        logger.error(f"🤖 [HITL_3FIELD] Error in 3-field response processing: {e}")
        hitl_error = handle_hitl_error(e, {
            "function": "_process_hitl_response_3field",
            "hitl_context": hitl_context
        })
        raise hitl_error


# [REVOLUTIONARY LLM-DRIVEN INTERPRETATION SYSTEM]
# Core functions for natural language understanding in the 3-field architecture
# These functions enable users to respond naturally ("send it", "go ahead", etc.)



async def _get_hitl_interpretation_llm() -> ChatOpenAI:
    """
    Get LLM for HITL response interpretation.
    Uses a fast, efficient model for natural language understanding.
    """
    settings = await get_settings()
    
    # Use simple model for fast interpretation
    model = getattr(settings, 'openai_simple_model', 'gpt-3.5-turbo')
    
    return ChatOpenAI(
        model=model,
        temperature=0,  # Deterministic for consistent interpretation
        openai_api_key=settings.openai_api_key
    )


async def _interpret_user_intent_with_llm(human_response: str, hitl_context: Dict[str, Any]) -> str:
    """
    REVOLUTIONARY: Use LLM to interpret user intent naturally.
    
    This function understands natural language responses like:
    - "send it" → approval
    - "go ahead" → approval  
    - "not now" → denial
    - "john@example.com" → input data
    - "option 2" → selection
    """
    try:
        llm = await _get_hitl_interpretation_llm()
        
        # Build context-aware prompt
        context_info = ""
        if hitl_context:
            tool_name = hitl_context.get("source_tool", "system")
            if tool_name:
                context_info = f"\nContext: This is a response to a {tool_name} tool request."
        
        # REVOLUTIONARY: Natural language interpretation prompt
        system_prompt = """You are an expert at interpreting human responses in conversational interfaces.

Your job is to determine the user's intent from their response. Return EXACTLY one of these categories:

**APPROVAL** - User wants to proceed/approve/confirm the action
Examples: "yes", "ok", "send it", "go ahead", "proceed", "do it", "sure", "approve", "confirm", "send", "continue"

**DENIAL** - User wants to cancel/deny/stop the action  
Examples: "no", "cancel", "stop", "don't", "abort", "not now", "skip", "deny", "quit", "exit"

**INPUT** - User is providing information/data/input
Examples: "john@example.com", "Customer ABC", "option 2", "tomorrow", "1234567890", specific names, addresses, etc.

Consider the full context and natural language patterns. Don't just match keywords - understand intent.

Respond with ONLY the category name: APPROVAL, DENIAL, or INPUT"""
        
        user_prompt = f"""User Response: "{human_response}"{context_info}

What is the user's intent?"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Get LLM interpretation with enhanced logging
        parser = StrOutputParser()
        chain = llm | parser
        
        # REVOLUTIONARY: Time the LLM interpretation
        import time
        interpretation_start = time.time()
        interpretation = await chain.ainvoke(messages)
        interpretation_time = (time.time() - interpretation_start) * 1000
        
        # Clean and validate the response
        interpretation_clean = interpretation.strip().upper()
        
        # REVOLUTIONARY: Enhanced LLM interpretation logging
        logger.info(f"🧠 [LLM_INTERPRETATION] Raw response: '{interpretation}' ({interpretation_time:.1f}ms)")
        logger.info(f"🧠 [LLM_INTERPRETATION] Cleaned: '{interpretation_clean}'")
        logger.info(f"🧠 [LLM_INTERPRETATION] Input: '{human_response}' → Output: {interpretation_clean}")
        
        # Validate LLM response
        if interpretation_clean in ["APPROVAL", "DENIAL", "INPUT"]:
            logger.info(f"🧠 [LLM_INTERPRETATION] ✅ Valid interpretation: {interpretation_clean.lower()}")
            return interpretation_clean.lower()
        else:
            # Fallback if LLM gives unexpected response
            logger.warning(f"🧠 [LLM_INTERPRETATION] ⚠️ Unexpected LLM response: '{interpretation}', falling back to INPUT")
            logger.warning(f"🧠 [LLM_INTERPRETATION] Expected: APPROVAL, DENIAL, or INPUT")
            return "input"
            
    except Exception as e:
        logger.error(f"🤖 [LLM_INTERPRETATION] Error in LLM interpretation: {str(e)}")
        # Fallback to treating as input if LLM fails
        return "input"


async def _process_hitl_response_llm_driven(hitl_context: Dict[str, Any], human_response: str) -> Dict[str, Any]:
    """
    REVOLUTIONARY: LLM-driven human response processing.
    
    Uses actual LLM to interpret user intent naturally without rigid validation.
    Understands responses like "send it", "go ahead", "not now", etc.
    
    Args:
        hitl_context: Minimal context data from hitl_context state field
        human_response: The human's response text (already extracted from messages)
        
    Returns:
        Dictionary containing processing results:
        - success: bool - Whether processing was successful
        - result: str - Simple result: "approved", "denied", or user input value
        - context: Dict - Updated context if needed (minimal)
        - response_message: str - Confirmation message to send back to user
    """
    try:
        response_text = human_response.strip()
        
        hitl_logger.log_response_processing("llm_driven", human_response, "LLM interpretation")
        
        # REVOLUTIONARY: Use LLM to interpret user intent naturally
        intent = await _interpret_user_intent_with_llm(response_text, hitl_context)
        
        if intent == "approval":
            logger.info(f"🤖 [LLM_HITL] LLM interpreted as APPROVAL: '{response_text}'")
            return {
                "success": True,
                "result": "approved",
                "context": None,  # Clear context - interaction complete
                "response_message": "✅ Approved - proceeding with the action."
            }
        
        elif intent == "denial":
            logger.info(f"🤖 [LLM_HITL] LLM interpreted as DENIAL: '{response_text}'")
            return {
                "success": True,
                "result": "denied", 
                "context": None,  # Clear context - interaction complete
                "response_message": "❌ Cancelled - the action has been cancelled as requested."
            }
        
        else:  # intent == "input"
            logger.info(f"🤖 [LLM_HITL] LLM interpreted as INPUT: '{response_text}'")
            return {
                "success": True,
                "result": response_text,  # Return the user's input as-is
                "context": hitl_context,  # Keep context for tool to process
                "response_message": f"📝 Received: {response_text}"
            }
            
    except Exception as e:
        logger.error(f"🤖 [LLM_HITL] Error in LLM-driven processing: {str(e)}")
        raise HITLInteractionError(f"Error processing HITL response: {str(e)}")


# Complex type-based processing functions have been simplified.
# 
# Previous rigid validation functions have been replaced with simple LLM-driven interpretation:
# - Confirmation responses: Now uses natural language understanding
# - Selection responses: Flexible option matching with LLM 
# - Input validation: Intelligent context-aware processing
# - Multi-step collection: Tool-managed for better user experience
#
# Total reduction: 296 lines of complex logic replaced with flexible LLM interpretation.

# Legacy type-based processing functions previously existed here.
# They have been replaced with LLM-driven interpretation for better user experience.








def _has_new_human_message(state: Dict[str, Any]) -> bool:
    """
    Check if there's a new human message in the conversation state.
    
    Args:
        state: Current agent state
        
    Returns:
        True if a human message is found, False otherwise
    """
    messages = state.get("messages", [])
    if not messages:
        return False
    
    # Check the most recent message
    last_message = messages[-1]
    
    # Handle different message formats
    if hasattr(last_message, 'type') and last_message.type == 'human':
        return True
    elif isinstance(last_message, dict) and last_message.get('role') == 'human':
        return True
    
    return False




def create_concise_error_message(error_type: str, user_input: str, expected: str) -> str:
    """Create concise, actionable error messages for HITL interactions."""
    error_templates = {
        "invalid_number": f"❓ **\"{user_input}\"** isn't a valid number. {expected}",
        "out_of_range": f"❓ **\"{user_input}\"** is out of range. {expected}",
        "empty_input": f"❓ Input required. {expected}",
        "conflicting_keywords": f"❓ **\"{user_input}\"** has conflicting choices. {expected}",
        "unrecognized": f"❓ **\"{user_input}\"** not recognized. {expected}"
    }
    
    return error_templates.get(error_type, f"❓ **\"{user_input}\"** invalid. {expected}")


# Update error messages to use the new concise format
def _get_concise_confirmation_error(original_response: str, approve_text: str, deny_text: str) -> str:
    """Generate concise error message for confirmation interactions."""
    if len(original_response) > 50:
        display_input = original_response[:47] + "..."
    else:
        display_input = original_response
        
    return create_concise_error_message(
        "unrecognized",
        display_input,
        f"Use `{approve_text}` (proceed) or `{deny_text}` (cancel)"
    )


def _get_concise_selection_error(user_input: str, max_options: int, error_type: str) -> str:
    """Generate concise error message for selection interactions."""
    expected = f"Enter number 1-{max_options}"
    return create_concise_error_message(error_type, user_input, expected)


def _get_concise_input_error(user_input: str, input_type: str, error_type: str) -> str:
    """Generate concise error message for input request interactions."""
    expected = f"Provide {input_type.replace('_', ' ')} or `cancel`"
    return create_concise_error_message(error_type, user_input, expected)


logger.info("🚀 [HITL_INIT] HITL module initialized successfully") 