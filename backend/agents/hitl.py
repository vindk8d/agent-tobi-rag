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
from langchain_core.messages import BaseMessage

# Configure logging for HITL module
logger = logging.getLogger(__name__)

# LangGraph interrupt functionality for HITL interactions
try:
    from langgraph.types import interrupt
    from langgraph.errors import GraphInterrupt
    logger.info("LangGraph interrupt functionality imported successfully")
except ImportError:
    # Fallback for development/testing - create a mock interrupt function and exception
    logger.warning("LangGraph not available, using mock interrupt functionality")
    def interrupt(message: str):
        logger.info(f"[MOCK_INTERRUPT] {message}")
        return f"[MOCK_INTERRUPT] {message}"
    
    class GraphInterrupt(Exception):
        """Mock GraphInterrupt for development/testing"""
        pass

# Type definitions for HITL interactions
HITLInteractionType = Literal[
    "confirmation",           # Yes/No confirmation
    "selection",             # Choose from options
    "input_request",         # Free text input
    "multi_step_input"       # Multiple inputs needed
]

# Module-level constants
HITL_REQUIRED_PREFIX = "HITL_REQUIRED"
DEFAULT_APPROVE_TEXT = "approve"
DEFAULT_DENY_TEXT = "deny"
DEFAULT_CANCEL_TEXT = "cancel"


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


def serialize_hitl_data(data: Dict[str, Any]) -> str:
    """
    Serialize HITL data using the custom encoder.
    
    Args:
        data: Dictionary containing HITL data to serialize
        
    Returns:
        JSON string representation of the data
        
    Raises:
        ValueError: If data cannot be serialized
    """
    try:
        return json.dumps(data, cls=HITLJSONEncoder)
    except Exception as e:
        logger.error(f"Failed to serialize HITL data: {e}")
        raise ValueError(f"Unable to serialize HITL data: {str(e)}")


def deserialize_hitl_data(json_str: str) -> Dict[str, Any]:
    """
    Deserialize HITL data from JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Dictionary containing deserialized HITL data
        
    Raises:
        ValueError: If JSON string cannot be parsed
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to deserialize HITL data: {e}")
        raise ValueError(f"Unable to deserialize HITL data: {str(e)}")


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


# HITLRequest Standardization System
class HITLRequest:
    """
    Standardized HITL request formatter that creates consistent response formats
    for tools to trigger different types of human interactions.
    
    All methods return standardized strings in the format:
    "HITL_REQUIRED:{interaction_type}:{json_data}"
    
    This allows the agent node to parse tool responses and automatically
    route to the HITL node when human interaction is needed.
    """
    
    @staticmethod
    def confirmation(
        prompt: str, 
        context: Dict[str, Any],
        approve_text: str = DEFAULT_APPROVE_TEXT,
        deny_text: str = DEFAULT_DENY_TEXT
    ) -> str:
        """
        Create a confirmation interaction request.
        
        Args:
            prompt: The question/message to show the user
            context: Tool-specific context data for post-interaction processing
            approve_text: Text for approval option (default: "approve")
            deny_text: Text for denial option (default: "deny")
            
        Returns:
            Formatted HITL request string
            
        Example:
            HITLRequest.confirmation(
                "Send this message to John?",
                {"tool": "trigger_customer_message", "customer_id": "123"}
            )
        """
        try:
            data = {
                "type": "confirmation",
                "prompt": prompt,
                "options": {
                    "approve": approve_text,
                    "deny": deny_text
                },
                "awaiting_response": False,
                "context": context
            }
            
            json_data = serialize_hitl_data(data)
            return f"{HITL_REQUIRED_PREFIX}:confirmation:{json_data}"
            
        except Exception as e:
            hitl_error = handle_hitl_error(e, {
                "method": "HITLRequest.confirmation",
                "prompt": prompt,
                "context": context
            })
            hitl_logger.log_error(hitl_error, "confirmation")
            raise hitl_error
    
    @staticmethod
    def selection(
        prompt: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any],
        allow_cancel: bool = True
    ) -> str:
        """
        Create a selection interaction request.
        
        Args:
            prompt: The question/instruction to show the user
            options: List of options to choose from. Each option should have at least
                    a 'display' or 'name' key for user-friendly display
            context: Tool-specific context data for post-interaction processing
            allow_cancel: Whether to allow cancellation of the selection
            
        Returns:
            Formatted HITL request string
            
        Example:
            HITLRequest.selection(
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
                "type": "selection",
                "prompt": prompt,
                "options": options,
                "allow_cancel": allow_cancel,
                "awaiting_response": False,
                "context": context
            }
            
            json_data = serialize_hitl_data(data)
            return f"{HITL_REQUIRED_PREFIX}:selection:{json_data}"
            
        except Exception as e:
            hitl_error = handle_hitl_error(e, {
                "method": "HITLRequest.selection",
                "prompt": prompt,
                "options_count": len(options) if options else 0,
                "context": context
            })
            hitl_logger.log_error(hitl_error, "selection")
            raise hitl_error
    
    @staticmethod
    def input_request(
        prompt: str,
        input_type: str,
        context: Dict[str, Any],
        validation_hints: Optional[List[str]] = None
    ) -> str:
        """
        Create an input request interaction.
        
        Args:
            prompt: The question/instruction to show the user
            input_type: Type of input being requested (e.g., "customer_name", "email", "description")
            context: Tool-specific context data for post-interaction processing
            validation_hints: Optional list of validation hints to show the user
            
        Returns:
            Formatted HITL request string
            
        Example:
            HITLRequest.input_request(
                "Please provide the customer's email address:",
                "customer_email",
                {"tool": "customer_lookup", "original_query": "john"},
                ["Must be a valid email format", "Use company domain if business customer"]
            )
        """
        try:
            data = {
                "type": "input_request",
                "prompt": prompt,
                "input_type": input_type,
                "validation_hints": validation_hints or [],
                "awaiting_response": False,
                "context": context
            }
            
            json_data = serialize_hitl_data(data)
            return f"{HITL_REQUIRED_PREFIX}:input_request:{json_data}"
            
        except Exception as e:
            hitl_error = handle_hitl_error(e, {
                "method": "HITLRequest.input_request",
                "prompt": prompt,
                "input_type": input_type,
                "context": context
            })
            hitl_logger.log_error(hitl_error, "input_request")
            raise hitl_error
    
    @staticmethod
    def multi_step_input(
        prompt: str,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        current_step: int = 0
    ) -> str:
        """
        Create a multi-step input interaction request.
        
        Args:
            prompt: The overall instruction/question for the multi-step process
            steps: List of step definitions, each with 'name', 'prompt', and optional 'type'
            context: Tool-specific context data for post-interaction processing
            current_step: Current step index (0-based)
            
        Returns:
            Formatted HITL request string
            
        Example:
            HITLRequest.multi_step_input(
                "Please provide customer details:",
                [
                    {"name": "name", "prompt": "Customer full name:", "type": "text"},
                    {"name": "email", "prompt": "Customer email:", "type": "email"},
                    {"name": "phone", "prompt": "Customer phone:", "type": "phone"}
                ],
                {"tool": "create_customer", "source": "manual_entry"}
            )
        """
        try:
            if not steps:
                raise HITLValidationError("Multi-step input must have at least one step")
            
            if current_step < 0 or current_step >= len(steps):
                raise HITLValidationError(f"Invalid current_step {current_step} for {len(steps)} steps")
            
            data = {
                "type": "multi_step_input",
                "prompt": prompt,
                "steps": steps,
                "current_step": current_step,
                "total_steps": len(steps),
                "awaiting_response": False,
                "context": context
            }
            
            json_data = serialize_hitl_data(data)
            return f"{HITL_REQUIRED_PREFIX}:multi_step_input:{json_data}"
            
        except Exception as e:
            hitl_error = handle_hitl_error(e, {
                "method": "HITLRequest.multi_step_input",
                "prompt": prompt,
                "steps_count": len(steps) if steps else 0,
                "current_step": current_step,
                "context": context
            })
            hitl_logger.log_error(hitl_error, "multi_step_input")
            raise hitl_error


def parse_tool_response(response: str, tool_name: str) -> Dict[str, Any]:
    """
    Parse tool responses and extract HITL requirements.
    
    Analyzes tool response strings to detect HITL interaction requests
    and extracts the interaction data for processing by the agent node.
    
    Args:
        response: The tool response string to parse
        tool_name: Name of the tool that generated the response
        
    Returns:
        Dictionary with parsed response information:
        - For normal responses: {"type": "normal", "content": response}
        - For HITL responses: {
            "type": "hitl_required",
            "hitl_type": "confirmation|selection|input_request|multi_step_input",
            "hitl_data": {...},
            "source_tool": tool_name
          }
        - For errors: {"type": "error", "content": error_message}
        
    Example:
        # Normal response
        result = parse_tool_response("Customer found: John Smith", "lookup_customer")
        # Returns: {"type": "normal", "content": "Customer found: John Smith"}
        
        # HITL response  
        result = parse_tool_response("HITL_REQUIRED:confirmation:{...}", "trigger_customer_message")
        # Returns: {"type": "hitl_required", "hitl_type": "confirmation", ...}
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
        
        # Parse HITL request format: HITL_REQUIRED:{type}:{json_data}
        try:
            # Split on first two colons to separate prefix, type, and data
            parts = response.split(":", 2)
            if len(parts) != 3:
                raise HITLValidationError(f"Invalid HITL response format: expected 3 parts, got {len(parts)}")
            
            prefix, hitl_type, json_data = parts
            
            # Validate prefix
            if prefix != HITL_REQUIRED_PREFIX:
                raise HITLValidationError(f"Invalid HITL prefix: expected '{HITL_REQUIRED_PREFIX}', got '{prefix}'")
            
            # Validate interaction type
            valid_types = ["confirmation", "selection", "input_request", "multi_step_input"]
            if hitl_type not in valid_types:
                raise HITLValidationError(f"Invalid HITL type '{hitl_type}', must be one of: {valid_types}")
            
            # Parse JSON data
            try:
                hitl_data = deserialize_hitl_data(json_data)
            except ValueError as e:
                raise HITLValidationError(f"Invalid HITL JSON data: {str(e)}")
            
            # Validate that the hitl_data contains required fields
            if not isinstance(hitl_data, dict):
                raise HITLValidationError("HITL data must be a dictionary")
            
            required_fields = ["type", "prompt", "context"]
            missing_fields = [field for field in required_fields if field not in hitl_data]
            if missing_fields:
                raise HITLValidationError(f"HITL data missing required fields: {missing_fields}")
            
            # Ensure the type in the data matches the type in the prefix
            if hitl_data.get("type") != hitl_type:
                raise HITLValidationError(
                    f"HITL type mismatch: prefix has '{hitl_type}', data has '{hitl_data.get('type')}'"
                )
            
            # Add source tool to context if not already present
            if "source_tool" not in hitl_data.get("context", {}):
                hitl_data.setdefault("context", {})["source_tool"] = tool_name
            
            hitl_logger.log_interaction_start(hitl_type, hitl_data.get("context", {}))
            
            return {
                "type": "hitl_required",
                "hitl_type": hitl_type,
                "hitl_data": hitl_data,
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


def validate_hitl_data_structure(hitl_data: Dict[str, Any], interaction_type: str) -> bool:
    """
    Validate that HITL data has the correct structure for the given interaction type.
    
    Args:
        hitl_data: The HITL data dictionary to validate
        interaction_type: The type of interaction to validate against
        
    Returns:
        True if valid, raises HITLValidationError if invalid
        
    Raises:
        HITLValidationError: If the data structure is invalid
    """
    try:
        # Common required fields for all interaction types
        common_required = ["type", "prompt", "context"]
        for field in common_required:
            if field not in hitl_data:
                raise HITLValidationError(f"Missing required field '{field}' in HITL data")
        
        # Type-specific validation
        if interaction_type == "confirmation":
            if "options" not in hitl_data:
                raise HITLValidationError("Confirmation HITL data must include 'options' field")
            options = hitl_data["options"]
            if not isinstance(options, dict) or "approve" not in options or "deny" not in options:
                raise HITLValidationError("Confirmation options must be a dict with 'approve' and 'deny' keys")
        
        elif interaction_type == "selection":
            if "options" not in hitl_data:
                raise HITLValidationError("Selection HITL data must include 'options' field")
            options = hitl_data["options"]
            if not isinstance(options, list) or len(options) == 0:
                raise HITLValidationError("Selection options must be a non-empty list")
        
        elif interaction_type == "input_request":
            if "input_type" not in hitl_data:
                raise HITLValidationError("Input request HITL data must include 'input_type' field")
        
        elif interaction_type == "multi_step_input":
            required_multi_step = ["steps", "current_step", "total_steps"]
            for field in required_multi_step:
                if field not in hitl_data:
                    raise HITLValidationError(f"Multi-step input HITL data must include '{field}' field")
            
            steps = hitl_data["steps"]
            if not isinstance(steps, list) or len(steps) == 0:
                raise HITLValidationError("Multi-step input steps must be a non-empty list")
            
            current_step = hitl_data["current_step"]
            total_steps = hitl_data["total_steps"]
            if not (0 <= current_step < total_steps):
                raise HITLValidationError(f"Invalid current_step {current_step} for {total_steps} total steps")
        
        return True
        
    except Exception as e:
        if isinstance(e, HITLValidationError):
            raise
        raise HITLValidationError(f"Error validating HITL data structure: {str(e)}")


# Single-Node HITL Handler Implementation
async def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    General-purpose HITL node that handles all human interaction types.
    
    This node determines whether to show a prompt or process a response based on
    the hitl_data.awaiting_response flag. It uses the interrupt() mechanism for
    human interaction and handles all interaction types through a unified interface.
    
    Args:
        state: The current agent state containing hitl_data
        
    Returns:
        Updated state after HITL interaction or response processing
        
    Flow:
        1. If hitl_data.awaiting_response is False: Show prompt → interrupt()
        2. If hitl_data.awaiting_response is True: Process human response
        3. Clear hitl_data when interaction is complete
        4. Re-prompt on invalid responses instead of failing
    """
    try:
        hitl_data = state.get("hitl_data")
        
        logger.info(f"🔍 [HITL_DEBUG] HITL node entry - hitl_data: {hitl_data}")
        
        if not hitl_data:
            # Shouldn't happen, but handle gracefully
            logger.warning("🔍 [HITL_DEBUG] Called without hitl_data, returning state unchanged")
            return state
        
        interaction_type = hitl_data.get("type")
        awaiting_response = hitl_data.get("awaiting_response", False)
        
        logger.info(f"🔍 [HITL_DEBUG] Interaction type: {interaction_type}, awaiting_response: {awaiting_response}")
        
        # Check if we have messages in state
        messages = state.get("messages", [])
        logger.info(f"🔍 [HITL_DEBUG] State has {len(messages)} messages")
        if messages:
            last_msg = messages[-1]
            msg_content = getattr(last_msg, 'content', str(last_msg))[:100] if hasattr(last_msg, 'content') else str(last_msg)[:100]
            logger.info(f"🔍 [HITL_DEBUG] Last message: {msg_content}...")
        
        log_hitl_operation(
            "hitl_node_entry",
            interaction_type=interaction_type,
            awaiting_response=awaiting_response,
            has_messages=len(state.get("messages", [])) > 0
        )
        
        if not awaiting_response:
            # First time running - show the prompt
            logger.info(f"🔍 [HITL_DEBUG] Showing HITL prompt for {interaction_type}")
            return await _show_hitl_prompt(state, hitl_data)
        else:
            # Processing human response
            logger.info(f"🔍 [HITL_DEBUG] Processing HITL response for {interaction_type}")
            result = await _process_hitl_response_node(state, hitl_data)
            logger.info(f"🔍 [HITL_DEBUG] HITL response processed. Result keys: {list(result.keys()) if result else 'None'}")
            logger.info(f"🔍 [HITL_DEBUG] hitl_data after processing: {result.get('hitl_data')}")
            logger.info(f"🔍 [HITL_DEBUG] hitl_result: {result.get('hitl_result')}")
            return result
            
    except GraphInterrupt:
        # GraphInterrupt is expected behavior - let it propagate naturally
        logger.info(f"🔍 [HITL_DEBUG] GraphInterrupt raised - this is expected for prompting")
        raise
    except HITLError as e:
        # Handle HITL-specific errors
        logger.error(f"🔍 [HITL_DEBUG] HITLError: {e}")
        hitl_logger.log_error(e, interaction_type=hitl_data.get("type") if hitl_data else None)
        return {
            **state,
            "hitl_data": None,  # Clear HITL state
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"I encountered an error during our interaction: {e.message}. Please try again."
            }]
        }
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"🔍 [HITL_DEBUG] Unexpected error: {e}")
        hitl_error = handle_hitl_error(e, {
            "function": "hitl_node",
            "hitl_data": hitl_data,
            "state_keys": list(state.keys()) if state else []
        })
        hitl_logger.log_error(hitl_error)
        return {
            **state,
            "hitl_data": None,  # Clear HITL state
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": "I encountered an unexpected error. Please try your request again."
            }]
        }


def _format_hitl_prompt(hitl_data: Dict[str, Any]) -> str:
    """
    Format concise, user-friendly prompts for HITL interactions.
    
    New approach: Concise prompts with clear expectations and context.
    """
    try:
        interaction_type = hitl_data.get("type")
        base_prompt = hitl_data.get("prompt", "")
        context = hitl_data.get("context", {})
        
        # Add agent context for why HITL is needed
        tool_name = context.get("tool", "system")
        agent_context = f"🤖 **Agent needs your input for:** `{tool_name}`\n\n"
        
        if interaction_type == "confirmation":
            options = hitl_data.get("options", {})
            approve_text = options.get("approve", DEFAULT_APPROVE_TEXT)
            deny_text = options.get("deny", DEFAULT_DENY_TEXT)
            
            return f"""{agent_context}{base_prompt}

**Options:** `{approve_text}` (proceed) | `{deny_text}` (cancel)"""
            
        elif interaction_type == "selection":
            options = hitl_data.get("options", [])
            allow_cancel = hitl_data.get("allow_cancel", True)
            
            if not options:
                raise HITLValidationError("Selection requires options")
            
            prompt = f"{agent_context}{base_prompt}\n\n**Options:**"
            
            # Concise numbered options
            for i, option in enumerate(options, 1):
                display_text = option.get("display", option.get("name", str(option)))
                prompt += f"\n`{i}` - {display_text}"
            
            if allow_cancel:
                prompt += f"\n`{len(options) + 1}` - Cancel"
            
            prompt += "\n\n**Enter number:**"
            return prompt
            
        elif interaction_type == "input_request":
            input_type = hitl_data.get("input_type", "information")
            validation_hints = hitl_data.get("validation_hints", [])
            
            prompt = f"{agent_context}{base_prompt}\n\n**Input needed:** {input_type.replace('_', ' ').title()}"
            
            if validation_hints:
                prompt += f"\n**Requirements:** {' | '.join(validation_hints)}"
            
            prompt += f"\n\n**Provide input** (or `cancel`):"
            return prompt
            
        elif interaction_type == "multi_step_input":
            steps = hitl_data.get("steps", [])
            current_step = hitl_data.get("current_step", 0)
            total_steps = hitl_data.get("total_steps", len(steps))
            
            if not steps or current_step >= len(steps):
                raise HITLValidationError("Invalid multi-step configuration")
            
            step_info = steps[current_step]
            step_name = step_info.get("name", f"step_{current_step + 1}")
            step_prompt = step_info.get("prompt", "Please provide information:")
            
            # Concise progress indicator
            progress = f"[{current_step + 1}/{total_steps}]"
            
            return f"""{agent_context}{base_prompt}

**Step {progress}:** {step_name.replace('_', ' ').title()}
{step_prompt}

**Provide input** (or `cancel`):**"""
            
        else:
            raise HITLValidationError(f"Unknown interaction type: {interaction_type}")
            
    except HITLError:
        raise
    except Exception as e:
        raise HITLValidationError(f"Error formatting HITL prompt: {str(e)}")


async def _show_hitl_prompt(state: Dict[str, Any], hitl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Show the appropriate prompt for the HITL interaction type.
    
    Args:
        state: Current agent state
        hitl_data: HITL interaction data
        
    Returns:
        Updated state with awaiting_response set to True
    """
    try:
        interaction_type = hitl_data["type"]
        
        # Format the prompt based on interaction type
        prompt = _format_hitl_prompt(hitl_data)
        
        hitl_logger.log_interaction_start(interaction_type, hitl_data.get("context", {}))
        
        # Add HITL prompt to messages so it gets persisted by centralized Memory Store
        updated_messages = list(state.get("messages", []))
        hitl_prompt_message = {
            "role": "assistant",
            "content": prompt,
            "type": "ai"  # Ensure it's recognized as AI message by Memory Store
        }
        updated_messages.append(hitl_prompt_message)
        
        # Update state to indicate we're waiting for response BEFORE interrupting
        updated_hitl_data = {**hitl_data, "awaiting_response": True}
        
        updated_state = {
            **state,
            "messages": updated_messages,  # Include the HITL prompt in messages
            "hitl_data": updated_hitl_data
        }
        
        # Interrupt to wait for human input
        interrupt(prompt)
        
        return updated_state
        
    except GraphInterrupt:
        # GraphInterrupt is expected behavior - let it propagate naturally
        raise
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "_show_hitl_prompt",
            "interaction_type": hitl_data.get("type"),
            "hitl_data": hitl_data
        })
        raise hitl_error


async def _process_hitl_response(hitl_data: Dict[str, Any], human_response: str) -> Dict[str, Any]:
    """
    Process human response for HITL interactions with validation and routing.
    
    This is the core response processing function that handles validation,
    routing, and result preparation for all HITL interaction types.
    
    Args:
        hitl_data: HITL interaction data containing type, options, context, etc.
        human_response: The human's response text (already extracted from messages)
        
    Returns:
        Dictionary containing processing results:
        - success: bool - Whether processing was successful
        - result: Any - The processed result (varies by interaction type)
        - hitl_data: Dict - Updated HITL data (None if complete, updated if continuing)
        - response_message: str - Message to send back to user
        - error_type: str - Type of error if unsuccessful ("validation", "processing", etc.)
        
    Raises:
        HITLValidationError: If response validation fails
        HITLInteractionError: If processing fails
    """
    try:
        interaction_type = hitl_data.get("type")
        if not interaction_type:
            raise HITLValidationError("Missing interaction type in HITL data")
        
        # Clean and normalize the human response
        response_text = human_response.strip().lower()
        
        hitl_logger.log_response_processing(interaction_type, human_response, "starting validation")
        
        # Route to appropriate handler based on interaction type
        if interaction_type == "confirmation":
            return await _process_confirmation_response(hitl_data, response_text, human_response)
            
        elif interaction_type == "selection":
            return await _process_selection_response(hitl_data, response_text, human_response)
            
        elif interaction_type == "input_request":
            return await _process_input_request_response(hitl_data, response_text, human_response)
            
        elif interaction_type == "multi_step_input":
            return await _process_multi_step_response(hitl_data, response_text, human_response)
            
        else:
            raise HITLValidationError(f"Unknown interaction type: {interaction_type}")
            
    except HITLError:
        # Re-raise HITL-specific errors
        raise
    except Exception as e:
        # Convert unexpected errors to HITL processing errors
        raise HITLInteractionError(f"Unexpected error processing HITL response: {str(e)}")


async def _process_confirmation_response(hitl_data: Dict[str, Any], response_text: str, original_response: str) -> Dict[str, Any]:
    """Process confirmation responses (approve/deny)."""
    try:
        logger.info(f"🔍 [HITL_CONFIRMATION] Processing confirmation response: '{response_text}'")
        
        options = hitl_data.get("options", {})
        approve_text = options.get("approve", DEFAULT_APPROVE_TEXT).lower()
        deny_text = options.get("deny", DEFAULT_DENY_TEXT).lower()
        
        logger.info(f"🔍 [HITL_CONFIRMATION] Approval keywords: [{approve_text}, approve, yes, y, ok, confirm, proceed]")
        logger.info(f"🔍 [HITL_CONFIRMATION] Denial keywords: [{deny_text}, no, n, cancel, abort, stop]")
        
        response_text_lower = response_text.lower()
        
        # Check for approval (exact word matching to avoid false positives)
        approval_keywords = [approve_text, "approve", "yes", "y", "ok", "confirm", "proceed"]
        response_words = response_text_lower.split()
        # Strip punctuation from words for better matching
        clean_words = [word.strip('.,!?";:()[]{}') for word in response_words]
        is_approval = any(keyword in clean_words or keyword == response_text_lower.strip() for keyword in approval_keywords)
        
        # Check for denial (exact word matching to avoid false positives)
        denial_keywords = [deny_text, "no", "n", "cancel", "abort", "stop"]
        # Strip punctuation from words for better matching  
        clean_words = [word.strip('.,!?";:()[]{}') for word in response_words]
        is_denial = any(keyword in clean_words or keyword == response_text_lower.strip() for keyword in denial_keywords)
        
        # Handle conflicting keywords (both approval and denial) as invalid
        if is_approval and is_denial:
            logger.warning(f"🔍 [HITL_CONFIRMATION] Conflicting keywords detected in '{original_response}' - re-prompting user")
            contextual_message = f"""❓ I found conflicting keywords in **"{original_response}"**.

**The message contains both:**
• Approval words: {', '.join([kw for kw in approval_keywords if kw in clean_words])}
• Denial words: {', '.join([kw for kw in denial_keywords if kw in clean_words])}

**Please be clear:**
• `{approve_text}` - to approve and proceed with the action?
• `{deny_text}` - to cancel and stop the action?

Please respond with one clear choice."""
            
            return {
                "success": False,
                "result": None,
                "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
                "response_message": contextual_message,
                "error_type": "validation"
            }
        
        if is_approval:
            logger.info(f"🔍 [HITL_CONFIRMATION] APPROVAL detected - clearing HITL data and proceeding")
            result = {
                "success": True,
                "result": "approved",
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": "✅ Approved - proceeding with the action.",
                "error_type": None
            }
            logger.info(f"🔍 [HITL_CONFIRMATION] Returning approval result: {result}")
            return result
        
        if is_denial:
            logger.info(f"🔍 [HITL_CONFIRMATION] DENIAL detected - clearing HITL data and cancelling")
            result = {
                "success": True,
                "result": "denied",
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": "❌ Cancelled - the action has been cancelled as requested.",
                "error_type": None
            }
            logger.info(f"🔍 [HITL_CONFIRMATION] Returning denial result: {result}")
            return result
        
        # Invalid response - re-prompt with concise context
        logger.warning(f"🔍 [HITL_CONFIRMATION] Invalid response '{original_response}' - re-prompting user")
        
        # Create concise error message
        contextual_message = _get_concise_confirmation_error(original_response, approve_text, deny_text)
        
        result = {
            "success": False,
            "result": None,
            "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
            "response_message": contextual_message,
            "error_type": "validation"
        }
        logger.info(f"🔍 [HITL_CONFIRMATION] Returning re-prompt result: {result}")
        return result
            
    except Exception as e:
        logger.error(f"🔍 [HITL_CONFIRMATION] Error processing confirmation response: {str(e)}")
        raise HITLInteractionError(f"Error processing confirmation response: {str(e)}")


async def _process_selection_response(hitl_data: Dict[str, Any], response_text: str, original_response: str) -> Dict[str, Any]:
    """Process selection responses (numbered choices)."""
    try:
        options = hitl_data.get("options", [])
        allow_cancel = hitl_data.get("allow_cancel", True)
        
        if not options:
            raise HITLValidationError("No options available for selection")
        
        # Check for cancellation first
        if allow_cancel and any(keyword in response_text for keyword in ["cancel", "abort", "quit", "exit"]):
            return {
                "success": True,
                "result": "cancelled",
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": "❌ Selection cancelled as requested.",
                "error_type": None
            }
        
        # Try to parse as number
        try:
            choice_num = int(response_text)
            
            # Check if it's a valid option number
            if 1 <= choice_num <= len(options):
                selected_option = options[choice_num - 1]
                return {
                    "success": True,
                    "result": {"index": choice_num - 1, "option": selected_option},
                    "hitl_data": None,  # Clear HITL data - interaction complete
                    "response_message": f"✅ Selected option {choice_num}",
                    "error_type": None
                }
            
            # Check if it's the cancel option number
            elif allow_cancel and choice_num == len(options) + 1:
                return {
                    "success": True,
                    "result": "cancelled",
                    "hitl_data": None,  # Clear HITL data - interaction complete
                    "response_message": "❌ Selection cancelled as requested.",
                    "error_type": None
                }
            
            # Number out of range
            else:
                max_num = len(options) + (1 if allow_cancel else 0)
                contextual_message = f"""❓ I received **"{choice_num}"** but that's not a valid option number.

**Valid options are:**
• Numbers **1 to {max_num}** (you entered {choice_num})

Please enter a number from the valid range above."""
                
                return {
                    "success": False,
                    "result": None,
                    "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
                    "response_message": contextual_message,
                    "error_type": "validation"
                }
                
        except ValueError:
            # Not a valid number
            contextual_message = f"""❓ I received **"{original_response}"** but I need a number to select an option.

**What I'm looking for:**
• A number (like 1, 2, 3) to select from the available options
• You entered: "{original_response}" (not a number)

Please enter the **number** of your choice from the options above."""
            
            return {
                "success": False,
                "result": None,
                "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
                "response_message": contextual_message,
                "error_type": "validation"
            }
            
    except Exception as e:
        raise HITLInteractionError(f"Error processing selection response: {str(e)}")


async def _process_input_request_response(hitl_data: Dict[str, Any], response_text: str, original_response: str) -> Dict[str, Any]:
    """Process input request responses (free text input)."""
    try:
        # Check for cancellation
        if response_text in ["cancel", "abort", "quit", "exit"]:
            return {
                "success": True,
                "result": "cancelled",
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": "❌ Input request cancelled as requested.",
                "error_type": None
            }
        
        # Validate input (basic validation - can be extended)
        if not original_response.strip():
            input_type = hitl_data.get("input_type", "information")
            contextual_message = f"""❓ I received an empty response, but I need some input from you.

**What I need:**
• {input_type.replace('_', ' ').title()} (cannot be empty)
• You sent: [empty message]

Please provide the requested {input_type.replace('_', ' ')} or type 'cancel' to abort."""
            
            return {
                "success": False,
                "result": None,
                "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
                "response_message": contextual_message,
                "error_type": "validation"
            }
        
        # Input accepted
        input_type = hitl_data.get("input_type", "information")
        return {
            "success": True,
            "result": {"input": original_response.strip(), "input_type": input_type},
            "hitl_data": None,  # Clear HITL data - interaction complete
            "response_message": f"✅ {input_type.replace('_', ' ').title()} received successfully.",
            "error_type": None
        }
        
    except Exception as e:
        raise HITLInteractionError(f"Error processing input request response: {str(e)}")


async def _process_multi_step_response(hitl_data: Dict[str, Any], response_text: str, original_response: str) -> Dict[str, Any]:
    """Process multi-step input responses."""
    try:
        steps = hitl_data.get("steps", [])
        current_step = hitl_data.get("current_step", 0)
        total_steps = hitl_data.get("total_steps", len(steps))
        
        # Check for cancellation
        if response_text in ["cancel", "abort", "quit", "exit"]:
            return {
                "success": True,
                "result": "cancelled",
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": "❌ Multi-step input cancelled as requested.",
                "error_type": None
            }
        
        # Validate input
        if not original_response.strip():
            current_step_info = steps[current_step]
            step_name = current_step_info.get("name", f"step_{current_step + 1}")
            contextual_message = f"""❓ I received an empty response for step {current_step + 1} of {total_steps}.

**What I need for this step:**
• {step_name.replace('_', ' ').title()} (cannot be empty)
• You sent: [empty message]

Please provide the requested information for this step or type 'cancel' to abort the entire process."""
            
            return {
                "success": False,
                "result": None,
                "hitl_data": {**hitl_data, "awaiting_response": False},  # Re-prompt
                "response_message": contextual_message,
                "error_type": "validation"
            }
        
        # Store current step response
        step_name = steps[current_step].get("name", f"step_{current_step + 1}")
        
        # Initialize collected_data if it doesn't exist
        collected_data = hitl_data.get("collected_data", {})
        collected_data[step_name] = original_response.strip()
        
        # Check if this was the last step
        if current_step >= total_steps - 1:
            # All steps complete
            return {
                "success": True,
                "result": {"collected_data": collected_data, "completed_steps": total_steps},
                "hitl_data": None,  # Clear HITL data - interaction complete
                "response_message": f"✅ All {total_steps} steps completed successfully!",
                "error_type": None
            }
        
        # Move to next step
        next_step = current_step + 1
        updated_hitl_data = {
            **hitl_data,
            "current_step": next_step,
            "collected_data": collected_data,
            "awaiting_response": False  # Will re-prompt for next step
        }
        
        return {
            "success": False,  # Not complete yet
            "result": {"partial_data": collected_data, "next_step": next_step},
            "hitl_data": updated_hitl_data,  # Continue with next step
            "response_message": f"✅ Step {current_step + 1} complete. Moving to step {next_step + 1}...",
            "error_type": None
        }
        
    except Exception as e:
        raise HITLInteractionError(f"Error processing multi-step response: {str(e)}")


async def _process_hitl_response_node(state: Dict[str, Any], hitl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node wrapper for processing HITL responses with state management.
    
    Args:
        state: Current agent state
        hitl_data: HITL interaction data
        
    Returns:
        Updated state after processing response
    """
    try:
        logger.info(f"🔍 [HITL_DEBUG] Processing HITL response node - hitl_data: {hitl_data}")
        
        # Get the latest human message
        messages = state.get("messages", [])
        if not messages:
            raise HITLResponseError("No messages found in state")
        
        logger.info(f"🔍 [HITL_DEBUG] Found {len(messages)} messages in state")
        
        # Find the most recent human message - this should be the approval/denial response
        latest_human_message = None
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', 'no_type') if hasattr(msg, 'type') else msg.get('type', 'no_type') if isinstance(msg, dict) else 'no_type'
            msg_content = getattr(msg, 'content', str(msg)) if hasattr(msg, 'content') else str(msg.get('content', msg)) if isinstance(msg, dict) else str(msg)
            
            logger.info(f"🔍 [HITL_DEBUG] Checking message from end: type={msg_type}, content='{msg_content[:50]}...'")
            
            if msg_type == 'human':
                latest_human_message = msg_content
                logger.info(f"🔍 [HITL_DEBUG] FOUND LATEST HUMAN MESSAGE: '{latest_human_message}'")
                break
        
        if not latest_human_message:
            raise HITLResponseError("No human message found in conversation state")
        
        logger.info(f"🔍 [HITL_DEBUG] Final selected human response: '{latest_human_message}'")
        
        # Process the response using the latest human message
        # This should be the user's approval/denial response
        response = latest_human_message
        
        # Process the response using the core function
        result = await _process_hitl_response(hitl_data, response)
        
        logger.info(f"🔍 [HITL_DEBUG] Core HITL processing result: {result}")
        
        # Prepare updated messages array
        updated_messages = list(messages)
        
        # Add response message if provided
        agent_response_message = None
        if result.get("response_message"):
            agent_response_message = {
                "role": "assistant",
                "content": result["response_message"],
                "type": "ai"  # Ensure it's recognized as AI message by Memory Store
            }
            updated_messages.append(agent_response_message)
            logger.info(f"🔍 [HITL_DEBUG] Added response message: '{result['response_message']}'")
        
        # Prepare execution data if action was approved  
        execution_data = None
        if result.get("success") and result.get("result") == "approved":
            logger.info(f"🔍 [HITL_DEBUG] Action APPROVED - checking for execution context")
            
            # Check if this was a confirmation with execution context
            context = hitl_data.get("context", {})
            logger.info(f"🔍 [HITL_DEBUG] Context available: {bool(context)}")
            logger.info(f"🔍 [HITL_DEBUG] Context keys: {list(context.keys()) if context else 'None'}")
            
            if context.get("tool"):
                execution_data = context
                logger.info(f"🔍 [HITL_DEBUG] Setting execution_data for approved {context.get('tool')} action")
                context_info = context.get('customer_info', context.get('meeting_info', context.get('workflow_data', {})))
                logger.info(f"🔍 [HITL_DEBUG] Execution data context: {type(context_info).__name__}")
            else:
                logger.warning(f"🔍 [HITL_DEBUG] APPROVED action missing tool in context - available keys: {list(context.keys())}")
        else:
            logger.info(f"🔍 [HITL_DEBUG] Action NOT approved - success: {result.get('success')}, result: {result.get('result')}")
        
        # Prepare final state
        final_state = {
            **state,
            "messages": updated_messages,
            "hitl_data": result.get("hitl_data"),  # None if complete, updated if continuing
            "hitl_result": result.get("result") if result.get("success") else None,
            "execution_data": execution_data  # Set execution data for approved actions
        }
        
        logger.info(f"🔍 [HITL_DEBUG] Final state prepared:")
        logger.info(f"🔍 [HITL_DEBUG]   - hitl_data: {final_state.get('hitl_data') is not None}")
        logger.info(f"🔍 [HITL_DEBUG]   - hitl_result: {final_state.get('hitl_result')}")
        logger.info(f"🔍 [HITL_DEBUG]   - execution_data: {final_state.get('execution_data') is not None}")
        if final_state.get('execution_data'):
            logger.info(f"🔍 [HITL_DEBUG]   - execution_data tool: {final_state.get('execution_data', {}).get('tool')}")
        
        # Return updated state
        return final_state
            
    except HITLError:
        # Re-raise HITL errors
        logger.error(f"🔍 [HITL_DEBUG] HITLError in response processing")
        raise
    except Exception as e:
        logger.error(f"🔍 [HITL_DEBUG] Exception in response processing: {e}")
        hitl_error = handle_hitl_error(e, {
            "function": "_process_hitl_response_node",
            "interaction_type": hitl_data.get("type"),
            "hitl_data": hitl_data
        })
        raise hitl_error


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


# Helper functions for better agent communication
def format_agent_hitl_transition(hitl_type: str, reason: str, context: Dict[str, Any]) -> str:
    """
    Format clear agent messages for HITL transitions.
    
    Args:
        hitl_type: Type of HITL interaction needed
        reason: Why the agent needs human input
        context: Context data for the interaction
    
    Returns:
        Clear transition message for the user
    """
    tool_name = context.get("tool", "system")
    
    transition_messages = {
        "confirmation": f"🤖 I need your approval to proceed with `{tool_name}`. {reason}",
        "selection": f"🤖 I found multiple options for `{tool_name}`. {reason}",  
        "input_request": f"🤖 I need additional information for `{tool_name}`. {reason}",
        "multi_step_input": f"🤖 I need to collect some details for `{tool_name}`. {reason}"
    }
    
    return transition_messages.get(hitl_type, f"🤖 I need your input for `{tool_name}`. {reason}")


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


logger.info("HITL module initialized successfully") 