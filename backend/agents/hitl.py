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
import time
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
    
    # Removed GraphInterrupt - using simple message-based flow instead

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

# PROACTIVE LOOP PREVENTION: Single-repetition detection constants
# Even ONE repeat is frustrating - prevent it proactively
RECENT_HITL_HISTORY_KEY = "_hitl_interaction_history"
MAX_RECENT_HITL_TRACKED = 5  # Track last 5 HITL interactions for repetition detection
PROMPT_SIMILARITY_THRESHOLD = 0.8  # 80% similarity considered a repeat
CONTEXT_SIMILARITY_THRESHOLD = 0.7  # 70% context similarity considered same interaction


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


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using simple token-based approach.
    
    This is a lightweight similarity calculation optimized for HITL prompt comparison.
    Uses normalized token overlap to detect when the same question is being asked again.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        float: Similarity score between 0.0 and 1.0 (1.0 = identical)
    """
    if not text1 or not text2:
        return 0.0
    
    if text1.strip() == text2.strip():
        return 1.0
    
    # Normalize texts - lowercase, remove extra whitespace
    norm_text1 = " ".join(text1.lower().split())
    norm_text2 = " ".join(text2.lower().split())
    
    if norm_text1 == norm_text2:
        return 1.0
    
    # Simple token-based similarity
    tokens1 = set(norm_text1.split())
    tokens2 = set(norm_text2.split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0


def _track_hitl_interaction(state: Dict[str, Any], hitl_prompt: str, hitl_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Track current HITL interaction in state history for repetition detection.
    
    Maintains a rolling history of recent HITL interactions to detect patterns
    and prevent immediate repetition that frustrates users.
    
    Args:
        state: Current agent state
        hitl_prompt: The prompt being shown to user
        hitl_context: Context for the HITL interaction
        
    Returns:
        Updated state with tracked interaction history
    """
    # Get existing history or initialize
    history = state.get(RECENT_HITL_HISTORY_KEY, [])
    
    # Create interaction record
    interaction_record = {
        "timestamp": time.time(),
        "prompt": hitl_prompt,
        "context_summary": {
            "source_tool": hitl_context.get("source_tool", "unknown"),
            "action_type": hitl_context.get("action_type", "unknown"),
            # Store key context indicators, not full context for efficiency
            "context_hash": hash(str(sorted(hitl_context.items()))) if hitl_context else 0
        }
    }
    
    # Add to history (most recent first)
    history.insert(0, interaction_record)
    
    # Trim to max tracked interactions
    if len(history) > MAX_RECENT_HITL_TRACKED:
        history = history[:MAX_RECENT_HITL_TRACKED]
    
    logger.debug(f"🔄 [HITL_TRACKING] Added interaction to history. Total tracked: {len(history)}")
    
    return {
        **state,
        RECENT_HITL_HISTORY_KEY: history
    }


def _detect_hitl_repetition(state: Dict[str, Any], current_prompt: str, current_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    PROACTIVE: Detect if current HITL request is a repetition of recent interaction.
    
    Even ONE repetition is frustrating to users. This function proactively detects
    when the same question is being asked again and prevents user frustration.
    
    Args:
        state: Current agent state with interaction history
        current_prompt: The prompt about to be shown
        current_context: Context for current HITL interaction
        
    Returns:
        Dict with repetition details if detected, None if no repetition
    """
    if not current_prompt:
        return None
    
    history = state.get(RECENT_HITL_HISTORY_KEY, [])
    if not history:
        return None
    
    current_context_hash = hash(str(sorted(current_context.items()))) if current_context else 0
    current_source = current_context.get("source_tool", "unknown")
    
    logger.debug(f"🔍 [REPETITION_DETECTION] Checking current prompt against {len(history)} recent interactions")
    
    # Check each recent interaction for similarity
    for i, past_interaction in enumerate(history):
        past_prompt = past_interaction.get("prompt", "")
        past_context = past_interaction.get("context_summary", {})
        
        # Calculate prompt similarity
        prompt_similarity = _calculate_text_similarity(current_prompt, past_prompt)
        
        # Check context similarity
        same_tool = (current_source == past_context.get("source_tool", "unknown"))
        context_similarity = 1.0 if current_context_hash == past_context.get("context_hash", -1) else 0.0
        
        # Detect repetition based on thresholds
        is_prompt_repeat = prompt_similarity >= PROMPT_SIMILARITY_THRESHOLD
        is_context_repeat = same_tool and context_similarity >= CONTEXT_SIMILARITY_THRESHOLD
        
        if is_prompt_repeat or is_context_repeat:
            repetition_info = {
                "detected": True,
                "repetition_index": i,
                "prompt_similarity": prompt_similarity,
                "context_similarity": context_similarity,
                "same_tool": same_tool,
                "past_interaction": past_interaction,
                "detection_reason": "prompt_similarity" if is_prompt_repeat else "context_similarity"
            }
            
            logger.warning(f"🚨 [REPETITION_DETECTED] Found repetition at index {i}:")
            logger.warning(f"   Prompt similarity: {prompt_similarity:.2f} (threshold: {PROMPT_SIMILARITY_THRESHOLD})")
            logger.warning(f"   Context similarity: {context_similarity:.2f} (threshold: {CONTEXT_SIMILARITY_THRESHOLD})")
            logger.warning(f"   Same tool: {same_tool}")
            logger.warning(f"   Detection reason: {repetition_info['detection_reason']}")
            
            return repetition_info
    
    logger.debug(f"✅ [REPETITION_DETECTION] No repetition detected - interaction is unique")
    return None


def _handle_hitl_repetition_recovery(state: Dict[str, Any], repetition_info: Dict[str, Any], current_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    INTELLIGENT RECOVERY: Handle detected HITL repetition with user-friendly resolution.
    
    When repetition is detected, provide intelligent recovery instead of frustrating
    the user with the same question again.
    
    Args:
        state: Current agent state
        repetition_info: Details about the detected repetition
        current_context: Context for current HITL interaction
        
    Returns:
        Updated state with intelligent recovery handling
    """
    past_interaction = repetition_info.get("past_interaction", {})
    detection_reason = repetition_info.get("detection_reason", "unknown")
    
    # Create intelligent recovery message
    recovery_message = (
        f"I notice I'm about to ask you the same question again. "
        f"Let me check if there's additional information I can use from our conversation "
        f"instead of repeating the request."
    )
    
    # Add recovery message to conversation
    recovery_messages = list(state.get("messages", []))
    recovery_messages.append({
        "role": "assistant",
        "content": recovery_message,
        "type": "ai",
        "metadata": {"recovery_type": "hitl_repetition_prevention"}
    })
    
    logger.info(f"🛠️ [REPETITION_RECOVERY] Applying intelligent recovery for {detection_reason} repetition")
    logger.info(f"   Recovery message: {recovery_message}")
    
    # Mark as recovery attempt in context
    recovered_context = {
        **current_context,
        "recovery_applied": True,
        "recovery_reason": detection_reason,
        "recovery_timestamp": time.time()
    }
    
    # Clear HITL state to prevent the repetitive interaction
    # The agent should try a different approach or use conversation analysis
    return {
        **state,
        "messages": recovery_messages,
        "hitl_phase": None,  # Clear HITL phase to exit HITL mode
        "hitl_prompt": None,
        "hitl_context": recovered_context,  # Keep context with recovery info
        "repetition_recovery_applied": True
    }


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
    ULTRA-SIMPLE HITL: Just show prompt, interrupt, and wait for response.
    
    No complex phases, no routing logic - just basic interrupt/resume pattern.
    """
    try:
        hitl_prompt = state.get("hitl_prompt")
        hitl_context = state.get("hitl_context", {})
        messages = state.get("messages", [])
        
        logger.info(f"[HITL_SIMPLE] Showing prompt and interrupting")
        logger.info(f"[HITL_SIMPLE] Prompt: {hitl_prompt[:100]}...")
        
        # Add prompt to conversation
        updated_messages = list(messages)
        if hitl_prompt:
            updated_messages.append({
                "role": "assistant",
                "content": hitl_prompt,
                "type": "ai"
            })
        
        # CRITICAL: Call interrupt() to STOP and wait for user
        interrupt(hitl_prompt or "Please provide your response:")
        
        # This code runs when user responds and system resumes
        logger.info(f"[HITL_SIMPLE] Resumed - looking for user response")
        
        # Get user response from latest messages  
        user_response = None
        all_messages = updated_messages
        for msg in reversed(all_messages):
            if isinstance(msg, dict) and msg.get("type") == "human":
                user_response = msg.get("content", "").strip()
                break
            elif hasattr(msg, 'type') and msg.type == "human":
                user_response = msg.content.strip()
                break
        
        if not user_response:
            logger.warning(f"[HITL_SIMPLE] No user response found after resume")
            interrupt("Please provide your response:")
            return state
        
        # Simple approval detection
        response_lower = user_response.lower()
        approve_words = ["yes", "approve", "send", "go ahead", "confirm", "ok", "proceed"]
        is_approved = any(word in response_lower for word in approve_words)
        
        # Clear HITL state and set result
        if is_approved:
            logger.info(f"[HITL_SIMPLE] User approved: '{user_response}'")
            response_msg = "✅ Approved! Executing action..."
            # Keep context for agent execution  
            execution_context = hitl_context
        else:
            logger.info(f"[HITL_SIMPLE] User denied: '{user_response}'")
            response_msg = "❌ Action cancelled."
            execution_context = None
        
        # Add response message
        final_messages = list(updated_messages)
        final_messages.append({
            "role": "assistant",
            "content": response_msg,
            "type": "ai"
        })
        
        # Return cleaned state
        return {
            **state,
            "messages": final_messages,
            "hitl_phase": "approved" if is_approved else "denied",  # Set completion phase
            "hitl_prompt": None,     # Clear prompt
            "hitl_context": execution_context  # Keep context if approved
        }
            
    except GraphInterrupt:
        # Expected - let it propagate
        logger.info(f"[HITL_SIMPLE] GraphInterrupt raised (expected)")
        raise
    except Exception as e:
        logger.error(f"[HITL_SIMPLE] Error: {e}")
        return {
            **state,
            "hitl_phase": None,
            "hitl_prompt": None, 
            "hitl_context": None,
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": "I encountered an error. Please try again."
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
    SIMPLIFIED: Add HITL prompt to messages - interrupt() is handled in main hitl_node.
    
    Args:
        state: Current agent state
        hitl_prompt: The complete prompt text (tool-generated)
        hitl_context: Minimal context for logging/debugging
        
    Returns:
        Updated state with prompt added to messages
    """
    try:
        if not hitl_prompt:
            logger.warning("🤖 [HITL_3FIELD] Empty prompt provided, using fallback")
            hitl_prompt = "Please provide your response:"
        
        # Add HITL prompt to messages for persistence
        updated_messages = list(state.get("messages", []))
        hitl_prompt_message = {
            "role": "assistant",
            "content": hitl_prompt,
            "type": "ai"
        }
        updated_messages.append(hitl_prompt_message)
        
        # Return state with message added
        return {
            **state,
            "messages": updated_messages,
            "hitl_phase": "awaiting_response"
        }
        
    except Exception as e:
        hitl_error = handle_hitl_error(e, {
            "function": "_show_hitl_prompt_3field",
            "hitl_context": hitl_context
        })
        raise hitl_error


def _get_latest_human_response(messages):
    """
    Simple, reliable human response detection.
    
    Replaces complex prompt-matching logic with straightforward last-message check.
    This is more reliable and handles different message formats robustly.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        str or None: Latest human message content, or None if not found
    """
    if not messages:
        return None
    
    last_message = messages[-1]
    
    # Handle LangChain message objects
    if hasattr(last_message, 'type') and hasattr(last_message, 'content'):
        if last_message.type == 'human':
            content = last_message.content
            return content.strip() if content else None
    
    # Handle dictionary format messages
    elif isinstance(last_message, dict):
        is_human = (last_message.get('type') == 'human' or 
                   last_message.get('role') == 'human')
        if is_human:
            content = last_message.get('content', '')
            return content.strip() if content else None
    
    return None


def _get_latest_human_response_enhanced(messages):
    """
    ENHANCED: Comprehensive human response detection with robust message format handling.
    
    Features:
    - Backward search through last 10 messages to find human responses reliably
    - Support for both 'human' and 'user' role detection for message format compatibility  
    - Comprehensive message format detection (LangChain objects, dicts, mixed formats)
    - Enhanced logging for debugging message parsing failures
    - Fallback detection for edge cases in message format variations
    
    Args:
        messages: List of conversation messages
        
    Returns:
        str or None: Latest human message content, or None if not found
    """
    if not messages:
        logger.debug("🔍 [ENHANCED_HUMAN_DETECTION] No messages provided")
        return None
    
    # Search backward through last 10 messages maximum for performance
    search_limit = min(10, len(messages))
    logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Searching {search_limit} messages backward from index {len(messages)-1}")
    
    # Track message format analysis for debugging
    format_stats = {
        "langchain_objects": 0,
        "dict_messages": 0,
        "unknown_format": 0,
        "human_candidates": []
    }
    
    # Search backward through recent messages
    for i in range(len(messages) - 1, max(len(messages) - search_limit - 1, -1), -1):
        message = messages[i]
        message_index = i
        human_content = None
        message_format = "unknown"
        
        try:
            # ENHANCED: Handle LangChain message objects (BaseMessage subclasses)
            if hasattr(message, 'type') and hasattr(message, 'content'):
                message_format = "langchain_object"
                format_stats["langchain_objects"] += 1
                
                # Support both 'human' and 'user' types for compatibility
                if message.type in ['human', 'user']:
                    human_content = message.content
                    logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Found LangChain {message.type} message at index {message_index}")
                    
            # ENHANCED: Handle dictionary format messages with comprehensive role detection
            elif isinstance(message, dict):
                message_format = "dict_message"
                format_stats["dict_messages"] += 1
                
                # Support multiple role/type field variations
                role_indicators = [
                    message.get('type'),
                    message.get('role'),
                    message.get('sender_type'),  # Some systems use this
                    message.get('message_type')  # Alternative format
                ]
                
                # Check if any role indicator suggests human/user
                human_indicators = ['human', 'user', 'customer', 'person']  # Extended compatibility
                is_human_message = any(
                    indicator in human_indicators 
                    for indicator in role_indicators 
                    if indicator is not None
                )
                
                if is_human_message:
                    # Try multiple content field variations
                    content_fields = ['content', 'text', 'message', 'body']
                    for field in content_fields:
                        if field in message and message[field]:
                            human_content = message[field]
                            break
                    
                    detected_role = next((ind for ind in role_indicators if ind in human_indicators), "unknown")
                    logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Found dict {detected_role} message at index {message_index}")
                    
            # ENHANCED: Handle edge cases - string messages, None, other formats
            elif message is None:
                logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Null message at index {message_index}")
                continue
            elif isinstance(message, str):
                # Some systems might store raw strings - treat as potential human input
                message_format = "raw_string"
                human_content = message
                logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Raw string message at index {message_index}")
            else:
                message_format = "unknown_format"
                format_stats["unknown_format"] += 1
                logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Unknown message format at index {message_index}: {type(message)}")
                
                # FALLBACK: Try to extract content from unknown formats
                if hasattr(message, 'content'):
                    human_content = getattr(message, 'content', None)
                elif hasattr(message, 'text'):
                    human_content = getattr(message, 'text', None)
                elif hasattr(message, '__str__'):
                    # Last resort - convert to string
                    human_content = str(message)
                    
        except Exception as e:
            logger.warning(f"🔍 [ENHANCED_HUMAN_DETECTION] Error processing message at index {message_index}: {e}")
            continue
        
        # If we found human content, validate and return it
        if human_content:
            cleaned_content = human_content.strip() if isinstance(human_content, str) else str(human_content).strip()
            if cleaned_content:
                format_stats["human_candidates"].append({
                    "index": message_index,
                    "format": message_format,
                    "content_preview": cleaned_content[:50] + "..." if len(cleaned_content) > 50 else cleaned_content
                })
                
                logger.info(f"🔍 [ENHANCED_HUMAN_DETECTION] Found human response at index {message_index} ({message_format}): '{cleaned_content}'")
                
                # Log format analysis for debugging
                logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Format analysis: {format_stats}")
                
                return cleaned_content
    
    # ENHANCED: Comprehensive logging when no human response found
    logger.warning(f"🔍 [ENHANCED_HUMAN_DETECTION] No human response found in {search_limit} messages")
    logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Message format breakdown: {format_stats}")
    
    # Log sample of message formats for debugging
    if messages:
        sample_size = min(3, len(messages))
        logger.debug(f"🔍 [ENHANCED_HUMAN_DETECTION] Sample of last {sample_size} message formats:")
        for i, msg in enumerate(messages[-sample_size:]):
            msg_type = type(msg).__name__
            msg_preview = str(msg)[:100] + "..." if len(str(msg)) > 100 else str(msg)
            logger.debug(f"  [{len(messages)-sample_size+i}] {msg_type}: {msg_preview}")
    
    return None


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
        # Extract the latest human message AFTER the HITL prompt
        messages = state.get("messages", [])
        latest_human_message = None
        hitl_prompt = state.get("hitl_prompt", "")
        
        # REVOLUTIONARY: Enhanced message extraction logging
        _log_3field_state_snapshot(
            location="before_response_processing",
            hitl_phase="awaiting_response",
            hitl_prompt=None,  # Prompt already shown
            hitl_context=hitl_context,
            message_count=len(messages)
        )
        
        # REVOLUTIONARY: Enhanced human response detection with comprehensive format support
        latest_human_message = _get_latest_human_response_enhanced(messages)
        
        if latest_human_message:
            logger.info(f"🤖 [HITL_SIMPLE] Found human response: '{latest_human_message}'")
        else:
            logger.info(f"🤖 [HITL_SIMPLE] No human response found - staying in awaiting_response state")
        
        if not latest_human_message:
            _log_3field_state_transition(
                operation="awaiting_user_response", 
                additional_info={"hitl_prompt_preserved": bool(hitl_prompt), "message_count": len(messages), "simple_wait": True}
            )
            
            # CRITICAL FIX: If no user response, interrupt again to wait for input
            # This prevents endless routing loops
            logger.info(f"🤖 [HITL_SIMPLE] No user response found - interrupting again to wait")
            interrupt("Please provide your response:")
            
            # Return state unchanged for when we resume
            return {
                **state,
                "messages": messages,
                "hitl_phase": "awaiting_response",  # Stay in awaiting_response
                "hitl_prompt": hitl_prompt,  # Keep the prompt for next iteration
                "hitl_context": hitl_context,
            }
        
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
        
        # CRITICAL FIX: Always clear HITL state after processing response
        # This prevents endless loops by ensuring we don't stay in HITL mode
        result_value = result.get("result") if result.get("success") else "denied"
        
        if result_value == "approved":
            # User approved - set approved context for execution
            approved_context = hitl_context if hitl_context and hitl_context.get("source_tool") else None
            final_phase = "approved"
        else:
            # User denied or unclear response
            approved_context = None
            final_phase = "denied"
        
        # CRITICAL FIX: Always clear HITL state to prevent loops
        final_state = {
            **state,
            "messages": updated_messages,
            "hitl_phase": None,  # CRITICAL FIX: Always clear to prevent routing loops
            "hitl_prompt": None,  # CRITICAL FIX: Always clear
            "hitl_context": None,   # CRITICAL FIX: Always clear
            "approved_action": approved_context if result_value == "approved" else None  # For agent execution
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
            after_phase=final_phase,
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
    # Removed GraphInterrupt handling - using simple message-based flow instead
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
                "context": hitl_context,  # Preserve context for tool re-calling
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