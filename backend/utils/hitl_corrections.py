"""
TASK 7.7.3.1: LLM Correction Processor for @hitl_recursive_tool decorator

Simple utility for processing natural language corrections and mapping them to parameter updates.
Designed to be used by the enhanced @hitl_recursive_tool decorator for universal correction capability.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class LLMCorrectionProcessor:
    """
    Simple LLM-based correction processor for universal HITL corrections.
    
    Processes natural language corrections like:
    - "change customer to John Smith"
    - "make it red Toyota Camry"  
    - "use email john@example.com instead"
    
    Maps corrections to parameter updates automatically.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use cost-effective model for simple corrections
            temperature=0.1,
            max_tokens=1000
        )
    
    async def process_correction(
        self,
        user_response: str,
        original_params: Dict[str, Any],
        tool_name: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process user correction and map to parameter updates.
        
        Args:
            user_response: User's natural language correction
            original_params: Original tool parameters
            tool_name: Name of the tool being corrected
            
        Returns:
            Tuple of (intent, updated_params):
            - intent: "approval", "denial", or "correction"
            - updated_params: Updated parameters if correction, None otherwise
        """
        try:
            logger.info(f"[LLM_CORRECTION] Processing correction for {tool_name}: '{user_response[:100]}...'")
            
            # First, detect intent
            intent = await self._detect_user_intent(user_response)
            logger.info(f"[LLM_CORRECTION] Detected intent: {intent}")
            
            if intent == "correction":
                # Process the correction and map to parameters
                updated_params = await self._map_correction_to_parameters(
                    user_response, original_params, tool_name
                )
                return intent, updated_params
            else:
                # Approval or denial - no parameter updates needed
                return intent, None
                
        except Exception as e:
            logger.error(f"[LLM_CORRECTION] Error processing correction: {e}")
            # Fallback to treating as input/correction
            return "correction", None
    
    async def _detect_user_intent(self, user_response: str) -> str:
        """Detect if user wants to approve, deny, or correct."""
        try:
            system_prompt = """You are an intent classifier for user responses to approval requests.

Classify the user's response into one of these categories:
- "approval": User wants to approve/proceed (yes, ok, approve, go ahead, send, confirm, etc.)
- "denial": User wants to cancel/deny (no, cancel, stop, deny, don't, etc.)  
- "correction": User wants to make changes or provide corrections (change, update, use, make it, etc.)

Respond with only the category name."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User response: {user_response}")
            ]
            
            response = await self.llm.ainvoke(messages)
            intent = response.content.strip().lower()
            
            # Validate response
            if intent not in ["approval", "denial", "correction"]:
                logger.warning(f"[LLM_CORRECTION] Invalid intent '{intent}', defaulting to correction")
                return "correction"
                
            return intent
            
        except Exception as e:
            logger.error(f"[LLM_CORRECTION] Error detecting intent: {e}")
            return "correction"  # Default to correction for safety
    
    async def _map_correction_to_parameters(
        self,
        user_response: str,
        original_params: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """Map natural language corrections to parameter updates."""
        try:
            # Create parameter mapping prompt
            params_json = json.dumps(original_params, indent=2)
            
            system_prompt = f"""You are a parameter correction mapper. The user wants to correct parameters for the {tool_name} tool.

Original parameters:
{params_json}

The user provided a correction. Map their natural language correction to updated parameter values.

Rules:
1. Only update parameters that the user explicitly mentions
2. Keep all other parameters unchanged
3. Understand natural language corrections like:
   - "change customer to John Smith" → update customer_identifier
   - "make it red Toyota Camry" → update vehicle_requirements  
   - "use email john@example.com" → update customer info
4. Return the COMPLETE updated parameters as valid JSON
5. If you can't understand the correction, return the original parameters unchanged

Respond with only the JSON object of updated parameters."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User correction: {user_response}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse the JSON response
            try:
                updated_params = json.loads(response.content.strip())
                logger.info(f"[LLM_CORRECTION] Mapped correction to parameter updates")
                return updated_params
            except json.JSONDecodeError as e:
                logger.error(f"[LLM_CORRECTION] Invalid JSON response: {e}")
                return original_params
                
        except Exception as e:
            logger.error(f"[LLM_CORRECTION] Error mapping correction: {e}")
            return original_params
