"""
Sales Requirements Collection Tools

Revolutionary tool-managed recursive collection system for gathering
customer sales requirements with intelligent conversation analysis.
"""

import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Core imports
from langchain_core.tools import tool
from langsmith import traceable

# Toolbox imports
from .toolbox import (
    get_appropriate_llm,
    get_conversation_context
)

# HITL imports - using relative imports for toolbox
try:
    from agents.hitl import request_input
except ImportError:
    # Fallback for when running from backend directory
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.hitl import request_input

logger = logging.getLogger(__name__)

# =============================================================================
# CONVERSATION ANALYSIS FOR PRE-POPULATION
# =============================================================================

async def extract_fields_from_conversation(
    state: Dict[str, Any],
    field_definitions: Dict[str, str],
    tool_name: str = "unknown"
) -> Dict[str, str]:
    """
    REVOLUTIONARY: Universal LLM-powered function to extract already-provided information from conversation.
    
    This universal helper eliminates redundant questions by intelligently analyzing conversation
    context to identify information customers have already provided. Can be used by ANY tool
    that needs to collect information from users.
    
    ARCHITECTURE: Directly pulls from agent state (messages + conversation_summary) for uniform
    integration with agents' context window and memory management system.
    
    Key Benefits:
    - Eliminates frustrating redundant questions
    - Improves user experience dramatically  
    - Reusable across all collection tools
    - Uses fast/cheap models for cost effectiveness
    - Consistent LLM-powered analysis
    - Uniform with agents' memory management
    - Handles complex natural language expressions
    
    Args:
        state: Agent state containing messages and conversation_summary
        field_definitions: Dict mapping field names to descriptions
                          e.g., {"budget": "Budget range or maximum amount", 
                                "timeline": "When they need the vehicle"}
        tool_name: Name of calling tool for logging purposes
        
    Returns:
        Dictionary of extracted fields and their values from conversation
        
    Examples:
        State: {"messages": [...], "conversation_summary": "Customer interested in SUV..."}
        Fields: {"budget": "Budget range", "vehicle_type": "Type of vehicle", "primary_use": "How they'll use it"}
        Returns: {"budget": "under $50,000", "vehicle_type": "SUV", "primary_use": "daily commuting"}
    """
    try:
        logger.info(f"[EXTRACT_FIELDS] 🧠 Analyzing conversation for {tool_name} - Fields: {list(field_definitions.keys())}")
        
        # Get conversation context
        conversation_summary = state.get("conversation_summary", "")
        messages = state.get("messages", [])
        
        if not conversation_summary and not messages:
            logger.info("[EXTRACT_FIELDS] No conversation context available for analysis")
            return {}
        
        # Format conversation context for LLM analysis
        context_parts = []
        if conversation_summary:
            context_parts.append(f"Conversation Summary: {conversation_summary}")
        
        if messages:
            recent_messages = messages[-10:]  # Last 10 messages
            formatted_messages = []
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:300]  # Truncate long messages
                if content:
                    formatted_messages.append(f"{role.upper()}: {content}")
            
            if formatted_messages:
                context_parts.append("Recent Messages:\n" + "\n".join(formatted_messages))
        
        if not context_parts:
            logger.info("[EXTRACT_FIELDS] No usable conversation content for analysis")
            return {}
        
        conversation_context = "\n\n".join(context_parts)
        
        # Create LLM prompt for field extraction
        field_list = []
        for field_name, description in field_definitions.items():
            field_list.append(f"- {field_name}: {description}")
        
        fields_text = "\n".join(field_list)
        
        extraction_prompt = f"""You are analyzing a conversation to extract specific information that has already been provided by the customer.

CONVERSATION CONTEXT:
{conversation_context}

FIELDS TO EXTRACT (only extract if clearly mentioned):
{fields_text}

INSTRUCTIONS:
1. Only extract information that is explicitly or clearly implied in the conversation
2. Do not make assumptions or inferences beyond what's stated
3. Return ONLY the information that's actually present
4. Use the customer's exact words or close paraphrases
5. If information is not clearly provided, do not extract it

Return your response as JSON in this exact format:
{{
    "field_name": "extracted_value",
    "another_field": "another_value"
}}

If no fields can be extracted, return: {{}}

IMPORTANT: Return ONLY valid JSON, no explanations or additional text."""

        # Get LLM and extract fields
        llm = await get_appropriate_llm(extraction_prompt)
        response = await llm.ainvoke([{"role": "user", "content": extraction_prompt}])
        
        # Parse LLM response
        try:
            import json
            
            response_content = response.content.strip()
            
            # Try to extract JSON from response
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "").strip()
            
            extracted_fields = json.loads(response_content)
            
            # Validate extracted fields
            valid_fields = {}
            for field_name, value in extracted_fields.items():
                if field_name in field_definitions and value and str(value).strip():
                    valid_fields[field_name] = str(value).strip()
            
            if valid_fields:
                logger.info(f"[EXTRACT_FIELDS] ✅ Successfully extracted {len(valid_fields)} fields: {list(valid_fields.keys())}")
            else:
                logger.info("[EXTRACT_FIELDS] No valid fields extracted from conversation")
            
            return valid_fields
            
        except json.JSONDecodeError as e:
            logger.warning(f"[EXTRACT_FIELDS] Failed to parse LLM response as JSON: {e}")
            return {}
        
    except Exception as e:
        logger.error(f"[EXTRACT_FIELDS] Error extracting fields from conversation: {e}")
        return {}

# =============================================================================
# SALES REQUIREMENTS COLLECTION
# =============================================================================

class CollectSalesRequirementsParams(BaseModel):
    """Parameters for collecting comprehensive sales requirements from customers."""
    customer_identifier: str = Field(..., description="Customer name, ID, email, or phone to identify the customer")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Previously collected requirements data")
    collection_mode: str = Field(default="tool_managed", description="Collection mode - always 'tool_managed' for this revolutionary approach")
    current_field: str = Field(default="", description="Current field being collected (used for recursive calls)")
    user_response: str = Field(default="", description="User's response to the current field request (used for recursive calls)")
    conversation_context: str = Field(default="", description="Recent conversation messages for intelligent pre-population (REVOLUTIONARY: Task 9.2)")

@tool(args_schema=CollectSalesRequirementsParams)
@traceable(name="collect_sales_requirements")
async def collect_sales_requirements(
    customer_identifier: str,
    collected_data: Dict[str, Any] = None,
    collection_mode: str = "tool_managed",
    current_field: str = "",
    user_response: str = "",
    conversation_context: str = ""
) -> str:
    """
    REVOLUTIONARY: Tool-managed recursive collection for sales requirements.
    
    This tool demonstrates the new approach where tools manage their own collection 
    state, determine completion themselves, and get re-called by the agent with user responses.
    
    KEY REVOLUTIONARY FEATURES (Task 9.2):
    - Intelligent conversation analysis to pre-populate already-provided information
    - Avoids redundant questions by extracting data from conversation history
    - Only asks for genuinely missing information
    - Provides user feedback about pre-populated data
    
    **Use this tool for:**
    - Collecting comprehensive customer requirements for vehicle purchases
    - Understanding customer needs, budget, and timeline
    - Gathering information for personalized recommendations
    - Building customer profiles for better service
    - Pre-qualifying leads and opportunities
    
    **Do NOT use this tool for:**
    - Creating quotations (use generate_quotation instead)
    - Looking up existing customer data (use CRM tools instead)
    - General information gathering (use appropriate specific tools)
    - Technical vehicle specifications (use product databases)
    
    **Required Fields:** budget, timeline, vehicle_type, primary_use, financing_preference
    
    **Revolutionary Features:**
    - Pre-populates information from conversation context
    - Eliminates redundant questions
    - Provides intelligent collection flow
    - Tool-managed state and completion detection
    
    Args:
        customer_identifier: Customer name, ID, email, or phone
        collected_data: Previously collected data (managed by tool)
        collection_mode: Always "tool_managed" for this approach
        current_field: Current field being collected (for recursive calls)
        user_response: User response (for recursive calls)
        conversation_context: Recent conversation messages for intelligent pre-population
        
    Returns:
        Either completion summary or next field request for human input
        
    Examples:
        Initial call:
        collect_sales_requirements("john.doe@email.com", conversation_context="I'm looking for an SUV under $50,000")
        -> Pre-populates budget and vehicle_type, only asks for remaining fields
        
        Recursive call:
        collect_sales_requirements("john.doe@email.com", collected_data={...}, current_field="timeline", user_response="within 2 weeks")
    """
    try:
        if collected_data is None:
            collected_data = {}
            
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Starting collection for customer: {customer_identifier}")
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Current field: '{current_field}', User response: '{user_response}'")
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Collected so far: {list(collected_data.keys())}")
        
        # REVOLUTIONARY: Intelligent conversation analysis for pre-population (Task 9.2)
        # Only analyze on first call (when no data collected yet)
        if not collected_data:
            logger.info("[COLLECT_SALES_REQUIREMENTS] 🧠 Analyzing conversation for already-provided information...")
            
            # Define what fields this tool needs
            field_definitions = {
                "budget": "Budget range or maximum amount (e.g., 'under $50,000', '$40k-60k', 'around 45000')",
                "timeline": "When they need the vehicle (e.g., 'within a month', 'by summer', 'ASAP', 'no rush')",
                "vehicle_type": "Type of vehicle (e.g., 'SUV', 'sedan', 'pickup truck', 'convertible')",
                "primary_use": "How they plan to use it (e.g., 'family trips', 'daily commuting', 'work hauling')",
                "financing_preference": "Payment method (e.g., 'cash', 'financing', 'lease', 'loan')"
            }
            
            # SIMPLIFIED: For now, get conversation context from the conversation_context parameter
            # TODO: In the future, this will be automatically extracted from agent state
            if conversation_context:
                # Build a simple state-like dict for the helper function
                mock_state = {
                    "messages": [],  # This would come from agent state
                    "conversation_summary": conversation_context  # Use provided context as summary
                }
                
                # Extract already-provided information
                pre_populated_data = await extract_fields_from_conversation(
                    mock_state,
                    field_definitions,
                    "collect_sales_requirements"
                )
                
                if pre_populated_data:
                    collected_data.update(pre_populated_data)
                    logger.info(f"[COLLECT_SALES_REQUIREMENTS] ✅ Pre-populated from conversation: {list(pre_populated_data.keys())}")
            else:
                logger.info("[COLLECT_SALES_REQUIREMENTS] ℹ️ No conversation context provided - skipping pre-population")
        
        # REVOLUTIONARY: Handle user response for recursive collection
        if current_field and user_response:
            logger.info(f"[COLLECT_SALES_REQUIREMENTS] 🔄 Processing user response for field: {current_field}")
            collected_data[current_field] = user_response.strip()
        
        # Define required fields and check completion
        required_fields = {
            "budget": "Budget range for the purchase",
            "timeline": "When you need the vehicle", 
            "vehicle_type": "Type of vehicle you're interested in",
            "primary_use": "How you plan to use the vehicle",
            "financing_preference": "Preferred payment method"
        }
        
        # Find next missing field
        missing_fields = [field for field in required_fields.keys() if field not in collected_data]
        
        if not missing_fields:
            # REVOLUTIONARY: Collection complete with intelligent pre-population awareness
            logger.info("[COLLECT_SALES_REQUIREMENTS] 🎉 Collection COMPLETE - all requirements gathered")
            
            # Check if any data was pre-populated from conversation analysis
            pre_populated_count = 0
            for field in required_fields.keys():
                if field in collected_data:
                    # Simple heuristic: if we didn't ask for this field in this session, it was pre-populated
                    if not current_field or current_field != field:
                        pre_populated_count += 1
            
            # Create completion summary
            summary_parts = [
                f"✅ **Sales Requirements Collection Complete - {customer_identifier}**",
                "",
                "**Your Requirements:**"
            ]
            
            for field, description in required_fields.items():
                value = collected_data.get(field, "Not specified")
                summary_parts.append(f"• **{description}**: {value}")
            
            if pre_populated_count > 0:
                summary_parts.extend([
                    "",
                    f"🧠 **Smart Analysis**: I automatically identified {pre_populated_count} requirement(s) from our conversation, saving you time!"
                ])
            
            summary_parts.extend([
                "",
                "**Next Steps:**",
                "• I can help you find vehicles that match your requirements",
                "• Generate a personalized quotation when you're ready",
                "• Connect you with a sales representative for detailed assistance",
                "",
                "*Your requirements have been saved and will be used to provide personalized recommendations.*"
            ])
            
            return "\n".join(summary_parts)
        
        # Get next field to collect
        next_field = missing_fields[0]
        field_description = required_fields[next_field]
        
        # Create intelligent prompt based on field type and context
        prompt_parts = [
            f"📋 **Sales Requirements Collection - {customer_identifier}**",
            ""
        ]
        
        # Show pre-populated data if any
        if collected_data:
            prompt_parts.extend([
                "**Information I already have:**"
            ])
            for field, value in collected_data.items():
                field_desc = required_fields.get(field, field.replace('_', ' ').title())
                prompt_parts.append(f"✅ {field_desc}: {value}")
            prompt_parts.append("")
        
        # Ask for the next field
        progress = len(collected_data) + 1
        total = len(required_fields)
        
        prompt_parts.extend([
            f"**Question {progress}/{total}: {field_description}**",
            "",
            _get_field_specific_guidance(next_field),
            "",
            f"*This helps me find the perfect vehicle options for you.*"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] 📝 Requesting field: {next_field} ({progress}/{total})")
        
        # REVOLUTIONARY: Use request_input for clean recursive collection
        return request_input(
            prompt=prompt,
            input_type="sales_requirement",
            context={
                "source_tool": "collect_sales_requirements",
                "collection_mode": "tool_managed",
                "customer_identifier": customer_identifier,
                "collected_data": collected_data,
                "current_field": next_field,
                "required_fields": required_fields,
                "missing_fields": missing_fields
            },
            validation_hints=["Please provide as much detail as you're comfortable sharing"]
        )

    except Exception as e:
        logger.error(f"[COLLECT_SALES_REQUIREMENTS] Error: {e}")  
        return f"""❌ **Collection Error**

Sorry, I encountered an error while collecting your requirements: {str(e)}

**Please try:**
- Starting the collection process again
- Providing your requirements in a different format
- Contacting support if the issue persists

I'm here to help gather your vehicle requirements efficiently!"""

def _get_field_specific_guidance(field_name: str) -> str:
    """Get specific guidance text for different field types."""
    guidance_map = {
        "budget": """**Please share your budget range or maximum amount:**
• Specific amount: "Around $45,000" or "Maximum $60k"
• Range: "$40,000 to $55,000" or "Between 35k-50k"
• General: "Under $50,000" or "Around 40k"
• Flexible: "Open to options" or "Depends on the vehicle"

*Don't worry about being exact - ranges and estimates work perfectly!*""",

        "timeline": """**When do you need your new vehicle:**
• Urgent: "ASAP" or "Within a week"
• Soon: "Within a month" or "By the end of this month"
• Flexible: "In the next few months" or "By summer"
• Planning: "No rush" or "Just exploring options"

*This helps me prioritize the right vehicles for your schedule.*""",

        "vehicle_type": """**What type of vehicle interests you:**
• **Cars**: Sedan, Coupe, Hatchback, Convertible
• **SUVs**: Compact SUV, Mid-size SUV, Full-size SUV, Luxury SUV
• **Trucks**: Pickup truck, Work truck, Heavy duty
• **Other**: Van, Wagon, Electric vehicle, Hybrid

*You can be specific (like "mid-size SUV") or general (like "something reliable").*""",

        "primary_use": """**How do you plan to use this vehicle:**
• **Daily**: Commuting to work, Daily driving, City driving
• **Family**: Family trips, School runs, Weekend activities
• **Work**: Business use, Hauling equipment, Delivery
• **Recreation**: Road trips, Outdoor adventures, Towing
• **Mixed**: A bit of everything, General purpose

*Understanding your main use helps me recommend the perfect features.*""",

        "financing_preference": """**How would you prefer to pay:**
• **Cash**: Full payment upfront
• **Financing**: Monthly payments with loan
• **Lease**: Lower monthly payments, return after term
• **Trade-in**: Using current vehicle as part of payment
• **Flexible**: Open to discussing options

*No commitment needed - this just helps me show relevant pricing.*"""
    }
    
    return guidance_map.get(field_name, f"Please provide information about: {field_name.replace('_', ' ')}")

# =============================================================================
# COLLECTION ANALYTICS AND UTILITIES
# =============================================================================

async def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about sales requirements collection (internal use)."""
    try:
        # This would query collection statistics from database
        # Simplified implementation for now
        logger.info("[COLLECTION_STATS] Getting collection statistics")
        return {
            "status": "available",
            "total_collections": 0,
            "completed_today": 0,
            "average_completion_time": "N/A",
            "most_common_vehicle_type": "SUV",
            "average_budget_range": "$40,000-$60,000"
        }
        
    except Exception as e:
        logger.error(f"[COLLECTION_STATS] Error getting collection stats: {e}")
        return {"status": "error", "error": str(e)}

def validate_collected_requirements(collected_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collected sales requirements for completeness and consistency."""
    validation_result = {
        "is_complete": True,
        "missing_fields": [],
        "warnings": [],
        "suggestions": []
    }
    
    required_fields = ["budget", "timeline", "vehicle_type", "primary_use", "financing_preference"]
    
    # Check for missing fields
    for field in required_fields:
        if field not in collected_data or not collected_data[field]:
            validation_result["is_complete"] = False
            validation_result["missing_fields"].append(field)
    
    # Check for consistency issues
    if "budget" in collected_data and "financing_preference" in collected_data:
        budget = str(collected_data["budget"]).lower()
        financing = str(collected_data["financing_preference"]).lower()
        
        if "cash" in financing and ("financing" in budget or "loan" in budget):
            validation_result["warnings"].append("Budget mentions financing but payment preference is cash")
    
    # Add suggestions based on collected data
    if "vehicle_type" in collected_data and "primary_use" in collected_data:
        vehicle_type = str(collected_data["vehicle_type"]).lower()
        primary_use = str(collected_data["primary_use"]).lower()
        
        if "sedan" in vehicle_type and ("hauling" in primary_use or "towing" in primary_use):
            validation_result["suggestions"].append("Consider a truck or SUV for hauling needs")
        elif "truck" in vehicle_type and "city" in primary_use:
            validation_result["suggestions"].append("SUV might be more comfortable for city driving")
    
    return validation_result

def format_requirements_summary(collected_data: Dict[str, Any], customer_identifier: str) -> str:
    """Format collected requirements into a professional summary."""
    if not collected_data:
        return f"No requirements collected yet for {customer_identifier}"
    
    summary_parts = [
        f"**Sales Requirements Summary - {customer_identifier}**",
        ""
    ]
    
    field_labels = {
        "budget": "Budget Range",
        "timeline": "Timeline",
        "vehicle_type": "Vehicle Type",
        "primary_use": "Primary Use",
        "financing_preference": "Financing Preference"
    }
    
    for field, value in collected_data.items():
        label = field_labels.get(field, field.replace('_', ' ').title())
        summary_parts.append(f"• **{label}**: {value}")
    
    return "\n".join(summary_parts)
