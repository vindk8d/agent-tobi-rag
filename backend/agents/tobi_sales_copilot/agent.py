"""
Unified Tool-calling RAG Agent following LangChain best practices.
Single agent node handles both tool calling and execution.
"""

from typing import Dict, Any, List, Optional  
from datetime import datetime
from uuid import uuid4, UUID
import logging
import sys
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphInterrupt
# NOTE: interrupt, Command removed in Task 4.11.5 - unused after function cleanup
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langsmith import traceable

# Smart path setup for deployment compatibility
# Development: running from /backend directory, need backend dir in path  
# Production: running from /app directory, current structure works directly
import pathlib
backend_path = pathlib.Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Use direct imports (no backend. prefix) for deployment compatibility
from agents.tobi_sales_copilot.state import AgentState
from agents.tobi_sales_copilot.language import detect_user_language_from_context, get_employee_system_prompt, get_customer_system_prompt
from agents.toolbox import get_all_tools, get_tool_names, UserContext, get_tools_for_user_type
from agents.memory import memory_manager, memory_scheduler
from agents.background_tasks import BackgroundTaskManager, BackgroundTask, TaskPriority
from agents.hitl import parse_tool_response, hitl_node
from core.config import get_settings, setup_langsmith_tracing
from core.database import db_client
# Import portable utilities
from utils.user_verification import verify_user_access, verify_employee_access, handle_access_denied
from utils.api_processing import process_agent_result_for_api, process_user_message
from utils.message_delivery import (
    execute_customer_message_delivery,
    send_via_chat_api,
    format_business_message,
    get_or_create_customer_user,
    get_or_create_customer_conversation
)

logger = logging.getLogger(__name__)


class UnifiedToolCallingRAGAgent:
    """
    Unified RAG agent that handles both tool calling and execution in one place.
    Follows LangChain best practices: tool creation → tool binding → tool calling → tool execution.
    """

    def __init__(self):
        self.settings = None  # Will be loaded asynchronously
        self.tools = None  # Will be loaded asynchronously
        self.tool_names = None  # Will be loaded asynchronously
        self.llm = None  # Will be initialized asynchronously
        self.tool_map = None  # Will be created asynchronously
        self.graph = None  # Will be built asynchronously
        self._initialized = False  # Track initialization status

    async def _ensure_initialized(self):
        """Ensure the agent is initialized asynchronously."""
        if self._initialized:
            return

        # Load settings asynchronously
        self.settings = await get_settings()

        # Initialize memory manager first
        await memory_manager._ensure_initialized()
        self.memory_manager = memory_manager

        # Initialize background task manager for non-blocking operations
        self.background_task_manager = BackgroundTaskManager()
        await self.background_task_manager._ensure_initialized()
        await self.background_task_manager.start()

        # Register RAG-specific memory plugins
        await self._register_memory_plugins()

        # Start background memory scheduler
        await memory_scheduler.start()

        # Initialize LangSmith tracing asynchronously
        await setup_langsmith_tracing()

        # Step 1: Tool creation (already done with @tool decorators)
        self.tools = get_all_tools()
        self.tool_names = get_tool_names()

        # Use simple model selection - tools handle their own model selection dynamically
        # Default to chat model for agent coordination

        # Create a tool lookup for execution
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Build graph with persistence
        self.graph = await self._build_graph()

        # Log tracing status
        if self.settings.langsmith.tracing_enabled:
            logger.info(
                f"LangSmith tracing enabled for project: {self.settings.langsmith.project}"
            )
        else:
            logger.info("LangSmith tracing disabled")

        self._initialized = True

    async def cleanup(self):
        """Clean up agent resources on shutdown."""
        logger.info("Cleaning up UnifiedToolCallingRAGAgent resources...")
        try:
            # Stop background task manager
            if hasattr(self, 'background_task_manager'):
                await self.background_task_manager.stop()
            
            # Stop memory scheduler
            await memory_scheduler.stop()
            
            # Clean up memory manager
            await memory_manager.cleanup()
            
            # Reset initialization flag
            self._initialized = False
            
            logger.info("Agent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")



    async def _build_graph(self) -> StateGraph:
        """Build a graph with automatic checkpointing and state persistence between agent steps."""
        # Get checkpointer for persistence
        checkpointer = await memory_manager.get_checkpointer()

        graph = StateGraph(AgentState)

        # Add nodes with automatic checkpointing between steps
        graph.add_node("user_verification", self._user_verification_node)
        
        # Agent processing nodes (simplified architecture with background tasks)
        graph.add_node("employee_agent", self._employee_agent_node)
        graph.add_node("customer_agent", self._customer_agent_node)
        
        # General-purpose HITL node for human interactions
        graph.add_node("hitl_node", hitl_node)

        # Simplified graph flow with clear separation between employee and customer workflows
        graph.add_edge(START, "user_verification")

        # Route from user verification directly to agent nodes (simplified architecture)
        graph.add_conditional_edges(
            "user_verification",
            self._route_to_agent,
            {
                "employee_agent": "employee_agent", 
                "customer_agent": "customer_agent", 
                "end": END
            },
        )
        
        # Employee workflow: agent → (hitl_node or END) - no memory storage nodes
        # Route from employee agent to HITL or END (background tasks handle persistence)
        graph.add_conditional_edges(
            "employee_agent",
            self.route_from_employee_agent,
            {
                "hitl_node": "hitl_node",
                "end": END,  # Direct to END instead of memory storage
                # Allow routing back to employee_agent for universal tool-managed collection
                # This enables the agent to re-call tools with HITL-provided input
                "employee_agent": "employee_agent",
            },
        )
        
        # Customer workflow: agent → END (simple, no HITL, background tasks handle persistence)
        graph.add_edge("customer_agent", END)
        
        # REVOLUTIONARY: HITL workflow NEVER routes back to itself - always goes to employee_agent
        # This eliminates all HITL recursion and implements tool-managed collection pattern
        graph.add_conditional_edges(
            "hitl_node",
            self.route_from_hitl,
            {
                "employee_agent": "employee_agent"  # ALWAYS back to employee agent (no self-routing)
            },
        )
        
        # No memory store nodes - background tasks handle persistence

        # Compile with checkpointer for automatic persistence between steps
        # Configure interrupt_before to enable human interaction at HITL node
        return graph.compile(checkpointer=checkpointer, interrupt_before=["hitl_node"])

    @traceable(name="user_verification_node")
    async def _user_verification_node(
        self, state: AgentState, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Streamlined user verification node that performs ALL identification processes.
        
        Input: user_id (from state)
        Output: Either customer_id OR employee_id populated in state
        
        This centralizes all identification logic in one place, making other nodes simpler.
        """
        try:
            user_id = state.get("user_id")
            
            # Strip whitespace from user_id to prevent UUID parsing errors
            if user_id and isinstance(user_id, str):
                user_id = user_id.strip()

            if not user_id:
                logger.warning("[USER_VERIFICATION_NODE] Access denied: No user_id provided")
                error_message = AIMessage(
                    content="Authentication required. Please ensure you're properly logged in."
                )
                return {
                    "messages": state.get("messages", []) + [error_message],
                    "conversation_id": state.get("conversation_id"),
                    "user_id": user_id,
                    "conversation_summary": state.get("conversation_summary"),
                    "customer_id": None,
                    "employee_id": None,
                }

            # Extract thread_id from config if no conversation_id is provided
            conversation_id = state.get("conversation_id")
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = str(thread_id)
                    logger.info(f"[USER_VERIFICATION_NODE] Using thread_id {thread_id} as conversation_id")

            logger.info(f"[USER_VERIFICATION_NODE] Starting identification for user: {user_id}")

            # Perform centralized user identification
            from core.database import db_client
            await db_client._async_initialize_client()

            # Look up user in database
            user_result = await asyncio.to_thread(
                lambda: db_client.client.table("users")
                .select("id, user_type, employee_id, customer_id, email, display_name, is_active")
                .eq("id", user_id)
                .execute()
            )

            if not user_result.data or len(user_result.data) == 0:
                logger.warning(f"[USER_VERIFICATION_NODE] No user found for user_id: {user_id}")
                return await handle_access_denied(state, conversation_id, "User not found in system")

            user_record = user_result.data[0]
            user_type = user_record.get("user_type", "").lower()
            is_active = user_record.get("is_active", False)
            user_employee_id = user_record.get("employee_id")
            user_customer_id = user_record.get("customer_id")

            logger.info(f"[USER_VERIFICATION_NODE] User found - Type: {user_type}, Active: {is_active}")

            # Check if user is active
            if not is_active:
                logger.warning(f"[USER_VERIFICATION_NODE] User {user_id} is inactive")
                return await handle_access_denied(state, conversation_id, "Account is inactive")

            # Handle employee users
            if user_type in ["employee", "admin"]:
                if not user_employee_id:
                    logger.warning(f"[USER_VERIFICATION_NODE] Employee user has no employee_id")
                    return await handle_access_denied(state, conversation_id, "Employee record incomplete")

                # Verify employee record exists and is active
                employee_result = await asyncio.to_thread(
                    lambda: db_client.client.table("employees")
                    .select("id, name, email, position, is_active")
                    .eq("id", user_employee_id)
                    .eq("is_active", True)
                    .execute()
                )

                if not employee_result.data or len(employee_result.data) == 0:
                    logger.warning(f"[USER_VERIFICATION_NODE] No active employee found for employee_id: {user_employee_id}")
                    return await handle_access_denied(state, conversation_id, "Employee record not found")

                employee = employee_result.data[0]
                logger.info(f"[USER_VERIFICATION_NODE] ✅ Employee access verified - {employee['name']} ({employee['position']})")
                
                return {
                    "messages": state.get("messages", []),
                    "conversation_id": conversation_id,
                    "user_id": user_id,  # Now using cleaned user_id
                    "conversation_summary": state.get("conversation_summary"),
                    "employee_id": user_employee_id,  # Populate employee_id
                    "customer_id": None,  # Explicitly set customer_id to None
                }

            # Handle customer users
            elif user_type == "customer":
                if not user_customer_id:
                    logger.warning(f"[USER_VERIFICATION_NODE] Customer user has no customer_id")
                    return await handle_access_denied(state, conversation_id, "Customer record incomplete")

                # Verify customer record exists
                customer_result = await asyncio.to_thread(
                    lambda: db_client.client.table("customers")
                    .select("id, name, email")
                    .eq("id", user_customer_id)
                    .execute()
                )

                if not customer_result.data or len(customer_result.data) == 0:
                    logger.warning(f"[USER_VERIFICATION_NODE] No customer found for customer_id: {user_customer_id}")
                    return await handle_access_denied(state, conversation_id, "Customer record not found")

                customer = customer_result.data[0]
                logger.info(f"[USER_VERIFICATION_NODE] ✅ Customer access verified - {customer['name']} ({customer['email']})")
                
                return {
                    "messages": state.get("messages", []),
                    "conversation_id": conversation_id,
                    "user_id": user_id,  # Now using cleaned user_id
                    "conversation_summary": state.get("conversation_summary"),
                    "customer_id": user_customer_id,  # Populate customer_id
                    "employee_id": None,  # Explicitly set employee_id to None
                }

            # Handle unknown user types
            else:
                logger.warning(f"[USER_VERIFICATION_NODE] Unknown user_type: {user_type}")
                return await handle_access_denied(state, conversation_id, f"Unsupported user type: {user_type}")

        except Exception as e:
            logger.error(f"[USER_VERIFICATION_NODE] Error in user verification node: {str(e)}")
            return await handle_access_denied(state, conversation_id, "System error during verification")



    # =================================================================
    # MEMORY PREPARATION NODES - Removed in Task 4.11 (Streamlined Architecture)
    # =================================================================

    # NOTE: _employee_memory_prep_node removed in Task 4.11.1
    # This function was unused after graph simplification in Task 2.7
    # Memory preparation is now handled directly in _employee_agent_node

    # NOTE: _customer_memory_prep_node removed in Task 4.11.1  
    # This function was unused after graph simplification in Task 2.7
    # Memory preparation is now handled directly in _customer_agent_node
    
    # NOTE: _is_customer_appropriate_context helper removed in Task 4.11.1
    # Context filtering is now handled by conversation summaries and background tasks

    # =================================================================
    # MEMORY STORAGE NODES - Comprehensive memory storage and consolidation
    # =================================================================

    # NOTE: _employee_memory_store_node removed in Task 4.11.2
    # This function was unused after graph simplification in Task 2.7  
    # Memory storage is now handled via background tasks in _employee_agent_node

    # NOTE: _customer_memory_store_node removed in Task 4.11.2
    # This function was unused after graph simplification in Task 2.7
    # Memory storage is now handled via background tasks in _customer_agent_node
    
    # NOTE: Customer insight and pattern tracking functions removed in Task 4.11.2
    # These functions were unused after graph simplification in Task 2.7:
    # - _extract_customer_appropriate_insights
    # - _track_customer_interaction_patterns  
    # - _analyze_communication_style
    # - _extract_interest_areas
    # Customer insights are now handled by conversation summaries and background tasks

    def _route_after_user_verification(self, state: Dict[str, Any]) -> str:
        """
        Route users directly from verification to the appropriate agent node based on customer_id/employee_id.
        
        NOTE: This function is not currently used in the graph - _route_to_agent is used instead.
        
        Args:
            state: Current agent state containing customer_id and employee_id
            
        Returns:
            str: Node name to route to ('employee_agent', 'customer_agent', or 'end')
        """
        # Route users based on presence of employee_id or customer_id (set by user_verification_node)
        employee_id = state.get("employee_id")
        customer_id = state.get("customer_id")
        user_id = state.get("user_id", "unknown")
        
        logger.info(f"[ROUTING] Routing user {user_id} - employee_id: {employee_id}, customer_id: {customer_id}")
        
        if employee_id:
            logger.info(f"[ROUTING] ✅ Routing employee user (ID: {employee_id}) to employee_agent")
            return "employee_agent"
        elif customer_id:
            logger.info(f"[ROUTING] ✅ Routing customer user (ID: {customer_id}) to customer_agent")
            return "customer_agent"
        else:
            # No employee_id or customer_id means verification failed
            logger.info(f"[ROUTING] No employee_id or customer_id found - user verification failed, ending conversation")
            return "end"

    def _route_to_agent(self, state: Dict[str, Any]) -> str:
        """
        Route users from verification directly to appropriate agent nodes (eliminate memory prep overhead).
        
        Args:
            state: Current agent state containing customer_id and employee_id
            
        Returns:
            str: Node name to route to ('employee_agent', 'customer_agent', or 'end')
        """
        # Route users based on presence of customer_id or employee_id (set by user_verification_node)
        employee_id = state.get("employee_id")
        customer_id = state.get("customer_id")
        user_id = state.get("user_id", "unknown")
        
        logger.info(f"[AGENT_ROUTING] Routing user {user_id} - employee_id: {employee_id}, customer_id: {customer_id}")
        
        if employee_id:
            logger.info(f"[AGENT_ROUTING] ✅ Routing employee user (ID: {employee_id}) directly to employee_agent")
            return "employee_agent"
        elif customer_id:
            logger.info(f"[AGENT_ROUTING] ✅ Routing customer user (ID: {customer_id}) directly to customer_agent")
            return "customer_agent"
        else:
            # No employee_id or customer_id means verification failed
            logger.info(f"[AGENT_ROUTING] No employee_id or customer_id found - user verification failed, ending conversation")
            return "end"

    # NOTE: _route_employee_to_confirmation_or_memory_store() method removed in Task 4.11
    # Replaced with simplified route_from_employee_agent() that uses the new HITL system

    def route_from_employee_agent(self, state: Dict[str, Any]) -> str:
        """
        REVOLUTIONARY: Ultra-simple routing using 3-field HITL architecture.
        
        Route from employee_agent to HITL node or employee memory storage based on hitl_phase.
        This function serves as a conditional edge after the employee_agent node to determine
        if HITL interaction is needed or if processing can continue to memory storage.
        
        Args:
            state: Current agent state containing possible hitl_phase
            
        Returns:
            str: Node name to route to:
                - "hitl_node" if HITL interaction is required (hitl_phase exists)
                - "end" if no HITL needed (background tasks handle persistence)
        """
        employee_id = state.get("employee_id")
        user_id = state.get("user_id", "unknown")
        
        # REVOLUTIONARY: Use ultra-simple 3-field architecture
        hitl_phase = state.get("hitl_phase")
        hitl_prompt = state.get("hitl_prompt")
        hitl_context = state.get("hitl_context")
        
        logger.info(f"[EMPLOYEE_ROUTING] Routing from employee_agent for employee_id: {employee_id}, user_id: {user_id}")
        logger.info(f"[EMPLOYEE_ROUTING] 3-field state: hitl_phase={hitl_phase}, prompt={bool(hitl_prompt)}, context={bool(hitl_context)}")
        
        # Validate this is actually an employee user (safety check)
        if not employee_id:
            logger.warning(f"[EMPLOYEE_ROUTING] No employee_id found in employee routing - this should not happen")
        
        # ULTRA-SIMPLE: Route to hitl_node if prompt exists, or back to agent if approved/denied
        if hitl_prompt:
            # Need to show prompt and get user response
            source_tool = hitl_context.get("source_tool", "unknown") if hitl_context else "unknown"
            logger.info(f"[EMPLOYEE_ROUTING] ✅ HITL prompt needed - tool: {source_tool}")
            return "hitl_node"
        elif hitl_phase in ["approved", "denied"]:
            # Check if this is a tool-managed collection that should continue
            if hitl_phase == "approved" and hitl_context and self._is_tool_managed_collection_needed(hitl_context, state):
                logger.info(f"[EMPLOYEE_ROUTING] ✅ HITL approved for tool-managed collection - routing back to employee_agent")
                return "employee_agent"
            
            # HITL completed, route to end (background tasks handle persistence)
            logger.info(f"[EMPLOYEE_ROUTING] ✅ HITL completed: {hitl_phase} - routing to end")
            return "end"
        
        # No HITL needed, route to end (background tasks handle persistence)
        logger.info(f"[EMPLOYEE_ROUTING] ✅ No HITL needed - routing to end")
        return "end"

    def route_from_hitl(self, state: Dict[str, Any]) -> str:
        """
        REVOLUTIONARY: Ultra-simple non-recursive HITL routing using 3-field architecture.
        
        Route from HITL node ALWAYS back to employee_agent - NEVER routes back to itself.
        This eliminates all HITL recursion complexity and implements the tool-managed
        recursion pattern where tools handle multi-step collection.
        
        Key principle: HITL node NEVER routes back to itself. All recursion is eliminated.
        The employee_agent will detect tool-managed collection and re-call tools as needed.
        
        Args:
            state: Current agent state after HITL processing
            
        Returns:
            str: Always "employee_agent" - HITL never routes to itself
        """
        employee_id = state.get("employee_id")
        user_id = state.get("user_id", "unknown")
        
        # REVOLUTIONARY: Use ultra-simple 3-field architecture for logging
        hitl_phase = state.get("hitl_phase")
        hitl_context = state.get("hitl_context")
        # REVOLUTIONARY: Using ultra-minimal 3-field architecture - no execution_data needed
        
        logger.info(f"[HITL_ROUTING] Routing from HITL for employee_id: {employee_id}, user_id: {user_id}")
        logger.info(f"[HITL_ROUTING] 3-field state: hitl_phase={hitl_phase}, context={bool(hitl_context)}")
        
        # Validate this is an employee user (safety check)
        if not employee_id:
            logger.warning(f"[HITL_ROUTING] No employee_id found in HITL routing - this should not happen")
        
        # REVOLUTIONARY: HITL NEVER routes back to itself - always goes to employee_agent
        # This eliminates all recursion complexity and implements tool-managed collection
        source_tool = hitl_context.get("source_tool", "unknown") if hitl_context else "unknown"
        logger.info(f"[HITL_ROUTING] ✅ HITL interaction processed - phase: {hitl_phase}, tool: {source_tool}")
        logger.info(f"[HITL_ROUTING] ✅ ALWAYS routing to employee_agent (no HITL recursion)")
        
        # The employee_agent will:
        # 1. Detect if tool-managed collection is needed and re-call tools
        # 2. Execute approved actions using ultra-minimal hitl_context
        # 3. Continue normal processing flow
        return "employee_agent"

    def _is_tool_managed_collection_needed(self, hitl_context: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        REVOLUTIONARY: Detect if universal tool-managed collection is needed (Task 14.5.1).
        
        Check if we're coming back from HITL with a context that indicates
        a tool needs to be re-called with user response for collection.
        
        Args:
            hitl_context: Context from HITL interaction
            state: Current agent state
            
        Returns:
            bool: True if tool should be re-called with user response
        """
        try:
            # Check if we have the necessary components for tool re-calling
            if not hitl_context:
                return False
            
            # Normalize state into a plain dict
            state_dict: Dict[str, Any]
            if isinstance(state, dict):
                state_dict = state
            else:
                state_dict = getattr(state, "values", {}) or {}

            # Get basic requirements
            source_tool = hitl_context.get("source_tool")
            hitl_phase = state_dict.get("hitl_phase")
            collection_mode = hitl_context.get("collection_mode")
            
            # Must have all universal requirements
            if not source_tool or hitl_phase not in ["approved", "denied"] or collection_mode != "tool_managed":
                return False
            
            # Universal tool-managed collection detected
            logger.info(f"[UNIVERSAL_HITL] ✅ Universal tool-managed collection detected:")
            logger.info(f"[UNIVERSAL_HITL]   └─ source_tool: {source_tool}")
            logger.info(f"[UNIVERSAL_HITL]   └─ collection_mode: {collection_mode}")
            logger.info(f"[UNIVERSAL_HITL]   └─ hitl_phase: {hitl_phase}")
            logger.info(f"[UNIVERSAL_HITL]   └─ original_params: {list(hitl_context.get('original_params', {}).keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"[UNIVERSAL_HITL] Error detecting tool collection need: {e}")
            return False

    async def _handle_tool_managed_collection(self, hitl_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        REVOLUTIONARY: Handle universal tool-managed collection by re-calling tools (Task 14.5.2).
        
        Re-call the specified tool with original parameters + user response,
        allowing @hitl_recursive_tool decorated tools to manage their own collection.
        
        Args:
            hitl_context: Universal HITL context with original_params and source_tool
            state: Current agent state
            
        Returns:
            Updated state after tool re-calling
        """
        try:
            # Normalize state into a plain dict for safe access
            state_dict: Dict[str, Any]
            if isinstance(state, dict):
                state_dict = state
            else:
                state_dict = getattr(state, "values", {}) or {}

            source_tool = hitl_context.get("source_tool")
            hitl_phase = state_dict.get("hitl_phase")
            messages = state_dict.get("messages", [])
            
            logger.info(f"[TOOL_COLLECTION_RECALL] 🔄 Re-calling tool: {source_tool}")
            logger.info(f"[TOOL_COLLECTION_RECALL] HITL phase: {hitl_phase}")
            
            # Extract the latest human response from messages
            user_response = None
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_response = msg.content
                    break
                elif isinstance(msg, dict) and msg.get('type') == 'human':
                    user_response = msg.get('content')
                    break
            
            if not user_response:
                logger.warning("[TOOL_COLLECTION_RECALL] No user response found in messages")
                user_response = ""
            
            logger.info(f"[TOOL_COLLECTION_RECALL] User response: '{user_response}'")
            
            # Get the tool from the initialized tool map to avoid wrapper mismatches
            await self._ensure_initialized()
            selected_tool = None
            if self.tool_map and source_tool in self.tool_map:
                selected_tool = self.tool_map[source_tool]
            else:
                # Fallback: try to match by underlying function name
                try:
                    for tool in (self.tools or []):
                        func = getattr(tool, 'func', None)
                        func_name = getattr(func, '__name__', None)
                        if func_name == source_tool:
                            selected_tool = tool
                            break
                except Exception:
                    selected_tool = None

            if not selected_tool:
                logger.error(f"[TOOL_COLLECTION_RECALL] Tool '{source_tool}' not found")
                return self._create_tool_recall_error_state(state_dict, f"Tool '{source_tool}' not found")
            
            # UNIVERSAL: Use original parameters + HITL resume parameters
            original_params = hitl_context.get("original_params", {})
            tool_params = original_params.copy()
            tool_params.update({
                "user_response": user_response,
                "hitl_phase": hitl_phase,
                # UNIVERSAL CONTEXT UNDERSTANDING (Task 15.5.3): 
                # Eliminated step-specific parameters in favor of universal context
                "conversation_context": ""  # Could be populated from messages if needed
            })
            
            logger.info(f"[UNIVERSAL_HITL] 🔄 Re-calling {source_tool} with universal parameters:")
            logger.info(f"[UNIVERSAL_HITL]   └─ original_params: {list(original_params.keys())}")
            logger.info(f"[UNIVERSAL_HITL]   └─ user_response: '{user_response}'")
            logger.info(f"[UNIVERSAL_HITL]   └─ hitl_phase: {hitl_phase}")
            logger.info(f"[UNIVERSAL_HITL]   └─ final_tool_params: {tool_params}")
            # Ensure employee context is present during tool execution
            user_id = state_dict.get("user_id")
            conversation_id = state_dict.get("conversation_id")
            employee_id = state_dict.get("employee_id")
            user_type = "employee" if employee_id else "unknown"

            # Prefer ainvoke/invoke on the tool wrapper to avoid .func mismatches
            from agents.toolbox import UserContext
            with UserContext(user_id=user_id, conversation_id=conversation_id, user_type=user_type, employee_id=employee_id):
                if hasattr(selected_tool, 'ainvoke'):
                    tool_result = await selected_tool.ainvoke(tool_params)
                elif hasattr(selected_tool, 'invoke'):
                    tool_result = selected_tool.invoke(tool_params)
                elif hasattr(selected_tool, 'func'):
                    # If underlying func is async, await it; otherwise call directly
                    func = selected_tool.func
                    result_candidate = func(**tool_params)
                    if asyncio.iscoroutine(result_candidate):
                        tool_result = await result_candidate
                    else:
                        tool_result = result_candidate
                else:
                    logger.error(f"[TOOL_COLLECTION_RECALL] Selected tool object has no callable interface: {selected_tool}")
                    return self._create_tool_recall_error_state(state_dict, f"Tool '{source_tool}' is not callable")
            
            # Process the tool result - it might return HITL_REQUIRED again for more collection
            # or return normal results indicating collection is complete
            parsed_response = parse_tool_response(tool_result, source_tool)
            
            if parsed_response["type"] == "hitl_required":
                # Tool still needs more information - set up for another HITL round
                logger.info(f"[TOOL_COLLECTION_RECALL] Tool still needs more info - setting up next HITL round")
                return {
                    **state_dict,
                    "hitl_phase": "needs_prompt",  # Set up for HITL prompt display
                    "hitl_prompt": parsed_response["hitl_prompt"],
                    "hitl_context": parsed_response["hitl_context"],
                    # REVOLUTIONARY: No execution_data needed - hitl_context is cleared when phase changes
                }
            else:
                # Tool collection complete - proceed with normal result
                logger.info(f"[TOOL_COLLECTION_RECALL] ✅ Tool collection complete - processing result")
                # Add tool result to messages and continue normal processing
                tool_message = {
                    "role": "assistant", 
                    "content": tool_result,
                    "type": "ai"
                }
                updated_messages = list(messages) + [tool_message]
                
                return {
                    **state_dict,
                    "messages": updated_messages,
                    "hitl_phase": None,      # Clear HITL state
                    "hitl_prompt": None,
                    "hitl_context": None,
                    # REVOLUTIONARY: No execution_data needed - using 3-field architecture
                }
                
        except Exception as e:
            logger.error(f"[TOOL_COLLECTION_RECALL] Error in tool re-calling: {e}")
            return self._create_tool_recall_error_state(state_dict, str(e))

    def _create_tool_recall_error_state(self, state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        REVOLUTIONARY: Create error state for tool recall failures.
        
        Helper function to create a clean error state when tool re-calling fails.
        """
        error_response = f"I encountered an error while processing your request: {error_message}. Please try again."
        
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": error_response,
            "type": "ai"
        })
        
        return {
            **state,
            "messages": messages,
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None,
                            # REVOLUTIONARY: No execution_data needed - using 3-field architecture
        }

    async def _handle_customer_message_execution(
        self, state: AgentState, hitl_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle customer message execution after confirmation approval.
        
        This consolidates execution within employee_agent for simplicity.
        """
        try:
            logger.info(f"🔍 [MESSAGE_EXECUTION] Starting approved message execution")
            logger.info(f"🔍 [MESSAGE_EXECUTION] HITL context keys: {list(hitl_context.keys())}")
            
            # Extract execution information from HITL context
            customer_info = hitl_context.get("customer_info", {})
            customer_name = customer_info.get("name", "Unknown Customer")
            customer_email = customer_info.get("email", "no-email")
            message_content = hitl_context.get("message_content", "")
            message_type = hitl_context.get("message_type", "follow_up")
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Executing delivery for {customer_name} ({customer_email})")
            logger.info(f"🔍 [MESSAGE_EXECUTION] Message type: {message_type}")
            logger.info(f"🔍 [MESSAGE_EXECUTION] Message content length: {len(message_content)} chars")
            
            # Execute the delivery
            logger.info(f"🔍 [MESSAGE_EXECUTION] Calling _execute_customer_message_delivery")
            delivery_success = await self._execute_customer_message_delivery(
                customer_id=hitl_context.get("customer_id"),
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
            logger.info(f"🔍 [MESSAGE_EXECUTION] Delivery result: {delivery_success}")
            
            # Generate feedback message
            if delivery_success:
                result_status = "delivered"
                feedback_message = f"""✅ **Message Delivered Successfully!**

Your message to {customer_name} has been sent successfully.

**Delivery Summary:**
- Recipient: {customer_name} ({customer_email})
- Message Type: {message_type.title()}
- Status: Delivered
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The customer will receive your message shortly."""
                
                logger.info(f"[EMPLOYEE_AGENT_NODE] SUCCESS: Message delivered to {customer_name}")
                logger.info(f"🔍 [MESSAGE_EXECUTION] Generated success feedback message")
                
            else:
                result_status = "failed"
                feedback_message = f"""❌ **Message Delivery Failed**

Failed to deliver your message to {customer_name}.

**Failure Details:**
- Recipient: {customer_name} ({customer_email})
- Message Type: {message_type.title()}
- Status: Delivery Failed
- Reason: System delivery error"""
                
                logger.error(f"[EMPLOYEE_AGENT_NODE] FAILED: Message delivery to {customer_name}")
                logger.error(f"🔍 [MESSAGE_EXECUTION] Generated failure feedback message")
            
            # Add feedback message to conversation
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=feedback_message))
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Delivery completed with status: {result_status}")
            
            # CRITICAL: Clear all HITL and execution state to prevent re-processing
            # This ensures the conversation is complete and no further tool calls occur
            return {
                "messages": messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                # CRITICAL: Clear HITL state after successful execution to prevent loops
                "hitl_phase": None,
                "hitl_prompt": None,
                "hitl_context": None
            }
            
        except Exception as e:
            logger.error(f"[EMPLOYEE_AGENT_NODE] Execution error: {e}")
            
            return {
                "messages": state.get("messages", []),
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                # CRITICAL: Clear HITL state even on error to prevent loops
                "hitl_phase": None,
                "hitl_prompt": None,
                "hitl_context": None
            }

    # NOTE: Context cache variables removed in Task 4.11.3
    # _user_context_cache and _long_term_context_cache were unused after function removal
    
    # NOTE: Context caching and scheduling functions removed in Task 4.11.3
    # These functions were redundant after simplification to use only messages and conversation summaries:
    # - _get_cached_user_context
    # - _cache_user_context  
    # - _get_cached_long_term_context
    # - _cache_long_term_context
    # - _schedule_context_loading
    # - _schedule_long_term_context_loading
    # Context management is now handled by MemoryManager with its own caching system

    async def _trigger_summary_generation_if_needed(
        self, messages: List, user_id: str, conversation_id: str, is_employee: bool = True
    ):
        """
        Trigger conversation summary generation when message limits are exceeded.
        Uses the background task system for non-blocking summary generation.
        
        Args:
            messages: Current message list
            user_id: User ID
            conversation_id: Conversation ID
            is_employee: Whether this is for an employee user
        """
        try:
            # Get the appropriate threshold
            summary_threshold = (self.settings.memory.summary_threshold 
                               if self.settings and hasattr(self.settings, 'memory') 
                               else 10)
            message_count = len(messages)
            
            logger.info(f"[SUMMARY_CHECK] Message count: {message_count}, threshold: {summary_threshold}")
            
            # Check if we meet the summary threshold
            if message_count >= summary_threshold:
                logger.info(f"[SUMMARY_GENERATION] Triggering summary for conversation {conversation_id} "
                           f"({message_count} messages >= {summary_threshold} threshold)")
                
                # Import background task manager
                from agents.background_tasks import background_task_manager, TaskPriority
                
                # Determine customer_id and employee_id based on user type
                customer_id = None if is_employee else user_id
                employee_id = user_id if is_employee else None
                
                # Schedule summary generation task (non-blocking)
                task_id = background_task_manager.schedule_summary_generation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    customer_id=customer_id,
                    employee_id=employee_id,
                    priority=TaskPriority.NORMAL
                )
                
                logger.info(f"[SUMMARY_GENERATION] Scheduled background summary task: {task_id}")
            else:
                logger.debug(f"[SUMMARY_CHECK] No summary needed ({message_count} < {summary_threshold})")
                
        except Exception as e:
            logger.error(f"[SUMMARY_GENERATION] Error triggering summary generation: {e}")
            # Don't fail the main context loading process if summary scheduling fails

    # NOTE: _load_context_for_employee removed in Task 4.11.3
    # This complex context loading function was replaced by simplified approach:
    # - Context loading now handled directly in _employee_agent_node
    # - Uses conversation summaries instead of complex context retrieval
    # - MemoryManager handles caching with its own optimized system

    async def _get_conversation_context_simple(self, conversation_id: str, user_id: str) -> str:
        """
        LangGraph-native simple conversation context loading with graceful fallback.
        
        This method provides conversation context for system prompt enhancement while
        following LangGraph best practices:
        - Uses conversation summaries as enhancement data, not critical path
        - Graceful degradation when summaries aren't available yet
        - Non-blocking, fast response times maintained
        
        Args:
            conversation_id: Conversation ID for context lookup
            user_id: User ID for context lookup
            
        Returns:
            Simple context string for system prompt enhancement
        """
        try:
            # Try to get existing summary from background task manager
            from agents.background_tasks import background_task_manager
            summary = await background_task_manager.get_conversation_summary(conversation_id)
            
            if summary and summary.strip():
                return f"Previous conversation context: {summary}"
            else:
                # Graceful fallback - no complex loading needed
                return "New conversation - no previous context available."
                
        except Exception as e:
            logger.debug(f"[SIMPLE_CONTEXT] Summary not available yet for {conversation_id}: {e}")
            # Never fail - graceful degradation is key
            return "Conversation context loading in progress..."

    @traceable(name="employee_agent_node")
    async def _employee_agent_node(
        self, state: AgentState, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Employee agent node with full tool access and customer messaging capabilities.
        
        Clean implementation following LangGraph best practices:
        - Tools handle their own logic without interrupt mixing
        - State management is clean and minimal
        - HITL routing is handled by conditional edges only
        - Memory storage for conversation persistence
        """
        try:
            logger.info("[EMPLOYEE_AGENT_NODE] Processing employee user request")
            
            # ULTRA-SIMPLE: Check for approved HITL actions
            hitl_phase = state.get("hitl_phase")
            hitl_context = state.get("hitl_context")
            
            if hitl_phase == "approved" and hitl_context:
                tool_name = hitl_context.get("source_tool")
                logger.info(f"[EMPLOYEE_AGENT_NODE] 🎯 EXECUTING APPROVED ACTION - tool: {tool_name}")
                
                if tool_name == "trigger_customer_message":
                    # Execute the approved action and return immediately
                    execution_result = await self._handle_customer_message_execution(state, hitl_context)
                    logger.info("[EMPLOYEE_AGENT_NODE] ✅ EXECUTION COMPLETE")
                    return execution_result
                else:
                    logger.warning(f"[EMPLOYEE_AGENT_NODE] Unknown approved tool: {tool_name}")
            
            else:
                logger.info("[EMPLOYEE_AGENT_NODE] No approved actions - processing as regular request")

            # REVOLUTIONARY: Check for tool-managed collection mode (Task 8.4)
            # Detect if we need to re-call a tool with user response from HITL
            # (hitl_context already retrieved above)
            if hitl_context and self._is_tool_managed_collection_needed(hitl_context, state):
                logger.info("[EMPLOYEE_AGENT_NODE] 🔄 TOOL-MANAGED COLLECTION detected - re-calling tool with user response")
                tool_recall_result = await self._handle_tool_managed_collection(hitl_context, state)
                return tool_recall_result

            # Get messages and validate this is an employee user
            messages = state.get("messages", [])
            employee_id = state.get("employee_id")
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            
            logger.info(f"🔍 [EMPLOYEE_AGENT_NODE] State keys: {list(state.keys())}")
            logger.info(f"🔍 [EMPLOYEE_AGENT_NODE] Employee ID: {employee_id}")
            logger.info(f"🔍 [EMPLOYEE_AGENT_NODE] Conversation ID: {conversation_id}")
            
            if not employee_id:
                logger.warning(f"[EMPLOYEE_AGENT_NODE] Non-employee user routed to employee node")
                return await handle_access_denied(state, conversation_id, "Employee access required")

            # Set user_type for this employee agent node
            user_type = "employee"
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Processing {len(messages)} messages for employee {employee_id}")

            # SIMPLIFIED: Direct message processing with conversation summaries
            # Context loading functions were removed in Task 4.11.3 for simplification
            trimmed_messages = messages  # Processing with current messages + conversation summary

            # Detect user language from conversation context
            current_query = ""
            for msg in reversed(trimmed_messages):
                if hasattr(msg, "type") and msg.type == "human":
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "human":
                    current_query = msg.get("content", "")
                    break
            
            user_language = detect_user_language_from_context(trimmed_messages, max_messages=10)
            logger.info(f"[EMPLOYEE_AGENT_NODE] Detected user language from context: {user_language}")

            # Get conversation context using LangGraph-native simple approach (Task 4.13.8)
            # Use state.conversation_summary as primary source, lazy load from DB as fallback
            conversation_summary = state.get("conversation_summary", "")
            if not conversation_summary:
                conversation_summary = await self._get_conversation_context_simple(conversation_id, user_id)
            
            # Simple user context building
            user_context = f"Employee user" + (f" working with customer ID: {state.get('customer_id')}" if state.get('customer_id') else "")

            # Create enhanced system prompt with context
            system_prompt = get_employee_system_prompt(
                self.tool_names or [], 
                user_language, 
                conversation_summary=conversation_summary or "",
                user_context=user_context or "",
                memory_manager=self.memory_manager
            )
            
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + trimmed_messages

            # Tool calling loop
            max_tool_iterations = 3
            iteration = 0

            # Get employee-specific tools (full access including generate_quotation)
            employee_tools = get_tools_for_user_type(user_type="employee")
            
            # Create LLM and bind employee tools
            selected_model = self.settings.openai_chat_model if self.settings else "gpt-4o-mini"
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature if self.settings else 0.3,
                max_tokens=self.settings.openai_max_tokens if self.settings else 4000,
                api_key=self.settings.openai_api_key if self.settings else "",
            ).bind_tools(employee_tools)

            logger.info(f"[EMPLOYEE_AGENT_NODE] Using model: {selected_model}")
            logger.info(f"[EMPLOYEE_AGENT_NODE] Employee tools available: {[tool.name for tool in employee_tools]}")

            hitl_required = False  # Flag to track if HITL interaction is needed
            
            while iteration < max_tool_iterations and not hitl_required:
                iteration += 1
                logger.info(f"[EMPLOYEE_AGENT_NODE] Agent iteration {iteration}")

                # LANGGRAPH BEST PRACTICE: Use conversation summarization instead of message trimming
                # Let LangGraph handle message management naturally through add_messages annotation
                # If we have too many messages, the background task system will generate summaries
                # This preserves message object integrity and follows LangGraph patterns
                max_context_messages = getattr(self.settings, 'memory_max_messages', 12)
                
                if len(processing_messages) > max_context_messages:
                    logger.info(f"[EMPLOYEE_AGENT_NODE] 🔄 Large context detected ({len(processing_messages)} messages) - relying on conversation summaries instead of trimming")
                    # Note: Background tasks will handle summarization when message limits are reached
                    # This approach preserves message object integrity and follows LangGraph best practices

                # Call the model
                response = await llm.ainvoke(processing_messages)
                processing_messages.append(response)

                # Check for tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.info(f"Model called {len(response.tool_calls)} tools")

                    # Execute tools
                    for tool_call in response.tool_calls:
                        # Handle both dict and object formats for tool_call
                        if isinstance(tool_call, dict):
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            tool_call_id = tool_call["id"]
                        else:
                            # OpenAI tool_call object with attributes
                            tool_name = tool_call.name
                            tool_args = tool_call.args
                            tool_call_id = tool_call.id

                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                        try:
                            tool = self.tool_map.get(tool_name) if self.tool_map else None
                            if tool:
                                # Set user context
                                user_id = state.get("user_id")
                                conversation_id = state.get("conversation_id")
                                employee_id = state.get("employee_id")
                                
                                # DEBUG: Log context values during agent execution
                                logger.info(f"🔍 [EMPLOYEE_AGENT_CONTEXT] Tool: {tool_name}")
                                logger.info(f"🔍 [EMPLOYEE_AGENT_CONTEXT] user_id: {user_id}")
                                logger.info(f"🔍 [EMPLOYEE_AGENT_CONTEXT] employee_id: {employee_id}")
                                logger.info(f"🔍 [EMPLOYEE_AGENT_CONTEXT] conversation_id: {conversation_id}")
                                
                                if not employee_id:
                                    logger.error(f"🚨 [EMPLOYEE_AGENT_ERROR] employee_id is None during tool execution! State: {state}")

                                with UserContext(
                                    user_id=user_id, 
                                    conversation_id=conversation_id, 
                                    user_type="employee" if employee_id else "unknown",
                                    employee_id=employee_id
                                ):
                                    # Execute tool based on type
                                    if tool_name in ["simple_rag", "simple_query_crm_data", "trigger_customer_message", "collect_sales_requirements", "generate_quotation"]:
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        tool_result = tool.invoke(tool_args)

                                # Parse tool response using standardized HITL detection
                                logger.info(f"🚨 [HITL_DEBUG] Tool {tool_name} returned: {str(tool_result)[:200]}...")
                                parsed_response = parse_tool_response(str(tool_result), tool_name)
                                logger.info(f"🚨 [HITL_DEBUG] Parsed response type: {parsed_response.get('type')}")
                                logger.info(f"🚨 [HITL_DEBUG] Parsed response keys: {list(parsed_response.keys())}")
                                
                                if parsed_response["type"] == "hitl_required":
                                    # ULTRA-SIMPLE: Set HITL state for routing to hitl_node
                                    logger.info(f"[HITL_SIMPLE] ✅ HITL DETECTED! Tool {tool_name} requires HITL")
                                    
                                    # Just set prompt and context - hitl_node will handle the rest
                                    state["hitl_prompt"] = parsed_response["hitl_prompt"] 
                                    state["hitl_context"] = parsed_response["hitl_context"]
                                    state["hitl_phase"] = "needs_confirmation"  # Simple flag for routing
                                    
                                    logger.info(f"[HITL_SIMPLE] Set HITL state - routing to hitl_node")
                                    
                                    # Create user-friendly tool response message
                                    tool_response = parsed_response["hitl_prompt"] or "Human interaction required"
                                    
                                    # Add tool message
                                    tool_message = ToolMessage(
                                        content=tool_response,
                                        tool_call_id=tool_call_id,
                                        name=tool_name,
                                    )
                                    processing_messages.append(tool_message)
                                    
                                    # Set flag to exit both loops - HITL node will handle next steps
                                    hitl_required = True
                                    break
                                else:
                                    # Normal tool response
                                    logger.info(f"🚨 [HITL_DEBUG] ❌ NO HITL DETECTED - Tool {tool_name} response parsed as: {parsed_response.get('type')}")
                                    tool_response = parsed_response["content"]
                                    
                                    # Add tool message
                                    tool_message = ToolMessage(
                                        content=tool_response,
                                        tool_call_id=tool_call_id,
                                        name=tool_name,
                                    )
                                    processing_messages.append(tool_message)

                            else:
                                # Tool not found
                                error_msg = f"Tool '{tool_name}' not found"
                                logger.error(error_msg)
                                tool_message = ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                                processing_messages.append(tool_message)

                        except Exception as e:
                            logger.error(f"[EMPLOYEE_AGENT_NODE] Error executing tool {tool_name}: {e}")
                            error_msg = f"Error executing {tool_name}: {str(e)}"
                            tool_message = ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                            processing_messages.append(tool_message)

                else:
                    # No more tool calls, break the loop
                    break

            # Extract final response
            final_response = processing_messages[-1]
            if hasattr(final_response, 'content'):
                final_content = final_response.content
            else:
                final_content = str(final_response)

            # Update messages with the conversation
            updated_messages = list(messages) + [AIMessage(content=final_content)]

            # Memory storage will be handled by background tasks
            # Prepare return state
            return_state = {
                "messages": updated_messages,
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "employee_id": employee_id,  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                # CRITICAL FIX: Include HITL fields in return state for proper routing
                "hitl_phase": state.get("hitl_phase"),
                "hitl_prompt": state.get("hitl_prompt"),
                "hitl_context": state.get("hitl_context"),
            }

            # HITL system now handles data automatically using 3-field architecture
            
            # Always clear execution state for clean routing    

            # PERFORMANCE: Schedule non-blocking background tasks (Task 2.6) using portable methods
            if final_content and len(final_content.strip()) > 0:
                # Schedule message storage for the AI response using portable method
                self.background_task_manager.schedule_message_from_agent_state(state, final_content, "assistant")
                
                # Check if we need to schedule summary generation (when message limits exceeded)
                message_count = len(updated_messages)
                max_messages = self.settings.memory.employee_max_messages if self.settings else 12
                if message_count >= max_messages:
                    logger.info(f"[EMPLOYEE_AGENT_NODE] Message limit reached ({message_count}/{max_messages}) - scheduling summary generation")
                    self.background_task_manager.schedule_summary_from_agent_state(state)

            logger.info(f"[EMPLOYEE_AGENT_NODE] Complete workflow finished successfully for {user_type}")
            return return_state

        except Exception as e:
            logger.error(f"[EMPLOYEE_AGENT_NODE] Error: {str(e)}")
            return await self._handle_processing_error(state, e)



    async def _handle_processing_error(self, state: AgentState, error: Exception) -> Dict[str, Any]:
        """Handle processing errors with proper error response."""
        error_message = AIMessage(content="I apologize, but I encountered an error while processing your request.")
        messages = list(state.get("messages", []))
        messages.append(error_message)
        
        logger.error(f"Processing error: {error}")
        
        return {
            "messages": messages,
            "sources": state.get("sources", []),
            "retrieved_docs": state.get("retrieved_docs", []),
            "conversation_id": state.get("conversation_id"),
            "user_id": state.get("user_id"),
            "conversation_summary": state.get("conversation_summary"),
            "long_term_context": state.get("long_term_context", []),
            "employee_id": state.get("employee_id"),  # Keep employee_id
            "customer_id": state.get("customer_id"),  # Keep customer_id
            # REVOLUTIONARY: No execution_data needed - using 3-field architecture,
            
        }



    # NOTE: _load_context_for_customer removed in Task 4.11.3
    # This complex context loading function was replaced by simplified approach:
    # - Context loading now handled directly in _customer_agent_node  
    # - Uses conversation summaries instead of complex context retrieval
    # - MemoryManager handles caching with its own optimized system


    @traceable(name="customer_agent_node")
    async def _customer_agent_node(
        self, state: AgentState, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Customer agent node with simplified context management:
        1. Uses current conversation messages and conversation summaries only
        2. Summary generation when message limits are exceeded  
        3. Tool calling with customer-restricted access (vehicles/pricing only)
        4. Customer-appropriate filtering of sensitive information
        
        SIMPLIFICATION:
        - Follows LangGraph best practices for memory management
        - Eliminates complex context loading and caching systems
        - Relies on conversation summaries for personalization
        
        Implements table-level restrictions for customer CRM access (vehicles/pricing only).
        Follows LangGraph best practices for tool calling workflow.
        """
        try:
            # Ensure initialization before processing
            await self._ensure_initialized()

            # Validate user context at node entry
            customer_id = state.get("customer_id")
            user_id = state.get("user_id")
            
            logger.info(f"[CUSTOMER_AGENT_NODE] Starting complete workflow for customer_id: {customer_id}, user_id: {user_id}")
            
            # Customer agent should only process customer users
            if not customer_id:
                logger.warning(f"[CUSTOMER_AGENT_NODE] No customer_id found - customer agent should only process customer users")
                error_message = AIMessage(
                    content="I apologize, but there seems to be a routing error. Please try your request again."
                )
                
                return {
                    "messages": state.get("messages", []) + [error_message],
                    "sources": [],
                    "retrieved_docs": [],
                    "conversation_id": state.get("conversation_id"),
                    "user_id": user_id,
                    "conversation_summary": state.get("conversation_summary"),
                    "long_term_context": state.get("long_term_context", []),
                    "customer_id": state.get("customer_id"),  # Keep customer_id
                    "employee_id": state.get("employee_id"),  # Keep employee_id
                    # REVOLUTIONARY: No execution_data needed - using ultra-minimal 3-field architecture
    
                }

            # SIMPLIFIED: Direct message processing with conversation summaries
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")

            logger.info(f"[CUSTOMER_AGENT_NODE] Processing {len(messages)} messages with customer-specific context management")

            # SIMPLIFIED: Use messages directly instead of complex context loading
            original_messages = messages
            
            # Extract current query for customer-specific processing
            current_query = "No query"
            for msg in reversed(original_messages):
                if hasattr(msg, "type") and msg.type == "human":
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "human":
                    current_query = msg.get("content", "")
                    break

            # Customer-specific context management enhancements

            
            # Detect user language from conversation context
            user_language = detect_user_language_from_context(original_messages, max_messages=10)
            logger.info(f"[CUSTOMER_AGENT_NODE] Detected user language from context: {user_language}")

            # Log thread_id and conversation_id for debugging
            conversation_id = state.get("conversation_id")
            thread_id = (
                config.get("configurable", {}).get("thread_id") if config else None
            )
            logger.info(f"[CUSTOMER_AGENT_NODE] Processing query: {current_query}")
            logger.info(f"[CUSTOMER_AGENT_NODE] Thread ID: {thread_id}, Conversation ID: {conversation_id}")

            # Use customer-specific model selection
            selected_model = self.settings.openai_chat_model if self.settings else "gpt-4o-mini"

            # Get conversation context using LangGraph-native simple approach (Task 4.13.8)
            # Use state.conversation_summary as primary source, lazy load from DB as fallback
            conversation_summary = state.get("conversation_summary", "")
            if not conversation_summary:
                conversation_summary = await self._get_conversation_context_simple(conversation_id, user_id)
            
            # Simple user context building
            user_context = f"Customer user interested in vehicle information" if customer_id else None

            # Get customer-specific system prompt with context enhancement
            system_prompt = get_customer_system_prompt(
                user_language,
                conversation_summary=conversation_summary or "",
                user_context=user_context or "",
                memory_manager=self.memory_manager
            )

            # Build processing messages with customer system prompt
            # Context management now handled by conversation summaries (simplified architecture)
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + original_messages

            # Get customer-specific tools (restricted access)
            customer_tools = get_tools_for_user_type(user_type="customer")
            
            # Create LLM with selected model and bind customer tools
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature if self.settings else 0.3,
                max_tokens=self.settings.openai_max_tokens if self.settings else 4000,
                api_key=self.settings.openai_api_key if self.settings else "",
            ).bind_tools(customer_tools)

            logger.info(f"[CUSTOMER_AGENT_NODE] Using model: {selected_model} for customer query processing")
            logger.info(f"[CUSTOMER_AGENT_NODE] Customer tools available: {[tool.name for tool in customer_tools]}")

            # Tool calling workflow for customers
            max_tool_iterations = 3  
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1
                logger.info(f"[CUSTOMER_AGENT_NODE] Agent iteration {iteration}")

                # LANGGRAPH BEST PRACTICE: Use conversation summarization instead of message trimming
                # Let LangGraph handle message management naturally through add_messages annotation
                # If we have too many messages, the background task system will generate summaries
                # This preserves message object integrity and follows LangGraph patterns
                max_context_messages = getattr(self.settings, 'memory_max_messages', 12)
                
                if len(processing_messages) > max_context_messages:
                    logger.info(f"[CUSTOMER_AGENT_NODE] 🔄 Large context detected ({len(processing_messages)} messages) - relying on conversation summaries instead of trimming")
                    # Note: Background tasks will handle summarization when message limits are reached
                    # This approach preserves message object integrity and follows LangGraph best practices

                # Call the model with customer tools
                response = await llm.ainvoke(processing_messages)
                processing_messages.append(response)

                # Check if model wants to use tools
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.info(f"[CUSTOMER_AGENT_NODE] Model called {len(response.tool_calls)} tools")

                    # Execute the tools with customer restrictions
                    for tool_call in response.tool_calls:
                        # Handle both dict and object formats for tool_call
                        if isinstance(tool_call, dict):
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            tool_call_id = tool_call["id"]
                        else:
                            # OpenAI tool_call object with attributes
                            tool_name = tool_call.name
                            tool_args = tool_call.args
                            tool_call_id = tool_call.id

                        logger.info(f"[CUSTOMER_AGENT_NODE] Executing tool: {tool_name} with args: {tool_args}")

                        try:
                            # Get the tool and execute it
                            customer_tool_map = {tool.name: tool for tool in customer_tools}
                            tool = customer_tool_map.get(tool_name)
                            if tool:
                                # Set user context for customer tools
                                user_id = state.get("user_id")
                                conversation_id = state.get("conversation_id")

                                with UserContext(
                                    user_id=user_id, conversation_id=conversation_id, user_type="customer"
                                ):
                                    # Handle async tools
                                    if tool_name in [
                                        "simple_rag",
                                        "simple_query_crm_data",
                                        "trigger_customer_message",
                                        "collect_sales_requirements",  # REVOLUTIONARY: Tool-managed recursive collection (Task 9.1)
                                        # "gather_further_details", # ELIMINATED - replaced by business-specific recursive collection tools
                                    ]:
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        tool_result = tool.invoke(tool_args)

                                # For customers, handle tool responses directly (no HITL)
                                tool_response = str(tool_result)
                                
                                # Add tool message
                                tool_message = ToolMessage(
                                    content=tool_response,
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                                processing_messages.append(tool_message)

                            else:
                                # Tool not found or not allowed for customers
                                error_msg = f"Tool '{tool_name}' not available for customer users"
                                logger.error(f"[CUSTOMER_AGENT_NODE] {error_msg}")
                                tool_message = ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                                processing_messages.append(tool_message)

                        except GraphInterrupt as e:
                            # Tool triggered human-in-the-loop interrupt (this is expected behavior)
                            logger.info(f"[CUSTOMER_AGENT_NODE] Tool '{tool_name}' triggered human-in-the-loop interrupt: {str(e)}")
                            # Re-raise the interrupt so LangGraph can handle it properly
                            raise e
                        except Exception as e:
                            # Tool execution failed
                            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(f"[CUSTOMER_AGENT_NODE] {error_msg}")
                            logger.error(f"[CUSTOMER_AGENT_NODE] Tool args that caused error: {tool_args}")
                            tool_message = ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                            processing_messages.append(tool_message)

                    # Continue to next iteration to let model process tool results
                    continue
                else:
                    # Model didn't call tools, we have the final response
                    logger.info("[CUSTOMER_AGENT_NODE] Model provided final response without tool calls")
                    break

            # If we exited the loop due to max iterations, ensure we get a final response
            if iteration >= max_tool_iterations:
                logger.info("[CUSTOMER_AGENT_NODE] Max tool iterations reached, getting final response")
                final_response = await llm.ainvoke(processing_messages)
                processing_messages.append(final_response)
                logger.info("[CUSTOMER_AGENT_NODE] Generated final response after tool execution")

            # Sources extraction removed - no longer needed for state bloat reduction

            # Update the original messages with the AI responses (excluding system message)
            updated_messages = list(original_messages)
            for msg in processing_messages[1:]:  # Skip system message
                if hasattr(msg, "type") and msg.type in ["ai", "tool", "human"]:
                    updated_messages.append(msg)

            # Extract thread_id from config if needed for conversation_id
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)

            # Memory storage will be handled by background tasks
            logger.info(f"[CUSTOMER_AGENT_NODE] Agent processing completed successfully")

            # PERFORMANCE: Schedule non-blocking background tasks (Task 2.6) using portable methods
            # Find the latest AI response for message storage
            latest_ai_message = None
            for msg in reversed(updated_messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    latest_ai_message = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "assistant":
                    latest_ai_message = msg.get("content")
                    break
            
            if latest_ai_message and len(latest_ai_message.strip()) > 0:
                # Schedule message storage for the AI response using portable method
                self.background_task_manager.schedule_message_from_agent_state(state, latest_ai_message, "assistant")
                
                # Check if we need to schedule summary generation (when message limits exceeded)
                message_count = len(updated_messages)
                max_messages = self.settings.memory.customer_max_messages if self.settings else 15
                if message_count >= max_messages:
                    logger.info(f"[CUSTOMER_AGENT_NODE] Message limit reached ({message_count}/{max_messages}) - scheduling summary generation")
                    self.background_task_manager.schedule_summary_from_agent_state(state)

            # Return final state (customers don't use HITL)
            return {
                "messages": updated_messages,
                "conversation_id": conversation_id,
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "customer_id": customer_id,  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                # HITL fields preserved (even though customers don't use HITL)
                "hitl_phase": state.get("hitl_phase"),
                "hitl_prompt": state.get("hitl_prompt"),
                "hitl_context": state.get("hitl_context"),
                # REVOLUTIONARY: No execution_data needed - using ultra-minimal 3-field architecture

            }

        # NOTE: GraphInterrupt handling removed since customers don't use HITL
        except Exception as e:
            logger.error(f"[CUSTOMER_AGENT_NODE] Error in customer agent node: {str(e)}")
            # Add error message to the conversation
            original_messages = state.get("messages", [])
            error_message = AIMessage(
                content="I apologize, but I encountered an error while processing your request."
            )
            updated_messages = list(original_messages)
            updated_messages.append(error_message)

            # Extract thread_id from config if needed for conversation_id
            conversation_id = state.get("conversation_id")
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)

            return {
                "messages": updated_messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                # REVOLUTIONARY: No execution_data needed - using ultra-minimal 3-field architecture

            }

    # NOTE: _confirmation_node() method removed - replaced with imported hitl_node from hitl.py
    # The general-purpose HITL system now handles all human interactions through hitl_node

    async def _execute_customer_message_delivery(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> bool:
        """
        Execute the actual customer message delivery via chat APIs and memory systems.
        
        This method implements real delivery by:
        1. Sending via chat API to create a proper conversation flow
        2. Integrating with LangGraph short-term memory (checkpointer)
        3. Integrating with long-term memory (Supabase messages table)
        4. Following LangGraph best practices for memory persistence
        
        Args:
            customer_id: Target customer ID
            message_content: Message content to deliver
            message_type: Type of message (follow_up, information, etc.)
            customer_info: Customer information for delivery
            
        Returns:
            bool: True if delivery successful, False otherwise
        """
        try:
            logger.info(f"[DELIVERY] Executing real delivery for customer {customer_id}")
            
            # =================================================================
            # REAL MESSAGE DELIVERY IMPLEMENTATION
            # =================================================================
            
            # Step 1: Send via chat API for real-time delivery
            chat_delivery_success = await self._send_via_chat_api(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
            # Step 2: Ensure memory integration (LangGraph + Long-term)
            memory_integration_success = await self._integrate_with_memory_systems(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
            # Step 3: Store audit trail (for compliance and tracking)
            audit_success = await self._store_customer_message_audit(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                delivery_status="delivered" if chat_delivery_success else "failed",
                customer_info=customer_info
            )
            
            # Overall success requires both chat delivery and memory integration
            overall_success = chat_delivery_success and memory_integration_success
            
            if overall_success:
                logger.info(f"[DELIVERY] Real message delivery completed successfully for customer {customer_id}")
            else:
                logger.error(f"[DELIVERY] Message delivery partially failed - Chat: {chat_delivery_success}, Memory: {memory_integration_success}")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"[DELIVERY] Message delivery failed for customer {customer_id}: {str(e)}")
            return False

    async def _send_via_chat_api(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> bool:
        """
        Send message via chat API to create real conversation flow.
        
        This leverages the existing chat infrastructure to deliver messages
        in a way that customers can actually receive and respond to.
        """
        try:
            logger.info(f"[CHAT_DELIVERY] Sending message via chat API to {customer_info.get('name', customer_id)}")
            
            # Get the actual customer UUID from customer_info (not the name)
            actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
            if not actual_customer_id:
                logger.error(f"[CHAT_DELIVERY] No valid customer UUID found in customer_info for {customer_info.get('name', customer_id)}")
                return False
            
            logger.info(f"[CHAT_DELIVERY] Using customer UUID: {actual_customer_id} for {customer_info.get('name', customer_id)}")
            
            # Get or create user record for the customer using the correct UUID
            user_id = await self._get_or_create_customer_user(actual_customer_id, customer_info)
            if not user_id:
                logger.error(f"[CHAT_DELIVERY] Failed to get user ID for customer UUID {actual_customer_id}")
                return False
            
            # Create or get existing conversation for this customer
            conversation_id = await self._get_or_create_customer_conversation(user_id, customer_info)
            if not conversation_id:
                logger.error(f"[CHAT_DELIVERY] Failed to get conversation ID for customer {customer_id}")
                return False
            
            # Format the message with proper business context
            formatted_message = await self._format_business_message(
                message_content, message_type, customer_info
            )
            
            # CRITICAL FIX: Actually store the message in the customer's conversation
            # This ensures the customer receives the message in their chat interface
            try:
                message_id = await self.memory_manager.store_message_from_agent(
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
                logger.info(f"[CHAT_DELIVERY] ✅ STORED message in customer conversation: {message_id}")
            except Exception as e:
                logger.error(f"[CHAT_DELIVERY] Failed to store message in customer conversation: {e}")
                return False
            
            if message_id:
                logger.info(f"[CHAT_DELIVERY] Message delivered successfully via chat API - Message ID: {message_id}")
                
                # Optional: Send real-time notification (if notification system exists)
                await self._send_real_time_notification(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_content=formatted_message,
                    customer_info=customer_info
                )
                
                return True
            else:
                logger.error(f"[CHAT_DELIVERY] Failed to store message via chat API for customer {customer_id}")
                return False
                
        except Exception as e:
            logger.error(f"[CHAT_DELIVERY] Error sending via chat API: {str(e)}")
            return False

    async def _integrate_with_memory_systems(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> bool:
        """
        Ensure message integrates with both LangGraph and long-term memory systems.
        
        Following LangGraph best practices:
        1. Messages are stored via memory_manager (handles checkpointer integration)
        2. Conversation state is maintained in LangGraph's short-term memory
        3. Long-term memory gets updated for future context retrieval
        """
        try:
            logger.info(f"[MEMORY_INTEGRATION] Integrating message with memory systems for customer {customer_id}")
            
            # Get the actual customer UUID from customer_info (not the name)
            actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
            if not actual_customer_id:
                logger.error(f"[MEMORY_INTEGRATION] No valid customer UUID found in customer_info for {customer_info.get('name', customer_id)}")
                return False
            
            # Get user and conversation IDs using the correct UUID
            user_id = await self._get_or_create_customer_user(actual_customer_id, customer_info)
            conversation_id = await self._get_or_create_customer_conversation(user_id, customer_info)
            
            if not user_id or not conversation_id:
                logger.error(f"[MEMORY_INTEGRATION] Missing user_id ({user_id}) or conversation_id ({conversation_id})")
                return False
            
            # Create LangGraph-compatible configuration
            langgraph_config = {
                'configurable': {
                    'thread_id': conversation_id,
                    'user_id': user_id,
                    'customer_id': customer_id
                }
            }
            
            # The message was already stored via _send_via_chat_api and handled by background tasks
            # Long-term memory indexing is handled automatically by the memory system
            
            logger.info(f"[MEMORY_INTEGRATION] Successfully integrated with memory systems for customer {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"[MEMORY_INTEGRATION] Error integrating with memory systems: {str(e)}")
            return False

    async def _get_or_create_customer_conversation(
        self, 
        user_id: str, 
        customer_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get or create a conversation for customer message delivery.
        
        This ensures continuity in customer conversations and proper
        integration with the existing chat system.
        """
        try:
            import asyncio
            from core.database import db_client
            import uuid
            
            # Look for existing active conversations for this user (async)
            existing_conversations = await asyncio.to_thread(
                lambda: db_client.client.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
            )
            
            if existing_conversations.data:
                conversation_id = existing_conversations.data[0]["id"]
                logger.info(f"[CONVERSATION] Using existing conversation {conversation_id} for customer {customer_info.get('name', user_id)}")
                
                # Update the conversation timestamp (async)
                await asyncio.to_thread(
                    lambda: db_client.client.table("conversations").update({
                        "updated_at": datetime.now().isoformat()
                    }).eq("id", conversation_id).execute()
                )
                
                return conversation_id
            
            # Create new conversation
            conversation_id = str(uuid.uuid4())
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
            
            # Insert new conversation (async)
            result = await asyncio.to_thread(
                lambda: db_client.client.table("conversations").insert(conversation_data).execute()
            )
            
            if result.data:
                logger.info(f"[CONVERSATION] Created new conversation {conversation_id} for customer {customer_info.get('name', user_id)}")
                return conversation_id
            else:
                logger.error(f"[CONVERSATION] Failed to create conversation for customer {customer_info.get('name', user_id)}")
                return None
                
        except Exception as e:
            logger.error(f"[CONVERSATION] Error getting or creating conversation: {str(e)}")
            return None

    async def _format_business_message(
        self,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> str:
        """
        Format the message with proper business context and personalization.
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
            logger.error(f"[MESSAGE_FORMAT] Error formatting message: {str(e)}")
            return message_content  # Return original content as fallback

    async def _send_real_time_notification(
        self,
        user_id: str,
        conversation_id: str,
        message_content: str,
        customer_info: Dict[str, Any]
    ) -> None:
        """
        Send real-time notification to customer (if notification system exists).
        
        This is optional and can be implemented later for email, SMS, 
        or in-app notifications.
        """
        try:
            # TODO: Implement real-time notifications
            # This could include:
            # - Email notifications
            # - SMS alerts  
            # - In-app push notifications
            # - Webhook calls to external systems
            
            logger.info(f"[NOTIFICATION] Real-time notification placeholder for customer {customer_info.get('name', user_id)}")
            
            # Example placeholder for future implementation:
            # await self._send_email_notification(customer_info, message_content)
            # await self._send_sms_notification(customer_info, message_content)
            # await self._send_webhook_notification(customer_info, message_content)
            
        except Exception as e:
            logger.error(f"[NOTIFICATION] Error sending real-time notification: {str(e)}")



    async def _store_customer_message_audit(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        delivery_status: str,
        customer_info: Dict[str, Any]
    ) -> bool:
        """
        Store audit trail for compliance and tracking purposes.
        
        This is separate from the main message storage and focuses on
        delivery tracking and compliance logging.
        """
        try:
            # This is the existing audit storage, kept for compliance
            await self._store_customer_message(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                delivery_status=delivery_status,
                customer_info=customer_info
            )
            
            logger.info(f"[AUDIT] Stored audit trail for customer {customer_info.get('name', customer_id)}")
            return True
            
        except Exception as e:
            logger.error(f"[AUDIT] Error storing audit trail: {str(e)}")
            return False

    async def _get_or_create_customer_user(self, customer_id: str, customer_info: Dict[str, Any]) -> Optional[str]:
        """
        Get or create a user record for a customer.
        
        Args:
            customer_id: The customer UUID from the customers table
            customer_info: Customer information dictionary
            
        Returns:
            The user_id (UUID) if successful, None otherwise
        """
        try:
            import asyncio
            from core.database import db_client
            
            # First, try to find an existing user for this customer (async)
            existing_user = await asyncio.to_thread(
                lambda: db_client.client.table("users").select("id").eq("customer_id", customer_id).execute()
            )
            
            if existing_user.data:
                # User already exists, return the user_id
                user_id = existing_user.data[0]["id"]
                logger.debug(f"[USER_LOOKUP] Found existing user {user_id} for customer {customer_info.get('name', customer_id)}")
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
            
            # Create the user record (async)
            user_result = await asyncio.to_thread(
                lambda: db_client.client.table("users").insert(user_data).execute()
            )
            
            if user_result.data:
                user_id = user_result.data[0]["id"]
                logger.info(f"[USER_LOOKUP] Created new user {user_id} for customer {customer_info.get('name', customer_id)}")
                return user_id
            else:
                logger.error(f"[USER_LOOKUP] Failed to create user for customer {customer_info.get('name', customer_id)}")
                return None
                
        except Exception as e:
            logger.error(f"[USER_LOOKUP] Error getting or creating user for customer {customer_info.get('name', customer_id)}: {str(e)}")
            return None

    async def _store_customer_message(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        delivery_status: str,
        customer_info: Dict[str, Any] = None
    ):
        """
        Store customer message in database for audit trail and persistence.
        
        This creates a record of all customer messages sent by the system.
        """
        try:
            # Use the actual UUID from customer_info if available
            actual_customer_id = None
            if customer_info:
                actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
            
            customer_identifier = actual_customer_id or customer_id
            
            # This would store in a customer_messages table
            # For now, just log the operation
            logger.info(f"[STORAGE] Storing message record - Customer: {customer_identifier}, Type: {message_type}, Status: {delivery_status}")
            
            # TODO: Implement actual database storage
            # await db_client.client.table("customer_messages").insert({
            #     "customer_id": customer_identifier,
            #     "message_content": message_content,
            #     "message_type": message_type,
            #     "delivery_status": delivery_status,
            #     "sent_at": datetime.now().isoformat()
            # }).execute()
            
        except Exception as e:
            logger.error(f"[STORAGE] Failed to store message record: {str(e)}")
            raise



    def _get_system_prompt(self) -> str:
        """Get the system prompt for the unified agent."""
        return f"""You are a helpful sales assistant with access to company documents, tools, and user context.

Available tools:
{', '.join(self.tool_names)}

Your workflow options:

**Option 1 - Complete RAG Pipeline (RECOMMENDED):**
1. Use simple_rag for complete RAG pipeline (retrieval + generation in one step)
2. This automatically optimizes search queries, retrieves documents, and generates comprehensive responses with source citations

**Option 2 - Step-by-step RAG approach:**
1. Use lcel_retrieval to find relevant documents with query optimization
2. Use lcel_generation to generate responses based on retrieved documents

**Option 3 - CRM Data Queries:**
1. Use simple_query_crm_data for specific business questions about sales, customers, vehicles, pricing, and performance metrics

**Option 4 - Customer Messaging (Employee Only):**
1. Use trigger_customer_message when asked to send messages, follow-ups, or contact customers
2. This tool requires human confirmation before sending messages
3. Common triggers: "send a follow-up to [customer]", "message [customer] about...", "contact [customer]"

Tool Usage Examples:
- simple_rag: {{"question": "user's complete question", "top_k": 5}}

- simple_query_crm_data: {{"question": "specific business question about sales, customers, vehicles, pricing"}}

- trigger_customer_message: {{"customer_id": "customer name, email, or ID", "message_content": "message to send", "message_type": "follow_up"}}

**CRM Database Schema Context:**
Your CRM database contains the following information that you can query using query_crm_data:

**EMPLOYEES & ORGANIZATION:**
- Employee count and details (positions: sales_agent, account_executive, manager, director, admin)
- Branch information (regions: north, south, east, west, central)
- Employee hierarchy and reporting structure

**VEHICLE INVENTORY:**
- Vehicle specifications (brand, model, year, type, color, power, acceleration, fuel_type, transmission)
- Stock quantities and availability status
- Vehicle types: sedan, suv, hatchback, pickup, van, motorcycle, truck
- Available brands: Toyota, Honda, Ford, Nissan, BMW, Mercedes, Audi
- Popular models: Camry, RAV4, Civic, CR-V, F-150, Altima, Prius

**PRICING INFORMATION:**
- Base prices, final prices, and discount amounts
- Insurance and LTO (Land Transport Office) fees
- Warranty information and promotional offers
- Add-on items and their prices

**CUSTOMER DATA:**
- Customer contact information and company details
- Business vs. individual customers
- Customer count and segmentation

**SALES PERFORMANCE:**
- Sales opportunities and pipeline stages (New, Contacted, Consideration, Purchase Intent, Won, Lost)
- Lead warmth levels (hot, warm, cold, dormant)
- Transaction status and revenue tracking
- Sales agent performance and conversion rates

**ACTIVITIES & INTERACTIONS:**
- Customer interaction history (call, email, meeting, demo, follow_up, proposal_sent, contract_signed)
- Activity scheduling and completion tracking

Use query_crm_data for questions about:
- "How many employees/customers/vehicles do we have?"
- "What's the price of [vehicle model]?"
- "How many [vehicle model] are available in inventory?"
- "Show me sales performance"
- "What opportunities are in the pipeline?"
- "Which branches do we have?"

Guidelines:
- Use simple_rag for comprehensive document-based answers (recommended approach)
- Use simple_query_crm_data for specific business data questions
- Use trigger_customer_message when asked to send messages, follow-ups, or contact customers
- Be concise but thorough in your responses
- LCEL tools automatically provide source attribution when documents are found
- If no relevant documents are found, clearly indicate this
- Choose the most appropriate tool based on the user's question type

**Important Context Handling:**
- Previous conversation context is automatically provided in your messages, so you don't need to ask for it
- Long-term context about user preferences and facts is available in the conversation state
- If a user asks a vague question like "How much is it?" or "What about both?", check the previous conversation context in your current message to understand what they're referring to
- For specific product/vehicle questions, use query_crm_data to get current pricing and availability
- Be helpful by suggesting common items they might be asking about if context is unclear
- Use any available long-term context to personalize responses based on user preferences

Important: Use the tools to help you provide the best possible assistance to the user."""

    def _extract_sources_from_messages(self, messages: List) -> List[Dict[str, Any]]:
        """Extract sources from tool call results in messages."""
        sources = []

        for message in messages:
            # Extract sources from LCEL-based RAG tools
            if isinstance(message, ToolMessage) and message.name in ["simple_rag"]:
                try:
                    # LCEL tools return formatted text with source information
                    # Extract sources from the formatted response
                    content = message.content
                    if "Sources (" in content:
                        # Parse sources from lcel_rag output format
                        sources_section = (
                            content.split("Sources (")[1].split("):")[1]
                            if "):" in content
                            else ""
                        )
                        for line in sources_section.split("\n"):
                            if line.strip().startswith("•"):
                                source = line.strip()[1:].strip()
                                if source:
                                    sources.append(
                                        {
                                            "source": source,
                                            "similarity": 0.0,  # LCEL tools don't expose individual similarity scores
                                        }
                                    )
                    elif "**Retrieved Documents (" in content:
                        # Parse sources from lcel_retrieval output format
                        lines = content.split("\n")
                        for line in lines:
                            if line.startswith("Source: "):
                                source = line.replace("Source: ", "").strip()
                                if source:
                                    sources.append(
                                        {"source": source, "similarity": 0.0}
                                    )
                except Exception:
                    continue

        return sources

    async def _register_memory_plugins(self) -> None:
        """Register RAG-specific memory plugins with the memory manager."""

        # Register RAG-specific memory filter
        async def rag_memory_filter(messages: List[BaseMessage]) -> bool:
            """RAG-specific logic for determining if memory should be stored."""
            if len(messages) < 2:
                return False

            # Check if recent messages contain storable information
            recent_messages = messages[-3:] if len(messages) >= 3 else messages

            for message in recent_messages:
                content = str(message.content).lower()

                # RAG-specific patterns (preferences, facts, tool-related insights)
                if any(
                    phrase in content
                    for phrase in [
                        "i prefer",
                        "i like",
                        "i want",
                        "i need",
                        "my name is",
                        "i am",
                        "i work at",
                        "i live in",
                        "remember",
                        "next time",
                        "always",
                        "never",
                        "search for",
                        "find me",
                        "look up",  # RAG-specific patterns
                    ]
                ):
                    return True

            return False

        # Register RAG-specific insight extractor
        async def rag_insight_extractor(
            user_id: str, messages: List[BaseMessage], conversation_id: UUID
        ) -> List[Dict[str, Any]]:
            """RAG-specific logic for extracting insights from conversations."""
            insights = []
            for message in messages:
                if hasattr(message, "type") and message.type == "human":
                    content = message.content.lower()

                    # Extract preferences
                    if any(
                        phrase in content for phrase in ["i prefer", "i like", "i want"]
                    ):
                        insights.append(
                            {
                                "type": "preference",
                                "content": message.content,
                                "extracted_at": datetime.now().isoformat(),
                                "agent_type": "rag",
                            }
                        )

                    # Extract facts
                    elif any(
                        phrase in content
                        for phrase in ["my name is", "i am", "i work at"]
                    ):
                        insights.append(
                            {
                                "type": "personal_fact",
                                "content": message.content,
                                "extracted_at": datetime.now().isoformat(),
                                "agent_type": "rag",
                            }
                        )

                    # Extract RAG-specific search patterns
                    elif any(
                        phrase in content
                        for phrase in ["search for", "find me", "look up"]
                    ):
                        insights.append(
                            {
                                "type": "search_pattern",
                                "content": message.content,
                                "extracted_at": datetime.now().isoformat(),
                                "agent_type": "rag",
                            }
                        )

            return insights

        # Register the plugins
        self.memory_manager.register_memory_filter("rag", rag_memory_filter)
        self.memory_manager.register_insight_extractor("rag", rag_insight_extractor)

        logger.info("RAG-specific memory plugins registered")



    @traceable(name="agent_invoke")
    async def invoke(
        self,
        user_query: str,  # Only accepts string queries now
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_history: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the unified tool-calling RAG agent with thread-based persistence.
        
        This method handles regular string queries only. For resuming interrupted 
        conversations, use resume_interrupted_conversation() instead.
        
        Args:
            user_query: String query from the user
            conversation_id: Conversation ID for persistence
            user_id: User ID for context
            conversation_history: Optional conversation history for new conversations
            config: Optional configuration overrides
        """
        # Ensure initialization before invoking
        await self._ensure_initialized()

        # Validate input - only accept string queries
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        
        query_text = user_query.strip()

        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid4())

        # Set up thread-based config for persistence
        if not config:
            config = await self.memory_manager.get_conversation_config(conversation_id)

        # Try to load existing conversation state from our memory manager
        existing_state = await self.memory_manager.load_conversation_state(
            conversation_id
        )

        # Initialize messages - start with existing messages or empty list
        messages = []

        if existing_state and existing_state.get("messages"):
            # Convert database messages to LangChain message objects
            # Database now uses LangChain-compatible role names directly
            for msg in existing_state.get("messages", []):
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if content:  # Only add messages with actual content
                        role = msg.get("role", "").lower()
                        if role in ["human", "user"]:
                            messages.append(HumanMessage(content=content))
                        elif role in [
                            "ai",
                            "assistant",
                        ]:  # Removed 'bot' since DB now uses 'ai'
                            messages.append(AIMessage(content=content))
                        elif role == "system":
                            messages.append(SystemMessage(content=content))
                elif isinstance(msg, BaseMessage):
                    messages.append(msg)
            logger.info(
                f"Continuing conversation with {len(messages)} existing messages"
            )
        elif conversation_history:
            # Only add conversation_history if this is a new conversation
            for msg in conversation_history:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if content:  # Only add messages with actual content
                        role = msg.get("role", "").lower()
                        if role in ["human", "user"]:
                            messages.append(HumanMessage(content=content))
                        elif role in [
                            "ai",
                            "assistant",
                        ]:  # Removed 'bot' since DB now uses 'ai'
                            messages.append(AIMessage(content=content))
                elif isinstance(msg, BaseMessage):
                    messages.append(msg)

        # Add the current user query as a human message
        messages.append(HumanMessage(content=query_text))

        # Initialize state with existing conversation context
        initial_state = AgentState(
            messages=messages,
            conversation_id=conversation_id,  # Keep as string, not UUID object
            user_id=user_id,
            customer_id=None,  # Will be set by user_verification node
            employee_id=None,  # Will be set by user_verification node
            conversation_summary=(
                existing_state.get("summary") if existing_state else None
            ),
            # REVOLUTIONARY: No execution_data needed - using ultra-minimal 3-field architecture
            
        )

        # Execute the graph with execution-scoped connection management
        from agents.memory import ExecutionScope
        execution_id = f"{user_id}_{conversation_id}_{int(asyncio.get_event_loop().time())}"
        
        with ExecutionScope(execution_id):
            # Background tasks handle persistence without blocking the conversation flow
            result = await self.graph.ainvoke(initial_state, config=config)

        # Memory management (sliding window, summarization, and persistence) is now handled
        # automatically by the graph nodes with checkpointing between each step

        return result

    @traceable(name="agent_resume")
    async def resume_interrupted_conversation(
        self, 
        conversation_id: str, 
        user_response: str
    ) -> Dict[str, Any]:
        """
        Resume an interrupted conversation with user response to HITL prompt
        """
        try:
            logger.info(f"🔄 [AGENT_RESUME] Resuming conversation {conversation_id} with response: {user_response}")
            
            # CRITICAL FIX: Persist human response before resuming
            # This ensures the approval/denial message gets stored in the database
            config = await self.memory_manager.get_conversation_config(conversation_id)
            if config:
                # CRITICAL FIX: Get the actual user_id from the conversation, not from thread_id
                # thread_id is the conversation_id, not the user_id
                actual_user_id = None
                try:
                    # Get user_id from the conversation record in database
                    from core.database import db_client
                    conversation_result = await asyncio.to_thread(
                        lambda: db_client.client.table("conversations")
                        .select("user_id")
                        .eq("id", conversation_id)
                        .execute()
                    )
                    
                    if conversation_result.data:
                        actual_user_id = conversation_result.data[0]["user_id"]
                        logger.info(f"🔄 [AGENT_RESUME] Retrieved correct user_id: {actual_user_id} for conversation {conversation_id}")
                    else:
                        logger.warning(f"🔄 [AGENT_RESUME] Conversation {conversation_id} not found in database")
                        
                except Exception as e:
                    logger.error(f"🔄 [AGENT_RESUME] Error getting user_id from conversation: {e}")
                
                if actual_user_id:
                    # Store the human response message using the correct user_id
                    try:
                        await self.memory_manager.store_message(
                            conversation_id=conversation_id,
                            user_id=actual_user_id,
                            message_text=user_response,
                            message_type="human",
                            agent_type="rag"
                        )
                        logger.info(f"🔄 [AGENT_RESUME] ✅ PERSISTED human response: '{user_response}'")
                    except Exception as e:
                        logger.error(f"🔄 [AGENT_RESUME] Error persisting human response: {e}")
                        # Continue anyway - we'll add it to the state manually
                else:
                    logger.warning(f"🔄 [AGENT_RESUME] Could not get valid user_id, message may not be persisted to DB")
            
            # CRITICAL FIX: Get the current state and add the human response to messages
            current_state = await self.graph.aget_state({"configurable": {"thread_id": conversation_id}})
            
            if current_state and current_state.values:
                # Add the human response to the messages array for HITL processing
                messages = list(current_state.values.get("messages", []))
                
                # Add the human response message to the messages array
                from langchain_core.messages import HumanMessage
                human_response_msg = HumanMessage(content=user_response)
                messages.append(human_response_msg)
                
                logger.info(f"🔄 [AGENT_RESUME] ✅ ADDED human response to messages array: '{user_response}'")
                logger.info(f"🔄 [AGENT_RESUME] Messages array now has {len(messages)} messages")
                
                # CRITICAL FIX: Update the state with the new messages and resume properly
                # Instead of calling ainvoke() which restarts the graph, we need to:
                # 1. Update the state with the human response
                # 2. Clear HITL state to process approval
                # 3. Resume with stream(None, config) to continue from where we left off
                
                # Process the approval using centralized HITL system
                try:
                    from agents.hitl import _interpret_user_intent_with_llm
                    
                    context = {
                        "source_tool": "agent_resume",
                        # UNIVERSAL CONTEXT UNDERSTANDING (Task 15.5.3):
                        # Eliminated step-specific "current_step" in favor of universal context
                        "interaction_type": "approval_request"
                    }
                    
                    user_intent = await _interpret_user_intent_with_llm(user_response, context)
                    
                    # UNIVERSAL HITL SYSTEM: Handle tool-managed collections
                    hitl_context = current_state.values.get("hitl_context", {})
                    is_tool_managed = (hitl_context.get("collection_mode") == "tool_managed")
                    
                    if is_tool_managed:
                        # For tool-managed collections, both "INPUT" and "approval" should continue the tool
                        is_approved = (user_intent in ["approval", "input"])
                        logger.info(f"🧠 [AGENT_RESUME] Tool-managed collection - LLM interpreted '{user_response}' as: {user_intent} → {'CONTINUE' if is_approved else 'DENY'}")
                    else:
                        # For regular HITL, only "approval" continues
                        is_approved = (user_intent == "approval")
                        logger.info(f"🧠 [AGENT_RESUME] Regular HITL - LLM interpreted '{user_response}' as: {user_intent} → {'APPROVED' if is_approved else 'DENIED'}")
                    
                except Exception as e:
                    logger.error(f"🧠 [AGENT_RESUME] Error in LLM interpretation: {e}")
                    # Fallback to keyword matching as backup
                    response_lower = user_response.lower()
                    approve_words = ["yes", "approve", "send", "go ahead", "confirm", "ok", "proceed", "send it", "do it"]
                    is_approved = any(word in response_lower for word in approve_words)
                    logger.warning(f"🧠 [AGENT_RESUME] Falling back to keyword matching: {is_approved}")
                
                # Update the state to include the human response and process approval
                hitl_update = {"messages": [human_response_msg]}
                
                if is_approved:
                    logger.info(f"🔄 [AGENT_RESUME] ✅ Processing approval: '{user_response}'")
                    
                    # EXECUTE APPROVED ACTION DIRECTLY
                    hitl_context = current_state.values.get("hitl_context")
                    if hitl_context and hitl_context.get("source_tool") == "trigger_customer_message":
                        logger.info(f"🔄 [AGENT_RESUME] 🎯 EXECUTING APPROVED ACTION - tool: trigger_customer_message")
                        
                        # Execute the customer message directly
                        execution_result = await self._handle_customer_message_execution(current_state.values, hitl_context)
                        
                        # CRITICAL: Clear HITL state in conversation state to prevent persistence
                        await self.graph.aupdate_state(
                            {"configurable": {"thread_id": conversation_id}},
                            {
                                "hitl_phase": None,
                                "hitl_prompt": None,
                                "hitl_context": None
                            }
                        )
                        
                        # Return the execution result directly (includes success message and cleared HITL state)
                        logger.info(f"🔄 [AGENT_RESUME] ✅ APPROVED ACTION EXECUTED - clearing conversation state")
                        return execution_result
                    
                    hitl_update.update({
                        "hitl_phase": "approved",
                        "hitl_prompt": None,
                        # Keep hitl_context for tool execution
                    })
                else:
                    logger.info(f"🔄 [AGENT_RESUME] ❌ Processing denial: '{user_response}'")
                    
                    # Generate appropriate cancellation message based on context
                    hitl_context = current_state.values.get("hitl_context", {})
                    source_tool = hitl_context.get("source_tool", "action")
                    
                    if source_tool == "generate_quotation":
                        cancellation_message = "I understand. I've cancelled the quotation request. Please let me know if you need anything else."
                    elif source_tool == "trigger_customer_message":
                        cancellation_message = "I understand. I've cancelled the message sending request. No message will be sent."
                    else:
                        cancellation_message = "I understand. I've cancelled the request. Please let me know if you need anything else."
                    
                    # Add cancellation message to conversation
                    cancellation_msg = AIMessage(content=cancellation_message)
                    hitl_update.update({
                        "hitl_phase": "denied", 
                        "hitl_prompt": None,
                        "hitl_context": None,
                        "messages": [human_response_msg, cancellation_msg]  # Include both user response and cancellation
                    })
                
                await self.graph.aupdate_state(
                    {"configurable": {"thread_id": conversation_id}},
                    hitl_update
                )
                
                logger.info(f"🔄 [AGENT_RESUME] ✅ UPDATED state with human response")
                
                # CRITICAL FIX: Instead of trying to resume with astream(None, config) which causes state consistency issues,
                # directly route to the appropriate next node based on the HITL approval/denial
                
                # Determine next node based on routing logic
                hitl_context = current_state.values.get("hitl_context", {})
                hitl_phase = hitl_update.get("hitl_phase", "denied")
                
                # Use the updated state (merge) for detection to avoid stale pre-update values
                merged_state_for_detection = {**current_state.values, **hitl_update}
                if hitl_phase == "approved" and hitl_context and self._is_tool_managed_collection_needed(hitl_context, merged_state_for_detection):
                    logger.info(f"🔄 [AGENT_RESUME] HITL approved for tool-managed collection - resuming with LangGraph stream")
                    
                    # Create complete updated state by merging current state with HITL updates
                    complete_updated_state = {**current_state.values, **hitl_update}
                    
                    result_messages = []
                    last_hitl_phase = None
                    last_hitl_prompt = None
                    last_hitl_context = None
                    async for chunk in self.graph.astream(
                        complete_updated_state,  # Pass the complete updated state
                        {"configurable": {"thread_id": conversation_id}},
                        stream_mode="values"
                    ):
                        if "messages" in chunk:
                            result_messages = chunk["messages"]
                        # Track latest HITL fields so we can return them for correct API handling
                        if "hitl_phase" in chunk:
                            last_hitl_phase = chunk.get("hitl_phase")
                        if "hitl_prompt" in chunk:
                            last_hitl_prompt = chunk.get("hitl_prompt")
                        if "hitl_context" in chunk:
                            last_hitl_context = chunk.get("hitl_context")
                    
                    result = {
                        "messages": result_messages,
                        "conversation_id": conversation_id,
                        "user_id": actual_user_id,
                        # Preserve HITL state so API can return a prompt instead of echoing user text
                        "hitl_phase": last_hitl_phase,
                        "hitl_prompt": last_hitl_prompt,
                        "hitl_context": last_hitl_context,
                    }
                else:
                    logger.info(f"🔄 [AGENT_RESUME] Routing to memory store - HITL completed")
                    
                    # Update the state with our changes
                    updated_state = {**current_state.values, **hitl_update}
                    
                    # If this is a tool-managed flow but detection failed, call tool-managed collection directly to avoid echoing user text
                    if hitl_context and hitl_context.get("collection_mode") == "tool_managed":
                        try:
                            logger.info("🔄 [AGENT_RESUME] Fallback: directly handling tool-managed collection")
                            tool_recall_state = await self._handle_tool_managed_collection(hitl_context, updated_state)
                            result = {
                                "messages": tool_recall_state.get("messages", []),
                                "conversation_id": conversation_id,
                                "user_id": actual_user_id,
                                "hitl_phase": tool_recall_state.get("hitl_phase"),
                                "hitl_prompt": tool_recall_state.get("hitl_prompt"),
                                "hitl_context": tool_recall_state.get("hitl_context"),
                            }
                        except Exception as e:
                            logger.error(f"🔄 [AGENT_RESUME] Fallback tool-managed handling error: {e}")
                            # Schedule background task for final processing as last resort
                            await self._schedule_message_persistence(updated_state)
                            result = {
                                "messages": updated_state.get("messages", []),
                                "conversation_id": conversation_id,
                                "user_id": actual_user_id,
                            }
                    else:
                        # Schedule background task for final processing
                        await self._schedule_message_persistence(updated_state)
                        
                        result = {
                            "messages": updated_state.get("messages", []),
                            "conversation_id": conversation_id,
                            "user_id": actual_user_id,
                        }
                
                logger.info(f"✅ [AGENT_RESUME] Conversation resumed successfully. Result keys: {list(result.keys())}")
                return result
            else:
                logger.error(f"🔄 [AGENT_RESUME] Could not retrieve current state for conversation {conversation_id}")
                return {"error": "Could not retrieve conversation state"}
                
        except Exception as e:
            logger.error(f"❌ [AGENT_RESUME] Error resuming conversation {conversation_id}: {e}")
            return {"error": f"Failed to resume conversation: {str(e)}"}

    # API processing methods now use portable utilities
    async def _process_agent_result_for_api(self, result: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """Use portable API processing utility."""
        return await process_agent_result_for_api(result, conversation_id)
    
    async def process_user_message(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Use portable user message processing utility."""
        return await process_user_message(
            agent_invoke_func=self.invoke,
            user_query=user_query,
            conversation_id=conversation_id,
            user_id=user_id,
            **kwargs
        )


# Update the aliases for easy import
ToolCallingRAGAgent = UnifiedToolCallingRAGAgent
SimpleRAGAgent = UnifiedToolCallingRAGAgent  # For backward compatibility
LinearToolCallingRAGAgent = UnifiedToolCallingRAGAgent  # For backward compatibility

# Export the actual agent's graph for LangGraph Studio
# This ensures Studio always shows the real implementation
_agent_instance = UnifiedToolCallingRAGAgent()


# Create a function to get the graph (lazy loading)
async def get_graph():
    """Get the agent graph, initializing if needed."""
    await _agent_instance._ensure_initialized()
    return _agent_instance.graph


# For LangGraph Studio compatibility, create the graph with persistence
async def create_graph():
    """Create a graph with full persistence for LangGraph Studio."""
    try:
        # Create a new agent instance
        agent = UnifiedToolCallingRAGAgent()

        # Initialize asynchronously with full persistence
        await agent._ensure_initialized()

        return agent.graph

    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        raise


# LangGraph Studio will call create_graph() directly
# No need to initialize graph here - keep it simple

# Export for LangGraph Studio
__all__ = [
    "UnifiedToolCallingRAGAgent",
    "ToolCallingRAGAgent",
    "SimpleRAGAgent",
    "LinearToolCallingRAGAgent",
    "create_graph",
    "get_graph",
]
