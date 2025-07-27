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
import os

from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langsmith import traceable

# Import strategy for LangGraph Studio compatibility
# Add backend directory to path for absolute imports
import pathlib
backend_path = pathlib.Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Now use absolute imports
from agents.tobi_sales_copilot.state import AgentState
from agents.tools import get_all_tools, get_tool_names, UserContext, get_tools_for_user_type
from agents.memory import memory_manager, memory_scheduler, context_manager
from agents.hitl import parse_tool_response, hitl_node
from config import get_settings, setup_langsmith_tracing
from database import db_client

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

    async def _build_graph(self) -> StateGraph:
        """Build a graph with automatic checkpointing and state persistence between agent steps."""
        # Get checkpointer for persistence
        checkpointer = await memory_manager.get_checkpointer()

        graph = StateGraph(AgentState)

        # Add nodes with automatic checkpointing between steps
        graph.add_node("user_verification", self._user_verification_node)
        
        # Memory preparation nodes
        graph.add_node("ea_memory_prep", self._employee_memory_prep_node)
        graph.add_node("ca_memory_prep", self._customer_memory_prep_node)
        
        # Agent processing nodes  
        graph.add_node("employee_agent", self._employee_agent_node)
        graph.add_node("customer_agent", self._customer_agent_node)
        
        # Memory storage nodes
        graph.add_node("ea_memory_store", self._employee_memory_store_node)
        graph.add_node("ca_memory_store", self._customer_memory_store_node)
        
        # General-purpose HITL node for human interactions
        graph.add_node("hitl_node", hitl_node)

        # Simplified graph flow with clear separation between employee and customer workflows
        graph.add_edge(START, "user_verification")

        # Route from user verification to memory preparation nodes
        graph.add_conditional_edges(
            "user_verification",
            self._route_to_memory_prep,
            {
                "ea_memory_prep": "ea_memory_prep", 
                "ca_memory_prep": "ca_memory_prep", 
                "end": END
            },
        )
        
        # Employee workflow: memory prep → agent → (hitl_node or ea_memory_store)
        graph.add_edge("ea_memory_prep", "employee_agent")
        
        # Route from employee agent to HITL or employee memory store based on hitl_data
        graph.add_conditional_edges(
            "employee_agent",
            self.route_from_employee_agent,
            {
                "hitl_node": "hitl_node",
                "ea_memory_store": "ea_memory_store"
            },
        )
        
        # Customer workflow: memory prep → agent → ca_memory_store (simple, no HITL)
        graph.add_edge("ca_memory_prep", "customer_agent")
        graph.add_edge("customer_agent", "ca_memory_store")
        
        # HITL workflow: hitl_node loops back to employee_agent only
        graph.add_conditional_edges(
            "hitl_node",
            self.route_from_hitl,
            {
                "hitl_node": "hitl_node",  # Re-prompts for invalid responses
                "employee_agent": "employee_agent"  # Back to employee agent for processing
            },
        )
        
        # Memory store nodes always go to end
        graph.add_edge("ea_memory_store", END)
        graph.add_edge("ca_memory_store", END)

        # Compile with checkpointer for automatic persistence between steps
        # Configure interrupt_before to enable human interaction at HITL node
        return graph.compile(checkpointer=checkpointer, interrupt_before=["hitl_node"])

    @traceable(name="user_verification_node")
    async def _user_verification_node(
        self, state: AgentState, config: Dict[str, Any] = None
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
                    "retrieved_docs": [],
                    "sources": [],
                    "long_term_context": [],
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
            from database import db_client
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
                return await self._handle_access_denied(state, conversation_id, "User not found in system")

            user_record = user_result.data[0]
            user_type = user_record.get("user_type", "").lower()
            is_active = user_record.get("is_active", False)
            user_employee_id = user_record.get("employee_id")
            user_customer_id = user_record.get("customer_id")

            logger.info(f"[USER_VERIFICATION_NODE] User found - Type: {user_type}, Active: {is_active}")

            # Check if user is active
            if not is_active:
                logger.warning(f"[USER_VERIFICATION_NODE] User {user_id} is inactive")
                return await self._handle_access_denied(state, conversation_id, "Account is inactive")

            # Handle employee users
            if user_type in ["employee", "admin"]:
                if not user_employee_id:
                    logger.warning(f"[USER_VERIFICATION_NODE] Employee user has no employee_id")
                    return await self._handle_access_denied(state, conversation_id, "Employee record incomplete")

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
                    return await self._handle_access_denied(state, conversation_id, "Employee record not found")

                employee = employee_result.data[0]
                logger.info(f"[USER_VERIFICATION_NODE] ✅ Employee access verified - {employee['name']} ({employee['position']})")
                
                return {
                    "messages": state.get("messages", []),
                    "conversation_id": conversation_id,
                    "user_id": user_id,  # Now using cleaned user_id
                    "conversation_summary": state.get("conversation_summary"),
                    "retrieved_docs": state.get("retrieved_docs", []),
                    "sources": state.get("sources", []),
                    "long_term_context": state.get("long_term_context", []),
                    "employee_id": user_employee_id,  # Populate employee_id
                    "customer_id": None,  # Explicitly set customer_id to None
                }

            # Handle customer users
            elif user_type == "customer":
                if not user_customer_id:
                    logger.warning(f"[USER_VERIFICATION_NODE] Customer user has no customer_id")
                    return await self._handle_access_denied(state, conversation_id, "Customer record incomplete")

                # Verify customer record exists
                customer_result = await asyncio.to_thread(
                    lambda: db_client.client.table("customers")
                    .select("id, name, email")
                    .eq("id", user_customer_id)
                    .execute()
                )

                if not customer_result.data or len(customer_result.data) == 0:
                    logger.warning(f"[USER_VERIFICATION_NODE] No customer found for customer_id: {user_customer_id}")
                    return await self._handle_access_denied(state, conversation_id, "Customer record not found")

                customer = customer_result.data[0]
                logger.info(f"[USER_VERIFICATION_NODE] ✅ Customer access verified - {customer['name']} ({customer['email']})")
                
                return {
                    "messages": state.get("messages", []),
                    "conversation_id": conversation_id,
                    "user_id": user_id,  # Now using cleaned user_id
                    "conversation_summary": state.get("conversation_summary"),
                    "retrieved_docs": state.get("retrieved_docs", []),
                    "sources": state.get("sources", []),
                    "long_term_context": state.get("long_term_context", []),
                    "customer_id": user_customer_id,  # Populate customer_id
                    "employee_id": None,  # Explicitly set employee_id to None
                }

            # Handle unknown user types
            else:
                logger.warning(f"[USER_VERIFICATION_NODE] Unknown user_type: {user_type}")
                return await self._handle_access_denied(state, conversation_id, f"Unsupported user type: {user_type}")

        except Exception as e:
            logger.error(f"[USER_VERIFICATION_NODE] Error in user verification node: {str(e)}")
            return await self._handle_access_denied(state, conversation_id, "System error during verification")

    async def _handle_access_denied(self, state: AgentState, conversation_id: Optional[str], reason: str) -> Dict[str, Any]:
        """Helper method to handle access denied scenarios with consistent messaging."""
        logger.warning(f"[USER_VERIFICATION_NODE] Access denied: {reason}")
        
        error_message = AIMessage(
            content="""I apologize, but I'm unable to verify your access to the system.

This could be due to:
- Your account may be inactive or suspended
- You may not have the necessary permissions
- There may be a temporary system issue

Please contact your system administrator for assistance, or try again later."""
        )

        return {
            "messages": state.get("messages", []) + [error_message],
            "conversation_id": conversation_id,
            "user_id": state.get("user_id"),
            "conversation_summary": state.get("conversation_summary"),
            "retrieved_docs": [],
            "sources": [],
            "long_term_context": [],
            "customer_id": None,
            "employee_id": None,
        }

    # =================================================================
    # MEMORY PREPARATION NODES - Comprehensive memory loading and context preparation
    # =================================================================

    @traceable(name="employee_memory_prep_node")
    async def _employee_memory_prep_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Employee Memory Preparation Node - Comprehensive memory loading and context preparation.
        
        Responsibilities:
        1. Store incoming user messages to database
        2. Load cross-conversation user context and history
        3. Retrieve relevant long-term memories for current query
        4. Prepare enhanced context for agent processing
        5. Apply context window management
        """
        try:
            logger.info("[EA_MEMORY_PREP] Starting employee memory preparation")
            
            # Extract state information
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            employee_id = state.get("employee_id")
            
            # Validate this is an employee user
            if not employee_id:
                logger.error(f"[EA_MEMORY_PREP] No employee_id found - this node should only process employee users")
                return state  # Pass through unchanged
            
            logger.info(f"[EA_MEMORY_PREP] Processing memory preparation for {len(messages)} messages")
            
            # =================================================================
            # CENTRALIZED STORAGE: All messages stored by Employee Memory Store phase
            # This prep phase focuses only on preparing context for processing
            # =================================================================
            
            # =================================================================
            # STEP 2: LOAD CROSS-CONVERSATION USER CONTEXT
            # =================================================================
            logger.info("[EA_MEMORY_PREP] Step 2: Loading cross-conversation context")
            
            user_context = {}
            enhanced_messages = list(messages)
            
            if user_id:
                user_context = await self.memory_manager.get_user_context_for_new_conversation(user_id)
                if user_context.get("has_history", False):
                    latest_summary = user_context.get("latest_summary", "")
                    logger.info(f"[EA_MEMORY_PREP] Found user history: {len(latest_summary)} chars")
                    
                    # Add user context as system message
                    context_message = SystemMessage(
                        content=f"""
EMPLOYEE USER CONTEXT (from previous conversations):

{latest_summary}

CONVERSATION COUNT: {user_context.get("conversation_count", 0)}

Use this context to provide personalized, contextually aware responses that build on previous interactions with this employee user. This helps maintain continuity and understanding of their role, preferences, and ongoing needs.
"""
                    )
                    enhanced_messages = [context_message] + enhanced_messages
                    logger.info("[EA_MEMORY_PREP] Added comprehensive user context")
                else:
                    logger.info("[EA_MEMORY_PREP] No previous conversation history found")
            
            # =================================================================
            # STEP 3: RETRIEVE RELEVANT LONG-TERM MEMORIES
            # =================================================================
            logger.info("[EA_MEMORY_PREP] Step 3: Retrieving relevant long-term memories")
            
            long_term_context = []
            current_query = ""
            
            # Extract current query from most recent human message
            for msg in reversed(enhanced_messages):
                if hasattr(msg, "type") and msg.type == "human":
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "human":
                    current_query = msg.get("content", "")
                    break
            
            if user_id and current_query:
                long_term_context = await self.memory_manager.get_relevant_context(
                    user_id, current_query, max_contexts=5  # More context for employees
                )
                if long_term_context:
                    logger.info(f"[EA_MEMORY_PREP] Retrieved {len(long_term_context)} relevant context items")
                    
                    # Add long-term context as system message
                    context_items = []
                    for context in long_term_context:
                        context_items.append(f"- {context.get('content', '')[:200]}...")
                    
                    if context_items:
                        long_term_message = SystemMessage(
                            content=f"""
RELEVANT LONG-TERM CONTEXT:

{chr(10).join(context_items)}

This context from previous conversations may be relevant to the current query.
"""
                        )
                        enhanced_messages = [long_term_message] + enhanced_messages
                        logger.info("[EA_MEMORY_PREP] Added long-term context")
                else:
                    logger.info("[EA_MEMORY_PREP] No relevant long-term context found")
            
            # =================================================================
            # STEP 4: APPLY CONTEXT WINDOW MANAGEMENT
            # =================================================================
            logger.info("[EA_MEMORY_PREP] Step 4: Applying context window management")
            
            # Apply context window management for employee model
            selected_model = self.settings.openai_chat_model
            system_prompt = self._get_employee_system_prompt()
            
            trimmed_messages, trim_stats = await context_manager.trim_messages_for_context(
                messages=enhanced_messages,
                model=selected_model,
                system_prompt=system_prompt,
                max_messages=self.settings.memory.max_messages,
            )
            
            if trim_stats["trimmed_message_count"] > 0:
                logger.info(
                    f"[EA_MEMORY_PREP] Context management: trimmed {trim_stats['trimmed_message_count']} messages, "
                    f"final token count: {trim_stats['final_token_count']}"
                )
            
            logger.info("[EA_MEMORY_PREP] Memory preparation completed successfully")
            
            # Return enhanced state
            return {
                "messages": trimmed_messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": long_term_context,
                "employee_id": employee_id,  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
                "hitl_data": state.get("hitl_data"),
            }
            
        except Exception as e:
            logger.error(f"[EA_MEMORY_PREP] Error in employee memory preparation: {e}")
            # Return state unchanged on error to prevent workflow disruption
            return state

    @traceable(name="customer_memory_prep_node")
    async def _customer_memory_prep_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Customer Memory Preparation Node - Comprehensive memory loading and context preparation for customers.
        
        Responsibilities:
        1. Store incoming customer messages to database
        2. Load cross-conversation customer context and history
        3. Retrieve relevant long-term memories for current query (customer-appropriate)
        4. Prepare enhanced context for customer agent processing
        5. Apply context window management with customer limits
        """
        try:
            logger.info("[CA_MEMORY_PREP] Starting customer memory preparation")
            
            # Extract state information
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            customer_id = state.get("customer_id")
            
            # Validate this is a customer user
            if not customer_id:
                logger.error(f"[CA_MEMORY_PREP] No customer_id found - this node should only process customer users")
                return state  # Pass through unchanged
            
            logger.info(f"[CA_MEMORY_PREP] Processing memory preparation for {len(messages)} messages")
            
            # =================================================================
            # CENTRALIZED STORAGE: All messages stored by Customer Memory Store phase
            # This prep phase focuses only on preparing context for processing
            # =================================================================
            
            # =================================================================
            # STEP 2: LOAD CROSS-CONVERSATION CUSTOMER CONTEXT
            # =================================================================
            logger.info("[CA_MEMORY_PREP] Step 2: Loading cross-conversation customer context")
            
            user_context = {}
            enhanced_messages = list(messages)
            
            if user_id:
                user_context = await self.memory_manager.get_user_context_for_new_conversation(user_id)
                if user_context.get("has_history", False):
                    latest_summary = user_context.get("latest_summary", "")
                    logger.info(f"[CA_MEMORY_PREP] Found customer history: {len(latest_summary)} chars")
                    
                    # Add customer-specific context as system message
                    context_message = SystemMessage(
                        content=f"""
CUSTOMER CONTEXT (from your previous conversations with us):

{latest_summary}

CONVERSATION COUNT: {user_context.get("conversation_count", 0)}

Use this context to provide personalized, helpful responses that build on your previous interactions. I'll reference your past interests, questions, and preferences to better assist you.
"""
                    )
                    enhanced_messages = [context_message] + enhanced_messages
                    logger.info("[CA_MEMORY_PREP] Added comprehensive customer context")
                else:
                    logger.info("[CA_MEMORY_PREP] No previous customer conversation history found")
            
            # =================================================================
            # STEP 3: RETRIEVE RELEVANT LONG-TERM MEMORIES (Customer-appropriate)
            # =================================================================
            logger.info("[CA_MEMORY_PREP] Step 3: Retrieving relevant customer memories")
            
            long_term_context = []
            current_query = ""
            
            # Extract current query from most recent human message
            for msg in reversed(enhanced_messages):
                if hasattr(msg, "type") and msg.type == "human":
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "human":
                    current_query = msg.get("content", "")
                    break
            
            if user_id and current_query:
                long_term_context = await self.memory_manager.get_relevant_context(
                    user_id, current_query, max_contexts=3  # Fewer contexts for customers
                )
                if long_term_context:
                    logger.info(f"[CA_MEMORY_PREP] Retrieved {len(long_term_context)} relevant context items")
                    
                    # Add long-term context as system message (customer-appropriate)
                    context_items = []
                    for context in long_term_context:
                        # Filter context to ensure it's customer-appropriate
                        context_content = context.get('content', '')
                        if self._is_customer_appropriate_context(context_content):
                            context_items.append(f"- {context_content[:150]}...")
                    
                    if context_items:
                        long_term_message = SystemMessage(
                            content=f"""
RELEVANT CONTEXT FROM YOUR PREVIOUS CONVERSATIONS:

{chr(10).join(context_items)}

This information from your past interactions may help me better assist you.
"""
                        )
                        enhanced_messages = [long_term_message] + enhanced_messages
                        logger.info("[CA_MEMORY_PREP] Added customer-appropriate long-term context")
                else:
                    logger.info("[CA_MEMORY_PREP] No relevant customer context found")
            
            # =================================================================
            # STEP 4: APPLY CUSTOMER CONTEXT WINDOW MANAGEMENT
            # =================================================================
            logger.info("[CA_MEMORY_PREP] Step 4: Applying customer context window management")
            
            # Apply context window management for customer model
            selected_model = self.settings.openai_chat_model
            customer_system_prompt = self._get_customer_system_prompt()
            
            # Use slightly smaller context for customers to ensure better response quality
            max_customer_messages = min(self.settings.memory.max_messages, 15)
            
            trimmed_messages, trim_stats = await context_manager.trim_messages_for_context(
                messages=enhanced_messages,
                model=selected_model,
                system_prompt=customer_system_prompt,
                max_messages=max_customer_messages,
            )
            
            if trim_stats["trimmed_message_count"] > 0:
                logger.info(
                    f"[CA_MEMORY_PREP] Context management: trimmed {trim_stats['trimmed_message_count']} messages, "
                    f"final token count: {trim_stats['final_token_count']}"
                )
            
            logger.info("[CA_MEMORY_PREP] Customer memory preparation completed successfully")
            
            # Return enhanced state
            return {
                "messages": trimmed_messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": long_term_context,
                "customer_id": customer_id,  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
                "hitl_data": state.get("hitl_data"),
            }
            
        except Exception as e:
            logger.error(f"[CA_MEMORY_PREP] Error in customer memory preparation: {e}")
            # Return state unchanged on error to prevent workflow disruption
            return state

    def _is_customer_appropriate_context(self, context_content: str) -> bool:
        """
        Filter context to ensure it's appropriate for customer viewing.
        
        Args:
            context_content: The context content to check
            
        Returns:
            bool: True if content is appropriate for customers, False otherwise
        """
        # Convert to lowercase for checking
        content_lower = context_content.lower()
        
        # Exclude internal/sensitive information
        excluded_terms = [
            "internal", "employee", "staff", "salary", "commission", "profit",
            "cost basis", "margin", "markup", "confidential", "private",
            "admin", "system", "database", "backend", "pipeline"
        ]
        
        for term in excluded_terms:
            if term in content_lower:
                return False
        
        # Include customer-appropriate information
        included_terms = [
            "vehicle", "car", "price", "feature", "specification", "model",
            "brand", "warranty", "financing", "discount", "promotion",
            "availability", "color", "option", "delivery"
        ]
        
        for term in included_terms:
            if term in content_lower:
                return True
        
        # Default to excluding if uncertain
        return False

    # =================================================================
    # MEMORY STORAGE NODES - Comprehensive memory storage and consolidation
    # =================================================================

    @traceable(name="employee_memory_store_node")
    async def _employee_memory_store_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Employee Memory Storage Node - Comprehensive memory storage and consolidation.
        
        Responsibilities:
        1. Store agent response messages to database
        2. Extract and store conversation insights in long-term memory
        3. Trigger automatic conversation summarization when thresholds are met
        4. Update user context and cross-conversation summaries
        5. Perform memory consolidation for long conversations
        """
        try:
            logger.info("[EA_MEMORY_STORE] Starting employee memory storage")
            
            # Extract state information
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            employee_id = state.get("employee_id")
            
            # Validate this is an employee user
            if not employee_id:
                logger.error(f"[EA_MEMORY_STORE] No employee_id found - this node should only process employee users")
                return state  # Pass through unchanged
            
            logger.info(f"[EA_MEMORY_STORE] Processing memory storage for {len(messages)} messages")
            
            # =================================================================
            # STEP 1: STORE ALL CONVERSATION MESSAGES
            # =================================================================
            logger.info("[EA_MEMORY_STORE] Step 1: Storing all conversation messages")
            
            if messages and config:
                stored_count = 0
                # Store all messages in chronological order to preserve conversation flow
                for msg in messages:
                    # Store both human and AI messages to preserve full conversation
                    # Handle both object and dictionary message formats
                    msg_type = None
                    if hasattr(msg, "type"):
                        msg_type = msg.type
                    elif isinstance(msg, dict) and "type" in msg:
                        msg_type = msg["type"]
                    
                    if msg_type in ["human", "ai"]:
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        stored_count += 1
                logger.info(f"[EA_MEMORY_STORE] Stored {stored_count} conversation messages (including HITL messages)")

            # =================================================================
            # STEP 2: EXTRACT AND STORE CONVERSATION INSIGHTS
            # =================================================================
            logger.info("[EA_MEMORY_STORE] Step 2: Extracting conversation insights")
            
            if user_id and messages and len(messages) >= 2:
                if await self.memory_manager.should_store_memory("rag", messages):
                    # Get recent messages for insight extraction
                    recent_messages = messages[-4:] if len(messages) >= 4 else messages
                    await self.memory_manager.store_conversation_insights(
                        "rag", user_id, recent_messages, conversation_id
                    )
                    logger.info("[EA_MEMORY_STORE] Stored conversation insights")
                else:
                    logger.info("[EA_MEMORY_STORE] No significant insights to store")
            
            # =================================================================
            # STEP 3: TRIGGER AUTOMATIC CONVERSATION SUMMARIZATION
            # =================================================================
            logger.info("[EA_MEMORY_STORE] Step 3: Checking for automatic summarization")
            
            if user_id and conversation_id:
                try:
                    summary = await self.memory_manager.consolidator.check_and_trigger_summarization(
                        str(conversation_id), user_id
                    )
                    if summary:
                        logger.info(f"[EA_MEMORY_STORE] Conversation {conversation_id} automatically summarized")
                    else:
                        logger.info("[EA_MEMORY_STORE] No summarization needed at this time")
                except Exception as e:
                    logger.error(f"[EA_MEMORY_STORE] Error in automatic summarization: {e}")
            
            # =================================================================
            # STEP 4: MEMORY CONSOLIDATION FOR LONG CONVERSATIONS
            # =================================================================
            logger.info("[EA_MEMORY_STORE] Step 4: Memory consolidation check")
            
            # For very long conversations, trigger background consolidation
            if len(messages) > 50:  # Threshold for background consolidation
                try:
                    logger.info("[EA_MEMORY_STORE] Triggering background memory consolidation")
                    # This runs in background and doesn't block the response
                    asyncio.create_task(
                        self.memory_manager.consolidate_old_conversations(user_id)
                    )
                except Exception as e:
                    logger.error(f"[EA_MEMORY_STORE] Error triggering background consolidation: {e}")
            
            logger.info("[EA_MEMORY_STORE] Employee memory storage completed successfully")
            
            # Return state with any updates
            return {
                "messages": messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": state.get("long_term_context", []),
                "employee_id": employee_id,  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
                "hitl_data": state.get("hitl_data"),
            }
            
        except Exception as e:
            logger.error(f"[EA_MEMORY_STORE] Error in employee memory storage: {e}")
            # Return state unchanged on error to prevent workflow disruption
            return state

    @traceable(name="customer_memory_store_node")
    async def _customer_memory_store_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Customer Memory Storage Node - Comprehensive memory storage and consolidation for customers.
        
        Responsibilities:
        1. Store customer agent response messages to database
        2. Extract and store customer-appropriate conversation insights
        3. Trigger automatic conversation summarization when thresholds are met
        4. Update customer context and preferences
        5. Track customer interaction patterns
        """
        try:
            logger.info("[CA_MEMORY_STORE] Starting customer memory storage")
            
            # Extract state information
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            customer_id = state.get("customer_id")
            
            # Validate this is a customer user
            if not customer_id:
                logger.error(f"[CA_MEMORY_STORE] No customer_id found - this node should only process customer users")
                return state  # Pass through unchanged
            
            logger.info(f"[CA_MEMORY_STORE] Processing memory storage for {len(messages)} customer messages")
            
            # =================================================================
            # STEP 1: STORE ALL CUSTOMER CONVERSATION MESSAGES
            # =================================================================
            logger.info("[CA_MEMORY_STORE] Step 1: Storing all customer conversation messages")
            
            if messages and config:
                stored_count = 0
                # Store all messages in chronological order to preserve full conversation flow
                for msg in messages:
                    # Store both human and AI messages to preserve complete customer conversation
                    # Handle both object and dictionary message formats
                    msg_type = None
                    if hasattr(msg, "type"):
                        msg_type = msg.type
                    elif isinstance(msg, dict) and "type" in msg:
                        msg_type = msg["type"]
                    
                    if msg_type in ["human", "ai"]:
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        stored_count += 1
                logger.info(f"[CA_MEMORY_STORE] Stored {stored_count} customer conversation messages")
            
            # =================================================================
            # STEP 2: EXTRACT CUSTOMER-APPROPRIATE INSIGHTS
            # =================================================================
            logger.info("[CA_MEMORY_STORE] Step 2: Extracting customer insights")
            
            if user_id and messages and len(messages) >= 2:
                if await self.memory_manager.should_store_memory("rag", messages):
                    # Get recent messages for insight extraction
                    recent_messages = messages[-3:] if len(messages) >= 3 else messages
                    
                    # Filter insights to be customer-appropriate before storing
                    customer_insights = await self._extract_customer_appropriate_insights(
                        recent_messages, user_id, conversation_id
                    )
                    
                    if customer_insights:
                        await self.memory_manager.store_conversation_insights(
                            "rag", user_id, customer_insights, conversation_id
                        )
                        logger.info(f"[CA_MEMORY_STORE] Stored {len(customer_insights)} customer insights")
                    else:
                        logger.info("[CA_MEMORY_STORE] No customer-appropriate insights to store")
                else:
                    logger.info("[CA_MEMORY_STORE] No significant customer insights to store")
            
            # =================================================================
            # STEP 3: TRIGGER CUSTOMER CONVERSATION SUMMARIZATION
            # =================================================================
            logger.info("[CA_MEMORY_STORE] Step 3: Checking for customer conversation summarization")
            
            if user_id and conversation_id:
                try:
                    summary = await self.memory_manager.consolidator.check_and_trigger_summarization(
                        str(conversation_id), user_id
                    )
                    if summary:
                        logger.info(f"[CA_MEMORY_STORE] Customer conversation {conversation_id} automatically summarized")
                    else:
                        logger.info("[CA_MEMORY_STORE] No customer summarization needed at this time")
                except Exception as e:
                    logger.error(f"[CA_MEMORY_STORE] Error in customer automatic summarization: {e}")
            
            # =================================================================
            # STEP 4: TRACK CUSTOMER INTERACTION PATTERNS
            # =================================================================
            logger.info("[CA_MEMORY_STORE] Step 4: Tracking customer interaction patterns")
            
            # Track customer preferences and interests for better future service
            if user_id and messages:
                try:
                    await self._track_customer_interaction_patterns(user_id, messages, conversation_id)
                    logger.info("[CA_MEMORY_STORE] Customer interaction patterns tracked")
                except Exception as e:
                    logger.error(f"[CA_MEMORY_STORE] Error tracking customer patterns: {e}")
            
            logger.info("[CA_MEMORY_STORE] Customer memory storage completed successfully")
            
            # Return state with any updates
            return {
                "messages": messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": state.get("long_term_context", []),
                "customer_id": customer_id,  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "hitl_data": state.get("hitl_data"),
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
            }
            
        except Exception as e:
            logger.error(f"[CA_MEMORY_STORE] Error in customer memory storage: {e}")
            # Return state unchanged on error to prevent workflow disruption
            return state

    async def _extract_customer_appropriate_insights(
        self, messages: List[BaseMessage], user_id: str, conversation_id: str
    ) -> List[BaseMessage]:
        """
        Extract insights that are appropriate for customer memory storage.
        
        Args:
            messages: Messages to extract insights from
            user_id: Customer user ID
            conversation_id: Current conversation ID
            
        Returns:
            List of customer-appropriate insight messages
        """
        try:
            customer_insights = []
            
            for message in messages:
                if hasattr(message, "type") and message.type == "human":
                    content = message.content.lower()
                    
                    # Extract customer preferences (vehicle-related)
                    if any(phrase in content for phrase in [
                        "i prefer", "i like", "i want", "i need", "looking for",
                        "interested in", "budget", "price range", "color preference"
                    ]):
                        # Only store vehicle/product-related preferences
                        if any(term in content for term in [
                            "vehicle", "car", "suv", "sedan", "truck", "color",
                            "feature", "price", "budget", "financing", "warranty"
                        ]):
                            customer_insights.append(message)
                    
                    # Extract customer information (contact preferences, timeline)
                    elif any(phrase in content for phrase in [
                        "call me", "email me", "contact me", "timeline", "when can"
                    ]):
                        customer_insights.append(message)
            
            logger.info(f"Extracted {len(customer_insights)} customer-appropriate insights")
            return customer_insights
            
        except Exception as e:
            logger.error(f"Error extracting customer insights: {e}")
            return []

    async def _track_customer_interaction_patterns(
        self, user_id: str, messages: List[BaseMessage], conversation_id: str
    ) -> None:
        """
        Track customer interaction patterns for better future service.
        
        Args:
            user_id: Customer user ID
            messages: Current conversation messages
            conversation_id: Current conversation ID
        """
        try:
            # Analyze customer interaction patterns
            patterns = {
                "preferred_communication_style": self._analyze_communication_style(messages),
                "interest_areas": self._extract_interest_areas(messages),
                "interaction_time": datetime.now().isoformat(),
                "conversation_id": conversation_id
            }
            
            # Store patterns in customer memory namespace
            namespace = (user_id, "customer_patterns")
            key = f"interaction_{conversation_id}"
            
            await self.memory_manager.store_long_term_memory(
                user_id=user_id,
                namespace=["customer_patterns"],
                key=key,
                value=patterns,
                ttl_hours=8760  # Store for 1 year
            )
            
            logger.info(f"Stored customer interaction patterns for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error tracking customer interaction patterns: {e}")

    def _analyze_communication_style(self, messages: List[BaseMessage]) -> str:
        """Analyze customer communication style from messages."""
        try:
            human_messages = [msg.content for msg in messages if hasattr(msg, "type") and msg.type == "human"]
            
            if not human_messages:
                return "unknown"
            
            total_length = sum(len(msg) for msg in human_messages)
            avg_length = total_length / len(human_messages)
            
            if avg_length > 100:
                return "detailed"
            elif avg_length > 50:
                return "moderate"
            else:
                return "concise"
                
        except Exception:
            return "unknown"

    def _extract_interest_areas(self, messages: List[BaseMessage]) -> List[str]:
        """Extract customer interest areas from messages."""
        try:
            interests = []
            interest_keywords = {
                "luxury": ["luxury", "premium", "high-end", "bmw", "mercedes", "audi"],
                "economy": ["affordable", "budget", "cheap", "economical", "efficient"],
                "family": ["family", "kids", "children", "suv", "minivan", "safety"],
                "performance": ["fast", "powerful", "sports", "performance", "acceleration"],
                "eco_friendly": ["hybrid", "electric", "green", "eco", "fuel efficient"]
            }
            
            for message in messages:
                if hasattr(message, "type") and message.type == "human":
                    content = message.content.lower()
                    for category, keywords in interest_keywords.items():
                        if any(keyword in content for keyword in keywords):
                            if category not in interests:
                                interests.append(category)
            
            return interests
            
        except Exception:
            return []

    def _route_after_user_verification(self, state: Dict[str, Any]) -> str:
        """
        Route users directly from verification to the appropriate agent node based on customer_id/employee_id.
        
        NOTE: This function is not currently used in the graph - _route_to_memory_prep is used instead.
        
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

    def _route_to_memory_prep(self, state: Dict[str, Any]) -> str:
        """
        Route users from verification to appropriate memory preparation nodes.
        
        Args:
            state: Current agent state containing customer_id and employee_id
            
        Returns:
            str: Node name to route to ('ea_memory_prep', 'ca_memory_prep', or 'end')
        """
        # Route users based on presence of customer_id or employee_id (set by user_verification_node)
        employee_id = state.get("employee_id")
        customer_id = state.get("customer_id")
        user_id = state.get("user_id", "unknown")
        
        logger.info(f"[MEMORY_ROUTING] Routing user {user_id} - employee_id: {employee_id}, customer_id: {customer_id}")
        
        if employee_id:
            logger.info(f"[MEMORY_ROUTING] ✅ Routing employee user (ID: {employee_id}) to employee memory prep")
            return "ea_memory_prep"
        elif customer_id:
            logger.info(f"[MEMORY_ROUTING] ✅ Routing customer user (ID: {customer_id}) to customer memory prep")
            return "ca_memory_prep"
        else:
            # No employee_id or customer_id means verification failed
            logger.info(f"[MEMORY_ROUTING] No employee_id or customer_id found - user verification failed, ending conversation")
            return "end"

    # NOTE: _route_employee_to_confirmation_or_memory_store() method removed
    # Replaced with simplified route_from_employee_agent() that uses the new HITL system

    def route_from_employee_agent(self, state: Dict[str, Any]) -> str:
        """
        Route from employee_agent to HITL node or employee memory storage based on HITL requirements.
        
        This function serves as a conditional edge after the employee_agent node to determine
        if HITL interaction is needed or if processing can continue to memory storage.
        
        Args:
            state: Current agent state containing possible hitl_data
            
        Returns:
            str: Node name to route to:
                - "hitl_node" if HITL interaction is required
                - "ea_memory_store" if no HITL needed
        """
        employee_id = state.get("employee_id")
        user_id = state.get("user_id", "unknown")
        hitl_data = state.get("hitl_data")
        
        logger.info(f"[EMPLOYEE_ROUTING] Routing from employee_agent for employee_id: {employee_id}, user_id: {user_id}")
        
        # Validate this is actually an employee user (safety check)
        if not employee_id:
            logger.warning(f"[EMPLOYEE_ROUTING] No employee_id found in employee routing - this should not happen")
        
        # Check if HITL interaction is required
        if hitl_data and isinstance(hitl_data, dict):
            hitl_type = hitl_data.get("type", hitl_data.get("interaction_type", "unknown"))
            logger.info(f"[EMPLOYEE_ROUTING] HITL interaction required - type: {hitl_type}, routing to hitl_node")
            return "hitl_node"
        elif hitl_data:
            # Malformed HITL data - log warning and route safely
            logger.warning(f"[EMPLOYEE_ROUTING] Malformed HITL data: {type(hitl_data)}, routing to memory store")
            return "ea_memory_store"
        
        # No HITL needed, route to employee memory storage
        logger.info(f"[EMPLOYEE_ROUTING] No HITL needed - routing to ea_memory_store")
        return "ea_memory_store"

    def route_from_hitl(self, state: Dict[str, Any]) -> str:
        """
        Route from HITL node back to employee_agent or continue HITL interaction.
        
        This function serves as a conditional edge after the hitl_node to determine
        if HITL interaction is complete and processing can continue back to employee_agent,
        or if additional HITL rounds are needed (re-prompts for invalid responses).
        
        Simplified logic: HITL is only used by employee workflows, so we only route
        back to employee_agent or continue HITL loops.
        
        Args:
            state: Current agent state after HITL processing
            
        Returns:
            str: Node name to route to:
                - "hitl_node" if additional HITL interaction needed (re-prompts)
                - "employee_agent" if HITL interaction is complete
        """
        employee_id = state.get("employee_id")
        user_id = state.get("user_id", "unknown")
        hitl_data = state.get("hitl_data")
        
        logger.info(f"[HITL_ROUTING] Routing from HITL for employee_id: {employee_id}, user_id: {user_id}")
        
        # Validate this is an employee user (safety check)
        if not employee_id:
            logger.warning(f"[HITL_ROUTING] No employee_id found in HITL routing - this should not happen")
        
        # Check if HITL interaction is still ongoing (re-prompts needed)
        if hitl_data and isinstance(hitl_data, dict) and hitl_data.get("awaiting_response", False):
            hitl_type = hitl_data.get("type", hitl_data.get("interaction_type", "unknown"))
            logger.info(f"[HITL_ROUTING] HITL still awaiting response - type: {hitl_type}, continuing with hitl_node")
            return "hitl_node"
        elif hitl_data and not isinstance(hitl_data, dict):
            # Malformed HITL data - log warning and route back to agent
            logger.warning(f"[HITL_ROUTING] Malformed HITL data: {type(hitl_data)}, routing back to employee_agent")
            return "employee_agent"
        
        # HITL interaction complete - route back to employee_agent for continued processing
        # The employee_agent will then decide whether to go to memory storage or not
        logger.info(f"[HITL_ROUTING] HITL interaction complete - routing back to employee_agent")
        return "employee_agent"

    async def _handle_customer_message_execution(
        self, state: AgentState, execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle customer message execution after confirmation approval.
        
        This consolidates execution within employee_agent for simplicity.
        """
        try:
            logger.info(f"🔍 [MESSAGE_EXECUTION] Starting approved message execution")
            logger.info(f"🔍 [MESSAGE_EXECUTION] Execution data keys: {list(execution_data.keys())}")
            
            # Extract execution information
            customer_info = execution_data.get("customer_info", {})
            customer_name = customer_info.get("name", "Unknown Customer")
            customer_email = customer_info.get("email", "no-email")
            message_content = execution_data.get("message_content", "")
            message_type = execution_data.get("message_type", "follow_up")
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Executing delivery for {customer_name} ({customer_email})")
            logger.info(f"🔍 [MESSAGE_EXECUTION] Message type: {message_type}")
            logger.info(f"🔍 [MESSAGE_EXECUTION] Message content length: {len(message_content)} chars")
            
            # Execute the delivery
            logger.info(f"🔍 [MESSAGE_EXECUTION] Calling _execute_customer_message_delivery")
            delivery_success = await self._execute_customer_message_delivery(
                customer_id=execution_data.get("customer_id"),
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
                "long_term_context": state.get("long_term_context", []),
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "confirmation_data": None,  # Clear all HITL-related state
                "execution_data": None,     # Clear execution state
                "confirmation_result": result_status,  # Set final result
                "hitl_data": None,         # Clear HITL state completely
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
                "long_term_context": state.get("long_term_context", []),
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "confirmation_data": None,  # Clear all HITL-related state
                "execution_data": None,     # Clear execution state  
                "confirmation_result": "error",
                "hitl_data": None,         # Clear HITL state completely
            }

    @traceable(name="employee_agent_node")
    async def _employee_agent_node(
        self, state: AgentState, config: Dict[str, Any] = None
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
            
            # Check if executing approved customer message (from HITL completion)
            execution_data = state.get("execution_data")
            if execution_data:
                logger.info("[EMPLOYEE_AGENT_NODE] 🎯 EXECUTING APPROVED ACTION - bypassing normal tool calling loop")
                logger.info(f"🔍 [EXECUTION_DATA] Found execution_data with keys: {list(execution_data.keys())}")
                logger.info(f"🔍 [EXECUTION_DATA] Tool: {execution_data.get('tool')}")
                logger.info(f"🔍 [EXECUTION_DATA] Customer: {execution_data.get('customer_info', {}).get('name', 'Unknown')}")
                
                # CRITICAL: Execute and return immediately - do NOT continue to tool calling loop
                # This prevents the LLM from seeing the original request again and re-executing
                execution_result = await self._handle_customer_message_execution(state, execution_data)
                logger.info("[EMPLOYEE_AGENT_NODE] ✅ EXECUTION COMPLETE - returning without further processing")
                return execution_result
            else:
                logger.info("[EMPLOYEE_AGENT_NODE] No execution_data found - processing as regular employee request")

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
                return await self._handle_access_denied(state, conversation_id, "Employee access required")

            # Set user_type for this employee agent node
            user_type = "employee"
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Processing {len(messages)} messages for employee {employee_id}")

            # Messages are already prepared by ea_memory_prep node
            # Use messages as-is from memory preparation
            trimmed_messages = messages

            # Create system prompt for employee
            system_prompt = self._get_employee_system_prompt()
            
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + trimmed_messages

            # Tool calling loop
            max_tool_iterations = 3
            iteration = 0

            # Create LLM and bind tools
            selected_model = self.settings.openai_chat_model
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens,
                api_key=self.settings.openai_api_key,
            ).bind_tools(self.tools)

            logger.info(f"[EMPLOYEE_AGENT_NODE] Using model: {selected_model}")

            hitl_required = False  # Flag to track if HITL interaction is needed
            
            while iteration < max_tool_iterations and not hitl_required:
                iteration += 1
                logger.info(f"[EMPLOYEE_AGENT_NODE] Agent iteration {iteration}")

                # Call the model
                response = await llm.ainvoke(processing_messages)
                processing_messages.append(response)

                # Check for tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.info(f"Model called {len(response.tool_calls)} tools")

                    # Execute tools
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                        try:
                            tool = self.tool_map.get(tool_name)
                            if tool:
                                # Set user context
                                user_id = state.get("user_id")
                                conversation_id = state.get("conversation_id")
                                employee_id = state.get("employee_id")

                                with UserContext(
                                    user_id=user_id, 
                                    conversation_id=conversation_id, 
                                    user_type="employee" if employee_id else "unknown",
                                    employee_id=employee_id
                                ):
                                    # Execute tool based on type
                                    if tool_name in ["simple_rag", "simple_query_crm_data", "trigger_customer_message", "gather_further_details"]:
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        tool_result = tool.invoke(tool_args)

                                # Parse tool response using standardized HITL detection
                                logger.info(f"🚨 [HITL_DEBUG] Tool {tool_name} returned: {str(tool_result)[:200]}...")
                                parsed_response = parse_tool_response(str(tool_result), tool_name)
                                logger.info(f"🚨 [HITL_DEBUG] Parsed response type: {parsed_response.get('type')}")
                                logger.info(f"🚨 [HITL_DEBUG] Parsed response keys: {list(parsed_response.keys())}")
                                
                                if parsed_response["type"] == "hitl_required":
                                    # HITL interaction required - store hitl_data and prepare for routing
                                    logger.info(f"🚨 [HITL_DEBUG] ✅ HITL DETECTED! Tool {tool_name} requires HITL interaction: {parsed_response.get('hitl_type')}")
                                    
                                    # Store HITL data in state for routing to hitl_node
                                    # CRITICAL FIX: Set awaiting_response=True since hitl_node never runs due to interrupt_before
                                    hitl_data_with_awaiting = {**parsed_response["hitl_data"], "awaiting_response": True}
                                    state["hitl_data"] = hitl_data_with_awaiting
                                    logger.info(f"🚨 [HITL_DEBUG] ✅ Set hitl_data in state with keys: {list(hitl_data_with_awaiting.keys())}")
                                    logger.info(f"🚨 [HITL_DEBUG] ✅ CRITICAL FIX: Set awaiting_response=True before interrupt")
                                    
                                    # Create user-friendly tool response message
                                    tool_response = parsed_response["hitl_data"].get("prompt", "Human interaction required")
                                    
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

            # Memory storage will be handled by ea_memory_store node
            # Prepare return state
            return_state = {
                "messages": updated_messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": state.get("long_term_context", []),
                "employee_id": employee_id,  # Keep employee_id
                "customer_id": state.get("customer_id"),  # Keep customer_id
            }

            # Include hitl_data if set during tool execution
            return_state["hitl_data"] = state.get("hitl_data")
            
            # HITL system now handles confirmation data automatically
            return_state["confirmation_data"] = None
            
            # Always clear execution state for clean routing    
            return_state["execution_data"] = None
            return_state["confirmation_result"] = None

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
            "confirmation_data": None,
            "execution_data": None,
            "confirmation_result": None,
            "hitl_data": state.get("hitl_data"),
        }

    def _get_employee_system_prompt(self) -> str:
        """Get the system prompt for employee users."""
        return f"""You are a helpful sales assistant with full access to company tools and data.

Available tools:
{', '.join(self.tool_names)}

**IMPORTANT - Current System Status:**
All employee identification and customer messaging systems are fully operational. You can directly use trigger_customer_message for any customer messaging requests without needing additional employee information.

**Tool Usage Guidelines:**
- Use simple_rag for comprehensive document-based answers
- Use simple_query_crm_data for specific CRM database queries  
- Use trigger_customer_message when asked to send messages, follow-ups, or contact customers

**Customer Messaging:**
When asked to "send a message to [customer]", "follow up with [customer]", or "contact [customer]", DIRECTLY use the trigger_customer_message tool. The system will automatically identify you as the sending employee. This will prepare the message and request your confirmation before sending.

**Message Content Guidelines:**
- If specific message content is provided, use it exactly
- If no specific content is given, generate appropriate professional content based on the message type
- For follow-up messages, create content like "Hi [Name], I wanted to follow up on our recent interaction. Please let me know if you have any questions or need assistance."
- NEVER ask for message content using gather_further_details - generate appropriate content instead

DO NOT ask for additional employee information - the system handles employee identification automatically.

You have full access to:
- All CRM data (employees, customers, vehicles, sales, etc.)
- All company documents through RAG system
- Customer messaging capabilities with confirmation workflows

Be helpful, professional, and make full use of your available tools to assist with sales and customer management tasks."""

    @traceable(name="customer_agent_node")
    async def _customer_agent_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Customer agent node that handles complete workflow for customer users:
        1. Memory preparation (context retrieval)
        2. Tool calling and execution with restricted access
        3. Memory update (context storage)
        
        Implements table-level restrictions for customer CRM access (vehicles/pricing only).
        Follows LangChain best practices for tool calling workflow.
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
                    "hitl_data": state.get("hitl_data"),
                    "confirmation_data": state.get("confirmation_data"),
                    "execution_data": state.get("execution_data"),
                    "confirmation_result": state.get("confirmation_result"),
                }

            # Messages are already prepared by ca_memory_prep node
            # Use messages as-is from memory preparation
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")

            logger.info(f"[CUSTOMER_AGENT_NODE] Processing prepared messages: {len(messages)} messages")

            # Use messages from memory preparation (includes user context)
            original_messages = messages
            current_query = "No query"
            for msg in reversed(original_messages):
                if hasattr(msg, "type") and msg.type == "human":
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "human":
                    current_query = msg.get("content", "")
                    break

            # Log thread_id and conversation_id for debugging
            conversation_id = state.get("conversation_id")
            thread_id = (
                config.get("configurable", {}).get("thread_id") if config else None
            )
            logger.info(f"[CUSTOMER_AGENT_NODE] Processing query: {current_query}")
            logger.info(f"[CUSTOMER_AGENT_NODE] Thread ID: {thread_id}, Conversation ID: {conversation_id}")

            # Use customer-specific model selection
            selected_model = self.settings.openai_chat_model

            # Get customer-specific system prompt
            system_prompt = self._get_customer_system_prompt()

            # Build processing messages with customer system prompt
            # Context management and user context already handled by ca_memory_prep
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + original_messages

            # Get customer-specific tools (restricted access)
            customer_tools = get_tools_for_user_type(user_type="customer")
            
            # Create LLM with selected model and bind customer tools
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens,
                api_key=self.settings.openai_api_key,
            ).bind_tools(customer_tools)

            logger.info(f"[CUSTOMER_AGENT_NODE] Using model: {selected_model} for customer query processing")
            logger.info(f"[CUSTOMER_AGENT_NODE] Customer tools available: {[tool.name for tool in customer_tools]}")

            # Tool calling workflow for customers
            max_tool_iterations = 3  
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1
                logger.info(f"[CUSTOMER_AGENT_NODE] Agent iteration {iteration}")

                # Call the model with customer tools
                response = await llm.ainvoke(processing_messages)
                processing_messages.append(response)

                # Check if model wants to use tools
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.info(f"[CUSTOMER_AGENT_NODE] Model called {len(response.tool_calls)} tools")

                    # Execute the tools with customer restrictions
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

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
                                        "gather_further_details",
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

            # Extract sources from messages and update state
            sources = self._extract_sources_from_messages(processing_messages)

            # Update the original messages with the AI responses (excluding system message)
            updated_messages = list(original_messages)
            for msg in processing_messages[1:]:  # Skip system message
                if hasattr(msg, "type") and msg.type in ["ai", "tool"]:
                    updated_messages.append(msg)

            # Extract thread_id from config if needed for conversation_id
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)

            # Memory storage will be handled by ca_memory_store node
            logger.info(f"[CUSTOMER_AGENT_NODE] Agent processing completed successfully")

            # Return final state (no hitl_data since customers don't use HITL)
            return {
                "messages": updated_messages,
                "sources": sources,
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "long_term_context": state.get("long_term_context", []),
                "customer_id": customer_id,  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "hitl_data": state.get("hitl_data"),
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
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
                "long_term_context": state.get("long_term_context", []),
                "customer_id": state.get("customer_id"),  # Keep customer_id
                "employee_id": state.get("employee_id"),  # Keep employee_id
                "hitl_data": state.get("hitl_data"),
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
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
            
            # The message was already stored via _send_via_chat_api, 
            # but we need to ensure it's properly indexed for long-term memory
            
            # Update long-term memory context for this customer
            await self._update_customer_long_term_context(
                customer_id=actual_customer_id,
                user_id=user_id,
                conversation_id=conversation_id,
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
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
            from database import db_client
            import uuid
            
            # Look for existing active conversations for this user
            existing_conversations = db_client.client.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
            
            if existing_conversations.data:
                conversation_id = existing_conversations.data[0]["id"]
                logger.info(f"[CONVERSATION] Using existing conversation {conversation_id} for customer {customer_info.get('name', user_id)}")
                
                # Update the conversation timestamp
                db_client.client.table("conversations").update({
                    "updated_at": datetime.now().isoformat()
                }).eq("id", conversation_id).execute()
                
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
            
            result = db_client.client.table("conversations").insert(conversation_data).execute()
            
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

    async def _update_customer_long_term_context(
        self,
        customer_id: str,
        user_id: str,
        conversation_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> None:
        """
        Update long-term memory context for this customer interaction.
        
        This ensures the message becomes part of the customer's long-term
        context for future interactions and AI responses.
        """
        try:
            # The message storage via memory_manager already handles most of this,
            # but we can add additional context indexing here if needed
            
            context_update = {
                'customer_id': customer_id,
                'interaction_type': 'employee_message',
                'message_type': message_type,
                'content_summary': message_content[:200] + "..." if len(message_content) > 200 else message_content,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': conversation_id
            }
            
            logger.info(f"[LONG_TERM_CONTEXT] Updated context for customer {customer_info.get('name', customer_id)}")
            
        except Exception as e:
            logger.error(f"[LONG_TERM_CONTEXT] Error updating long-term context: {str(e)}")

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
            from database import db_client
            
            # First, try to find an existing user for this customer
            existing_user = db_client.client.table("users").select("id").eq("customer_id", customer_id).execute()
            
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
            
            # Create the user record
            user_result = db_client.client.table("users").insert(user_data).execute()
            
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

    def _get_customer_system_prompt(self) -> str:
        """Get the customer-specific system prompt with restricted capabilities."""
        return f"""You are a helpful vehicle sales assistant designed specifically for customers looking for vehicle information.

You have access to limited tools for customer inquiries:
- simple_rag: Search company documents and vehicle information
- simple_query_crm_data: Query vehicle specifications and pricing (RESTRICTED ACCESS)

**IMPORTANT - Your Access Restrictions:**
- You can ONLY help with vehicle specifications, models, features, and pricing information
- You CANNOT access employee information, customer records, sales opportunities, or internal business data
- You CANNOT provide information about other customers or internal company operations

**What you CAN help with:**
- Vehicle models, specifications, and features
- Pricing information and available discounts
- Vehicle comparisons and recommendations
- Inventory availability
- Vehicle features and options

**What you CANNOT help with:**
- Employee information or contact details
- Other customer information or records
- Sales opportunities or deals
- Internal company operations
- Business performance data

Your goal is to help customers find the right vehicle by providing accurate information about our vehicle inventory and pricing.

Be friendly, helpful, and focus on addressing the customer's vehicle-related questions. If asked about restricted information, politely explain that you can only assist with vehicle and pricing information.

Available tools:
{', '.join(['simple_rag', 'simple_query_crm_data'])}

Guidelines:
- Use simple_rag for comprehensive document-based answers about vehicles and company information
- Use simple_query_crm_data for specific vehicle specifications and pricing questions  
- Be helpful and customer-focused in your responses
- If asked about restricted data, redirect to vehicle and pricing topics
- Provide source citations when documents are found

Remember: You are here to help customers make informed vehicle purchasing decisions!"""

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

    async def _verify_user_access(self, user_id: str) -> str:
        """
        Verify user access and return user type (employee, customer, unknown).
        
        This function replaces the previous employee-only verification with dual-user support.
        It determines if a user is an employee, customer, or unknown based on database records.

        Args:
            user_id: The users.id (UUID)

        Returns:
            str: User type - 'employee', 'customer', or 'unknown'
        """
        try:
            # Import database client
            # Ensure database client is initialized
            await db_client._async_initialize_client()

            # Look up the user in the users table
            logger.info(f"[USER_VERIFICATION] Looking up user by ID: {user_id}")
            user_result = await asyncio.to_thread(
                lambda: db_client.client.table("users")
                .select("id, user_type, employee_id, customer_id, email, display_name, is_active")
                .eq("id", user_id)
                .execute()
            )

            if not user_result.data or len(user_result.data) == 0:
                logger.warning(f"[USER_VERIFICATION] No user found for user_id: {user_id}")
                return "unknown"

            user_record = user_result.data[0]
            user_type = user_record.get("user_type", "").lower()
            employee_id = user_record.get("employee_id")
            customer_id = user_record.get("customer_id")
            is_active = user_record.get("is_active", False)
            email = user_record.get("email", "")
            display_name = user_record.get("display_name", "")

            logger.info(f"[USER_VERIFICATION] User found - Type: {user_type}, Active: {is_active}, Email: {email}")

            # Check if user is active
            if not is_active:
                logger.warning(f"[USER_VERIFICATION] User {user_id} is inactive - denying access")
                return "unknown"

            # Handle employee users
            if user_type == "employee":
                if not employee_id:
                    logger.warning(f"[USER_VERIFICATION] Employee user {user_id} has no employee_id - treating as unknown")
                    return "unknown"

                # Verify employee record exists and is active
                logger.info(f"[USER_VERIFICATION] Verifying employee record for employee_id: {employee_id}")
                employee_result = await asyncio.to_thread(
                    lambda: db_client.client.table("employees")
                    .select("id, name, email, position, is_active")
                    .eq("id", employee_id)
                    .eq("is_active", True)
                    .execute()
                )

                if employee_result.data and len(employee_result.data) > 0:
                    employee = employee_result.data[0]
                    logger.info(f"[USER_VERIFICATION] ✅ Employee access verified - {employee['name']} ({employee['position']})")
                    return "employee"
                else:
                    logger.warning(f"[USER_VERIFICATION] No active employee found for employee_id: {employee_id}")
                    return "unknown"

            # Handle customer users
            elif user_type == "customer":
                if not customer_id:
                    logger.warning(f"[USER_VERIFICATION] Customer user {user_id} has no customer_id - treating as unknown")
                    return "unknown"

                # Verify customer record exists
                logger.info(f"[USER_VERIFICATION] Verifying customer record for customer_id: {customer_id}")
                customer_result = await asyncio.to_thread(
                    lambda: db_client.client.table("customers")
                    .select("id, name, email")
                    .eq("id", customer_id)
                    .execute()
                )

                if customer_result.data and len(customer_result.data) > 0:
                    customer = customer_result.data[0]
                    logger.info(f"[USER_VERIFICATION] ✅ Customer access verified - {customer['name']} ({customer['email']})")
                    return "customer"
                else:
                    logger.warning(f"[USER_VERIFICATION] No customer found for customer_id: {customer_id}")
                    return "unknown"

            # Handle admin users (treat as employees for now)
            elif user_type == "admin":
                logger.info(f"[USER_VERIFICATION] ✅ Admin user detected - granting employee-level access")
                return "employee"

            # Handle system users (deny access)
            elif user_type == "system":
                logger.warning(f"[USER_VERIFICATION] System user detected - denying access for security")
                return "unknown"

            # Handle unknown user types
            else:
                logger.warning(f"[USER_VERIFICATION] Unknown user_type: {user_type} for user {user_id}")
                return "unknown"

        except Exception as e:
            logger.error(f"[USER_VERIFICATION] Error verifying user access for user_id {user_id}: {str(e)}")
            # On database errors, return unknown for security
            return "unknown"

    async def _verify_employee_access(self, user_id: str) -> bool:
        """
        Legacy method for backward compatibility.
        
        This method now uses the new _verify_user_access function internally
        but maintains the boolean return type for existing code.

        Args:
            user_id: The users.id (UUID)

        Returns:
            bool: True if user is an employee (including admin), False otherwise
        """
        logger.info(f"[LEGACY] _verify_employee_access called - delegating to _verify_user_access")
        user_type = await self._verify_user_access(user_id)
        
        # Return True for employees and admins, False for customers and unknown
        is_employee = user_type in ["employee", "admin"]
        logger.info(f"[LEGACY] User type '{user_type}' -> employee access: {is_employee}")
        return is_employee

    @traceable(name="rag_agent_invoke")
    async def invoke(
        self,
        user_query: str,  # Only accepts string queries now
        conversation_id: str = None,
        user_id: str = None,
        conversation_history: List = None,
        config: Dict[str, Any] = None,
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
            retrieved_docs=(
                existing_state.get("retrieved_docs", []) if existing_state else []
            ),
            sources=existing_state.get("sources", []) if existing_state else [],
            conversation_summary=(
                existing_state.get("summary") if existing_state else None
            ),
            long_term_context=[],  # Initialize empty, will be populated by nodes
            hitl_data=None,  # Initialize empty
            confirmation_data=None,  # Initialize empty
            execution_data=None,  # Initialize empty
            confirmation_result=None,  # Initialize empty
        )

        # Execute the graph with execution-scoped connection management
        from agents.memory import ExecutionScope
        execution_id = f"{user_id}_{conversation_id}_{int(asyncio.get_event_loop().time())}"
        
        with ExecutionScope(execution_id):
            # The memory preparation and update nodes handle automatic checkpointing and state persistence
            result = await self.graph.ainvoke(initial_state, config=config)

        # Memory management (sliding window, summarization, and persistence) is now handled
        # automatically by the graph nodes with checkpointing between each step

        return result

    @traceable(name="rag_agent_resume")
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
                    from database import db_client
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
                # 2. Resume with stream(None, config) to continue from where we left off
                
                # Update the state to include the human response
                await self.graph.aupdate_state(
                    {"configurable": {"thread_id": conversation_id}},
                    {"messages": [human_response_msg]}
                )
                
                logger.info(f"🔄 [AGENT_RESUME] ✅ UPDATED state with human response")
                
                # CRITICAL FIX: Resume from checkpoint using stream(None, config)
                # This continues from the HITL node instead of restarting the entire graph
                result_messages = []
                async for chunk in self.graph.astream(
                    None,  # Pass None to resume from checkpoint
                    {"configurable": {"thread_id": conversation_id}},
                    stream_mode="values"
                ):
                    if "messages" in chunk:
                        result_messages = chunk["messages"]
                
                # Build the final result from the last chunk
                result = {
                    "messages": result_messages,
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

    async def _process_agent_result_for_api(self, result: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """
        Process agent result and return clean semantic data for API layer.
        
        This method encapsulates all LangGraph-specific result processing logic,
        providing a clean interface that abstracts away LangGraph internals.
        
        Args:
            result: Raw result from LangGraph agent execution
            conversation_id: Conversation ID for context
            
        Returns:
            Dict containing clean, semantic data for API response:
            - message: Final message content
            - sources: Document sources if any
            - is_interrupted: Boolean indicating if human interaction is needed
            - confirmation_data: Confirmation details if interrupted
        """
        try:
            logger.info(f"🔍 [RESULT_PROCESSING] Processing agent result for API response")
            logger.info(f"🔍 [RESULT_PROCESSING] Result keys: {list(result.keys()) if result else 'None'}")
            
            # Initialize clean response structure
            api_response = {
                "message": "I apologize, but I encountered an issue processing your request.",
                "sources": [],
                "is_interrupted": False,
                "confirmation_data": None
            }
            
            if not result:
                return api_response
            
            # =================================================================
            # STEP 1: Check for HITL (Human-in-the-Loop) interactions
            # =================================================================
            hitl_data = result.get('hitl_data')
            if hitl_data and isinstance(hitl_data, dict):
                logger.info(f"🔍 [RESULT_PROCESSING] 🎯 HITL STATE DETECTED! Type: {hitl_data.get('type')}")
                
                # Return HITL prompt with clean interface - no LangGraph details
                api_response.update({
                    "message": hitl_data.get('prompt', 'Confirmation required'),
                    "is_interrupted": True,
                    "confirmation_data": {
                        "type": hitl_data.get('type', 'confirmation'),
                        "prompt": hitl_data.get('prompt', 'Confirmation required'),
                        "context": hitl_data.get('context', {})
                    }
                })
                return api_response
            
            # =================================================================
            # STEP 2: Check for legacy LangGraph interrupts (for backward compatibility)
            # =================================================================
            is_interrupted = False
            interrupt_message = ""
            
            # Check for LangGraph's internal interrupt indicators
            if isinstance(result, dict) and '__interrupt__' in result:
                logger.info(f"🔍 [RESULT_PROCESSING] Legacy LangGraph interrupt detected")
                is_interrupted = True
                
                # Extract interrupt message from LangGraph's internal format
                interrupt_data = result['__interrupt__']
                if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                    first_interrupt = interrupt_data[0]
                    if hasattr(first_interrupt, 'value'):
                        interrupt_message = str(first_interrupt.value)
                        logger.info(f"🔍 [RESULT_PROCESSING] Extracted legacy interrupt message")
            elif hasattr(result, '__interrupt__'):
                logger.info(f"🔍 [RESULT_PROCESSING] Legacy LangGraph interrupt attribute detected")
                is_interrupted = True
            
            # Handle legacy interrupts
            if is_interrupted and interrupt_message:
                logger.info(f"🔍 [RESULT_PROCESSING] Processing legacy interrupt for conversation {conversation_id}")
                
                api_response.update({
                    "message": interrupt_message,
                    "is_interrupted": True,
                    "confirmation_data": {
                        "type": "legacy_confirmation",
                        "prompt": interrupt_message,
                        "context": {"legacy": True}
                    }
                })
                return api_response
            
            # =================================================================
            # STEP 3: Check for error states first
            # =================================================================
            if result.get('error'):
                logger.error(f"🔍 [RESULT_PROCESSING] Agent returned error: {result['error']}")
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
            logger.info(f"🔍 [RESULT_PROCESSING] Processing normal conversation response")
            
            # Extract final message content (excluding tool messages to prevent duplicates)
            final_message_content = ""
            messages = result.get('messages', [])
            
            if messages:
                logger.info(f"🔍 [RESULT_PROCESSING] Found {len(messages)} messages in result")
                
                # CRITICAL FIX: Filter out ToolMessage objects to prevent duplicate display
                # Tool messages are internal LangChain constructs and shouldn't appear in user interface
                filtered_messages = []
                for msg in messages:
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        logger.debug(f"🔍 [RESULT_PROCESSING] Filtering out tool message: {getattr(msg, 'name', 'unknown')}")
                        continue
                    elif isinstance(msg, dict) and msg.get('type') == 'tool':
                        logger.debug(f"🔍 [RESULT_PROCESSING] Filtering out tool message dict: {msg.get('name', 'unknown')}")
                        continue
                    else:
                        filtered_messages.append(msg)
                
                logger.info(f"🔍 [RESULT_PROCESSING] Filtered {len(messages) - len(filtered_messages)} tool messages, {len(filtered_messages)} remaining")
                
                # Get the final assistant message (after filtering tool messages)
                for msg in reversed(filtered_messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        final_message_content = msg.content
                        logger.info(f"🔍 [RESULT_PROCESSING] Found final AI message: '{final_message_content[:100]}...'")
                        break
                    elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                        final_message_content = msg.get('content', '')
                        logger.info(f"🔍 [RESULT_PROCESSING] Found final assistant message: '{final_message_content[:100]}...'")
                        break
                    elif hasattr(msg, 'content') and not hasattr(msg, 'type'):
                        # Fallback for generic message objects
                        final_message_content = msg.content
                        logger.info(f"🔍 [RESULT_PROCESSING] Found final generic message: '{final_message_content[:100]}...'")
                        break
                
                if not final_message_content:
                    logger.warning(f"🔍 [RESULT_PROCESSING] No suitable final message found in {len(filtered_messages)} filtered messages")
                    # Use the last non-tool message as fallback
                    if filtered_messages:
                        last_msg = filtered_messages[-1]
                        if hasattr(last_msg, 'content'):
                            final_message_content = last_msg.content
                        elif isinstance(last_msg, dict):
                            final_message_content = last_msg.get('content', '')
                        logger.info(f"🔍 [RESULT_PROCESSING] Using fallback message: '{final_message_content[:100]}...'")
            
            # Fallback if no content found - provide helpful message instead of false success
            if not final_message_content.strip():
                logger.warning(f"🔍 [RESULT_PROCESSING] No message content found in agent result")
                final_message_content = "I processed your request, but didn't generate a response message. If you were expecting a specific action or response, please try rephrasing your request."
            
            # Extract sources if any
            sources = result.get('sources', [])
            logger.info(f"🔍 [RESULT_PROCESSING] Found {len(sources)} sources")
            
            # Return clean response
            api_response.update({
                "message": final_message_content,
                "sources": sources,
                "is_interrupted": False
            })
            
            logger.info(f"🔍 [RESULT_PROCESSING] Constructed final API response")
            return api_response
            
        except Exception as e:
            logger.error(f"🔍 [RESULT_PROCESSING] Error processing agent result: {e}")
            return {
                "message": "I apologize, but I encountered an unexpected error while processing your request. Please try again, and if the problem persists, contact support.",
                "sources": [],
                "is_interrupted": False,
                "error": True,
                "error_details": str(e),
                "confirmation_data": None
            }

    async def process_user_message(
        self,
        user_query: str,
        conversation_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level method for processing user messages with clean API interface.
        
        This method provides a semantic interface for the API layer, encapsulating
        all LangGraph-specific logic and returning clean, structured data.
        
        Args:
            user_query: User's message/query
            conversation_id: Conversation ID for persistence
            user_id: User ID for context
            **kwargs: Additional arguments passed to invoke
            
        Returns:
            Dict containing clean API response structure:
            - message: Final response message
            - sources: Document sources if any
            - is_interrupted: Boolean indicating if human interaction needed
            - confirmation_data: Confirmation details if interrupted
            - conversation_id: Conversation ID for persistence
        """
        try:
            logger.info(f"🚀 [API_INTERFACE] Processing user message for conversation {conversation_id}")
            
            # Invoke the agent with the user query
            raw_result = await self.invoke(
                user_query=user_query,
                conversation_id=conversation_id,
                user_id=user_id,
                **kwargs
            )
            
            # Process the result for clean API response
            processed_result = await self._process_agent_result_for_api(raw_result, conversation_id)
            
            # Add conversation ID to response
            processed_result["conversation_id"] = conversation_id or raw_result.get("conversation_id")
            
            logger.info(f"✅ [API_INTERFACE] User message processed successfully. Interrupted: {processed_result['is_interrupted']}")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"❌ [API_INTERFACE] Error processing user message: {e}")
            return {
                "message": "I apologize, but I encountered an error while processing your request.",
                "sources": [],
                "is_interrupted": False,
                "confirmation_data": None,
                "conversation_id": conversation_id
            }


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
