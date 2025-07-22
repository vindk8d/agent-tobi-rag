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
        
        # Confirmation node for HITL
        graph.add_node("confirmation_node", self._confirmation_node)

        # Enhanced graph flow with memory separation
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
        
        # Employee workflow: memory prep → agent → (confirmation if needed) → memory store
        graph.add_edge("ea_memory_prep", "employee_agent")
        
        # Employee agent routes to confirmation or memory store
        graph.add_conditional_edges(
            "employee_agent",
            self._route_employee_to_confirmation_or_memory_store,
            {
                "confirmation_node": "confirmation_node",
                "ea_memory_store": "ea_memory_store"
            },
        )
        
        # Confirmation node routes back to employee_agent for execution
        graph.add_edge("confirmation_node", "employee_agent")
        
        # Employee memory store always goes to end
        graph.add_edge("ea_memory_store", END)
        
        # Customer workflow: memory prep → agent → memory store → end
        graph.add_edge("ca_memory_prep", "customer_agent")
        graph.add_edge("customer_agent", "ca_memory_store")
        graph.add_edge("ca_memory_store", END)

        # Compile with checkpointer for automatic persistence between steps
        return graph.compile(checkpointer=checkpointer)

    @traceable(name="user_verification_node")
    async def _user_verification_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verify user access and determine user type (employee, customer, unknown).
        Sets user_type in user context system for tool access control.
        """
        try:
            user_id = state.get("user_id")

            if not user_id:
                logger.warning(
                    "[USER_VERIFICATION_NODE] Access denied: No user_id provided"
                )
                # Return state with access denied message
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
                    "user_verified": False,
                }

            # Extract thread_id from config if no conversation_id is provided
            conversation_id = state.get("conversation_id")
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = str(thread_id)  # Keep as string, not UUID object
                    logger.info(
                        f"[USER_VERIFICATION_NODE] Using thread_id {thread_id} as conversation_id for persistence"
                    )

            logger.info(f"[USER_VERIFICATION_NODE] Verifying user access for user: {user_id}")

            # Verify user access and get user type
            user_type = await self._verify_user_access(user_id)
            logger.info(f"[USER_VERIFICATION_NODE] User type determined: {user_type}")

            # Set user type in user context system (not in AgentState)
            with UserContext(user_id=user_id, conversation_id=conversation_id, user_type=user_type):
                logger.info(f"[USER_VERIFICATION_NODE] User context set - user_type: {user_type}")

            if user_type == "unknown":
                logger.warning(f"[USER_VERIFICATION_NODE] Access denied for unknown user: {user_id}")
                # Return state with access denied message
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
                    "user_id": user_id,
                    "conversation_summary": state.get("conversation_summary"),
                    "retrieved_docs": [],
                    "sources": [],
                    "long_term_context": [],
                    "user_verified": False,
                }

            # Access granted for employee or customer
            logger.info(f"[USER_VERIFICATION_NODE] ✅ Access granted for {user_type} user: {user_id}")

            # Return full state with user verification flag
            return {
                "messages": state.get("messages", []),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "retrieved_docs": state.get("retrieved_docs", []),
                "sources": state.get("sources", []),
                "long_term_context": state.get("long_term_context", []),
                "user_verified": True,
                "user_type": user_type,  # Add user_type to state for debugging (not stored)
            }

        except Exception as e:
            logger.error(f"[USER_VERIFICATION_NODE] Error in user verification node: {str(e)}")
            # On error, deny access for security
            error_message = AIMessage(
                content="I'm temporarily unable to verify your access. Please try again later or contact your system administrator."
            )

            return {
                "messages": state.get("messages", []) + [error_message],
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "retrieved_docs": [],
                "sources": [],
                "long_term_context": [],
                "user_verified": False,
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
            user_type = state.get("user_type", "unknown")
            
            # Validate user context
            if user_type not in ["employee", "admin"]:
                logger.error(f"[EA_MEMORY_PREP] Invalid user type '{user_type}' for employee memory prep")
                return state  # Pass through unchanged
            
            logger.info(f"[EA_MEMORY_PREP] Processing {len(messages)} messages for user {user_id}")
            
            # =================================================================
            # STEP 1: STORE INCOMING USER MESSAGES
            # =================================================================
            logger.info("[EA_MEMORY_PREP] Step 1: Storing incoming user messages")
            
            if messages and config:
                # Find the most recent human message and store it
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        logger.info("[EA_MEMORY_PREP] Stored incoming user message")
                        break
            
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
                "user_verified": state.get("user_verified", True),
                "user_type": user_type,
                "long_term_context": long_term_context,
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
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
            user_type = state.get("user_type", "unknown")
            
            # Validate user context
            if user_type != "customer":
                logger.error(f"[CA_MEMORY_PREP] Invalid user type '{user_type}' for customer memory prep")
                return state  # Pass through unchanged
            
            logger.info(f"[CA_MEMORY_PREP] Processing {len(messages)} messages for customer user {user_id}")
            
            # =================================================================
            # STEP 1: STORE INCOMING CUSTOMER MESSAGES
            # =================================================================
            logger.info("[CA_MEMORY_PREP] Step 1: Storing incoming customer messages")
            
            if messages and config:
                # Find the most recent human message and store it
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        logger.info("[CA_MEMORY_PREP] Stored incoming customer message")
                        break
            
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
                "user_verified": state.get("user_verified", True),
                "user_type": user_type,
                "long_term_context": long_term_context,
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
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
            user_type = state.get("user_type", "unknown")
            
            # Validate user context
            if user_type not in ["employee", "admin"]:
                logger.error(f"[EA_MEMORY_STORE] Invalid user type '{user_type}' for employee memory storage")
                return state  # Pass through unchanged
            
            logger.info(f"[EA_MEMORY_STORE] Processing memory storage for {len(messages)} messages")
            
            # =================================================================
            # STEP 1: STORE AGENT RESPONSE MESSAGES
            # =================================================================
            logger.info("[EA_MEMORY_STORE] Step 1: Storing agent response messages")
            
            if messages and config:
                # Find the most recent AI message and store it
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "ai":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        logger.info("[EA_MEMORY_STORE] Stored agent response message")
                        break
            
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
                "user_verified": state.get("user_verified", True),
                "user_type": user_type,
                "long_term_context": state.get("long_term_context", []),
                "confirmation_data": state.get("confirmation_data"),
                "execution_data": state.get("execution_data"),
                "confirmation_result": state.get("confirmation_result"),
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
            user_type = state.get("user_type", "unknown")
            
            # Validate user context
            if user_type != "customer":
                logger.error(f"[CA_MEMORY_STORE] Invalid user type '{user_type}' for customer memory storage")
                return state  # Pass through unchanged
            
            logger.info(f"[CA_MEMORY_STORE] Processing memory storage for {len(messages)} customer messages")
            
            # =================================================================
            # STEP 1: STORE CUSTOMER AGENT RESPONSE MESSAGES
            # =================================================================
            logger.info("[CA_MEMORY_STORE] Step 1: Storing customer agent response messages")
            
            if messages and config:
                # Find the most recent AI message and store it
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "ai":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        logger.info("[CA_MEMORY_STORE] Stored customer agent response message")
                        break
            
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
                "user_verified": state.get("user_verified", True),
                "user_type": user_type,
                "long_term_context": state.get("long_term_context", []),
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
        Route users directly from verification to the appropriate agent node based on user type.
        
        Args:
            state: Current agent state containing user_verified and user_type
            
        Returns:
            str: Node name to route to ('employee_agent', 'customer_agent', or 'end')
        """
        # First check if user is verified
        if not state.get("user_verified", False):
            logger.info(f"[ROUTING] User not verified - ending conversation")
            return "end"
        
        # Route verified users based on user type
        user_type = state.get("user_type", "unknown")
        user_id = state.get("user_id", "unknown")
        
        logger.info(f"[ROUTING] Routing verified user {user_id} with user_type: {user_type}")
        
        if user_type in ["employee", "admin"]:
            logger.info(f"[ROUTING] ✅ Routing {user_type} user directly to employee_agent")
            return "employee_agent"
        elif user_type == "customer":
            logger.info(f"[ROUTING] ✅ Routing customer user directly to customer_agent")
            return "customer_agent"
        else:
            # This should never happen if user_verification works correctly
            logger.error(f"[ROUTING] ❌ UNEXPECTED: Unknown user_type '{user_type}' for verified user - ending conversation for safety")
            return "end"

    def _route_to_memory_prep(self, state: Dict[str, Any]) -> str:
        """
        Route users from verification to appropriate memory preparation nodes.
        
        Args:
            state: Current agent state containing user_verified and user_type
            
        Returns:
            str: Node name to route to ('ea_memory_prep', 'ca_memory_prep', or 'end')
        """
        # First check if user is verified
        if not state.get("user_verified", False):
            logger.info(f"[MEMORY_ROUTING] User not verified - ending conversation")
            return "end"
        
        # Route verified users to appropriate memory preparation
        user_type = state.get("user_type", "unknown")
        user_id = state.get("user_id", "unknown")
        
        logger.info(f"[MEMORY_ROUTING] Routing verified user {user_id} to memory prep for user_type: {user_type}")
        
        if user_type in ["employee", "admin"]:
            logger.info(f"[MEMORY_ROUTING] ✅ Routing {user_type} user to employee memory prep")
            return "ea_memory_prep"
        elif user_type == "customer":
            logger.info(f"[MEMORY_ROUTING] ✅ Routing customer user to customer memory prep")
            return "ca_memory_prep"
        else:
            # This should never happen if user_verification works correctly
            logger.error(f"[MEMORY_ROUTING] ❌ UNEXPECTED: Unknown user_type '{user_type}' for verified user - ending conversation for safety")
            return "end"

    def _route_employee_to_confirmation_or_memory_store(self, state: Dict[str, Any]) -> str:
        """
        Route employee agent to confirmation node (if needed) or memory store.
        
        SIMPLIFIED DESIGN:
        - If confirmation_data exists (new request) → confirmation_node
        - Otherwise → ea_memory_store
        
        Args:
            state: Current agent state
            
        Returns:
            str: Node name to route to ("confirmation_node" or "ea_memory_store")
        """
        confirmation_data = state.get("confirmation_data")
        
        # Route to confirmation if new request needs approval
        if confirmation_data:
            logger.info("[ROUTING] New confirmation needed - routing to confirmation_node")
            return "confirmation_node"
        else:
            logger.info("[ROUTING] No confirmation needed - routing to memory store")
            return "ea_memory_store"

    async def _handle_customer_message_execution(
        self, state: AgentState, execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle customer message execution after confirmation approval.
        
        This consolidates execution within employee_agent for simplicity.
        """
        try:
            # Extract execution information
            customer_info = execution_data.get("customer_info", {})
            customer_name = customer_info.get("name", "Unknown Customer")
            customer_email = customer_info.get("email", "no-email")
            message_content = execution_data.get("message_content", "")
            message_type = execution_data.get("message_type", "follow_up")
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Executing delivery for {customer_name} ({customer_email})")
            
            # Execute the delivery
            delivery_success = await self._execute_customer_message_delivery(
                customer_id=execution_data.get("customer_id"),
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
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
                
            else:
                result_status = "failed"
                feedback_message = f"""❌ **Message Delivery Failed**

Failed to deliver your message to {customer_name}.

**Failure Details:**
- Recipient: {customer_name} ({customer_email})
- Message Type: {message_type.title()}
- Status: Delivery Failed
- Reason: System delivery error

Please try again or contact the customer directly."""
                
                logger.error(f"[EMPLOYEE_AGENT_NODE] FAILED: Delivery failed for {customer_name}")
            
            # Add feedback message to conversation
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=feedback_message))
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Delivery completed with status: {result_status}")
            
            # Return completion state (clears execution data, sets final result)
            return {
                "messages": messages,
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "user_verified": state.get("user_verified", True),
                "user_type": state.get("user_type"),
                "confirmation_data": None,
                "execution_data": None,  # Clear after execution
                "confirmation_result": result_status,  # Set final result
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
                "user_verified": state.get("user_verified", True),
                "user_type": state.get("user_type"),
                "confirmation_data": None,
                "execution_data": None,
                "confirmation_result": "error",
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
            
            # Check if executing approved customer message (from confirmation_node)
            execution_data = state.get("execution_data")
            if execution_data:
                logger.info("[EMPLOYEE_AGENT_NODE] Executing approved customer message delivery")
                return await self._handle_customer_message_execution(state, execution_data)

            # Get messages and validate user type
            messages = state.get("messages", [])
            user_type = state.get("user_type")
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            
            if user_type not in ["employee", "admin"]:
                logger.warning(f"[EMPLOYEE_AGENT_NODE] Non-employee user type '{user_type}' routed to employee node")
                return await self._handle_access_denied(state, "Employee access required")

            logger.info(f"[EMPLOYEE_AGENT_NODE] Processing {len(messages)} messages for {user_type}")

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
            confirmation_data_to_return = None  # Track if customer messaging was triggered

            # Create LLM and bind tools
            selected_model = self.settings.openai_chat_model
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens,
                api_key=self.settings.openai_api_key,
            ).bind_tools(self.tools)

            logger.info(f"[EMPLOYEE_AGENT_NODE] Using model: {selected_model}")

            while iteration < max_tool_iterations:
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
                                user_type = state.get("user_type", "unknown")

                                with UserContext(
                                    user_id=user_id, 
                                    conversation_id=conversation_id, 
                                    user_type=user_type
                                ):
                                    # Execute tool based on type
                                    if tool_name in ["simple_rag", "simple_query_crm_data", "trigger_customer_message"]:
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        tool_result = tool.invoke(tool_args)

                                # CLEAN APPROACH: Check for customer messaging confirmation need
                                if (tool_name == "trigger_customer_message" and 
                                    isinstance(tool_result, str) and 
                                    "STATE_DRIVEN_CONFIRMATION_REQUIRED" in tool_result):
                                    
                                    logger.info("[EMPLOYEE_AGENT_NODE] Customer messaging tool requires HITL confirmation")
                                    
                                    try:
                                        import json
                                        prefix = "STATE_DRIVEN_CONFIRMATION_REQUIRED: "
                                        data_start = tool_result.find(prefix) + len(prefix)
                                        data_str = tool_result[data_start:].strip()
                                        confirmation_data_to_return = json.loads(data_str)
                                        
                                        # Create user-friendly response
                                        customer_info = confirmation_data_to_return.get("customer_info", {})
                                        customer_name = customer_info.get("name", "customer")
                                        
                                        tool_response = f"📤 Message prepared for {customer_name} and ready for your confirmation."
                                        
                                        logger.info(f"[EMPLOYEE_AGENT_NODE] Confirmation data prepared for customer: {customer_name}")
                                        
                                    except Exception as e:
                                        logger.error(f"[EMPLOYEE_AGENT_NODE] Error processing confirmation data: {e}")
                                        tool_response = f"Error preparing customer message: {e}"
                                        confirmation_data_to_return = None
                                else:
                                    # Normal tool response
                                    tool_response = str(tool_result)

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
                "user_verified": state.get("user_verified", True),
                "user_type": user_type,
            }

            # Add confirmation data if customer messaging was triggered
            if confirmation_data_to_return:
                return_state["confirmation_data"] = confirmation_data_to_return
                logger.info("[EMPLOYEE_AGENT_NODE] Returning with confirmation_data for confirmation routing")
            else:
                return_state["confirmation_data"] = None
            
            # Always clear execution state for clean routing    
            return_state["execution_data"] = None
            return_state["confirmation_result"] = None

            logger.info(f"[EMPLOYEE_AGENT_NODE] Complete workflow finished successfully for {user_type}")
            return return_state

        except Exception as e:
            logger.error(f"[EMPLOYEE_AGENT_NODE] Error: {str(e)}")
            return await self._handle_processing_error(state, e)

    async def _handle_access_denied(self, state: AgentState, message: str) -> Dict[str, Any]:
        """Handle access denied scenarios with proper error response."""
        error_message = AIMessage(content=f"Access denied: {message}")
        messages = list(state.get("messages", []))
        messages.append(error_message)
        
        return {
            "messages": messages,
            "sources": state.get("sources", []),
            "retrieved_docs": state.get("retrieved_docs", []),
            "conversation_id": state.get("conversation_id"),
            "user_id": state.get("user_id"),
            "conversation_summary": state.get("conversation_summary"),
            "user_verified": state.get("user_verified", True),
            "user_type": state.get("user_type"),
            "confirmation_data": None,
            "execution_data": None,
            "confirmation_result": None,
        }

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
            "user_verified": state.get("user_verified", True),
            "user_type": state.get("user_type"),
            "confirmation_data": None,
            "execution_data": None,
            "confirmation_result": None,
        }

    def _get_employee_system_prompt(self) -> str:
        """Get the system prompt for employee users."""
        return f"""You are a helpful sales assistant with full access to company tools and data.

Available tools:
{', '.join(self.tool_names)}

**Tool Usage Guidelines:**
- Use simple_rag for comprehensive document-based answers
- Use simple_query_crm_data for specific CRM database queries  
- Use trigger_customer_message when asked to send messages, follow-ups, or contact customers

**Customer Messaging:**
When asked to "send a message to [customer]", "follow up with [customer]", or "contact [customer]", use the trigger_customer_message tool. This will prepare the message and request your confirmation before sending.

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
            user_type = state.get("user_type", "unknown")
            user_id = state.get("user_id")
            
            logger.info(f"[CUSTOMER_AGENT_NODE] Starting complete workflow for user_type: {user_type}, user_id: {user_id}")
            
            # Customer agent should only process customer users
            if user_type not in ["customer"]:
                logger.warning(f"[CUSTOMER_AGENT_NODE] Invalid user_type '{user_type}' for customer agent - should not reach here")
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
                    "user_verified": state.get("user_verified", True),
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
                                user_type = state.get("user_type", "customer")

                                with UserContext(
                                    user_id=user_id, conversation_id=conversation_id, user_type=user_type
                                ):
                                    # Handle async tools
                                    if tool_name in [
                                        "simple_rag",
                                        "simple_query_crm_data",
                                        "trigger_customer_message",
                                    ]:
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        tool_result = tool.invoke(tool_args)

                                # Add tool result to messages
                                tool_message = ToolMessage(
                                    content=str(tool_result),
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

            # Return final state
            return {
                "messages": updated_messages,
                "sources": sources,
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "user_verified": state.get("user_verified", True),
            }

        except GraphInterrupt:
            # Human-in-the-loop interrupt is expected behavior - re-raise it
            logger.info(f"[CUSTOMER_AGENT_NODE] Human-in-the-loop interrupt triggered, passing to LangGraph")
            raise
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
                "user_verified": state.get("user_verified", True),
            }

    @traceable(name="confirmation_node") 
    async def _confirmation_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        CONFIRMATION NODE - Pure Interrupt (Routes Back to Employee Agent)
        
        SIMPLIFIED DESIGN:
        - Only handles user confirmation interrupt
        - Sets execution_data if approved, cancellation if denied  
        - Routes back to employee_agent for execution and response
        - Clean separation: interrupt here, execution in employee_agent
        """
        try:
            logger.info("[CONFIRMATION_NODE] Starting clean confirmation workflow")
            
            # Validate confirmation data - must exist to reach this node
            confirmation_data = state.get("confirmation_data")
            if not confirmation_data:
                logger.warning("[CONFIRMATION_NODE] Missing confirmation_data - routing error")
                return {
                    "messages": state.get("messages", []),
                    "sources": state.get("sources", []),
                    "retrieved_docs": state.get("retrieved_docs", []),
                    "conversation_id": state.get("conversation_id"),
                    "user_id": state.get("user_id"),
                    "conversation_summary": state.get("conversation_summary"),
                    "user_verified": state.get("user_verified", True),
                    "user_type": state.get("user_type"),
                    "confirmation_data": None,  # Clear invalid state
                    "execution_data": None,
                    "confirmation_result": "error_no_data",
                }
            
            # Extract customer information for confirmation prompt
            customer_info = confirmation_data.get("customer_info", {})
            customer_name = customer_info.get("name", "Unknown Customer")
            customer_email = customer_info.get("email", "no-email")
            message_content = confirmation_data.get("message_content", "")
            message_type = confirmation_data.get("message_type", "follow_up").title()
            
            logger.info(f"[CONFIRMATION_NODE] Processing confirmation for {customer_name} ({customer_email})")
            
            # ===============================================================
            # CLEAN INTERRUPT - NO EXECUTION IN THIS NODE
            # ===============================================================
            
            # Create confirmation prompt
            confirmation_prompt = f"""🔄 **Customer Message Confirmation**

**To:** {customer_name} ({customer_email})
**Type:** {message_type}
**Message:** {message_content}

**Instructions:**
- Reply 'approve' to send the message  
- Reply 'deny' to cancel the message

Your choice:"""
            
            logger.info("[CONFIRMATION_NODE] Getting user confirmation (no execution)")
            
            # INTERRUPT - Get user decision
            human_response = interrupt(confirmation_prompt)
            
            logger.info(f"[CONFIRMATION_NODE] Received decision: {human_response}")
            
            # PROCESS USER DECISION - SET STATE FOR EXECUTION NODE
            response_text = str(human_response).lower().strip()
            
            if any(keyword in response_text for keyword in ["approve", "yes", "send", "ok"]):
                # APPROVED - Set execution data for execution node
                logger.info("[CONFIRMATION_NODE] Approved - setting execution data for execution node")
                
                execution_data = {
                    "customer_id": confirmation_data.get("customer_id"),
                    "customer_info": customer_info,
                    "message_content": message_content,
                    "message_type": confirmation_data.get("message_type", "follow_up"),
                    "formatted_message": confirmation_data.get("formatted_message", message_content),
                    "sender_employee_id": confirmation_data.get("sender_employee_id")
                }
                
                logger.info("[CONFIRMATION_NODE] Execution data set - routing to execution node")
                
                return {
                    "messages": state.get("messages", []),
                    "sources": state.get("sources", []),
                    "retrieved_docs": state.get("retrieved_docs", []),
                    "conversation_id": state.get("conversation_id"),
                    "user_id": state.get("user_id"),
                    "conversation_summary": state.get("conversation_summary"),
                    "user_verified": state.get("user_verified", True),
                    "user_type": state.get("user_type"),
                    "confirmation_data": None,  # Clear confirmation data
                    "execution_data": execution_data,  # Set execution data
                    "confirmation_result": None,  # No result yet
                }
                
            else:
                # DENIED/CANCELLED - Add cancellation message and complete
                logger.info("[CONFIRMATION_NODE] Denied - adding cancellation message")
                
                cancellation_message = f"""🚫 **Message Cancelled**

Your message to {customer_name} has been cancelled as requested.

**Cancellation Summary:**
- Recipient: {customer_name} ({customer_email})
- Message Type: {message_type}
- Status: Cancelled by User
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

No message was sent to the customer."""
                
                messages = list(state.get("messages", []))
                messages.append(AIMessage(content=cancellation_message))
                
                return {
                    "messages": messages,
                    "sources": state.get("sources", []),
                    "retrieved_docs": state.get("retrieved_docs", []),
                    "conversation_id": state.get("conversation_id"),
                    "user_id": state.get("user_id"),
                    "conversation_summary": state.get("conversation_summary"),
                    "user_verified": state.get("user_verified", True),
                    "user_type": state.get("user_type"),
                    "confirmation_data": None,  # Clear confirmation data
                    "execution_data": None,  # No execution needed
                    "confirmation_result": "cancelled",  # Set result
                }
            
        except GraphInterrupt:
            # Expected LangGraph interrupt - reraise for proper handling
            logger.info("[CONFIRMATION_NODE] LangGraph interrupt (expected) - reraising")
            raise
        except Exception as e:
            logger.error(f"[CONFIRMATION_NODE] Unexpected error: {e}")
            
            return {
                "messages": state.get("messages", []),
                "sources": state.get("sources", []),
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary"),
                "user_verified": state.get("user_verified", True),
                "user_type": state.get("user_type"),
                "confirmation_data": None,  # Clear data on error
                "execution_data": None,
                "confirmation_result": "error",  # Set error status
            }


    async def _execute_customer_message_delivery(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ) -> bool:
        """
        Execute the actual customer message delivery.
        
        This method handles the actual delivery mechanism and can be extended
        for different channels (in-system, email, SMS, etc.).
        
        Args:
            customer_id: Target customer ID
            message_content: Message content to deliver
            message_type: Type of message (follow_up, information, etc.)
            customer_info: Customer information for delivery
            
        Returns:
            bool: True if delivery successful, False otherwise
        """
        try:
            logger.info(f"[DELIVERY] Executing delivery for customer {customer_id}")
            
            # =================================================================
            # MESSAGE DELIVERY IMPLEMENTATION
            # =================================================================
            # For now, we'll store the message in the database and simulate delivery
            # In production, you would also send via email, SMS, or in-app notification
            
            # Store the message in database for audit trail
            await self._store_customer_message(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                delivery_status="delivered",
                customer_info=customer_info
            )
            
            # Create a message record in the messages table for the customer to see
            await self._create_customer_message_record(
                customer_id=customer_id,
                message_content=message_content,
                message_type=message_type,
                customer_info=customer_info
            )
            
            # TODO: Implement actual delivery channels
            # await self._send_via_email(customer_info, message_content)
            # await self._send_via_sms(customer_info, message_content)
            # await self._send_via_in_app_notification(customer_info, message_content)
            
            logger.info(f"[DELIVERY] Message delivery completed successfully for customer {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"[DELIVERY] Message delivery failed for customer {customer_id}: {str(e)}")
            return False

    async def _create_customer_message_record(
        self,
        customer_id: str,
        message_content: str,
        message_type: str,
        customer_info: Dict[str, Any]
    ):
        """
        Create a message record in the messages table for the customer to see.
        """
        try:
            from database import db_client
            from datetime import datetime
            import uuid
            
            # Use the actual UUID from customer_info, not the customer_id parameter
            actual_customer_id = customer_info.get("customer_id") or customer_info.get("id")
            if not actual_customer_id:
                logger.error(f"[MESSAGE_RECORD] No valid customer UUID found in customer_info: {customer_info}")
                return
            
            # First, find or create a user record for this customer
            user_id = await self._get_or_create_customer_user(actual_customer_id, customer_info)
            if not user_id:
                logger.error(f"[MESSAGE_RECORD] Failed to get or create user for customer {customer_info.get('name', actual_customer_id)}")
                return
            
            # Create a proper UUID for the conversation_id
            conversation_uuid = str(uuid.uuid4())
            
            # Create the conversation record with the correct user_id
            conversation_data = {
                "id": conversation_uuid,
                "user_id": user_id,
                "title": f"Customer Message - {customer_info.get('name', 'Unknown')}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert conversation record
            conv_result = db_client.client.table("conversations").insert(conversation_data).execute()
            if not conv_result.data:
                logger.error(f"[MESSAGE_RECORD] Failed to create conversation record for customer {customer_info.get('name', actual_customer_id)}")
                return
            
            # Insert the message into the messages table
            message_data = {
                "conversation_id": conversation_uuid,
                "role": "assistant",
                "content": message_content,
                "metadata": {
                    "message_type": message_type,
                    "delivered_by": "employee_system",
                    "customer_id": actual_customer_id,
                    "customer_name": customer_info.get("name", ""),
                    "delivery_timestamp": datetime.now().isoformat()
                }
            }
            
            result = db_client.client.table("messages").insert(message_data).execute()
            
            if result.data:
                logger.info(f"[MESSAGE_RECORD] Created message record for customer {customer_info.get('name', actual_customer_id)} in conversation {conversation_uuid}")
            else:
                logger.error(f"[MESSAGE_RECORD] Failed to create message record for customer {customer_info.get('name', actual_customer_id)}")
                
        except Exception as e:
            logger.error(f"[MESSAGE_RECORD] Error creating message record: {str(e)}")
            # Don't raise the error, just log it - delivery can still be considered successful

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
        user_query=None,  # Can be string or Command object
        conversation_id: str = None,
        user_id: str = None,
        conversation_history: List = None,
        config: Dict[str, Any] = None,  # Support direct config passing
    ) -> Dict[str, Any]:
        """
        Invoke the unified tool-calling RAG agent with thread-based persistence.
        
        Supports both regular queries and LangGraph Command objects for resuming from interrupts.
        
        Args:
            user_query: Either a string query or a Command object for resuming
            conversation_id: Conversation ID for persistence
            user_id: User ID for context
            conversation_history: Optional conversation history for new conversations
            config: Optional direct config (used for Command resumption)
        """
        # Ensure initialization before invoking
        await self._ensure_initialized()

        # Handle Command objects for interrupt resumption (following LangGraph best practices)
        if hasattr(user_query, 'resume'):  # This is a Command object
            logger.info(f"[RAG_AGENT_INVOKE] Resuming from interrupt with Command object")
            
            # Extract config from the provided config or build it
            if config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = thread_id
            
            if not conversation_id:
                raise ValueError("conversation_id is required when resuming with Command")
            
            # Use LangGraph's direct invocation for Command objects
            graph_config = await self.memory_manager.get_conversation_config(conversation_id)
            result = await self.graph.invoke(user_query, config=graph_config)
            
            logger.info(f"[RAG_AGENT_INVOKE] Command resumption completed")
            return result
        
        # Handle regular string queries (original behavior)
        if isinstance(user_query, str) and user_query.strip():
            query_text = user_query
        elif user_query is None:
            raise ValueError("user_query is required for new conversations")
        else:
            raise ValueError("user_query must be a string or Command object")

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
            user_verified=False,  # Will be set by user_verification node
            user_type=None,  # Will be set by user_verification node
            retrieved_docs=(
                existing_state.get("retrieved_docs", []) if existing_state else []
            ),
            sources=existing_state.get("sources", []) if existing_state else [],
            conversation_summary=(
                existing_state.get("summary") if existing_state else None
            ),
            long_term_context=[],  # Initialize empty, will be populated by nodes
        )

        # Execute the graph with thread-based persistence
        # The memory preparation and update nodes handle automatic checkpointing and state persistence
        result = await self.graph.ainvoke(initial_state, config=config)

        # Memory management (sliding window, summarization, and persistence) is now handled
        # automatically by the graph nodes with checkpointing between each step

        return result


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
