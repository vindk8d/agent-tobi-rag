"""
Unified Tool-calling RAG Agent following LangChain best practices.
Single agent node handles both tool calling and execution.
"""

from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4, UUID
import logging
import sys
import asyncio
import os

from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphInterrupt
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
        graph.add_node("employee_agent", self._employee_agent_node)
        graph.add_node("customer_agent", self._customer_agent_node)

        # Simplified graph flow with direct routing from user verification
        graph.add_edge(START, "user_verification")

        # Direct conditional routing from user verification to appropriate agent
        graph.add_conditional_edges(
            "user_verification",
            self._route_after_user_verification,
            {"employee_agent": "employee_agent", "customer_agent": "customer_agent", "end": END},
        )
        
        # Both agents go directly to END (interrupts handled within nodes)
        graph.add_edge("employee_agent", END)
        graph.add_edge("customer_agent", END)

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



    @traceable(name="employee_agent_node")
    async def _employee_agent_node(
        self, state: AgentState, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Employee agent node that handles complete workflow for employee users:
        1. Memory preparation (context retrieval)
        2. Tool calling and execution with full access
        3. Memory update (context storage)
        
        Follows LangChain best practices for tool calling workflow.
        """
        try:
            # Ensure initialization before processing
            await self._ensure_initialized()

            # Validate user context at node entry
            user_type = state.get("user_type", "unknown")
            user_id = state.get("user_id")
            
            logger.info(f"[EMPLOYEE_AGENT_NODE] Starting complete workflow for user_type: {user_type}, user_id: {user_id}")
            
            # Employee agent should only process employee users
            if user_type not in ["employee", "admin"]:
                logger.warning(f"[EMPLOYEE_AGENT_NODE] Invalid user_type '{user_type}' for employee agent - should not reach here")
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

            # =================================================================
            # STEP 1: MEMORY PREPARATION (integrated from memory_preparation_node)
            # =================================================================
            logger.info(f"[EMPLOYEE_AGENT_NODE] Step 1: Memory preparation")
            
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")

            # Extract thread_id from config if no conversation_id is provided
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = str(thread_id)
                    logger.info(f"[EMPLOYEE_AGENT_NODE] Using thread_id {thread_id} as conversation_id for persistence")

            logger.info(f"[EMPLOYEE_AGENT_NODE] Memory preparation for conversation {conversation_id}: {len(messages)} messages")

            # Store incoming user messages to database
            if messages and config:
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        break

            # Get cross-conversation user context
            user_context = {}
            long_term_context = []
            if user_id:
                user_context = await self.memory_manager.get_user_context_for_new_conversation(user_id)
                if user_context.get("has_history", False):
                    latest_summary = user_context.get("latest_summary", "")
                    logger.info(f"[EMPLOYEE_AGENT_NODE] Found user history: {len(latest_summary)} chars")
                    
                    system_context = SystemMessage(
                        content=f"""
USER CONTEXT (from latest conversation summary):

{latest_summary}

CONVERSATION COUNT: {user_context.get("conversation_count", 0)}

Use this context to provide personalized, contextually aware responses that build on previous interactions.
"""
                    )
                    messages = [system_context] + messages
                    logger.info("[EMPLOYEE_AGENT_NODE] Added comprehensive user context to conversation")

            # Get relevant long-term context for current query
            if user_id and messages:
                current_query = ""
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        current_query = msg.content
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "human":
                        current_query = msg.get("content", "")
                        break

                if current_query:
                    long_term_context = await self.memory_manager.get_relevant_context(
                        user_id, current_query, max_contexts=3
                    )
                    if long_term_context:
                        logger.info(f"[EMPLOYEE_AGENT_NODE] Retrieved {len(long_term_context)} relevant context items")
            
            # =================================================================
            # STEP 2: TOOL EXECUTION (main agent logic)
            # =================================================================
            logger.info(f"[EMPLOYEE_AGENT_NODE] Step 2: Tool execution")

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
            logger.info(f"[EMPLOYEE_AGENT_NODE] Processing query: {current_query}")
            logger.info(f"[EMPLOYEE_AGENT_NODE] Thread ID: {thread_id}, Conversation ID: {conversation_id}")

            # Simple model selection - use default chat model
            # Tools handle their own dynamic model selection internally
            selected_model = self.settings.openai_chat_model

            # Get system prompt
            system_prompt = self._get_system_prompt()

            # Apply context window management to prevent overflow
            trimmed_messages, trim_stats = (
                await context_manager.trim_messages_for_context(
                    messages=original_messages,
                    model=selected_model,
                    system_prompt=system_prompt,
                    max_messages=self.settings.memory.max_messages,
                )
            )

            # Log context management information
            if trim_stats["trimmed_message_count"] > 0:
                logger.info(
                    f"[EMPLOYEE_AGENT_NODE] Context management: trimmed {trim_stats['trimmed_message_count']} messages, "
                    f"final token count: {trim_stats['final_token_count']}"
                )

            # Build processing messages with system prompt and trimmed conversation
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + trimmed_messages

            # Step 3: Tool calling - let model decide when to use tools
            max_tool_iterations = 3  # Allow more tool calls if needed
            iteration = 0

            # Create LLM with selected model and bind tools
            llm = ChatOpenAI(
                model=selected_model,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens,
                api_key=self.settings.openai_api_key,
            ).bind_tools(self.tools)

            logger.info(f"[EMPLOYEE_AGENT_NODE] Using model: {selected_model} for employee query processing")

            while iteration < max_tool_iterations:
                iteration += 1
                logger.info(f"[EMPLOYEE_AGENT_NODE] Agent iteration {iteration}")

                # Call the model with dynamically selected LLM
                response = await llm.ainvoke(processing_messages)
                processing_messages.append(response)

                # Check if model wants to use tools
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.info(f"Model called {len(response.tool_calls)} tools")

                    # Step 4: Tool execution - execute the tools
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        logger.info(
                            f"Executing tool: {tool_name} with args: {tool_args}"
                        )

                        try:
                            # Get the tool and execute it
                            tool = self.tool_map.get(tool_name)
                            if tool:
                                # Set user context for tools that need access to user information
                                user_id = state.get("user_id")
                                conversation_id = state.get("conversation_id")
                                user_type = state.get("user_type", "unknown")

                                with UserContext(
                                    user_id=user_id, conversation_id=conversation_id, user_type=user_type
                                ):
                                    # Check if tool is async by examining the underlying function
                                    if tool_name in [
                                        "simple_rag",
                                        "simple_query_crm_data",
                                        "trigger_customer_message",
                                    ]:
                                        # Handle async tools
                                        tool_result = await tool.ainvoke(tool_args)
                                    else:
                                        # Handle sync tools
                                        tool_result = tool.invoke(tool_args)



                                # Add tool result to messages (for non-interrupt tools)
                                tool_message = ToolMessage(
                                    content=str(tool_result),
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

                        except GraphInterrupt as e:
                            # Tool triggered human-in-the-loop interrupt (this is expected behavior)
                            logger.info(f"Tool '{tool_name}' triggered human-in-the-loop interrupt: {str(e)}")
                            # Re-raise the interrupt so LangGraph can handle it properly
                            raise e
                        except Exception as e:
                            # Tool execution failed
                            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(error_msg)
                            logger.error(f"Tool args that caused error: {tool_args}")
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
                    logger.info("Model provided final response without tool calls")
                    break

            # If we exited the loop due to max iterations, ensure we get a final response
            if iteration >= max_tool_iterations:
                logger.info("Max tool iterations reached, getting final response")
                # Make one final call to get a proper response
                final_response = await llm.ainvoke(processing_messages)
                processing_messages.append(final_response)
                logger.info("Generated final response after tool execution")

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

            # =================================================================
            # STEP 3: MEMORY UPDATE (integrated from memory_update_node)
            # =================================================================
            logger.info(f"[EMPLOYEE_AGENT_NODE] Step 3: Memory update")
            
            # Store agent response messages to database
            if updated_messages and config:
                for msg in reversed(updated_messages):
                    if hasattr(msg, "type") and msg.type == "ai":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        break

            # Store important information in long-term memory
            if user_id and updated_messages and len(updated_messages) >= 2:
                if await self.memory_manager.should_store_memory("rag", updated_messages):
                    recent_messages = updated_messages[-3:] if len(updated_messages) >= 3 else updated_messages
                    await self.memory_manager.store_conversation_insights(
                        "rag", user_id, recent_messages, conversation_id
                    )

            # Check if automatic conversation summarization should be triggered
            if user_id and conversation_id:
                try:
                    summary = await self.memory_manager.consolidator.check_and_trigger_summarization(
                        str(conversation_id), user_id
                    )
                    if summary:
                        logger.info(f"[EMPLOYEE_AGENT_NODE] Conversation {conversation_id} automatically summarized")
                except Exception as e:
                    logger.error(f"[EMPLOYEE_AGENT_NODE] Error in automatic summarization: {e}")

            logger.info(f"[EMPLOYEE_AGENT_NODE] Complete workflow finished successfully")

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
            logger.info(f"[EMPLOYEE_AGENT_NODE] Human-in-the-loop interrupt triggered, passing to LangGraph")
            raise
        except Exception as e:
            logger.error(f"[EMPLOYEE_AGENT_NODE] Error in employee agent node: {str(e)}")
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
                "conversation_summary": state.get(
                    "conversation_summary"
                ),  # Preserve existing summary
                "user_verified": state.get(
                    "user_verified", True
                ),  # Preserve verification status
            }

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

            # =================================================================
            # STEP 1: MEMORY PREPARATION (integrated from memory_preparation_node)
            # =================================================================
            logger.info(f"[CUSTOMER_AGENT_NODE] Step 1: Memory preparation")
            
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")

            # Extract thread_id from config if no conversation_id is provided
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = str(thread_id)
                    logger.info(f"[CUSTOMER_AGENT_NODE] Using thread_id {thread_id} as conversation_id for persistence")

            logger.info(f"[CUSTOMER_AGENT_NODE] Memory preparation for conversation {conversation_id}: {len(messages)} messages")

            # Store incoming user messages to database
            if messages and config:
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        break

            # Get cross-conversation user context
            user_context = {}
            long_term_context = []
            if user_id:
                user_context = await self.memory_manager.get_user_context_for_new_conversation(user_id)
                if user_context.get("has_history", False):
                    latest_summary = user_context.get("latest_summary", "")
                    logger.info(f"[CUSTOMER_AGENT_NODE] Found user history: {len(latest_summary)} chars")
                    
                    system_context = SystemMessage(
                        content=f"""
USER CONTEXT (from latest conversation summary):

{latest_summary}

CONVERSATION COUNT: {user_context.get("conversation_count", 0)}

Use this context to provide personalized, contextually aware responses that build on previous interactions.
"""
                    )
                    messages = [system_context] + messages
                    logger.info("[CUSTOMER_AGENT_NODE] Added comprehensive user context to conversation")

            # Get relevant long-term context for current query
            if user_id and messages:
                current_query = ""
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        current_query = msg.content
                        break

                if current_query:
                    long_term_context = await self.memory_manager.get_relevant_context(
                        user_id, current_query, max_contexts=3
                    )
                    if long_term_context:
                        logger.info(f"[CUSTOMER_AGENT_NODE] Retrieved {len(long_term_context)} relevant context items")
            
            # =================================================================
            # STEP 2: TOOL EXECUTION (main agent logic)
            # =================================================================
            logger.info(f"[CUSTOMER_AGENT_NODE] Step 2: Tool execution")

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

            # Apply context window management to prevent overflow
            trimmed_messages, trim_stats = (
                await context_manager.trim_messages_for_context(
                    messages=original_messages,
                    model=selected_model,
                    system_prompt=system_prompt,
                    max_messages=self.settings.memory.max_messages,
                )
            )

            # Log context management information
            if trim_stats["trimmed_message_count"] > 0:
                logger.info(
                    f"[CUSTOMER_AGENT_NODE] Context management: trimmed {trim_stats['trimmed_message_count']} messages, "
                    f"final token count: {trim_stats['final_token_count']}"
                )

            # Build processing messages with customer system prompt and trimmed conversation
            processing_messages = [
                SystemMessage(content=system_prompt)
            ] + trimmed_messages

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

            # =================================================================
            # STEP 3: MEMORY UPDATE (integrated from memory_update_node)
            # =================================================================
            logger.info(f"[CUSTOMER_AGENT_NODE] Step 3: Memory update")
            
            # Store agent response messages to database
            if updated_messages and config:
                for msg in reversed(updated_messages):
                    if hasattr(msg, "type") and msg.type == "ai":
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag",
                            user_id=user_id,
                        )
                        break

            # Store important information in long-term memory
            if user_id and updated_messages and len(updated_messages) >= 2:
                if await self.memory_manager.should_store_memory("rag", updated_messages):
                    recent_messages = updated_messages[-3:] if len(updated_messages) >= 3 else updated_messages
                    await self.memory_manager.store_conversation_insights(
                        "rag", user_id, recent_messages, conversation_id
                    )

            # Check if automatic conversation summarization should be triggered
            if user_id and conversation_id:
                try:
                    summary = await self.memory_manager.consolidator.check_and_trigger_summarization(
                        str(conversation_id), user_id
                    )
                    if summary:
                        logger.info(f"[CUSTOMER_AGENT_NODE] Conversation {conversation_id} automatically summarized")
                except Exception as e:
                    logger.error(f"[CUSTOMER_AGENT_NODE] Error in automatic summarization: {e}")

            logger.info(f"[CUSTOMER_AGENT_NODE] Complete workflow finished successfully")

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

Tool Usage Examples:
- simple_rag: {{"question": "user's complete question", "top_k": 5}}

- simple_query_crm_data: {{"question": "specific business question about sales, customers, vehicles, pricing"}}

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
            messages: List[BaseMessage], conversation_id: UUID
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
        user_query: str,
        conversation_id: str = None,
        user_id: str = None,
        conversation_history: List = None,
    ) -> Dict[str, Any]:
        """
        Invoke the unified tool-calling RAG agent with thread-based persistence.
        """
        # Ensure initialization before invoking
        await self._ensure_initialized()

        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid4())

        # Set up thread-based config for persistence
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
        messages.append(HumanMessage(content=user_query))

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
