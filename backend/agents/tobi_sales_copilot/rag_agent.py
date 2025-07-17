"""
Unified Tool-calling RAG Agent following LangChain best practices.
Single agent node handles both tool calling and execution.
"""

from typing import Dict, Any, List, Union
from datetime import datetime
from uuid import uuid4, UUID
import logging
import json
import sys
import os
import asyncio

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langsmith import traceable

# Handle both relative and absolute imports for LangGraph Studio compatibility
try:
    from .state import AgentState
    from ..tools import get_all_tools, get_tool_names
    from ..memory import memory_manager, memory_scheduler
    from ...config import get_settings, setup_langsmith_tracing
except ImportError:
    # Fallback to absolute imports when loaded directly by LangGraph Studio
    # Use explicit path resolution WITHOUT calling resolve() to avoid os.getcwd()
    import pathlib
    
    # Get the absolute path of this file without using os.getcwd()
    current_file = pathlib.Path(__file__).absolute()
    
    # Navigate to parent directories without using resolve()
    # This file is in backend/agents/tobi_sales_copilot/rag_agent.py, so we need to go up three levels
    copilot_dir = current_file.parent   # backend/agents/tobi_sales_copilot/
    agents_dir = copilot_dir.parent     # backend/agents/
    backend_dir = agents_dir.parent     # backend/
    project_root = backend_dir.parent   # project root
    
    # Use absolute path without resolve() to avoid blocking calls
    project_root_str = str(project_root)
    
    # Only add to sys.path if not already present
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Now import with absolute paths
    from agents.tobi_sales_copilot.state import AgentState
    from agents.tools import get_all_tools, get_tool_names
    from agents.memory import memory_manager, memory_scheduler
    from config import get_settings, setup_langsmith_tracing

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
        
        # Step 2: Tool binding - connect tools to model
        self.llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            temperature=self.settings.openai_temperature,
            max_tokens=self.settings.openai_max_tokens,
            api_key=self.settings.openai_api_key
        ).bind_tools(self.tools)
        
        # Create a tool lookup for execution
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Build graph with persistence
        self.graph = await self._build_graph()
        
        # Log tracing status
        if self.settings.langsmith.tracing_enabled:
            logger.info(f"LangSmith tracing enabled for project: {self.settings.langsmith.project}")
        else:
            logger.info("LangSmith tracing disabled")
            
        self._initialized = True
    

    
    async def _build_graph(self) -> StateGraph:
        """Build a graph with automatic checkpointing and state persistence between agent steps."""
        # Import memory manager using absolute import for LangGraph Studio compatibility
        try:
            from ..memory import memory_manager
        except ImportError:
            # Fallback to absolute import when loaded directly by LangGraph Studio
            from agents.memory import memory_manager
        
        # Get checkpointer for persistence
        checkpointer = await memory_manager.get_checkpointer()
        
        graph = StateGraph(AgentState)
        
        # Add nodes with automatic checkpointing between steps
        graph.add_node("memory_preparation", self._memory_preparation_node)
        graph.add_node("agent", self._unified_agent_node)
        graph.add_node("memory_update", self._memory_update_node)
        
        # Graph flow with automatic checkpointing between each step
        graph.add_edge(START, "memory_preparation")
        graph.add_edge("memory_preparation", "agent")
        graph.add_edge("agent", "memory_update")
        graph.add_edge("memory_update", END)
        
        # Compile with checkpointer for automatic persistence between steps
        return graph.compile(checkpointer=checkpointer)
    

    

    

    


    @traceable(name="memory_preparation_node")
    async def _memory_preparation_node(self, state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare conversation memory with long-term context retrieval.
        LangGraph handles short-term persistence automatically.
        """
        try:
            await self._ensure_initialized()
            
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")
            user_id = state.get("user_id")
            
            # Extract thread_id from config if no conversation_id is provided
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)
                    logger.info(f"Using thread_id {thread_id} as conversation_id for persistence")
            
            logger.info(f"Memory preparation for conversation {conversation_id}: {len(messages)} messages")
            
            # Store incoming user messages to database
            if messages and config:
                # Find the most recent human message to store
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'human':
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag"
                        )
                        break
            
            # Get relevant long-term context if we have a user query
            long_term_context = []
            if user_id and messages:
                # Extract current query for context retrieval
                current_query = ""
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'human':
                        current_query = msg.content
                        break
                
                if current_query:
                    long_term_context = await self.memory_manager.get_relevant_context(
                        user_id, current_query, max_contexts=3
                    )
                    if long_term_context:
                        logger.info(f"Retrieved {len(long_term_context)} relevant context items")
            
            # Return state with long-term context
            return {
                "messages": messages,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_summary": state.get("conversation_summary"),
                "retrieved_docs": state.get("retrieved_docs", []),
                "sources": state.get("sources", []),
                "long_term_context": long_term_context
            }
            
        except Exception as e:
            logger.error(f"Error in memory preparation node: {str(e)}")
            # Return original state on error
            return state
    
    @traceable(name="memory_update_node")
    async def _memory_update_node(self, state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update long-term memory after agent execution.
        LangGraph handles short-term persistence automatically.
        """
        try:
            await self._ensure_initialized()
            
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")
            user_id = state.get("user_id")
            current_summary = state.get("conversation_summary")
            
            # Extract thread_id from config if no conversation_id is provided
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)
                    logger.info(f"Using thread_id {thread_id} as conversation_id for persistence")
            
            logger.info(f"Memory update for conversation {conversation_id}: {len(messages)} messages")
            
            # Store agent response messages to database
            if messages and config:
                # Find the most recent AI message to store
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        await self.memory_manager.store_message_from_agent(
                            message=msg,
                            config=config,
                            agent_type="rag"
                        )
                        break
            
            # Store important information in long-term memory
            if user_id and messages and len(messages) >= 2:
                # Check if this conversation contains storable information using generic infrastructure
                if await self.memory_manager.should_store_memory("rag", messages):
                    # Extract and store insights using RAG-specific plugins
                    recent_messages = messages[-3:] if len(messages) >= 3 else messages
                    await self.memory_manager.store_conversation_insights(
                        "rag", user_id, recent_messages, conversation_id
                    )
            
            logger.info(f"Memory update complete: long-term context processed")
            
            # Return state unchanged - LangGraph handles persistence automatically
            return state
            
        except Exception as e:
            logger.error(f"Error in memory update node: {str(e)}")
            # Return original state on error
            return state
    
    @traceable(name="unified_agent_node")
    async def _unified_agent_node(self, state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Unified agent node that handles tool calling and execution.
        Follows LangChain best practices for tool calling workflow.
        """
        try:
            # Ensure initialization before processing
            await self._ensure_initialized()
            
            # Extract current query from messages for logging
            original_messages = state.get("messages", [])
            current_query = "No query"
            for msg in reversed(original_messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    current_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'human':
                    current_query = msg.get('content', '')
                    break
            
            # Log thread_id and conversation_id for debugging
            conversation_id = state.get("conversation_id")
            thread_id = config.get("configurable", {}).get("thread_id") if config else None
            logger.info(f"Agent processing query: {current_query}")
            logger.info(f"Thread ID: {thread_id}, Conversation ID: {conversation_id}")
            
            # Preserve original messages and add system prompt
            # Instead of creating a new messages list, work with the original state
            processing_messages = [
                SystemMessage(content=self._get_system_prompt())
            ]
            
            # Add original messages to preserve the conversation context
            processing_messages.extend(original_messages)
            
            # Step 3: Tool calling - let model decide when to use tools
            max_tool_iterations = 3  # Allow more tool calls if needed
            iteration = 0
            
            while iteration < max_tool_iterations:
                iteration += 1
                logger.info(f"Agent iteration {iteration}")
                
                # Call the model
                response = await self.llm.ainvoke(processing_messages)
                processing_messages.append(response)
                
                # Check if model wants to use tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"Model called {len(response.tool_calls)} tools")
                    
                    # Step 4: Tool execution - execute the tools
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]
                        
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        
                        try:
                            # Get the tool and execute it
                            tool = self.tool_map.get(tool_name)
                            if tool:
                                # Check if tool is async by examining the underlying function
                                if tool_name in ["semantic_search", "query_crm_data"]:
                                    # Handle async tools
                                    tool_result = await tool.ainvoke(tool_args)
                                else:
                                    # Handle sync tools
                                    tool_result = tool.invoke(tool_args)
                                
                                # Add tool result to messages
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call_id,
                                    name=tool_name
                                )
                                processing_messages.append(tool_message)
                                
                            else:
                                # Tool not found
                                error_msg = f"Tool '{tool_name}' not found"
                                logger.error(error_msg)
                                tool_message = ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id,
                                    name=tool_name
                                )
                                processing_messages.append(tool_message)
                                
                        except Exception as e:
                            # Tool execution failed
                            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(error_msg)
                            logger.error(f"Tool args that caused error: {tool_args}")
                            tool_message = ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id,
                                name=tool_name
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
                final_response = await self.llm.ainvoke(processing_messages)
                processing_messages.append(final_response)
                logger.info("Generated final response after tool execution")
            
            # Extract sources from messages and update state
            sources = self._extract_sources_from_messages(processing_messages)
            
            # Update the original messages with the AI responses (excluding system message)
            updated_messages = list(original_messages)
            for msg in processing_messages[1:]:  # Skip system message
                if hasattr(msg, 'type') and msg.type in ['ai', 'tool']:
                    updated_messages.append(msg)
            
            # Extract thread_id from config if needed for conversation_id
            conversation_id = state.get("conversation_id")
            if not conversation_id and config:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    conversation_id = UUID(thread_id)
            
            # Update state with original messages preserved plus new responses
            return {
                "messages": updated_messages,
                "sources": sources,
                "retrieved_docs": state.get("retrieved_docs", []),
                "conversation_id": conversation_id,
                "user_id": state.get("user_id"),
                "conversation_summary": state.get("conversation_summary")  # Preserve existing summary
            }
            
        except Exception as e:
            logger.error(f"Error in unified agent node: {str(e)}")
            # Add error message to the conversation
            original_messages = state.get("messages", [])
            error_message = AIMessage(content="I apologize, but I encountered an error while processing your request.")
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
                "conversation_summary": state.get("conversation_summary")  # Preserve existing summary
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the unified agent."""
        return f"""You are a helpful sales assistant with access to company documents, tools, and user context.

Available tools:
{', '.join(self.tool_names)}

Your workflow:
1. When a user asks a question, use semantic_search to find relevant documents
2. If documents are found, use build_context to create a comprehensive context
3. Answer the user's question based on the context
4. If you want to format sources, use format_sources with the documents list from semantic_search results

Tool Usage Examples:
- semantic_search: {{"query": "your search query", "top_k": 5}}
- build_context: {{"documents": [list of documents from semantic_search]}}
- format_sources: {{"sources": [list of documents from semantic_search]}}
- query_crm_data: {{"question": "specific business question about sales, customers, vehicles, pricing"}}

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
- Always search for documents first when answering questions
- Use the context from documents to provide accurate answers
- Be concise but thorough in your responses
- Always provide source attribution when you have sources
- If no relevant documents are found, clearly indicate this
- You can use multiple tools in sequence as needed
- When using format_sources, pass the documents array from semantic_search results as the sources parameter

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
            if isinstance(message, ToolMessage) and message.name == "semantic_search":
                try:
                    content = json.loads(message.content)
                    if isinstance(content, dict) and "documents" in content:
                        for doc in content["documents"]:
                            if doc.get("source"):
                                sources.append({
                                    "source": doc["source"],
                                    "similarity": doc.get("similarity", 0.0)
                                })
                except (json.JSONDecodeError, KeyError):
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
                if any(phrase in content for phrase in [
                    "i prefer", "i like", "i want", "i need",
                    "my name is", "i am", "i work at", "i live in",
                    "remember", "next time", "always", "never",
                    "search for", "find me", "look up"  # RAG-specific patterns
                ]):
                    return True
            
            return False
        
        # Register RAG-specific insight extractor
        async def rag_insight_extractor(messages: List[BaseMessage], conversation_id: UUID) -> List[Dict[str, Any]]:
            """RAG-specific logic for extracting insights from conversations."""
            insights = []
            for message in messages:
                if hasattr(message, 'type') and message.type == 'human':
                    content = message.content.lower()
                    
                    # Extract preferences
                    if any(phrase in content for phrase in ["i prefer", "i like", "i want"]):
                        insights.append({
                            "type": "preference",
                            "content": message.content,
                            "extracted_at": datetime.now().isoformat(),
                            "agent_type": "rag"
                        })
                    
                    # Extract facts
                    elif any(phrase in content for phrase in ["my name is", "i am", "i work at"]):
                        insights.append({
                            "type": "personal_fact",
                            "content": message.content,
                            "extracted_at": datetime.now().isoformat(),
                            "agent_type": "rag"
                        })
                    
                    # Extract RAG-specific search patterns
                    elif any(phrase in content for phrase in ["search for", "find me", "look up"]):
                        insights.append({
                            "type": "search_pattern",
                            "content": message.content,
                            "extracted_at": datetime.now().isoformat(),
                            "agent_type": "rag"
                        })
            
            return insights
        
        # Register the plugins
        self.memory_manager.register_memory_filter("rag", rag_memory_filter)
        self.memory_manager.register_insight_extractor("rag", rag_insight_extractor)
        
        logger.info("RAG-specific memory plugins registered")
    

    
    @traceable(name="rag_agent_invoke")
    async def invoke(self, user_query: str, conversation_id: str = None, user_id: str = None, conversation_history: List = None) -> Dict[str, Any]:
        """
        Invoke the unified tool-calling RAG agent with thread-based persistence.
        """
        # Ensure initialization before invoking
        await self._ensure_initialized()
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid4())
            
        # Set up thread-based config for persistence
        config = await self.memory_manager.get_conversation_config(UUID(conversation_id))
        
        # Try to load existing conversation state from our memory manager
        existing_state = await self.memory_manager.load_conversation_state(UUID(conversation_id))
        
        # Initialize messages - start with existing messages or empty list
        messages = []
        
        if existing_state and existing_state.get('messages'):
            # Use existing messages from our conversation state
            messages = existing_state.get('messages', [])
            logger.info(f"Continuing conversation with {len(messages)} existing messages")
        elif conversation_history:
            # Only add conversation_history if this is a new conversation
            for msg in conversation_history:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if content:  # Only add messages with actual content
                        if msg.get("role") == "human":
                            messages.append(HumanMessage(content=content))
                        elif msg.get("role") == "bot":
                            messages.append(AIMessage(content=content))
                elif isinstance(msg, BaseMessage):
                    messages.append(msg)
        
        # Add the current user query as a human message
        messages.append(HumanMessage(content=user_query))
        
        # Initialize state with existing conversation context
        initial_state = AgentState(
            messages=messages,
            conversation_id=UUID(conversation_id),
            user_id=user_id,
            retrieved_docs=existing_state.get('retrieved_docs', []) if existing_state else [],
            sources=existing_state.get('sources', []) if existing_state else [],
            conversation_summary=existing_state.get('summary') if existing_state else None
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
    "get_graph"
] 