"""
Unified Tool-calling RAG Agent following LangChain best practices.
Single agent node handles both tool calling and execution.
"""

from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4
import logging
import json
import sys
import os
import asyncio

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langsmith import traceable

# Handle both relative and absolute imports for LangGraph Studio compatibility
try:
    from .state import AgentState
    from .tools import get_all_tools, get_tool_names
    from config import get_settings, setup_langsmith_tracing
except ImportError:
    # Fallback to absolute imports when loaded directly by LangGraph Studio
    # Use explicit path resolution WITHOUT calling resolve() to avoid os.getcwd()
    import pathlib
    
    # Get the absolute path of this file without using os.getcwd()
    current_file = pathlib.Path(__file__).absolute()
    
    # Navigate to parent directories without using resolve()
    # This file is in backend/agents/rag_agent.py, so we need to go up two levels
    agents_dir = current_file.parent  # backend/agents/
    backend_dir = agents_dir.parent   # backend/
    project_root = backend_dir.parent  # project root
    
    # Use absolute path without resolve() to avoid blocking calls
    project_root_str = str(project_root)
    
    # Only add to sys.path if not already present
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Now import with absolute paths
    from backend.agents.state import AgentState
    from backend.agents.tools import get_all_tools, get_tool_names
    from backend.config import get_settings, setup_langsmith_tracing

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
        
        self.graph = self._build_graph()
        
        # Log tracing status
        if self.settings.langsmith.tracing_enabled:
            logger.info(f"LangSmith tracing enabled for project: {self.settings.langsmith.project}")
        else:
            logger.info("LangSmith tracing disabled")
            
        self._initialized = True
    
    def _build_graph(self) -> StateGraph:
        """Build a simple graph with unified agent node."""
        graph = StateGraph(AgentState)
        
        # Single agent node that handles everything
        graph.add_node("agent", self._unified_agent_node)
        
        # Simple linear flow
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        
        return graph.compile()
    
    @traceable(name="unified_agent_node")
    async def _unified_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Unified agent node that handles tool calling and execution.
        Follows LangChain best practices for tool calling workflow.
        """
        try:
            # Ensure initialization before processing
            await self._ensure_initialized()
            
            logger.info(f"Agent processing query: {state.get('user_query', 'No query')}")
            
            # Build initial messages
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=self._build_user_prompt(state))
            ]
            
            # Step 3: Tool calling - let model decide when to use tools
            max_iterations = 3  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Agent iteration {iteration}")
                
                # Call the model
                response = await self.llm.ainvoke(messages)
                messages.append(response)
                
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
                                # Add extra validation for specific tools
                                if tool_name == "format_sources":
                                    logger.info(f"format_sources called with args: {tool_args}")
                                    logger.info(f"args type: {type(tool_args)}")
                                    if isinstance(tool_args, dict):
                                        logger.info(f"args keys: {list(tool_args.keys())}")
                                        if 'sources' in tool_args:
                                            logger.info(f"sources type: {type(tool_args['sources'])}")
                                            logger.info(f"sources length: {len(tool_args['sources']) if isinstance(tool_args['sources'], list) else 'Not a list'}")
                                
                                elif tool_name == "get_conversation_summary":
                                    logger.info(f"get_conversation_summary called with args: {tool_args}")
                                    logger.info(f"args type: {type(tool_args)}")
                                    if isinstance(tool_args, dict):
                                        logger.info(f"args keys: {list(tool_args.keys())}")
                                        if 'conversation_history' in tool_args:
                                            logger.info(f"conversation_history type: {type(tool_args['conversation_history'])}")
                                            logger.info(f"conversation_history length: {len(tool_args['conversation_history']) if isinstance(tool_args['conversation_history'], list) else 'Not a list'}")
                                
                                elif tool_name == "build_context":
                                    logger.info(f"build_context called with args: {tool_args}")
                                    logger.info(f"args type: {type(tool_args)}")
                                    if isinstance(tool_args, dict):
                                        logger.info(f"args keys: {list(tool_args.keys())}")
                                        if 'documents' in tool_args:
                                            logger.info(f"documents type: {type(tool_args['documents'])}")
                                            logger.info(f"documents length: {len(tool_args['documents']) if isinstance(tool_args['documents'], list) else 'Not a list'}")
                                
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
                                messages.append(tool_message)
                                
                            else:
                                # Tool not found
                                error_msg = f"Tool '{tool_name}' not found"
                                logger.error(error_msg)
                                tool_message = ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id,
                                    name=tool_name
                                )
                                messages.append(tool_message)
                                
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
                            messages.append(tool_message)
                    
                    # Continue to next iteration to let model process tool results
                    continue
                else:
                    # Model didn't call tools, we have the final response
                    logger.info("Model provided final response without tool calls")
                    break
            
            # Extract final response
            final_response = self._extract_final_response(messages)
            sources = self._extract_sources_from_messages(messages)
            tools_used = self._get_tools_used_from_messages(messages)
            
            return {
                "messages": messages,
                "response": final_response,
                "sources": sources,
                "response_metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": self.settings.openai_chat_model if self.settings else "gpt-4o-mini",
                    "tools_used": tools_used,
                    "iterations": iteration
                },
                "current_step": "agent_complete",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in unified agent node: {str(e)}")
            return {
                "messages": state.get("messages", []),
                "response": "I apologize, but I encountered an error while processing your request.",
                "error_message": f"Agent error: {str(e)}",
                "current_step": "agent_failed"
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the unified agent."""
        return f"""You are a helpful sales assistant with access to company documents and tools.

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
- get_conversation_summary: {{"conversation_history": [list of conversation messages]}} or {{}} for no history

Guidelines:
- Always search for documents first when answering questions
- Use the context from documents to provide accurate answers
- Be concise but thorough in your responses
- Always provide source attribution when you have sources
- If no relevant documents are found, clearly indicate this
- You can use multiple tools in sequence as needed
- When using format_sources, pass the documents array from semantic_search results as the sources parameter
- Use get_conversation_summary to understand previous conversation context when needed

Important: Use the tools to help you provide the best possible assistance to the user."""
    
    def _build_user_prompt(self, state: AgentState) -> str:
        """Build the user prompt with context."""
        user_query = state.get("user_query", "")
        
        # If user_query is empty, try to extract from messages (LangGraph Studio format)
        if not user_query:
            messages = state.get("messages", [])
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'user':
                    user_query = msg.get('content', '')
                    break
        
        conversation_history = state.get("conversation_history", [])
        
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append("Previous conversation context:")
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User's question: {user_query}")
        
        return "\n".join(prompt_parts)
    
    def _extract_final_response(self, messages: List) -> str:
        """Extract the final response from the message chain."""
        # Get the last AI message that's not a tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                return msg.content
        
        # If no final response found, return a default message
        return "I apologize, but I couldn't generate a proper response."
    
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
    
    def _get_tools_used_from_messages(self, messages: List) -> List[str]:
        """Get list of tools used from the messages."""
        tools_used = []
        
        for message in messages:
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name")
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        return tools_used
    
    @traceable(name="rag_agent_invoke")
    async def invoke(self, user_query: str, conversation_id: str = None, user_id: str = None, conversation_history: List = None) -> Dict[str, Any]:
        """
        Invoke the unified tool-calling RAG agent.
        """
        # Ensure initialization before invoking
        await self._ensure_initialized()
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            conversation_id=conversation_id or str(uuid4()),
            user_id=user_id,
            retrieved_docs=[],
            retrieval_metadata={},
            conversation_history=conversation_history or [],
            response="",
            response_metadata={},
            current_step="initialized",
            error_message=None,
            timestamp=datetime.now(),
            sources=[],
            confidence_score=None,
            messages=[]
        )
        
        # Execute the graph
        result = await self.graph.ainvoke(initial_state)
        
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

# For LangGraph Studio compatibility, create a synchronous wrapper
# This will be used by Studio to access the graph
def create_sync_graph():
    """Create a graph synchronously for LangGraph Studio."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we can't use asyncio.run()
            # LangGraph Studio should handle this differently
            logger.warning("Cannot create graph synchronously in async context. Graph will be created on first use.")
            return None
        else:
            return asyncio.run(get_graph())
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(get_graph())

# For LangGraph Studio - try to create graph, but handle async context gracefully
try:
    graph = create_sync_graph()
except Exception as e:
    logger.warning(f"Could not create graph synchronously: {e}. Graph will be created on first use.")
    graph = None

# Export for LangGraph Studio
__all__ = ["UnifiedToolCallingRAGAgent", "ToolCallingRAGAgent", "SimpleRAGAgent", "LinearToolCallingRAGAgent", "graph", "get_graph"] 