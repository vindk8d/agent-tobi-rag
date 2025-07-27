"""
LangGraph state schema for the RAG agent following best practices.
Uses proper message history management with add_messages annotation.
Enhanced with long-term memory integration.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from uuid import UUID
from datetime import datetime
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph state schema for conversational RAG agent with proper message persistence.

    Follows LangGraph best practices:
    - Uses add_messages annotation for automatic message history management
    - Supports conversation persistence via PostgreSQL checkpointer
    - Maintains only essential state that needs to survive across requests
    - Other data (context, preferences, stats) is retrieved by nodes when needed
    
    Streamlined identification approach:
    - user_verification_node takes user_id and populates either customer_id OR employee_id
    - All other nodes use these IDs directly instead of user_type checks
    """

    # Core conversation state with proper message history management
    # Using add_messages annotation for automatic message appending
    messages: Annotated[List[AnyMessage], add_messages]

    # Persistent memory identifiers
    conversation_id: Optional[str]
    user_id: Optional[str]

    # Direct user identification (populated by user_verification_node)
    customer_id: Optional[str]  # Populated for customer users
    employee_id: Optional[str]  # Populated for employee users

    # RAG context and results (current session)
    retrieved_docs: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]

    # Long-term memory context
    long_term_context: Optional[List[Dict[str, Any]]]

    # Optional conversation summary (for very long conversations)
    conversation_summary: Optional[str]

    # HITL (Human-in-the-Loop) unified interaction state - SINGLE NODE DESIGN
    # 
    # hitl_data: When present, triggers routing to general-purpose HITL node
    # Contains all HITL interaction data in a single, self-contained structure:
    # {
    #   "type": "confirmation" | "selection" | "input_request" | "multi_step_input",
    #   "prompt": "User-facing prompt text",
    #   "options": {...},  # Interaction-specific options (buttons, choices, etc.)
    #   "awaiting_response": bool,  # Internal state tracking
    #   "context": {  # Tool-specific context for post-interaction processing
    #     "source_tool": "tool_name",
    #     "original_args": {...},
    #     "customer_info": {...},
    #     ...
    #   }
    # }
    hitl_data: Optional[Dict[str, Any]]

    # Legacy HITL fields (to be deprecated after migration)
    # PHASE 1: Employee Agent → Confirmation Node
    # confirmation_data: When present, triggers routing to confirmation node for user approval
    # Contains customer message details: customer_id, message_content, message_type, customer_name
    confirmation_data: Optional[Dict[str, Any]]
    
    # PHASE 2: Confirmation Node → Execution Node  
    # execution_data: When present, triggers routing to execution node for message delivery
    # Contains approved delivery details: customer_id, message_content, message_type, customer_info
    execution_data: Optional[Dict[str, Any]]
    
    # PHASE 3: Execution Node → END
    # confirmation_result: When present, indicates completion (prevents re-execution)
    # Contains delivery status: 'delivered', 'cancelled', 'failed' with optional details
    confirmation_result: Optional[str]


class ConversationMemory(TypedDict):
    """
    Conversation memory structure for persistent storage.
    Maps to database tables: conversations and messages.
    """

    conversation_id: str
    user_id: str
    messages: List[BaseMessage]
    conversation_summary: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class LongTermMemoryItem(TypedDict):
    """
    Long-term memory item structure for cross-conversation context.
    Maps to the long_term_memories table.
    """

    id: UUID
    namespace: List[str]
    key: str
    value: Dict[str, Any]
    embedding: List[float]
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int
    last_accessed: datetime
    similarity_score: Optional[float]  # For search results


class ConversationSummary(TypedDict):
    """
    Conversation summary structure for episodic memory.
    Maps to the conversation_summaries table.
    """

    id: UUID
    user_id: str
    conversation_id: UUID
    summary: str
    topics: List[str]
    message_count: int
    date_range: Dict[str, str]
    embedding: List[float]
    created_at: datetime
    consolidated_at: datetime


class MemoryContext(TypedDict):
    """
    Memory context structure for providing relevant background information.
    Used to enhance agent responses with long-term memory.
    """

    type: str  # Type of context (e.g., "conversation_summary", "preference", "fact")
    content: str  # Context content
    relevance_score: float  # How relevant this context is (0-1)
    source: str  # Source of the context
    created_at: datetime  # When this context was created
    namespace: List[str]  # Memory namespace


class MemoryConsolidationStatus(TypedDict):
    """
    Status of memory consolidation for a user.
    Used to track consolidation progress and results.
    """

    user_id: str
    total_conversations: int
    conversations_to_consolidate: int
    consolidated_count: int
    consolidation_in_progress: bool
    last_consolidation: Optional[datetime]
    errors: Optional[List[str]]


class RetrievalContext(TypedDict):
    """
    Context structure for document retrieval results.
    Used internally by RAG tools.
    """

    query: str
    documents: List[Dict[str, Any]]
    similarity_threshold: float
    top_k: int
    search_metadata: Dict[str, Any]


class MemoryRetrievalContext(TypedDict):
    """
    Context structure for long-term memory retrieval results.
    Used internally by memory retrieval tools.
    """

    query: str
    memories: List[LongTermMemoryItem]
    conversation_summaries: List[ConversationSummary]
    user_preferences: Dict[str, Any]
    search_metadata: Dict[str, Any]
    retrieval_time: datetime
