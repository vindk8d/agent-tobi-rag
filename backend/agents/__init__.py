"""
Agents package for RAG-Tobi system.

This package contains the various AI agents and their supporting infrastructure.

Key components:
- UnifiedToolCallingRAGAgent: Main agent for RAG-based conversations
- ConversationMemory: Handles conversation persistence and summarization
- Tools: Collection of tools agents can use (SQL, RAG, etc.)
"""

from .tobi_sales_copilot.state import ConversationMemory
from .tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent

# Export the key classes for easy importing
__all__ = [
    "ConversationMemory",
    "UnifiedToolCallingRAGAgent",
]
