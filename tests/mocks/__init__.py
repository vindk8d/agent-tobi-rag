"""
Mock modules for token-efficient testing.

This package provides mock implementations for external services
to enable comprehensive testing without consuming API tokens or
making external API calls.

Modules:
- mock_llm_responses: Mock LLM responses for summary generation and analysis
"""

from .mock_llm_responses import (
    MockSummaryGenerator,
    MockEmbeddingGenerator,
    MockConversationAnalyzer
)

__all__ = [
    "MockSummaryGenerator",
    "MockEmbeddingGenerator", 
    "MockConversationAnalyzer"
]
