"""
Simple test script for the tool-calling RAG agent.
Demonstrates how the agent decides when to use tools.
"""

import asyncio
import logging
from .rag_agent import UnifiedToolCallingRAGAgent
from .tools import get_all_tools, get_tool_names

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tool_calling_agent():
    """Test the tool-calling RAG agent with sample queries."""
    
    print("🤖 Testing Unified Tool-Calling RAG Agent")
    print("=" * 50)
    
    # Initialize the agent
    agent = UnifiedToolCallingRAGAgent()
    
    # Show available tools
    print(f"Available tools: {get_tool_names()}")
    print()
    
    # Test queries
    test_queries = [
        "What is our company's return policy?",
        "How do I contact customer support?",
        "What are the pricing options for premium features?",
        "Tell me about the latest product updates"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"📝 Test Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Invoke the agent
            result = await agent.invoke(
                user_query=query,
                conversation_id=f"test-conversation-{i}",
                user_id="test-user"
            )
            
            # Display results
            print(f"✅ Response: {result.get('response', 'No response')}")
            print(f"🔧 Tools used: {result.get('response_metadata', {}).get('tools_used', [])}")
            print(f"📚 Sources: {len(result.get('sources', []))} sources found")
            print(f"🎯 Current step: {result.get('current_step', 'unknown')}")
            
            if result.get('error_message'):
                print(f"❌ Error: {result.get('error_message')}")
            
        except Exception as e:
            print(f"❌ Error testing query: {str(e)}")
        
        print()
    
    print("🎉 Tool-calling agent test complete!")


async def test_individual_tools():
    """Test individual tools directly."""
    
    print("🛠️ Testing Individual Tools")
    print("=" * 50)
    
    from .tools import semantic_search, format_sources, build_context
    
    # Test semantic search
    print("Testing semantic_search tool:")
    try:
        search_result = await semantic_search(
            query="customer support contact information",
            top_k=3
        )
        print(f"✅ Search result: {search_result[:200]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # Test format_sources
    print("Testing format_sources tool:")
    try:
        sample_sources = [
            {"source": "FAQ.pdf", "similarity": 0.95},
            {"source": "Support Guide.docx", "similarity": 0.87},
            {"source": "Company Policy.pdf", "similarity": 0.82}
        ]
        formatted = format_sources(sample_sources)
        print(f"✅ Formatted sources:\n{formatted}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # Test build_context
    print("Testing build_context tool:")
    try:
        sample_documents = [
            {
                "content": "Our customer support team is available 24/7 via email and chat.",
                "source": "Support Guide.docx",
                "similarity": 0.9
            },
            {
                "content": "For technical issues, please use our online ticketing system.",
                "source": "FAQ.pdf",
                "similarity": 0.85
            }
        ]
        context = build_context(sample_documents)
        print(f"✅ Built context:\n{context}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    print("🎉 Individual tools test complete!")


if __name__ == "__main__":
    # Run both tests
    asyncio.run(test_individual_tools())
    asyncio.run(test_tool_calling_agent()) 