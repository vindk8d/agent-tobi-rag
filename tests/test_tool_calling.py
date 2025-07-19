"""
Simple test script for the tool-calling RAG agent.
Demonstrates how the agent decides when to use tools.
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from agents.tools import get_all_tools, get_tool_names

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tool_calling_agent():
    """Test the tool-calling RAG agent with sample queries."""
    
    print("🤖 Testing Unified Tool-Calling RAG Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = UnifiedToolCallingRAGAgent()
    await agent.initialize()
    
    # Test queries that should trigger different tools
    test_queries = [
        "What documents do we have about customer support?",  # Should use lcel_rag
        "How many employees work in sales?",                  # Should use query_crm_data
        "Tell me about our pricing structure",               # Should use lcel_rag  
        "What vehicles do we have in stock?"                # Should use query_crm_data
    ]
    
    print(f"Available tools: {get_tool_names()}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Test with a simple conversation
            config = {
                "configurable": {
                    "thread_id": f"test-thread-{i}"
                }
            }
            
            response = await agent.chat(query, config=config)
            
            print(f"✅ Response received: {len(response)} characters")
            print(f"Preview: {response[:200]}...")
            print()
            
        except Exception as e:
            print(f"❌ Error with query: {str(e)}")
            print()
    
    print("🎉 Tool-calling agent test complete!")


async def test_individual_tools():
    """Test individual tools directly."""
    
    print("🛠️ Testing Individual LCEL Tools")
    print("=" * 50)
    
    from agents.tools import lcel_rag, lcel_retrieval, lcel_generation, query_crm_data
    
    # Test LCEL RAG (complete pipeline)
    print("Testing lcel_rag tool (complete RAG pipeline):")
    try:
        rag_result = await lcel_rag(
            question="What is customer support contact information?",
            top_k=3,
            similarity_threshold=0.7
        )
        print(f"✅ LCEL RAG result:\n{rag_result[:300]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # Test LCEL Retrieval
    print("Testing lcel_retrieval tool:")
    try:
        retrieval_result = await lcel_retrieval(
            question="customer support contact information",
            top_k=3,
            similarity_threshold=0.7
        )
        print(f"✅ LCEL retrieval result:\n{retrieval_result[:300]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # Test LCEL Generation (with sample documents)
    print("Testing lcel_generation tool:")
    try:
        sample_documents = [
            {
                "content": "Our customer support team is available 24/7 via email and chat.",
                "source": "Support Guide.docx",
                "similarity_score": 0.9
            },
            {
                "content": "For technical issues, please use our online ticketing system.",
                "source": "FAQ.pdf",
                "similarity_score": 0.85
            }
        ]
        
        generation_result = await lcel_generation(
            question="How can I contact customer support?",
            documents=sample_documents
        )
        print(f"✅ LCEL generation result:\n{generation_result[:300]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # Test CRM Query
    print("Testing query_crm_data tool:")
    try:
        crm_result = await query_crm_data(
            question="How many employees do we have?"
        )
        print(f"✅ CRM query result:\n{crm_result[:300]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    print("🎉 Individual LCEL tools test complete!")


if __name__ == "__main__":
    # Run both tests
    asyncio.run(test_individual_tools())
    asyncio.run(test_tool_calling_agent()) 