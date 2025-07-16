#!/usr/bin/env python3
"""
Debug script to test if memory_preparation_node is working correctly.
"""

import asyncio
import sys
import os
from uuid import uuid4

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_core.messages import HumanMessage, AIMessage
from backend.agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from backend.agents.tobi_sales_copilot.state import AgentState

async def test_memory_preparation():
    """Test the memory preparation node directly."""
    print("🔍 Testing Memory Preparation Node")
    print("=" * 50)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state similar to what the user showed
    test_state = AgentState(
        messages=[
            HumanMessage(content="Hello there"),
            AIMessage(content="Hi! How can I help you?"),
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="I don't have access to weather information."),
            HumanMessage(content="can you list them all down?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print(f"Original state:")
    print(f"  - Messages: {len(test_state['messages'])}")
    print(f"  - Conversation ID: {test_state['conversation_id']}")
    print(f"  - Summary: {test_state['conversation_summary']}")
    
    # Call memory preparation node directly
    try:
        result_state = await agent._memory_preparation_node(test_state)
        
        print(f"\nAfter memory preparation:")
        print(f"  - Messages: {len(result_state['messages'])}")
        print(f"  - Conversation ID: {result_state['conversation_id']}")
        print(f"  - Summary: {result_state['conversation_summary']}")
        
        # Check if the last message was enhanced with context
        if result_state['messages']:
            last_message = result_state['messages'][-1]
            print(f"\nLast message type: {type(last_message)}")
            print(f"Last message content (first 200 chars):")
            print(f"  {last_message.content[:200]}...")
            
            # Check if it contains previous conversation context
            if "Previous conversation:" in last_message.content:
                print("✅ Context appending is working!")
            else:
                print("❌ Context appending is NOT working!")
        
    except Exception as e:
        print(f"❌ Error in memory preparation: {e}")
        import traceback
        traceback.print_exc()

async def test_full_graph():
    """Test the full graph flow."""
    print("\n🔍 Testing Full Graph Flow")
    print("=" * 50)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state
    test_state = AgentState(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help you?"),
            HumanMessage(content="can you list them all down?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print(f"Initial state:")
    print(f"  - Messages: {len(test_state['messages'])}")
    
    # Test with graph
    try:
        config = {
            "configurable": {
                "thread_id": str(test_state['conversation_id'])
            }
        }
        
        result = await agent.graph.ainvoke(test_state, config)
        
        print(f"\nFinal result:")
        print(f"  - Messages: {len(result['messages'])}")
        print(f"  - Summary: {result['conversation_summary']}")
        
        # Check the final messages
        if result['messages']:
            print(f"\nFinal message types:")
            for i, msg in enumerate(result['messages']):
                print(f"  {i}: {type(msg).__name__}")
                if hasattr(msg, 'content'):
                    print(f"     Content: {msg.content[:100]}...")
        
    except Exception as e:
        print(f"❌ Error in full graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_preparation())
    asyncio.run(test_full_graph()) 