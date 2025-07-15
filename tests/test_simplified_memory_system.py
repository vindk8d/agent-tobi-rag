#!/usr/bin/env python3
"""
Test the simplified memory system without get_conversation_summary tool.
Context is now directly appended to messages in the moving context window.
"""

import asyncio
import sys
import os
from uuid import uuid4

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_core.messages import HumanMessage, AIMessage
from agents.rag_agent import UnifiedToolCallingRAGAgent
from agents.state import AgentState

async def test_simplified_context_handling():
    """Test that context is properly handled without the get_conversation_summary tool."""
    print("🧪 Testing Simplified Context Handling")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state with a realistic conversation
    test_state = AgentState(
        messages=[
            HumanMessage(content="What Toyota models are available?"),
            AIMessage(content="We have Toyota Camry, RAV4, Prius, and Corolla available. The Camry is a popular sedan, RAV4 is great for families, Prius is hybrid, and Corolla is our most affordable option."),
            HumanMessage(content="What are the prices for these?"),
            AIMessage(content="Here are the Toyota prices: Camry starts at $25,000, RAV4 at $28,000, Prius at $24,000, and Corolla at $22,000."),
            HumanMessage(content="can you list all the models with their prices?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print("Scenario:")
    print("  - User asked about Toyota models")
    print("  - Agent provided model information")
    print("  - User asked about prices")
    print("  - Agent provided pricing")
    print("  - User now asks: 'can you list all the models with their prices?'")
    print("  - Agent should understand context without needing to call get_conversation_summary tool")
    
    # Test with full graph
    try:
        config = {
            "configurable": {
                "thread_id": str(test_state['conversation_id'])
            }
        }
        
        result = await agent.graph.ainvoke(test_state, config)
        
        print(f"\nResults:")
        print(f"  - Total messages: {len(result['messages'])}")
        
        # Check that no get_conversation_summary tool was called
        tool_calls = []
        for msg in result['messages']:
            if hasattr(msg, 'name'):
                tool_calls.append(msg.name)
        
        print(f"  - Tools called: {tool_calls}")
        
        if 'get_conversation_summary' in tool_calls:
            print("❌ get_conversation_summary tool was called - it should have been removed!")
            return False
        else:
            print("✅ No get_conversation_summary tool calls - tool successfully removed!")
        
        # Get the final AI response
        final_response = None
        for msg in reversed(result['messages']):
            if hasattr(msg, 'type') and msg.type == 'ai':
                # Check if this is a final response (no tool calls or empty tool calls)
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    final_response = msg.content
                    break
        
        if final_response:
            print(f"\nFinal AI response:")
            print(f"  {final_response}")
            
            # Check if the response addresses Toyota models and prices
            toyota_models = ['Camry', 'RAV4', 'Prius', 'Corolla']
            models_mentioned = [model for model in toyota_models if model in final_response]
            prices_mentioned = ['$25,000', '$28,000', '$24,000', '$22,000']
            price_info = any(price in final_response for price in prices_mentioned)
            
            if models_mentioned and price_info:
                print(f"✅ Agent correctly used context! Found models: {models_mentioned}")
                print("✅ Agent provided pricing information from context!")
                return True
            else:
                print("❌ Agent did not properly use the conversation context")
                return False
        else:
            print("❌ Could not find final AI response")
            return False
        
    except Exception as e:
        print(f"❌ Error in simplified context test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_message_format():
    """Test that enhanced messages are properly formatted with context."""
    print("\n🧪 Testing Enhanced Message Format")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state
    test_state = AgentState(
        messages=[
            HumanMessage(content="What SUVs do you have?"),
            AIMessage(content="We have several SUVs: Toyota RAV4, Honda CR-V, and Ford Escape."),
            HumanMessage(content="which one is cheapest?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print("Scenario:")
    print("  - User asked about SUVs")
    print("  - Agent listed SUV options")
    print("  - User asks 'which one is cheapest?' - should use context to understand SUVs")
    
    # Test memory preparation directly
    try:
        result_state = await agent._memory_preparation_node(test_state)
        
        print(f"\nMemory preparation results:")
        print(f"  - Messages: {len(result_state['messages'])}")
        
        # Check if the last message was enhanced with context
        if result_state['messages']:
            last_message = result_state['messages'][-1]
            print(f"  - Last message type: {type(last_message).__name__}")
            
            if "Previous conversation:" in last_message.content:
                print("✅ Message enhanced with previous conversation context!")
                print(f"  - Enhanced message content preview:")
                print(f"    {last_message.content[:200]}...")
                
                # Check if it contains SUV information
                if "SUVs" in last_message.content and "RAV4" in last_message.content:
                    print("✅ Context properly includes SUV information!")
                    return True
                else:
                    print("❌ Context missing SUV information")
                    return False
            else:
                print("❌ Message not enhanced with context")
                return False
        else:
            print("❌ No messages found")
            return False
            
    except Exception as e:
        print(f"❌ Error in enhanced message test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_no_tool_dependency():
    """Test that the agent works without any dependency on get_conversation_summary tool."""
    print("\n🧪 Testing No Tool Dependency")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Check that get_conversation_summary is not in the available tools
    available_tools = [tool.name for tool in agent.tools]
    
    print(f"Available tools: {available_tools}")
    
    if 'get_conversation_summary' in available_tools:
        print("❌ get_conversation_summary tool is still available - removal failed!")
        return False
    else:
        print("✅ get_conversation_summary tool successfully removed from available tools!")
    
    # Test that the agent can still handle context-dependent queries
    test_state = AgentState(
        messages=[
            HumanMessage(content="Tell me about hybrid vehicles"),
            AIMessage(content="Hybrid vehicles combine an electric motor with a gasoline engine. They're more fuel-efficient and environmentally friendly. Popular hybrids include Toyota Prius and Honda Insight."),
            HumanMessage(content="what's the price of the first one?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    try:
        config = {
            "configurable": {
                "thread_id": str(test_state['conversation_id'])
            }
        }
        
        result = await agent.graph.ainvoke(test_state, config)
        
        # Check for any tool calls
        tool_calls = []
        for msg in result['messages']:
            if hasattr(msg, 'name'):
                tool_calls.append(msg.name)
        
        print(f"Tools called during execution: {tool_calls}")
        
        # Verify no get_conversation_summary calls
        if 'get_conversation_summary' not in tool_calls:
            print("✅ No get_conversation_summary calls - system works independently!")
            return True
        else:
            print("❌ get_conversation_summary was still called somehow!")
            return False
            
    except Exception as e:
        print(f"❌ Error in no tool dependency test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests for the simplified memory system."""
    print("🚀 Testing Simplified Memory System")
    print("=" * 70)
    print("Context is now directly appended to messages without tool dependency")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(await test_simplified_context_handling())
    results.append(await test_enhanced_message_format())
    results.append(await test_no_tool_dependency())
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Simplified memory system is working correctly.")
    else:
        print(f"❌ {total - passed} tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 