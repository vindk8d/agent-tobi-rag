#!/usr/bin/env python3
"""
Comprehensive test for moving context window with realistic scenarios.
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

async def test_realistic_context_reference():
    """Test a realistic scenario where the user refers to previous context."""
    print("🔍 Testing Realistic Context Reference")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state with a realistic conversation
    test_state = AgentState(
        messages=[
            HumanMessage(content="What Toyota models do you have available?"),
            AIMessage(content="We have several Toyota models available: Camry, RAV4, Prius, and Corolla. The Camry is our most popular sedan, while the RAV4 is great for families. Would you like more details about any specific model?"),
            HumanMessage(content="What about the prices?"),
            AIMessage(content="Here are the starting prices: Camry starts at $25,000, RAV4 at $28,000, Prius at $24,000, and Corolla at $22,000. These are base prices and can vary with options and trim levels."),
            HumanMessage(content="can you list them all down?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print(f"Initial conversation scenario:")
    print(f"  - User asked about Toyota models")
    print(f"  - Agent listed: Camry, RAV4, Prius, Corolla")
    print(f"  - User asked about prices")
    print(f"  - Agent provided prices for each model")
    print(f"  - User now asks: 'can you list them all down?'")
    
    # Test with full graph
    try:
        config = {
            "configurable": {
                "thread_id": str(test_state['conversation_id'])
            }
        }
        
        result = await agent.graph.ainvoke(test_state, config)
        
        print(f"\nFinal result:")
        print(f"  - Total messages: {len(result['messages'])}")
        
        # Get the final AI response
        final_response = None
        for msg in reversed(result['messages']):
            if hasattr(msg, 'type') and msg.type == 'ai' and not hasattr(msg, 'tool_calls'):
                final_response = msg.content
                break
        
        if final_response:
            print(f"\nFinal AI response:")
            print(f"  {final_response}")
            
            # Check if the response properly addresses the Toyota models
            if any(model in final_response for model in ['Camry', 'RAV4', 'Prius', 'Corolla']):
                print("✅ Agent correctly understood the context and listed the Toyota models!")
            else:
                print("❌ Agent did not properly reference the Toyota models from context")
        else:
            print("❌ Could not find final AI response")
        
    except Exception as e:
        print(f"❌ Error in realistic context test: {e}")
        import traceback
        traceback.print_exc()

async def test_context_window_reset():
    """Test that the context window resets properly at 12 messages."""
    print("\n🔍 Testing Context Window Reset at 12 Messages")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create 12 messages to trigger reset
    messages = []
    for i in range(6):
        messages.append(HumanMessage(content=f"Question {i+1}: What about vehicle {i+1}?"))
        messages.append(AIMessage(content=f"Answer {i+1}: Vehicle {i+1} is a great choice with these features..."))
    
    # Add the final message that should trigger reset
    messages.append(HumanMessage(content="Can you summarize all the vehicles we discussed?"))
    
    test_state = AgentState(
        messages=messages,
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print(f"Test scenario:")
    print(f"  - Created {len(messages)} messages")
    print(f"  - This should trigger summarization and reset")
    
    # Test memory preparation directly
    try:
        result_state = await agent._memory_preparation_node(test_state)
        
        print(f"\nAfter memory preparation:")
        print(f"  - Messages: {len(result_state['messages'])}")
        print(f"  - Summary exists: {result_state['conversation_summary'] is not None}")
        
        # Check if summarization was triggered
        if result_state['conversation_summary']:
            print(f"  - Summary length: {len(result_state['conversation_summary'])} chars")
            print("✅ Context window reset and summarization worked!")
        else:
            print("❌ Summarization was not triggered")
            
        # Check if the last message has context
        if result_state['messages']:
            last_message = result_state['messages'][-1]
            if "Previous conversation summary:" in last_message.content:
                print("✅ Summary was properly appended to current message!")
            else:
                print("❌ Summary was not appended to current message")
        
    except Exception as e:
        print(f"❌ Error in context window reset test: {e}")
        import traceback
        traceback.print_exc()

async def test_conversation_summary_tool():
    """Test the updated conversation summary tool."""
    print("\n🔍 Testing Updated Conversation Summary Tool")
    print("=" * 60)
    
    # Create agent instance
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create test state with context
    test_state = AgentState(
        messages=[
            HumanMessage(content="What cars do you recommend?"),
            AIMessage(content="I recommend the Toyota Camry for reliability and the Honda Civic for fuel efficiency."),
            HumanMessage(content="What about their prices?")
        ],
        conversation_id=uuid4(),
        user_id="test_user",
        conversation_summary=None,
        retrieved_docs=[],
        sources=[]
    )
    
    print(f"Test scenario:")
    print(f"  - User asked about car recommendations")
    print(f"  - Agent recommended Camry and Civic")
    print(f"  - User now asks about prices")
    
    # Test with full graph to see if tools work correctly
    try:
        config = {
            "configurable": {
                "thread_id": str(test_state['conversation_id'])
            }
        }
        
        result = await agent.graph.ainvoke(test_state, config)
        
        print(f"\nFinal result:")
        print(f"  - Total messages: {len(result['messages'])}")
        
        # Find tool messages from get_conversation_summary
        summary_tool_messages = []
        for msg in result['messages']:
            if hasattr(msg, 'name') and msg.name == 'get_conversation_summary':
                summary_tool_messages.append(msg.content)
        
        if summary_tool_messages:
            print(f"\nConversation summary tool responses:")
            for i, summary in enumerate(summary_tool_messages):
                print(f"  Tool call {i+1}: {summary}")
            
            # Check if the tool properly extracted context
            if any("Toyota Camry" in summary or "Honda Civic" in summary for summary in summary_tool_messages):
                print("✅ Tool correctly extracted previous car recommendations!")
            else:
                print("❌ Tool did not properly extract previous context")
        else:
            print("ℹ️  No conversation summary tool calls found")
        
    except Exception as e:
        print(f"❌ Error in conversation summary tool test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_realistic_context_reference())
    asyncio.run(test_context_window_reset())
    asyncio.run(test_conversation_summary_tool()) 