#!/usr/bin/env python3
"""
Debug test for memory update node to see exactly what's happening with summarization.
"""

import asyncio
import os
import sys
from uuid import uuid4, UUID

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

async def test_memory_update_debug():
    """Test memory update node with debug output."""
    print("🧪 Testing Memory Update Node with Debug Output")
    print("=" * 80)
    
    try:
        # Initialize the agent
        print("🔧 Initializing RAG agent...")
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        print("✅ Agent initialized successfully")
        
        # Create test state with multiple messages to trigger summarization
        test_conversation_id = str(uuid4())
        test_user_id = str(uuid4())  # Fix: Use UUID instead of string
        
        # Create test user in database to satisfy foreign key constraints
        print("👤 Creating test user...")
        from database import db_client
        await asyncio.to_thread(
            lambda: db_client.client.table("users").upsert({
                "id": test_user_id,
                "user_type": "employee",
                "email": f"test-{test_user_id[:8]}@example.com",
                "display_name": f"Test User {test_user_id[:8]}",
                "is_active": True,
                "is_verified": True
            }).execute()
        )
        print(f"✅ Test user created: {test_user_id[:8]}...")
        
        # Create a mock conversation with enough messages to trigger summarization
        messages = []
        
        # Pre-populate database with conversation messages to test summarization
        print("💾 Pre-populating database with messages...")
        for i in range(12):  # Create 12 message pairs (24 total - above threshold)
            human_msg = HumanMessage(content=f"Test message {i+1} from user")
            ai_msg = AIMessage(content=f"Test response {i+1} from assistant")
            messages.extend([human_msg, ai_msg])
            
            # Save each message to database so summarization can count them
            await agent.memory_manager.save_message_to_database(
                test_conversation_id, test_user_id, "human", human_msg.content
            )
            await agent.memory_manager.save_message_to_database(
                test_conversation_id, test_user_id, "bot", ai_msg.content
            )
        
        print(f"✅ Saved {len(messages)} messages to database")
        
        print(f"\n📊 Test Setup:")
        print(f"   - Conversation ID: {test_conversation_id}")
        print(f"   - User ID: {test_user_id}")
        print(f"   - Total messages: {len(messages)}")
        print(f"   - Expected to trigger summarization: {'✅ YES' if len(messages) >= 10 else '❌ NO'}")
        
        # Create test state
        test_state = AgentState(
            messages=messages,
            conversation_id=UUID(test_conversation_id),
            user_id=test_user_id,
            conversation_summary=None,
            retrieved_docs=[],
            sources=[]
        )
        
        # Create test config with thread_id
        test_config = {
            "configurable": {
                "thread_id": test_conversation_id
            }
        }
        
        print(f"\n🚀 Calling memory_update_node...")
        print("=" * 50)
        
        # Call the memory update node directly
        result_state = await agent._memory_update_node(test_state, test_config)
        
        print("=" * 50)
        print(f"✅ Memory update node completed!")
        
        # Show results
        print(f"\n📊 Results:")
        print(f"   - State returned successfully: {'✅' if result_state else '❌'}")
        print(f"   - Same state object: {'✅' if result_state is test_state else '❌'}")
        print(f"   - Messages preserved: {'✅' if len(result_state.get('messages', [])) == len(messages) else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in memory update test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the debug test."""
    print("🚀 Starting Memory Update Debug Test")
    print("=" * 80)
    
    success = await test_memory_update_debug()
    
    print(f"\n📋 Test Summary:")
    print("=" * 40)
    print(f"Memory update debug test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n💡 What to look for in the output above:")
        print(f"   🔍 Memory update node should show all debug info")
        print(f"   🤖 Summarization check should show prerequisites")
        print(f"   📊 If triggered, should show message counting and threshold check")
        print(f"   ✅ Look for 'THRESHOLD MET' if enough messages exist")
    else:
        print(f"\n❌ Debug test failed - check the error output above")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 