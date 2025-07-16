"""
Test script for message storage functionality.
Tests the new row-by-row message storage in the messages table.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from uuid import uuid4, UUID
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.memory import memory_manager
from backend.agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from backend.database import db_client
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_memory_manager_direct():
    """Test the MemoryManager message storage functions directly."""
    print("🧪 Testing MemoryManager message storage directly...")
    
    # Test data
    conversation_id = str(uuid4())
    user_id = str(uuid4())  # Use UUID format
    
    try:
        # First create a user record (required for foreign key constraint)
        user_data = {
            'id': user_id,
            'name': 'Test User',
            'email': 'test@example.com',
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            db_client.client.table('users').insert(user_data).execute()
        except Exception as e:
            # User might already exist, ignore duplicate errors
            if 'duplicate key' not in str(e):
                print(f"Warning: Could not create user: {e}")
        
        # Test 1: Save a user message
        user_message_id = await memory_manager.save_message_to_database(
            conversation_id=conversation_id,
            user_id=user_id,
            role="human",
            content="Hello, this is a test message from a user.",
            metadata={"test": True, "message_type": "user"}
        )
        
        if user_message_id:
            print(f"✅ User message saved successfully: {user_message_id}")
        else:
            print("❌ Failed to save user message")
            return False
        
        # Test 2: Save an agent response
        agent_message_id = await memory_manager.save_message_to_database(
            conversation_id=conversation_id,
            user_id=user_id,
            role="bot",
            content="Hello! I'm the RAG agent. How can I help you today?",
            metadata={"test": True, "message_type": "agent", "agent_type": "rag"}
        )
        
        if agent_message_id:
            print(f"✅ Agent message saved successfully: {agent_message_id}")
        else:
            print("❌ Failed to save agent message")
            return False
        
        # Test 3: Verify messages are in the database
        result = db_client.client.table('messages').select('*').eq('conversation_id', conversation_id).execute()
        
        if result.data and len(result.data) == 2:
            print(f"✅ Found {len(result.data)} messages in database")
            for msg in result.data:
                print(f"   - {msg['role']}: {msg['content'][:50]}...")
        else:
            print(f"❌ Expected 2 messages, found {len(result.data) if result.data else 0}")
            return False
        
        # Test 4: Verify conversation was created
        conv_result = db_client.client.table('conversations').select('*').eq('id', conversation_id).execute()
        
        if conv_result.data and len(conv_result.data) == 1:
            print(f"✅ Conversation record created: {conv_result.data[0]['title']}")
        else:
            print("❌ Conversation record not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in direct memory manager test: {e}")
        return False


async def test_store_message_from_agent():
    """Test the store_message_from_agent method with LangChain messages."""
    print("\n🧪 Testing store_message_from_agent method...")
    
    # Test data
    conversation_id = str(uuid4())
    user_id = str(uuid4())  # Use UUID format
    
    try:
        # First create a user record (required for foreign key constraint)
        user_data = {
            'id': user_id,
            'name': 'Test User 2',
            'email': 'test2@example.com',
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            db_client.client.table('users').insert(user_data).execute()
        except Exception as e:
            # User might already exist, ignore duplicate errors
            if 'duplicate key' not in str(e):
                print(f"Warning: Could not create user: {e}")
        
        # Create config like LangGraph would
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        # Test 1: Store a HumanMessage
        human_msg = HumanMessage(
            content="What is the weather like today?",
            metadata={"user_id": user_id}
        )
        
        user_msg_id = await memory_manager.store_message_from_agent(
            message=human_msg,
            config=config,
            agent_type="rag"
        )
        
        if user_msg_id:
            print(f"✅ HumanMessage stored successfully: {user_msg_id}")
        else:
            print("❌ Failed to store HumanMessage")
            return False
        
        # Test 2: Store an AIMessage
        ai_msg = AIMessage(
            content="I don't have access to real-time weather data, but I can help you find weather information.",
            metadata={"sources": ["weather_api"], "confidence": 0.9}
        )
        
        ai_msg_id = await memory_manager.store_message_from_agent(
            message=ai_msg,
            config=config,
            agent_type="rag"
        )
        
        if ai_msg_id:
            print(f"✅ AIMessage stored successfully: {ai_msg_id}")
        else:
            print("❌ Failed to store AIMessage")
            return False
        
        # Test 3: Verify messages are properly stored
        result = db_client.client.table('messages').select('*').eq('conversation_id', conversation_id).execute()
        
        if result.data and len(result.data) == 2:
            print(f"✅ Found {len(result.data)} messages in database")
            for msg in result.data:
                print(f"   - {msg['role']}: {msg['content'][:50]}...")
                print(f"     Agent type: {msg['metadata'].get('agent_type', 'N/A')}")
        else:
            print(f"❌ Expected 2 messages, found {len(result.data) if result.data else 0}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in store_message_from_agent test: {e}")
        return False


async def test_agent_integration():
    """Test the full agent integration with message storage."""
    print("\n🧪 Testing full agent integration...")
    
    try:
        # First create a user record (required for foreign key constraint)
        user_data = {
            'id': user_id,
            'name': 'Test User 3',
            'email': 'test3@example.com',
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            db_client.client.table('users').insert(user_data).execute()
        except Exception as e:
            # User might already exist, ignore duplicate errors
            if 'duplicate key' not in str(e):
                print(f"Warning: Could not create user: {e}")
        
        # Initialize the agent
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Create a conversation
        conversation_id = UUID(str(uuid4()))
        user_id = str(uuid4())  # Use UUID format
        
        # Test message
        test_message = HumanMessage(
            content="Hello, can you help me with a simple question?",
            metadata={"user_id": user_id}
        )
        
        # Create config for the agent
        config = {
            "configurable": {
                "thread_id": str(conversation_id)
            }
        }
        
        # Create initial state
        initial_state = {
            "messages": [test_message],
            "conversation_id": conversation_id,
            "user_id": user_id,
            "retrieved_docs": [],
            "sources": []
        }
        
        print(f"🤖 Running agent with conversation ID: {conversation_id}")
        
        # Run the agent
        result = await agent.graph.ainvoke(initial_state, config)
        
        if result:
            print(f"✅ Agent completed successfully")
            print(f"   Messages in result: {len(result.get('messages', []))}")
            
            # Wait a moment for database writes to complete
            await asyncio.sleep(2)
            
            # Check if messages were stored
            db_result = db_client.client.table('messages').select('*').eq('conversation_id', str(conversation_id)).execute()
            
            if db_result.data:
                print(f"✅ Found {len(db_result.data)} messages stored in database:")
                for msg in db_result.data:
                    print(f"   - {msg['role']}: {msg['content'][:50]}...")
                    print(f"     Agent type: {msg['metadata'].get('agent_type', 'N/A')}")
                    print(f"     Created: {msg['created_at']}")
            else:
                print("❌ No messages found in database")
                return False
            
            return True
        else:
            print("❌ Agent failed to complete")
            return False
            
    except Exception as e:
        print(f"❌ Error in agent integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_data():
    """Clean up test data from the database."""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Delete test messages
        db_client.client.table('messages').delete().eq('metadata->>test', 'true').execute()
        
        # Delete test conversations
        db_client.client.table('conversations').delete().eq('metadata->>created_by', 'memory_manager').execute()
        
        # Delete test users
        db_client.client.table('users').delete().like('email', 'test%@example.com').execute()
        
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"❌ Error cleaning up test data: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting message storage tests...")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Direct memory manager functions
    test1_passed = await test_memory_manager_direct()
    all_passed = all_passed and test1_passed
    
    # Test 2: store_message_from_agent method
    test2_passed = await test_store_message_from_agent()
    all_passed = all_passed and test2_passed
    
    # Test 3: Full agent integration
    test3_passed = await test_agent_integration()
    all_passed = all_passed and test3_passed
    
    # Cleanup
    await cleanup_test_data()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Message storage is working correctly.")
    else:
        print("❌ Some tests failed. Please check the logs above.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main()) 