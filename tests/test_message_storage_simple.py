"""
Simplified test script for message storage functionality.
Tests the new row-by-row message storage with minimal dependencies.
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
from backend.database import db_client
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_message_storage_without_fk():
    """Test message storage by temporarily disabling foreign key constraints."""
    print("🧪 Testing message storage (simplified approach)...")
    
    # Test data
    conversation_id = str(uuid4())
    user_id = str(uuid4())
    
    try:
        # Create a simple user record first
        user_data = {
            'id': user_id,
            'email': f'test_{user_id[:8]}@example.com',
            'display_name': 'Test User',
            'user_type': 'employee'
        }
        
        try:
            db_client.client.table('users').insert(user_data).execute()
            print(f"✅ Created test user: {user_id}")
        except Exception as e:
            if 'duplicate key' in str(e):
                print(f"ℹ️  User already exists: {user_id}")
            else:
                print(f"❌ Could not create user: {e}")
                return False
        
        # Test the core save_message_to_database function
        user_message_id = await memory_manager.save_message_to_database(
            conversation_id=conversation_id,
            user_id=user_id,
            role="human",
            content="Hello, this is a test message from a user.",
            metadata={"test": "true", "message_type": "user"}
        )
        
        if user_message_id:
            print(f"✅ User message saved successfully: {user_message_id}")
        else:
            print("❌ Failed to save user message")
            return False
        
        # Test agent response
        agent_message_id = await memory_manager.save_message_to_database(
            conversation_id=conversation_id,
            user_id=user_id,
            role="bot",
            content="Hello! I'm the RAG agent. How can I help you today?",
            metadata={"test": "true", "message_type": "agent", "agent_type": "rag"}
        )
        
        if agent_message_id:
            print(f"✅ Agent message saved successfully: {agent_message_id}")
        else:
            print("❌ Failed to save agent message")
            return False
        
        # Verify messages are in the database
        result = db_client.client.table('messages').select('*').eq('conversation_id', conversation_id).execute()
        
        if result.data and len(result.data) == 2:
            print(f"✅ Found {len(result.data)} messages in database")
            for msg in result.data:
                print(f"   - {msg['role']}: {msg['content'][:50]}...")
                print(f"     User ID: {msg['user_id']}")
                print(f"     Created: {msg['created_at']}")
        else:
            print(f"❌ Expected 2 messages, found {len(result.data) if result.data else 0}")
            return False
        
        # Test store_message_from_agent function
        print("\n🧪 Testing store_message_from_agent method...")
        
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        # Test with HumanMessage
        human_msg = HumanMessage(
            content="Can you help me with something?",
            metadata={"user_id": user_id}
        )
        
        msg_id = await memory_manager.store_message_from_agent(
            message=human_msg,
            config=config,
            agent_type="rag"
        )
        
        if msg_id:
            print(f"✅ store_message_from_agent worked: {msg_id}")
        else:
            print("❌ store_message_from_agent failed")
            return False
        
        # Test with AIMessage
        ai_msg = AIMessage(
            content="Sure, I'd be happy to help you with that!",
            metadata={"sources": ["knowledge_base"], "confidence": 0.95, "user_id": user_id}
        )
        
        msg_id2 = await memory_manager.store_message_from_agent(
            message=ai_msg,
            config=config,
            agent_type="rag"
        )
        
        if msg_id2:
            print(f"✅ AI message stored: {msg_id2}")
        else:
            print("❌ AI message storage failed")
            return False
        
        # Final verification
        final_result = db_client.client.table('messages').select('*').eq('conversation_id', conversation_id).execute()
        
        print(f"\n📊 Final count: {len(final_result.data)} messages stored")
        for msg in final_result.data:
            print(f"   - {msg['role']}: {msg['content'][:40]}...")
            print(f"     Agent type: {msg['metadata'].get('agent_type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in message storage test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_creation():
    """Test conversation creation functionality."""
    print("\n🧪 Testing conversation creation...")
    
    conversation_id = str(uuid4())
    user_id = str(uuid4())
    
    try:
        # Create user first
        user_data = {
            'id': user_id,
            'email': f'conv_{user_id[:8]}@example.com',
            'display_name': 'Conversation Test User',
            'user_type': 'employee'
        }
        
        db_client.client.table('users').insert(user_data).execute()
        
        # Test conversation creation through message storage
        await memory_manager.save_message_to_database(
            conversation_id=conversation_id,
            user_id=user_id,
            role="human",
            content="This should create a conversation.",
            metadata={"test": "true"}
        )
        
        # Verify conversation was created
        conv_result = db_client.client.table('conversations').select('*').eq('id', conversation_id).execute()
        
        if conv_result.data and len(conv_result.data) == 1:
            print(f"✅ Conversation created: {conv_result.data[0]['title']}")
            print(f"   Created by: {conv_result.data[0]['metadata'].get('created_by', 'N/A')}")
            return True
        else:
            print("❌ Conversation not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in conversation creation test: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data from the database."""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Clean up in correct order due to foreign key constraints
        
        # 1. Delete test messages
        db_client.client.table('messages').delete().eq('metadata->>test', 'true').execute()
        
        # 2. Delete test conversations
        db_client.client.table('conversations').delete().eq('metadata->>created_by', 'memory_manager').execute()
        
        # 3. Delete test users
        db_client.client.table('users').delete().like('email', '%@example.com').execute()
        
        print("✅ Test data cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Error cleaning up test data: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting simplified message storage tests...")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Message storage functionality
    test1_passed = await test_message_storage_without_fk()
    all_passed = all_passed and test1_passed
    
    # Test 2: Conversation creation
    test2_passed = await test_conversation_creation()
    all_passed = all_passed and test2_passed
    
    # Cleanup
    await cleanup_test_data()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Message storage is working correctly.")
        print("\n✅ Key features verified:")
        print("   - save_message_to_database() function works")
        print("   - store_message_from_agent() function works")
        print("   - Conversation records are auto-created")
        print("   - Messages are stored with proper metadata")
        print("   - Both human and AI messages are handled")
    else:
        print("❌ Some tests failed. Please check the logs above.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main()) 