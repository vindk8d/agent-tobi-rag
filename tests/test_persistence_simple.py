#!/usr/bin/env python3
"""
Simple test for LangGraph persistence layer configuration.
Tests only the database connection and checkpointer setup.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.memory_manager import memory_manager
from backend.config import get_settings

async def test_persistence_setup():
    """Test the persistence layer setup."""
    print("🔧 Testing LangGraph persistence setup...")
    
    # Test 1: Get settings
    print("\n1. Testing settings...")
    try:
        settings = await get_settings()
        print("✅ Settings loaded successfully")
        print(f"✅ Supabase URL: {settings.supabase.url}")
        print(f"✅ PostgreSQL connection string configured: {bool(settings.supabase.postgresql_connection_string)}")
    except Exception as e:
        print(f"❌ Settings failed: {e}")
        return False
    
    # Test 2: Test memory manager initialization
    print("\n2. Testing memory manager initialization...")
    try:
        await memory_manager._ensure_initialized()
        checkpointer = await memory_manager.get_checkpointer()
        print("✅ Memory manager initialized successfully")
        print(f"✅ Checkpointer type: {type(checkpointer)}")
        print(f"✅ Checkpointer has setup method: {hasattr(checkpointer, 'setup')}")
    except Exception as e:
        print(f"❌ Memory manager initialization failed: {e}")
        return False
    
    # Test 3: Test conversation configuration
    print("\n3. Testing conversation configuration...")
    try:
        from uuid import uuid4
        test_conversation_id = uuid4()
        config = await memory_manager.get_conversation_config(test_conversation_id)
        print(f"✅ Conversation config created: {config}")
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["configurable"]["thread_id"] == str(test_conversation_id)
        print("✅ Thread-based configuration validated")
    except Exception as e:
        print(f"❌ Conversation configuration failed: {e}")
        return False
    
    # Test 4: Test conversation summary logic
    print("\n4. Testing conversation summary logic...")
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Test with fewer than 8 messages (should not update summary)
        messages = [HumanMessage(content="Hello")] * 4
        should_update = await memory_manager.should_update_summary(messages)
        print(f"✅ Should update summary with 4 messages: {should_update}")
        assert should_update is False
        
        # Test with 8 messages (should update summary)
        messages = [HumanMessage(content="Hello")] * 8
        should_update = await memory_manager.should_update_summary(messages)
        print(f"✅ Should update summary with 8 messages: {should_update}")
        assert should_update is True
        
        # Test summary generation
        summary = await memory_manager.generate_conversation_summary(messages)
        print(f"✅ Generated summary: {summary[:100]}...")
        assert summary is not None and len(summary) > 0
        
    except Exception as e:
        print(f"❌ Conversation summary logic failed: {e}")
        return False
    
    print("\n✅ All persistence setup tests passed!")
    return True

async def test_basic_state_schema():
    """Test the basic state schema."""
    print("\n🧪 Testing basic state schema...")
    
    try:
        from backend.agents.state import AgentState
        from langchain_core.messages import HumanMessage
        from uuid import uuid4
        
        # Create a test state
        test_state = AgentState(
            messages=[HumanMessage(content="Test message")],
            conversation_id=uuid4(),
            user_id="test-user",
            retrieved_docs=[],
            sources=[],
            conversation_summary="Test summary"
        )
        
        print("✅ AgentState created successfully")
        print(f"✅ State has {len(test_state['messages'])} messages")
        print(f"✅ State has conversation_id: {test_state['conversation_id']}")
        print(f"✅ State has conversation_summary: {test_state['conversation_summary']}")
        
        return True
    except Exception as e:
        print(f"❌ State schema test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting LangGraph persistence layer simple tests...")
    
    # Test setup
    setup_success = await test_persistence_setup()
    if not setup_success:
        print("❌ Setup tests failed")
        return
    
    # Test state schema
    schema_success = await test_basic_state_schema()
    if not schema_success:
        print("❌ State schema tests failed")
        return
    
    print("\n🎉 All simple tests passed! LangGraph persistence layer is configured correctly.")

if __name__ == "__main__":
    asyncio.run(main()) 