#!/usr/bin/env python3
"""
Test script for LangGraph persistence layer configuration.
Tests thread-based conversation management with PostgreSQL checkpointing.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.rag_agent import UnifiedToolCallingRAGAgent
from agents.memory_manager import memory_manager
from config import validate_all_configs

async def test_persistence_configuration():
    """Test the persistence layer configuration."""
    print("🔧 Testing LangGraph persistence configuration...")
    
    # Test 1: Validate configurations
    print("\n1. Testing configuration validation...")
    config_valid = await validate_all_configs()
    if not config_valid:
        print("❌ Configuration validation failed")
        return False
    
    # Test 2: Test memory manager initialization
    print("\n2. Testing memory manager initialization...")
    try:
        await memory_manager._ensure_initialized()
        checkpointer = await memory_manager.get_checkpointer()
        print("✅ Memory manager initialized successfully")
        print(f"✅ Checkpointer type: {type(checkpointer)}")
    except Exception as e:
        print(f"❌ Memory manager initialization failed: {e}")
        return False
    
    # Test 3: Test agent initialization with persistence
    print("\n3. Testing agent initialization with persistence...")
    try:
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        print("✅ Agent initialized with persistence layer")
        print(f"✅ Graph type: {type(agent.graph)}")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Test 4: Test conversation configuration
    print("\n4. Testing conversation configuration...")
    try:
        from uuid import uuid4, UUID
        test_conversation_id = uuid4()
        config = await memory_manager.get_conversation_config(test_conversation_id)
        print(f"✅ Conversation config created: {config}")
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["configurable"]["thread_id"] == str(test_conversation_id)
    except Exception as e:
        print(f"❌ Conversation configuration failed: {e}")
        return False
    
    # Test 5: Test conversation summary logic
    print("\n5. Testing conversation summary logic...")
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
    
    print("\n✅ All persistence configuration tests passed!")
    return True

async def test_basic_conversation_flow():
    """Test basic conversation flow with persistence."""
    print("\n🧪 Testing basic conversation flow with persistence...")
    
    try:
        agent = UnifiedToolCallingRAGAgent()
        
        # Test conversation 1
        result1 = await agent.invoke(
            user_query="Hello, can you help me with sales information?",
            conversation_id="test-conv-1",
            user_id="test-user"
        )
        print("✅ First conversation message processed")
        print(f"✅ Response: {result1.get('messages', [])[-1].content[:100]}...")
        
        # Test conversation 2 (same thread)
        result2 = await agent.invoke(
            user_query="What was my previous question?",
            conversation_id="test-conv-1",
            user_id="test-user"
        )
        print("✅ Second conversation message processed")
        print(f"✅ Response: {result2.get('messages', [])[-1].content[:100]}...")
        
        # Verify conversation persistence
        assert len(result2.get('messages', [])) > len(result1.get('messages', []))
        print("✅ Conversation state persisted across messages")
        
    except Exception as e:
        print(f"❌ Conversation flow test failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("🚀 Starting LangGraph persistence layer tests...")
    
    # Test configuration
    config_success = await test_persistence_configuration()
    if not config_success:
        print("❌ Configuration tests failed")
        return
    
    # Test conversation flow
    flow_success = await test_basic_conversation_flow()
    if not flow_success:
        print("❌ Conversation flow tests failed")
        return
    
    print("\n🎉 All tests passed! LangGraph persistence layer is configured correctly.")

if __name__ == "__main__":
    asyncio.run(main()) 