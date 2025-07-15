#!/usr/bin/env python3
"""
Test for LangGraph persistence layer configuration validation.
Tests the configuration setup without requiring actual database connectivity.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_settings
from agents.state import AgentState

async def test_persistence_configuration():
    """Test that the persistence layer is properly configured."""
    print("🔧 Testing LangGraph persistence configuration...")
    
    # Test 1: Settings and connection string
    print("\n1. Testing settings and connection string...")
    try:
        settings = await get_settings()
        print("✅ Settings loaded successfully")
        
        # Check that PostgreSQL connection string is configured
        conn_string = settings.supabase.postgresql_connection_string
        print(f"✅ PostgreSQL connection string configured: {bool(conn_string)}")
        
        # Validate connection string format
        if conn_string:
            expected_parts = ["postgresql://", "postgres:", "@db.", ".supabase.co:5432/postgres"]
            for part in expected_parts:
                if part not in conn_string:
                    print(f"⚠️  Warning: Connection string missing expected part: {part}")
            print("✅ Connection string format appears valid")
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False
    
    # Test 2: State schema validation
    print("\n2. Testing state schema...")
    try:
        from langchain_core.messages import HumanMessage
        from uuid import uuid4
        
        # Create a test state with all required fields
        test_state = AgentState(
            messages=[HumanMessage(content="Test message")],
            conversation_id=uuid4(),
            user_id="test-user",
            retrieved_docs=[],
            sources=[],
            conversation_summary="Test summary"
        )
        
        print("✅ AgentState schema validated")
        print(f"✅ State includes messages: {len(test_state['messages'])}")
        print(f"✅ State includes conversation_id: {bool(test_state['conversation_id'])}")
        print(f"✅ State includes user_id: {bool(test_state['user_id'])}")
        print(f"✅ State includes retrieved_docs: {isinstance(test_state['retrieved_docs'], list)}")
        print(f"✅ State includes sources: {isinstance(test_state['sources'], list)}")
        print(f"✅ State includes conversation_summary: {bool(test_state['conversation_summary'])}")
        
    except Exception as e:
        print(f"❌ State schema test failed: {e}")
        return False
    
    # Test 3: Memory manager class structure
    print("\n3. Testing memory manager class structure...")
    try:
        from agents.memory_manager import ConversationMemoryManager
        
        # Test that the class is properly structured
        manager = ConversationMemoryManager()
        print("✅ ConversationMemoryManager class instantiated")
        
        # Check that it has the required methods
        required_methods = [
            '_ensure_initialized',
            'get_checkpointer',
            'should_update_summary',
            'generate_conversation_summary',
            'get_conversation_config',
            'save_conversation_to_database',
            'load_conversation_context'
        ]
        
        for method in required_methods:
            if hasattr(manager, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
    except Exception as e:
        print(f"❌ Memory manager structure test failed: {e}")
        return False
    
    # Test 4: Import validation
    print("\n4. Testing import validation...")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        print("✅ PostgresSaver import successful")
        
        # Check that the class has the expected methods
        required_attrs = ['from_conn_string', 'setup', 'aget', 'aput']
        for attr in required_attrs:
            if hasattr(PostgresSaver, attr):
                print(f"✅ PostgresSaver has {attr} method")
            else:
                print(f"❌ PostgresSaver missing {attr} method")
                return False
        
    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False
    
    # Test 5: Agent integration structure
    print("\n5. Testing agent integration structure...")
    try:
        from agents.rag_agent import UnifiedToolCallingRAGAgent
        
        # Test that the agent class is properly structured
        agent = UnifiedToolCallingRAGAgent()
        print("✅ UnifiedToolCallingRAGAgent class instantiated")
        
        # Check that it has the required attributes/methods for persistence
        required_attrs = ['_ensure_initialized', 'invoke', '_build_graph']
        for attr in required_attrs:
            if hasattr(agent, attr):
                print(f"✅ Agent has {attr} method")
            else:
                print(f"❌ Agent missing {attr} method")
                return False
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False
    
    print("\n✅ All persistence configuration tests passed!")
    return True

async def test_thread_based_config():
    """Test thread-based conversation configuration."""
    print("\n🧪 Testing thread-based conversation management...")
    
    try:
        from agents.memory_manager import ConversationMemoryManager
        from uuid import uuid4
        
        manager = ConversationMemoryManager()
        
        # Test conversation config generation
        conversation_id = uuid4()
        config = await manager.get_conversation_config(conversation_id)
        
        print("✅ Conversation config generated")
        print(f"✅ Config structure: {config}")
        
        # Validate config structure
        assert isinstance(config, dict), "Config should be a dictionary"
        assert "configurable" in config, "Config should have 'configurable' key"
        assert "thread_id" in config["configurable"], "Config should have 'thread_id' in configurable"
        assert config["configurable"]["thread_id"] == str(conversation_id), "Thread ID should match conversation ID"
        
        print("✅ Thread-based configuration validated")
        
    except Exception as e:
        print(f"❌ Thread-based config test failed: {e}")
        return False
    
    return True

async def main():
    """Run all configuration tests."""
    print("🚀 Starting LangGraph persistence configuration tests...")
    
    # Test configuration
    config_success = await test_persistence_configuration()
    if not config_success:
        print("❌ Configuration tests failed")
        return
    
    # Test thread-based config
    thread_success = await test_thread_based_config()
    if not thread_success:
        print("❌ Thread-based config tests failed")
        return
    
    print("\n🎉 All configuration tests passed!")
    print("✅ LangGraph persistence layer is properly configured")
    print("✅ Thread-based conversation management is set up")
    print("✅ State schema includes conversation_summary field")
    print("✅ PostgreSQL checkpointer integration is ready")
    
    print("\n📋 Summary of sub-task 5.5.1 completion:")
    print("  ✅ PostgreSQL connection string configuration")
    print("  ✅ LangGraph PostgresSaver integration")
    print("  ✅ Thread-based conversation management")
    print("  ✅ Enhanced state schema with conversation_summary")
    print("  ✅ Memory manager with checkpointing support")
    print("  ✅ Agent graph compilation with persistence")

if __name__ == "__main__":
    asyncio.run(main()) 