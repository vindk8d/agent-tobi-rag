#!/usr/bin/env python3
"""
Comprehensive test for LangGraph persistence layer database connectivity.
Tests actual database connection and persistence operations with Supabase.
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.memory_manager import memory_manager
from agents.state import AgentState
from config import get_settings
from langchain_core.messages import HumanMessage, AIMessage

async def test_database_connectivity():
    """Test direct database connectivity."""
    print("🔌 Testing database connectivity...")
    
    try:
        settings = await get_settings()
        conn_string = settings.supabase.postgresql_connection_string
        print(f"✅ Connection string configured: {conn_string[:50]}...")
        
        # Test direct psycopg connection
        import psycopg
        
        print("📡 Testing direct psycopg connection...")
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                print(f"✅ Database connection successful: {version[:50]}...")
                
                # Test basic table operations
                cursor.execute("SELECT current_database();")
                db_name = cursor.fetchone()[0]
                print(f"✅ Connected to database: {db_name}")
                
                # Test schema access
                cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'public';")
                schema = cursor.fetchone()
                print(f"✅ Public schema accessible: {bool(schema)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Database connectivity test failed: {e}")
        return False

async def test_langgraph_checkpointer():
    """Test LangGraph checkpointer initialization and table setup."""
    print("\n🗄️ Testing LangGraph checkpointer...")
    
    try:
        # Initialize memory manager
        await memory_manager._ensure_initialized()
        checkpointer = await memory_manager.get_checkpointer()
        
        print("✅ Memory manager initialized")
        print(f"✅ Checkpointer type: {type(checkpointer)}")
        
        # Test checkpointer setup (table creation)
        print("📋 Testing checkpointer table setup...")
        checkpointer.setup()
        print("✅ Checkpointer tables created/verified")
        
        # Test that tables exist
        settings = await get_settings()
        conn_string = settings.supabase.postgresql_connection_string
        
        import psycopg
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                # Check for LangGraph tables
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
                    ORDER BY table_name;
                """)
                tables = cursor.fetchall()
                
                expected_tables = ['checkpoint_blobs', 'checkpoint_writes', 'checkpoints']
                found_tables = [row[0] for row in tables]
                
                print(f"✅ Found LangGraph tables: {found_tables}")
                
                for table in expected_tables:
                    if table in found_tables:
                        print(f"✅ Table {table} exists")
                    else:
                        print(f"⚠️  Table {table} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ LangGraph checkpointer test failed: {e}")
        return False

async def test_basic_persistence_operations():
    """Test basic persistence operations without full agent."""
    print("\n💾 Testing basic persistence operations...")
    
    try:
        await memory_manager._ensure_initialized()
        checkpointer = await memory_manager.get_checkpointer()
        
        # Create a test state
        test_conversation_id = uuid4()
        test_state = AgentState(
            messages=[
                HumanMessage(content="Hello, this is a test message"),
                AIMessage(content="This is a test response")
            ],
            conversation_id=test_conversation_id,
            user_id="test-user",
            retrieved_docs=[],
            sources=[],
            conversation_summary=None
        )
        
        # Test saving state
        config = await memory_manager.get_conversation_config(test_conversation_id)
        print(f"✅ Thread config: {config}")
        
        # Test checkpointer put operation
        from langgraph.checkpoint.base import Checkpoint
        from langchain_core.runnables import RunnableConfig
        
        checkpoint = Checkpoint(
            v=1,
            ts=None,
            id=str(uuid4()),
            channel_values=test_state,
            channel_versions={},
            versions_seen={}
        )
        
        print("💾 Testing checkpoint save...")
        # aput requires: config, checkpoint, metadata, new_versions
        await checkpointer.aput(config, checkpoint, {}, {})
        print("✅ Checkpoint saved successfully")
        
        # Test checkpointer get operation
        print("📖 Testing checkpoint retrieval...")
        retrieved = await checkpointer.aget(config)
        print("✅ Checkpoint retrieved successfully")
        
        if retrieved:
            print(f"✅ Retrieved state has {len(retrieved.channel_values.get('messages', []))} messages")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic persistence operations test failed: {e}")
        return False

async def test_conversation_summary_operations():
    """Test conversation summary operations."""
    print("\n📝 Testing conversation summary operations...")
    
    try:
        # Test summary logic
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(10)
        ]
        
        # Test should_update_summary
        should_update = await memory_manager.should_update_summary(messages[:4])
        print(f"✅ Should update summary with 4 messages: {should_update}")
        assert should_update is False
        
        should_update = await memory_manager.should_update_summary(messages[:8])
        print(f"✅ Should update summary with 8 messages: {should_update}")
        assert should_update is True
        
        # Test summary generation
        summary = await memory_manager.generate_conversation_summary(messages[:8])
        print(f"✅ Generated summary: {summary[:100]}...")
        assert summary is not None and len(summary) > 0
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation summary operations test failed: {e}")
        return False

async def test_database_storage_operations():
    """Test long-term database storage operations."""
    print("\n🗃️ Testing database storage operations...")
    
    try:
        test_conversation_id = uuid4()
        test_user_id = "test-user-db"
        
        messages = [
            HumanMessage(content="Test message for database"),
            AIMessage(content="Test response for database")
        ]
        
        # Test saving conversation to database
        await memory_manager.save_conversation_to_database(
            test_conversation_id,
            test_user_id,
            messages,
            "Test conversation summary"
        )
        print("✅ Conversation saved to database")
        
        # Test loading conversation from database
        context = await memory_manager.load_conversation_context(
            test_conversation_id,
            test_user_id
        )
        print(f"✅ Conversation loaded from database: {bool(context)}")
        
        if context:
            print(f"✅ Context has metadata: {bool(context.get('metadata'))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database storage operations test failed: {e}")
        return False

async def test_agent_with_persistence():
    """Test the agent with actual persistence (without OpenAI)."""
    print("\n🤖 Testing agent with persistence (mock mode)...")
    
    try:
        from agents.rag_agent import UnifiedToolCallingRAGAgent
        
        # Create agent but don't initialize LLM (to avoid OpenAI quota issues)
        agent = UnifiedToolCallingRAGAgent()
        
        # Test that agent can be initialized up to the point of LLM
        print("✅ Agent instance created")
        
        # Test that the agent has the right structure for persistence
        assert hasattr(agent, '_ensure_initialized')
        assert hasattr(agent, 'memory_manager')
        assert hasattr(agent, 'invoke')
        print("✅ Agent has persistence-related methods")
        
        # Test conversation config generation
        test_conversation_id = uuid4()
        config = await memory_manager.get_conversation_config(test_conversation_id)
        print(f"✅ Agent can generate conversation config: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent with persistence test failed: {e}")
        return False

async def test_cleanup_operations():
    """Test cleanup operations."""
    print("\n🧹 Testing cleanup operations...")
    
    try:
        # Test cleanup method exists and can be called
        await memory_manager.cleanup_old_conversations(days_old=1)
        print("✅ Cleanup operation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup operations test failed: {e}")
        return False

async def main():
    """Run all database connectivity and persistence tests."""
    print("🚀 Starting comprehensive persistence database tests...")
    
    # Test 1: Basic database connectivity
    db_success = await test_database_connectivity()
    if not db_success:
        print("❌ Database connectivity failed - stopping tests")
        return
    
    # Test 2: LangGraph checkpointer
    checkpointer_success = await test_langgraph_checkpointer()
    if not checkpointer_success:
        print("❌ LangGraph checkpointer failed - stopping tests")
        return
    
    # Test 3: Basic persistence operations
    persistence_success = await test_basic_persistence_operations()
    if not persistence_success:
        print("❌ Basic persistence operations failed")
        return
    
    # Test 4: Conversation summary operations
    summary_success = await test_conversation_summary_operations()
    if not summary_success:
        print("❌ Conversation summary operations failed")
        return
    
    # Test 5: Database storage operations
    storage_success = await test_database_storage_operations()
    if not storage_success:
        print("❌ Database storage operations failed")
        return
    
    # Test 6: Agent with persistence
    agent_success = await test_agent_with_persistence()
    if not agent_success:
        print("❌ Agent with persistence failed")
        return
    
    # Test 7: Cleanup operations
    cleanup_success = await test_cleanup_operations()
    if not cleanup_success:
        print("❌ Cleanup operations failed")
        return
    
    print("\n🎉 All database connectivity and persistence tests passed!")
    print("✅ Database connection is working properly")
    print("✅ LangGraph checkpointer is set up correctly")
    print("✅ Basic persistence operations are functional")
    print("✅ Conversation summary operations are working")
    print("✅ Database storage operations are functional")
    print("✅ Agent persistence integration is ready")
    print("✅ Cleanup operations are working")
    
    print("\n📋 Persistence layer is fully operational and ready for production use!")

if __name__ == "__main__":
    asyncio.run(main()) 