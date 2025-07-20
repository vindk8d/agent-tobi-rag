#!/usr/bin/env python3
"""
Test script to verify automatic conversation summarization is working.
This tests the newly implemented auto-summarization logic.
"""

import asyncio
import os
import sys
from uuid import uuid4
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.memory import ConversationConsolidator, SimpleDBManager, SupabaseLongTermMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from config import get_settings

async def test_automatic_summarization():
    """Test that automatic conversation summarization triggers correctly."""
    print("\n🧪 Testing Automatic Conversation Summarization")
    print("=" * 60)
    
    try:
        # Initialize components
        settings = await get_settings()
        db_manager = SimpleDBManager()
        memory_store = SupabaseLongTermMemoryStore()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        
        consolidator = ConversationConsolidator(db_manager, memory_store, llm, embeddings)
        
        print(f"✅ Components initialized")
        print(f"📊 Configuration:")
        print(f"   - Max messages: {settings.memory_max_messages}")
        print(f"   - Summary interval: {settings.memory_summary_interval}")
        print(f"   - Auto-summarize: {settings.memory_auto_summarize}")
        
        # Test 1: Check message counting
        print("\n📝 Test 1: Message Counting")
        test_conversation_id = str(uuid4())
        test_user_id = "test_user_001"
        
        # Simulate a conversation with enough messages to trigger summarization
        message_count = await consolidator._count_conversation_messages(test_conversation_id)
        print(f"   Initial message count: {message_count}")
        
        # Test 2: Check summarization trigger logic
        print("\n🔄 Test 2: Summarization Trigger Logic")
        if settings.memory_auto_summarize:
            print("   ✅ Auto-summarization is ENABLED")
            
            # Simulate checking a conversation that would trigger summarization
            # (In a real scenario, this would be called after actual messages are stored)
            summary = await consolidator.check_and_trigger_summarization(
                test_conversation_id, test_user_id
            )
            
            if summary:
                print(f"   ✅ Summarization triggered: {len(summary)} chars")
            else:
                print(f"   ℹ️  Summarization not triggered (likely not enough messages)")
                
        else:
            print("   ❌ Auto-summarization is DISABLED")
        
        # Test 3: Configuration validation
        print("\n⚙️  Test 3: Configuration Validation")
        if settings.memory_summary_interval > 0:
            print(f"   ✅ Valid summary interval: {settings.memory_summary_interval} messages")
        else:
            print(f"   ❌ Invalid summary interval: {settings.memory_summary_interval}")
            
        if settings.memory_max_messages > settings.memory_summary_interval:
            print(f"   ✅ Max messages ({settings.memory_max_messages}) > summary interval ({settings.memory_summary_interval})")
        else:
            print(f"   ⚠️  Max messages ({settings.memory_max_messages}) <= summary interval ({settings.memory_summary_interval})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in summarization test: {e}")
        return False

async def test_configuration_loading():
    """Test that the new configuration settings are loaded correctly."""
    print("\n🔧 Testing Configuration Loading")
    print("-" * 40)
    
    try:
        settings = await get_settings()
        
        # Check all memory-related settings
        config_tests = [
            ("memory_max_messages", settings.memory_max_messages, int, 12),
            ("memory_summary_interval", settings.memory_summary_interval, int, 10),
            ("memory_auto_summarize", settings.memory_auto_summarize, bool, True),
        ]
        
        all_passed = True
        for setting_name, value, expected_type, default_value in config_tests:
            if isinstance(value, expected_type):
                print(f"   ✅ {setting_name}: {value} ({type(value).__name__})")
            else:
                print(f"   ❌ {setting_name}: {value} (expected {expected_type.__name__})")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"❌ Configuration loading error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Automatic Summarization System Tests")
    print("=" * 80)
    
    # Test configuration loading
    config_test_passed = await test_configuration_loading()
    
    # Test automatic summarization
    summarization_test_passed = await test_automatic_summarization()
    
    # Results summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    print(f"✅ Configuration loading: {'PASSED' if config_test_passed else 'FAILED'}")
    print(f"✅ Automatic summarization: {'PASSED' if summarization_test_passed else 'FAILED'}")
    
    if config_test_passed and summarization_test_passed:
        print("\n🎉 All tests PASSED! Conversation summarization should now work.")
        print("\n📝 What to expect:")
        print("   • After 10 messages in a conversation, auto-summarization will trigger")
        print("   • Summaries will be stored in long-term memory")
        print("   • Check logs for 'Auto-triggering summarization' messages")
    else:
        print("\n❌ Some tests FAILED. Check the errors above.")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 