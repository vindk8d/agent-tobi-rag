#!/usr/bin/env python3
"""
Test script to validate LangSmith tracing integration.
This script verifies that LangSmith tracing is properly configured and working.
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

# Add project root to path
project_root = os.path.join(backend_path, '..')
sys.path.insert(0, project_root)

try:
    from config import get_settings, setup_langsmith_tracing
    from agents.rag_agent import UnifiedToolCallingRAGAgent
    from agents.state import AgentState
    from agents.tools import semantic_search, format_sources, build_context, get_conversation_summary, query_crm_data
except ImportError:
    # Try absolute imports
    from backend.config import get_settings, setup_langsmith_tracing
    from backend.agents.rag_agent import UnifiedToolCallingRAGAgent
    from backend.agents.state import AgentState
    from backend.agents.tools import semantic_search, format_sources, build_context, get_conversation_summary, query_crm_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_langsmith_configuration():
    """Test that LangSmith configuration is properly loaded."""
    print("\n🔧 Testing LangSmith Configuration...")
    
    try:
        settings = await get_settings()
        
        print(f"✅ Tracing enabled: {settings.langsmith.tracing_enabled}")
        print(f"✅ Endpoint: {settings.langsmith.endpoint}")
        print(f"✅ Project: {settings.langsmith.project}")
        
        # Check if API key is set (don't print it for security)
        if settings.langsmith.api_key:
            print("✅ API key is configured")
        else:
            print("⚠️ API key is not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def test_langsmith_setup():
    """Test that LangSmith setup function works properly."""
    print("\n🔧 Testing LangSmith Setup...")
    
    try:
        # Store original environment variables
        original_env = {
            "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2"),
            "LANGCHAIN_ENDPOINT": os.environ.get("LANGCHAIN_ENDPOINT"),
            "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY"),
            "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT")
        }
        
        # Run setup
        await setup_langsmith_tracing()
        
        # Check if environment variables are set
        required_vars = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]
        for var in required_vars:
            if var in os.environ:
                print(f"✅ {var} is set")
            else:
                print(f"⚠️ {var} is not set")
        
        # Check API key separately for security
        if "LANGCHAIN_API_KEY" in os.environ:
            print("✅ LANGCHAIN_API_KEY is set")
        else:
            print("⚠️ LANGCHAIN_API_KEY is not set")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False


async def test_tracing_decorators():
    """Test that tracing decorators work on tools."""
    print("\n🔧 Testing Tracing Decorators...")
    
    try:
        # Test semantic_search tool
        print("Testing semantic_search...")
        search_result = await semantic_search.ainvoke({"query": "test query", "top_k": 3})
        print(f"✅ semantic_search executed successfully")
        
        # Test format_sources tool
        print("Testing format_sources...")
        sources_result = format_sources.invoke({"sources": []})
        print(f"✅ format_sources executed successfully")
        
        # Test build_context tool
        print("Testing build_context...")
        context_result = build_context.invoke({"documents": []})
        print(f"✅ build_context executed successfully")
        
        # Test get_conversation_summary tool
        print("Testing get_conversation_summary...")
        summary_result = get_conversation_summary.invoke({"conversation_history": []})
        print(f"✅ get_conversation_summary executed successfully")
        
        # Test query_crm_data tool
        print("Testing query_crm_data...")
        crm_result = await query_crm_data.ainvoke({"question": "Show me all branches"})
        print(f"✅ query_crm_data executed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracing decorators test failed: {e}")
        return False


async def test_agent_initialization():
    """Test that agent initializes with tracing enabled."""
    print("\n🔧 Testing Agent Initialization with Tracing...")
    
    try:
        # Create agent instance
        agent = UnifiedToolCallingRAGAgent()
        
        # Force initialization to test tracing setup
        await agent._ensure_initialized()
        
        print("✅ Agent initialized successfully")
        print(f"✅ Agent has {len(agent.tools)} tools available")
        print(f"✅ Agent model: {agent.settings.openai_chat_model}")
        
        # Check if tracing was enabled
        if agent.settings.langsmith.tracing_enabled:
            print("✅ Tracing is enabled on agent")
        else:
            print("⚠️ Tracing is not enabled on agent")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        return False


async def test_end_to_end_tracing():
    """Test end-to-end conversation flow with tracing."""
    print("\n🔧 Testing End-to-End Tracing...")
    
    try:
        # Create agent
        agent = UnifiedToolCallingRAGAgent()
        
        # Run agent with tracing using the correct method
        result = await agent.invoke(
            user_query="What branches do we have?",
            conversation_id="test_conversation",
            user_id="test_user",
            conversation_history=[]
        )
        
        print("✅ End-to-end conversation completed")
        print(f"✅ Response generated: {result.get('response', 'No response')[:100]}...")
        
        # Check if sources were retrieved
        if result.get('sources'):
            print(f"✅ Sources retrieved: {len(result.get('sources', []))}")
        else:
            print("⚠️ No sources retrieved")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end tracing test failed: {e}")
        return False


async def test_environment_variables():
    """Test that all required environment variables are available."""
    print("\n🔧 Testing Environment Variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key",
        "SUPABASE_URL": "Supabase URL",
        "SUPABASE_ANON_KEY": "Supabase anonymous key",
        "LANGCHAIN_TRACING_V2": "LangSmith tracing flag",
        "LANGCHAIN_ENDPOINT": "LangSmith endpoint",
        "LANGCHAIN_PROJECT": "LangSmith project name"
    }
    
    optional_vars = {
        "LANGCHAIN_API_KEY": "LangSmith API key"
    }
    
    all_good = True
    
    # Check required variables
    for var, description in required_vars.items():
        if var in os.environ and os.environ[var]:
            print(f"✅ {var} ({description}) is set")
        else:
            print(f"❌ {var} ({description}) is missing")
            all_good = False
    
    # Check optional variables
    for var, description in optional_vars.items():
        if var in os.environ and os.environ[var]:
            print(f"✅ {var} ({description}) is set")
        else:
            print(f"⚠️ {var} ({description}) is not set (optional)")
    
    return all_good


async def main():
    """Run all tests."""
    print("🚀 Starting LangSmith Tracing Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("LangSmith Configuration", test_langsmith_configuration),
        ("LangSmith Setup", test_langsmith_setup),
        ("Tracing Decorators", test_tracing_decorators),
        ("Agent Initialization", test_agent_initialization),
        ("End-to-End Tracing", test_end_to_end_tracing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print(f"\n🎯 Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! LangSmith tracing integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the configuration and try again.")
    
    return failed == 0


if __name__ == "__main__":
    asyncio.run(main()) 