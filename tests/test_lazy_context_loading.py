#!/usr/bin/env python3
"""
Test script to validate lazy user context loading within agent nodes.
Tests caching, background task scheduling, and performance improvements.
"""

import os
import sys
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


async def test_lazy_context_caching():
    """Test that context caching works correctly with TTL."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {
            "has_history": True,
            "latest_summary": "Test user context",
            "conversation_count": 3
        }
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create agent with test settings
        test_env = {'EMPLOYEE_MAX_MESSAGES': '10'}
        
        with patch.dict(os.environ, test_env, clear=False):
            settings = Settings()
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = settings
            agent.tool_names = []
            
            # Clear any existing cache
            agent._user_context_cache = {}
            
            # Test 1: First call should cache the context
            test_messages = [{"role": "human", "content": "test message"}]
            
            # Call with lazy=False to populate cache
            await agent._load_context_for_employee(test_messages, "user_123", "conv_456", lazy=False)
            
            # Verify context was loaded and cached
            mock_memory_manager.get_user_context_for_new_conversation.assert_called_once_with("user_123")
            
            # Check cache was populated
            cache_key = "user_123:emp"
            assert cache_key in agent._user_context_cache
            cached_data = agent._user_context_cache[cache_key]
            assert cached_data['data']['has_history'] == True
            assert cached_data['data']['latest_summary'] == "Test user context"
            
            print("✅ Context caching works correctly")
            
            # Reset mock for lazy test
            mock_memory_manager.reset_mock()
            
            # Test 2: Second call with lazy=True should use cache
            await agent._load_context_for_employee(test_messages, "user_123", "conv_456", lazy=True)
            
            # Verify memory manager was NOT called (used cache)
            mock_memory_manager.get_user_context_for_new_conversation.assert_not_called()
            
            print("✅ Lazy loading uses cache correctly")
            
            # Test 3: Cache expiration
            # Manually expire the cache by setting old timestamp
            agent._user_context_cache[cache_key]['timestamp'] = time.time() - 400  # Older than 5 minutes
            
            # Mock background task manager for scheduling
            with patch('agents.background_tasks.background_task_manager') as mock_task_manager, \
                 patch('agents.background_tasks.TaskPriority') as mock_priority:
                
                mock_task_manager.schedule_context_loading.return_value = "task_123"
                mock_priority.LOW = "low"
                
                # Call lazy loading with expired cache
                await agent._load_context_for_employee(test_messages, "user_123", "conv_456", lazy=True)
                
                # Verify background task was scheduled
                mock_task_manager.schedule_context_loading.assert_called_once()
                call_args = mock_task_manager.schedule_context_loading.call_args
                assert call_args[1]['user_id'] == "user_123"
                assert call_args[1]['user_type'] == "employee"
                
                print("✅ Cache expiration triggers background loading")


async def test_long_term_context_caching():
    """Test that long-term context caching works with query hashing."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {"has_history": False}
        mock_memory_manager.get_relevant_context.return_value = [
            {"content": "Previous discussion about vehicles"},
            {"content": "Customer preferences for SUVs"}
        ]
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create agent with test settings
        test_env = {'EMPLOYEE_MAX_MESSAGES': '10'}
        
        with patch.dict(os.environ, test_env, clear=False):
            settings = Settings()
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = settings
            agent.tool_names = []
            
            # Clear any existing cache
            agent._long_term_context_cache = {}
            
            # Test messages with specific query
            test_messages = [{"role": "human", "content": "What cars do you recommend?"}]
            
            # Call with lazy=False to populate cache
            await agent._load_context_for_employee(test_messages, "user_456", "conv_789", lazy=False)
            
            # Verify long-term context was loaded
            mock_memory_manager.get_relevant_context.assert_called_once_with("user_456", "What cars do you recommend?", max_contexts=5)
            
            # Check cache was populated
            import hashlib
            query_hash = hashlib.md5("What cars do you recommend?".encode()).hexdigest()[:8]
            cache_key = f"user_456:{query_hash}:emp"
            assert cache_key in agent._long_term_context_cache
            
            print("✅ Long-term context caching works with query hashing")
            
            # Reset mock for lazy test
            mock_memory_manager.reset_mock()
            
            # Test lazy loading uses cache
            await agent._load_context_for_employee(test_messages, "user_456", "conv_789", lazy=True)
            
            # Verify memory manager was NOT called for long-term context (used cache)
            mock_memory_manager.get_relevant_context.assert_not_called()
            
            print("✅ Lazy long-term context loading uses cache correctly")


async def test_customer_context_lazy_loading():
    """Test lazy loading works for customer context with filtering."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_customer_system_prompt') as mock_customer_prompt:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {
            "has_history": True,
            "latest_summary": "Customer interested in SUVs",
            "conversation_count": 2
        }
        mock_memory_manager.get_relevant_context.return_value = [
            {"content": "Customer looking for vehicle with good fuel economy"},  # Should be included
            {"content": "Internal company policy discussion"},  # Should be filtered out
            {"content": "Customer budget is around $30k for a car"}  # Should be included
        ]
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_customer_prompt.return_value = "Customer system prompt"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create agent with test settings
        test_env = {'CUSTOMER_MAX_MESSAGES': '12'}
        
        with patch.dict(os.environ, test_env, clear=False):
            settings = Settings()
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = settings
            agent.tool_names = []
            
            # Clear any existing cache
            agent._user_context_cache = {}
            agent._long_term_context_cache = {}
            
            # Test messages
            test_messages = [{"role": "human", "content": "Show me some cars"}]
            
            # Call with lazy=False to populate cache
            await agent._load_context_for_customer(test_messages, "cust_789", "conv_012", lazy=False)
            
            # Verify customer context was loaded
            mock_memory_manager.get_user_context_for_new_conversation.assert_called_once_with("cust_789")
            mock_memory_manager.get_relevant_context.assert_called_once_with("cust_789", "Show me some cars", max_contexts=3)
            
            # Check that customer-appropriate filtering was applied and cached
            import hashlib
            query_hash = hashlib.md5("Show me some cars".encode()).hexdigest()[:8]
            cache_key = f"cust_789:{query_hash}:cust"
            assert cache_key in agent._long_term_context_cache
            
            cached_context = agent._long_term_context_cache[cache_key]['data']
            # Should have 2 customer-appropriate items (filtered from 3)
            assert len(cached_context) == 2
            
            # Verify content contains customer-appropriate keywords
            cached_contents = [item['content'] for item in cached_context]
            assert any('vehicle' in content or 'car' in content or 'budget' in content for content in cached_contents)
            assert not any('company policy' in content for content in cached_contents)
            
            print("✅ Customer context filtering and caching works correctly")
            
            # Test lazy loading uses filtered cache
            mock_memory_manager.reset_mock()
            
            await agent._load_context_for_customer(test_messages, "cust_789", "conv_012", lazy=True)
            
            # Verify no new memory manager calls (used cache)
            mock_memory_manager.get_user_context_for_new_conversation.assert_not_called()
            mock_memory_manager.get_relevant_context.assert_not_called()
            
            print("✅ Customer lazy loading uses filtered cache correctly")


async def test_performance_improvement():
    """Test that lazy loading provides performance improvements."""
    
    # Mock dependencies with artificial delays
    async def slow_get_user_context(user_id):
        await asyncio.sleep(0.1)  # 100ms delay
        return {"has_history": True, "latest_summary": "Test context", "conversation_count": 1}
    
    async def slow_get_relevant_context(user_id, query, max_contexts=5):
        await asyncio.sleep(0.15)  # 150ms delay
        return [{"content": "Test long-term context"}]
    
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt:
        
        # Setup mocks with delays
        mock_memory_manager.get_user_context_for_new_conversation.side_effect = slow_get_user_context
        mock_memory_manager.get_relevant_context.side_effect = slow_get_relevant_context
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        test_env = {'EMPLOYEE_MAX_MESSAGES': '10'}
        
        with patch.dict(os.environ, test_env, clear=False):
            settings = Settings()
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = settings
            agent.tool_names = []
            
            # Clear cache
            agent._user_context_cache = {}
            agent._long_term_context_cache = {}
            
            test_messages = [{"role": "human", "content": "test query"}]
            
            # Test 1: Full loading (should be slow)
            start_time = time.time()
            await agent._load_context_for_employee(test_messages, "user_perf", "conv_perf", lazy=False)
            full_load_time = time.time() - start_time
            
            print(f"✅ Full context loading time: {full_load_time:.3f}s")
            
            # Test 2: Lazy loading with cache (should be fast)
            start_time = time.time()
            await agent._load_context_for_employee(test_messages, "user_perf", "conv_perf", lazy=True)
            lazy_load_time = time.time() - start_time
            
            print(f"✅ Lazy context loading time: {lazy_load_time:.3f}s")
            
            # Verify performance improvement
            performance_improvement = (full_load_time - lazy_load_time) / full_load_time
            assert performance_improvement > 0.8, f"Expected >80% improvement, got {performance_improvement:.1%}"
            
            print(f"✅ Performance improvement: {performance_improvement:.1%} faster with lazy loading")


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing lazy user context loading...")
        
        try:
            await test_lazy_context_caching()
            await test_long_term_context_caching()
            await test_customer_context_lazy_loading()
            await test_performance_improvement()
            
            print("\n✅ All lazy context loading tests passed!")
            print("📝 Summary:")
            print("   - Context caching works with proper TTL (5 min user, 10 min long-term)")
            print("   - Cache miss triggers background task scheduling")
            print("   - Long-term context uses query hashing for cache keys")
            print("   - Customer context filtering is applied and cached")
            print("   - Lazy loading provides significant performance improvement (>80%)")
            print("   - Background task scheduling handles cache misses gracefully")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

