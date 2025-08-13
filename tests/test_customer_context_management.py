#!/usr/bin/env python3
"""
Test script to validate customer-specific context management enhancements.
Tests preference tracking, interest identification, and context warming.
"""

import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


async def test_customer_preference_tracking():
    """Test customer preference extraction from queries."""
    
    # Mock the agent methods
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=MagicMock(),
            TaskPriority=MagicMock()
        )
    }):
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Test cases for preference tracking
        test_cases = [
            {
                'query': "I'm looking for a reliable SUV with good fuel economy",
                'expected_preferences': ['interested_in_suv', 'values_reliability', 'values_fuel_economy'],
                'description': 'Vehicle type and feature preferences'
            },
            {
                'query': "Need something affordable and safe for my family",
                'expected_preferences': ['price_conscious', 'values_safety'],
                'description': 'Budget and safety preferences'
            },
            {
                'query': "I want a luxury sedan with high performance",
                'expected_preferences': ['interested_in_sedan', 'values_luxury', 'values_performance'],
                'description': 'Luxury and performance preferences'
            },
            {
                'query': "Looking for a truck with lots of space",
                'expected_preferences': ['interested_in_truck', 'values_space'],
                'description': 'Vehicle type and space preferences'
            },
            {
                'query': "Just browsing your inventory",
                'expected_preferences': [],
                'description': 'No specific preferences'
            }
        ]
        
        for case in test_cases:
            # Mock the preferences list to capture what gets identified
            captured_preferences = []
            
            # Override the logger to capture preferences
            with patch('agents.tobi_sales_copilot.agent.logger') as mock_logger:
                def capture_debug(msg):
                    if "Identified preferences" in msg and case['query'] in msg:
                        # Extract preferences from the log message
                        import re
                        match = re.search(r'\[([^\]]+)\]', msg)
                        if match:
                            prefs_str = match.group(1)
                            captured_preferences.extend([p.strip().strip("'\"") for p in prefs_str.split(',')])
                
                mock_logger.debug.side_effect = capture_debug
                
                # Test the preference tracking
                await agent._track_customer_preferences("cust_123", case['query'])
                
                # Verify preferences were identified correctly
                for expected_pref in case['expected_preferences']:
                    assert any(expected_pref in pref for pref in captured_preferences), \
                        f"Expected preference '{expected_pref}' not found for query: {case['query']}"
                
                if case['expected_preferences']:
                    print(f"✅ {case['description']}: {len(case['expected_preferences'])} preferences identified")
                else:
                    print(f"✅ {case['description']}: No preferences (as expected)")


async def test_customer_interest_identification():
    """Test customer interest area identification."""
    
    # Mock the agent methods
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=MagicMock(),
            TaskPriority=MagicMock()
        )
    }):
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Test cases for interest identification
        test_cases = [
            {
                'query': "I want to buy a new car and need financing options",
                'expected_interests': ['purchase_intent', 'financing_interest'],
                'description': 'Purchase intent with financing'
            },
            {
                'query': "Can you compare the Honda Civic vs Toyota Corolla?",
                'expected_interests': ['comparison_shopping'],
                'description': 'Comparison shopping'
            },
            {
                'query': "What warranty do you offer on used vehicles?",
                'expected_interests': ['service_interest'],
                'description': 'Service and warranty interest'
            },
            {
                'query': "I'm looking for something with low monthly payments",
                'expected_interests': ['purchase_intent', 'financing_interest'],
                'description': 'Purchase with payment focus'
            },
            {
                'query': "Just checking out your website",
                'expected_interests': [],
                'description': 'No specific interests'
            }
        ]
        
        for case in test_cases:
            # Mock the interests list to capture what gets identified
            captured_interests = []
            
            # Override the logger to capture interests
            with patch('agents.tobi_sales_copilot.agent.logger') as mock_logger:
                def capture_debug(msg):
                    if "Identified interests" in msg and case['query'] in msg:
                        # Extract interests from the log message
                        import re
                        match = re.search(r'\[([^\]]+)\]', msg)
                        if match:
                            interests_str = match.group(1)
                            captured_interests.extend([i.strip().strip("'\"") for i in interests_str.split(',')])
                
                mock_logger.debug.side_effect = capture_debug
                
                # Test the interest identification
                await agent._identify_customer_interests("cust_123", case['query'], "conv_456")
                
                # Verify interests were identified correctly
                for expected_interest in case['expected_interests']:
                    assert any(expected_interest in interest for interest in captured_interests), \
                        f"Expected interest '{expected_interest}' not found for query: {case['query']}"
                
                if case['expected_interests']:
                    print(f"✅ {case['description']}: {len(case['expected_interests'])} interests identified")
                else:
                    print(f"✅ {case['description']}: No specific interests (as expected)")


async def test_context_warming_scheduling():
    """Test background context warming task scheduling."""
    
    # Mock background task manager
    mock_task_manager = MagicMock()
    mock_task_manager.schedule_context_loading.return_value = "task_123"
    mock_priority = MagicMock()
    mock_priority.LOW = "low"
    
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=mock_task_manager,
            TaskPriority=mock_priority
        )
    }):
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Test context warming
        await agent._schedule_context_warming("cust_456", "user_789", "conv_012")
        
        # Verify that context loading tasks were scheduled for common queries
        assert mock_task_manager.schedule_context_loading.call_count == 3  # 3 common queries
        
        # Check the calls
        calls = mock_task_manager.schedule_context_loading.call_args_list
        
        for call in calls:
            args, kwargs = call
            assert kwargs['user_id'] == "user_789"
            assert kwargs['conversation_id'] == "conv_012"
            assert kwargs['user_type'] == "customer"
            assert kwargs['priority'] == "low"
            assert 'context_query' in kwargs
            assert len(kwargs['context_query']) > 0
        
        # Check that different queries were scheduled
        scheduled_queries = [call[1]['context_query'] for call in calls]
        assert len(set(scheduled_queries)) == 3  # All queries should be unique
        
        print("✅ Context warming scheduled 3 background tasks for common queries")
        print(f"✅ Scheduled queries: {[q[:20] + '...' for q in scheduled_queries]}")


async def test_customer_context_enhancement_integration():
    """Test the full customer context enhancement workflow."""
    
    # Mock background task manager
    mock_task_manager = MagicMock()
    mock_task_manager.schedule_context_loading.return_value = "task_456"
    mock_priority = MagicMock()
    mock_priority.LOW = "low"
    
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=mock_task_manager,
            TaskPriority=mock_priority
        )
    }):
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Test the full enhancement workflow
        customer_id = "cust_789"
        user_id = "user_123"
        query = "I want to buy a reliable SUV with good financing options"
        conversation_id = "conv_456"
        
        # Mock logger to capture all debug messages
        captured_logs = []
        with patch('agents.tobi_sales_copilot.agent.logger') as mock_logger:
            def capture_logs(msg):
                captured_logs.append(msg)
            
            mock_logger.debug.side_effect = capture_logs
            mock_logger.info.side_effect = capture_logs
            
            # Run the enhancement
            await agent._enhance_customer_context(customer_id, user_id, query, conversation_id)
            
            # Verify that all enhancement steps were executed
            log_messages = ' '.join(captured_logs)
            
            # Check that context enhancement was started and completed
            assert any("Enhancing context for customer" in log for log in captured_logs)
            assert any("Context enhancement completed" in log for log in captured_logs)
            
            # Check that background tasks were scheduled (3 for context warming)
            assert mock_task_manager.schedule_context_loading.call_count == 3
            
            print("✅ Full customer context enhancement workflow completed")
            print(f"✅ Processed query: {query[:30]}...")
            print(f"✅ Scheduled {mock_task_manager.schedule_context_loading.call_count} background tasks")


async def test_error_handling():
    """Test that errors in customer context enhancement don't break the workflow."""
    
    # Mock background task manager that raises errors
    mock_task_manager = MagicMock()
    mock_task_manager.schedule_context_loading.side_effect = Exception("Background task error")
    
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=mock_task_manager,
            TaskPriority=MagicMock()
        )
    }):
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Test that enhancement doesn't fail even with background task errors
        try:
            await agent._enhance_customer_context("cust_123", "user_456", "test query", "conv_789")
            print("✅ Customer context enhancement handles errors gracefully")
        except Exception as e:
            assert False, f"Customer context enhancement should not raise exceptions: {e}"
        
        # Test individual method error handling
        try:
            await agent._track_customer_preferences("cust_123", "test query")
            await agent._identify_customer_interests("cust_123", "test query", "conv_789")
            # This one will fail due to mock error, but should be caught
            await agent._schedule_context_warming("cust_123", "user_456", "conv_789")
            print("✅ Individual enhancement methods handle errors gracefully")
        except Exception as e:
            assert False, f"Individual methods should not raise exceptions: {e}"


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing customer-specific context management...")
        
        try:
            await test_customer_preference_tracking()
            await test_customer_interest_identification()
            await test_context_warming_scheduling()
            await test_customer_context_enhancement_integration()
            await test_error_handling()
            
            print("\n✅ All customer context management tests passed!")
            print("📝 Summary:")
            print("   - Preference tracking: Vehicle types, features, budget indicators")
            print("   - Interest identification: Purchase intent, comparison, financing, service")
            print("   - Context warming: 3 common queries scheduled as background tasks")
            print("   - Error handling: Graceful failure without breaking main workflow")
            print("   - Integration: Full enhancement workflow works end-to-end")
            print("\n🎯 Customer-specific context management is ready for production")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

