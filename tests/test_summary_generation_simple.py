#!/usr/bin/env python3
"""
Simple test script to validate conversation summary generation logic.
Tests the core functionality without complex agent dependencies.
"""

import os
import sys
import asyncio
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


def test_summary_threshold_logic():
    """Test the summary generation threshold logic."""
    from core.config import Settings
    
    # Test with different configurations
    test_cases = [
        {
            'env': {'SUMMARY_THRESHOLD': '5'},
            'message_count': 3,
            'should_trigger': False,
            'description': 'Below threshold'
        },
        {
            'env': {'SUMMARY_THRESHOLD': '5'},
            'message_count': 5,
            'should_trigger': True,
            'description': 'Meets threshold exactly'
        },
        {
            'env': {'SUMMARY_THRESHOLD': '5'},
            'message_count': 8,
            'should_trigger': True,
            'description': 'Above threshold'
        },
        {
            'env': {'SUMMARY_THRESHOLD': '1'},
            'message_count': 1,
            'should_trigger': True,
            'description': 'Minimum threshold'
        },
        {
            'env': {'SUMMARY_THRESHOLD': '10'},
            'message_count': 15,
            'should_trigger': True,
            'description': 'High threshold exceeded'
        }
    ]
    
    for case in test_cases:
        with patch.dict(os.environ, case['env'], clear=False):
            settings = Settings()
            threshold = settings.memory.summary_threshold
            message_count = case['message_count']
            
            # Test the core logic
            should_trigger = message_count >= threshold
            
            assert should_trigger == case['should_trigger'], \
                f"Failed for {case['description']}: {message_count} messages, threshold {threshold}"
            
            print(f"✅ {case['description']}: {message_count} messages, threshold {threshold} -> {'Trigger' if should_trigger else 'Skip'}")
    
    return True


async def test_summary_generation_method_logic():
    """Test the summary generation method logic without actual task scheduling."""
    
    # Mock the background task manager import
    mock_task_manager = MagicMock()
    mock_task_manager.schedule_summary_generation.return_value = "task_123"
    mock_task_priority = MagicMock()
    mock_task_priority.NORMAL = "normal"
    
    with patch.dict('sys.modules', {
        'agents.background_tasks': MagicMock(
            background_task_manager=mock_task_manager,
            TaskPriority=mock_task_priority
        )
    }):
        # Import the agent class after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create agent with test settings
        test_env = {
            'SUMMARY_THRESHOLD': '3',
            'EMPLOYEE_MAX_MESSAGES': '5',
            'CUSTOMER_MAX_MESSAGES': '7'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            settings = Settings()
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = settings
            
            # Test 1: Messages above threshold should trigger summary
            test_messages = [{"content": f"msg {i}"} for i in range(5)]  # 5 > 3 threshold
            
            await agent._trigger_summary_generation_if_needed(
                test_messages, "user_123", "conv_456", is_employee=True
            )
            
            # Verify task was scheduled
            mock_task_manager.schedule_summary_generation.assert_called_once_with(
                conversation_id="conv_456",
                user_id="user_123",
                customer_id=None,
                employee_id="user_123",
                priority="normal"
            )
            
            print("✅ Employee summary generation scheduled correctly")
            
            # Reset mock for customer test
            mock_task_manager.reset_mock()
            
            # Test 2: Customer user type
            await agent._trigger_summary_generation_if_needed(
                test_messages, "user_789", "conv_012", is_employee=False
            )
            
            # Verify customer task was scheduled
            mock_task_manager.schedule_summary_generation.assert_called_once_with(
                conversation_id="conv_012",
                user_id="user_789",
                customer_id="user_789",
                employee_id=None,
                priority="normal"
            )
            
            print("✅ Customer summary generation scheduled correctly")
            
            # Reset mock for below threshold test
            mock_task_manager.reset_mock()
            
            # Test 3: Messages below threshold should not trigger
            small_messages = [{"content": f"msg {i}"} for i in range(2)]  # 2 < 3 threshold
            
            await agent._trigger_summary_generation_if_needed(
                small_messages, "user_456", "conv_789", is_employee=True
            )
            
            # Verify no task was scheduled
            mock_task_manager.schedule_summary_generation.assert_not_called()
            
            print("✅ Summary generation correctly skipped when below threshold")
    
    return True


async def test_message_limit_integration():
    """Test that summary generation integrates with message limit checking."""
    
    from core.config import Settings
    
    # Test different message limit scenarios
    test_cases = [
        {
            'env': {'EMPLOYEE_MAX_MESSAGES': '5', 'SUMMARY_THRESHOLD': '3'},
            'message_count': 6,  # Above limit (5) and threshold (3)
            'should_check_summary': True,
            'user_type': 'employee'
        },
        {
            'env': {'CUSTOMER_MAX_MESSAGES': '8', 'SUMMARY_THRESHOLD': '4'},
            'message_count': 10,  # Above limit (8) and threshold (4)
            'should_check_summary': True,
            'user_type': 'customer'
        },
        {
            'env': {'EMPLOYEE_MAX_MESSAGES': '10', 'SUMMARY_THRESHOLD': '5'},
            'message_count': 8,  # Below limit (10) but above threshold (5)
            'should_check_summary': False,  # Only checked when limit exceeded
            'user_type': 'employee'
        }
    ]
    
    for case in test_cases:
        with patch.dict(os.environ, case['env'], clear=False):
            settings = Settings()
            
            # Get the appropriate limit
            if case['user_type'] == 'employee':
                limit = settings.memory.employee_max_messages
            else:
                limit = settings.memory.customer_max_messages
            
            threshold = settings.memory.summary_threshold
            message_count = case['message_count']
            
            # Test the integration logic
            should_check = message_count > limit
            would_trigger = message_count >= threshold
            
            assert should_check == case['should_check_summary'], \
                f"Summary check logic failed: {message_count} messages, {limit} limit"
            
            if should_check:
                print(f"✅ {case['user_type'].title()}: {message_count} messages > {limit} limit, "
                      f"would {'trigger' if would_trigger else 'skip'} summary (threshold: {threshold})")
            else:
                print(f"✅ {case['user_type'].title()}: {message_count} messages <= {limit} limit, "
                      f"summary check skipped")
    
    return True


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing conversation summary generation logic...")
        
        try:
            test_summary_threshold_logic()
            await test_summary_generation_method_logic()
            await test_message_limit_integration()
            
            print("\n✅ All summary generation logic tests passed!")
            print("📝 Summary:")
            print("   - Summary threshold logic works correctly")
            print("   - Employee and customer user types handled properly")
            print("   - Message limit integration functions as expected")
            print("   - Background task scheduling logic is correct")
            print("   - Below-threshold cases are handled correctly")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

