#!/usr/bin/env python3
"""
Test script to validate conversation summary generation when message limits are exceeded.
Tests the integration between context loading and background task scheduling.
"""

import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


async def test_summary_generation_trigger():
    """Test that summary generation is triggered when message limits are exceeded."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt, \
         patch('agents.background_tasks.background_task_manager') as mock_task_manager:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {"has_history": False}
        mock_memory_manager.get_relevant_context.return_value = []
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        mock_task_manager.schedule_summary_generation.return_value = "task_123"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create test settings with specific thresholds
        test_env = {
            'EMPLOYEE_MAX_MESSAGES': '5',
            'CUSTOMER_MAX_MESSAGES': '7',
            'SUMMARY_THRESHOLD': '3'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            test_settings = Settings()
            
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = test_settings
            agent.tool_names = []
            
            # Test 1: Employee context with messages exceeding limit (5) and threshold (3)
            test_messages = [
                {"role": "human", "content": f"message {i}"} for i in range(6)  # 6 messages > 5 limit and > 3 threshold
            ]
            
            await agent._load_context_for_employee(test_messages, "emp_user_123", "conv_456")
            
            # Verify summary generation was triggered
            mock_task_manager.schedule_summary_generation.assert_called_with(
                conversation_id="conv_456",
                user_id="emp_user_123",
                customer_id=None,  # Employee user
                employee_id="emp_user_123",
                priority=mock_task_manager.TaskPriority.NORMAL
            )
            
            print("✅ Employee summary generation triggered correctly")
            
            # Reset mock for customer test
            mock_task_manager.reset_mock()
            
            # Test 2: Customer context with messages exceeding limit (7) and threshold (3)
            test_messages = [
                {"role": "human", "content": f"message {i}"} for i in range(8)  # 8 messages > 7 limit and > 3 threshold
            ]
            
            await agent._load_context_for_customer(test_messages, "cust_user_789", "conv_012")
            
            # Verify summary generation was triggered
            mock_task_manager.schedule_summary_generation.assert_called_with(
                conversation_id="conv_012",
                user_id="cust_user_789",
                customer_id="cust_user_789",  # Customer user
                employee_id=None,
                priority=mock_task_manager.TaskPriority.NORMAL
            )
            
            print("✅ Customer summary generation triggered correctly")
            
            # Reset mock for threshold test
            mock_task_manager.reset_mock()
            
            # Test 3: Messages below threshold should not trigger summary
            test_messages = [
                {"role": "human", "content": f"message {i}"} for i in range(2)  # 2 messages < 3 threshold
            ]
            
            await agent._load_context_for_employee(test_messages, "emp_user_456", "conv_789")
            
            # Verify summary generation was NOT triggered
            mock_task_manager.schedule_summary_generation.assert_not_called()
            
            print("✅ Summary generation correctly skipped when below threshold")


async def test_summary_generation_error_handling():
    """Test that errors in summary generation don't break context loading."""
    
    # Mock dependencies with task manager that raises an error
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt, \
         patch('agents.background_tasks.background_task_manager') as mock_task_manager:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {"has_history": False}
        mock_memory_manager.get_relevant_context.return_value = []
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        
        # Make task manager raise an error
        mock_task_manager.schedule_summary_generation.side_effect = Exception("Task manager error")
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Create test settings
        test_env = {
            'EMPLOYEE_MAX_MESSAGES': '3',
            'SUMMARY_THRESHOLD': '2'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            test_settings = Settings()
            
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = test_settings
            agent.tool_names = []
            
            # Test that context loading still works despite summary generation error
            test_messages = [
                {"role": "human", "content": f"message {i}"} for i in range(4)  # Above threshold and limit
            ]
            
            # This should not raise an exception despite the task manager error
            result = await agent._load_context_for_employee(test_messages, "user_123", "conv_456")
            
            # Verify context loading still completed successfully
            assert result is not None
            assert len(result) == 0  # Empty due to mocked trim_messages_for_context
            
            print("✅ Context loading continues successfully despite summary generation errors")


async def test_configurable_summary_thresholds():
    """Test that different summary thresholds work correctly."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt, \
         patch('agents.background_tasks.background_task_manager') as mock_task_manager:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {"has_history": False}
        mock_memory_manager.get_relevant_context.return_value = []
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        mock_task_manager.schedule_summary_generation.return_value = "task_456"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        from core.config import Settings
        
        # Test with different threshold values
        test_cases = [
            {'threshold': '1', 'messages': 2, 'should_trigger': True},
            {'threshold': '5', 'messages': 4, 'should_trigger': False},
            {'threshold': '5', 'messages': 5, 'should_trigger': True},
            {'threshold': '10', 'messages': 15, 'should_trigger': True},
        ]
        
        for case in test_cases:
            mock_task_manager.reset_mock()
            
            test_env = {
                'EMPLOYEE_MAX_MESSAGES': '20',  # High limit so we test threshold, not limit
                'SUMMARY_THRESHOLD': case['threshold']
            }
            
            with patch.dict(os.environ, test_env, clear=False):
                test_settings = Settings()
                
                agent = UnifiedToolCallingRAGAgent()
                agent.settings = test_settings
                agent.tool_names = []
                
                # Create test messages
                test_messages = [
                    {"role": "human", "content": f"message {i}"} for i in range(case['messages'])
                ]
                
                await agent._load_context_for_employee(test_messages, "user_123", f"conv_{case['threshold']}")
                
                if case['should_trigger']:
                    mock_task_manager.schedule_summary_generation.assert_called_once()
                    print(f"✅ Threshold {case['threshold']}: Summary triggered with {case['messages']} messages")
                else:
                    mock_task_manager.schedule_summary_generation.assert_not_called()
                    print(f"✅ Threshold {case['threshold']}: Summary correctly skipped with {case['messages']} messages")


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing conversation summary generation on message limits...")
        
        try:
            await test_summary_generation_trigger()
            await test_summary_generation_error_handling()
            await test_configurable_summary_thresholds()
            
            print("\n✅ All summary generation tests passed!")
            print("📝 Summary:")
            print("   - Summary generation triggered when message count exceeds limits and threshold")
            print("   - Employee and customer contexts handled correctly")
            print("   - Error handling prevents context loading failures")
            print("   - Configurable thresholds work as expected")
            print("   - Background task scheduling integrated successfully")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

