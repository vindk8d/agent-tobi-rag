#!/usr/bin/env python3
"""
Test script to validate configurable message limits for employee and customer users.
Validates that environment variables are properly loaded and used in agent nodes.
"""

import os
import sys
import asyncio
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.config import get_settings


async def test_default_message_limits():
    """Test that default message limits are correctly set."""
    settings = await get_settings()
    
    # Test default values
    assert settings.employee_max_messages == 12
    assert settings.customer_max_messages == 15
    assert settings.summary_threshold == 10
    
    # Test memory property access
    assert settings.memory.employee_max_messages == 12
    assert settings.memory.customer_max_messages == 15
    assert settings.memory.summary_threshold == 10
    
    print("✅ Default message limits validated successfully")


async def test_environment_variable_override():
    """Test that environment variables properly override default values."""
    
    # Set test environment variables
    test_env = {
        'EMPLOYEE_MAX_MESSAGES': '8',
        'CUSTOMER_MAX_MESSAGES': '20',
        'SUMMARY_THRESHOLD': '5'
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        # Clear the settings cache to force reload
        from core.config import _settings_cache, _settings_lock
        async with _settings_lock:
            # Force settings reload
            global _settings_cache
            _settings_cache = None
        
        # Create new settings instance
        from core.config import Settings
        settings = Settings()
        
        # Validate overridden values
        assert settings.employee_max_messages == 8
        assert settings.customer_max_messages == 20
        assert settings.summary_threshold == 5
        
        # Test memory property access
        assert settings.memory.employee_max_messages == 8
        assert settings.memory.customer_max_messages == 20
        assert settings.memory.summary_threshold == 5
    
    print("✅ Environment variable override validated successfully")


async def test_agent_uses_configurable_limits():
    """Test that agent context loading methods use the configurable limits."""
    
    # Mock dependencies
    with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory_manager, \
         patch('agents.tobi_sales_copilot.agent.context_manager') as mock_context_manager, \
         patch('agents.tobi_sales_copilot.agent.detect_user_language_from_context') as mock_detect_lang, \
         patch('agents.tobi_sales_copilot.agent.get_employee_system_prompt') as mock_employee_prompt, \
         patch('agents.tobi_sales_copilot.agent.get_customer_system_prompt') as mock_customer_prompt:
        
        # Setup mocks
        mock_memory_manager.get_user_context_for_new_conversation.return_value = {"has_history": False}
        mock_memory_manager.get_relevant_context.return_value = []
        mock_context_manager.trim_messages_for_context.return_value = ([], {"trimmed_message_count": 0, "final_token_count": 100})
        mock_detect_lang.return_value = "english"
        mock_employee_prompt.return_value = "Employee system prompt"
        mock_customer_prompt.return_value = "Customer system prompt"
        
        # Import agent after mocking
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        # Create agent instance with test settings
        test_env = {
            'EMPLOYEE_MAX_MESSAGES': '6',
            'CUSTOMER_MAX_MESSAGES': '18'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            from core.config import Settings
            test_settings = Settings()
            
            agent = UnifiedToolCallingRAGAgent()
            agent.settings = test_settings
            agent.tool_names = []
            
            # Test employee context loading
            test_messages = [{"role": "human", "content": "test message"}]
            await agent._load_context_for_employee(test_messages, "user_123", "conv_456")
            
            # Verify that trim_messages_for_context was called with the configured limit
            mock_context_manager.trim_messages_for_context.assert_called()
            call_args = mock_context_manager.trim_messages_for_context.call_args
            assert call_args[1]['max_messages'] == 6  # Employee limit
            
            # Reset mock for customer test
            mock_context_manager.reset_mock()
            
            # Test customer context loading
            await agent._load_context_for_customer(test_messages, "user_789", "conv_012")
            
            # Verify that trim_messages_for_context was called with the configured customer limit
            mock_context_manager.trim_messages_for_context.assert_called()
            call_args = mock_context_manager.trim_messages_for_context.call_args
            assert call_args[1]['max_messages'] == 18  # Customer limit
    
    print("✅ Agent configurable limits validated successfully")


def test_configuration_documentation():
    """Test that configuration fields have proper documentation."""
    from core.config import Settings
    
    # Get field info
    fields = Settings.model_fields
    
    # Check that new fields exist and have descriptions
    assert 'employee_max_messages' in fields
    assert 'customer_max_messages' in fields
    assert 'summary_threshold' in fields
    
    # Check descriptions
    assert 'employee users' in fields['employee_max_messages'].description.lower()
    assert 'customer users' in fields['customer_max_messages'].description.lower()
    assert 'conversation summary' in fields['summary_threshold'].description.lower()
    
    print("✅ Configuration documentation validated successfully")


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing configurable message limits...")
        
        await test_default_message_limits()
        await test_environment_variable_override()
        await test_agent_uses_configurable_limits()
        test_configuration_documentation()
        
        print("\n✅ All configurable message limit tests passed!")
    
    asyncio.run(run_tests())
