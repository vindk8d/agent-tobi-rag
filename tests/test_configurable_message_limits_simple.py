#!/usr/bin/env python3
"""
Simple test script to validate configurable message limits configuration.
Tests the core functionality without complex agent dependencies.
"""

import os
import sys
import asyncio
from unittest.mock import patch

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
    return True


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
    return True


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
    return True


def test_environment_variable_names():
    """Test that the correct environment variable names are used."""
    from core.config import Settings
    import os
    
    # Test that environment variables are actually read
    test_env = {
        'EMPLOYEE_MAX_MESSAGES': '99',
        'CUSTOMER_MAX_MESSAGES': '88',
        'SUMMARY_THRESHOLD': '77'
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        settings = Settings()
        
        # Verify the environment variables are being read
        assert settings.employee_max_messages == 99
        assert settings.customer_max_messages == 88
        assert settings.summary_threshold == 77
    
    print("✅ Environment variable names validated successfully")
    return True


async def test_memory_property_integration():
    """Test that the memory property correctly exposes all new configuration values."""
    settings = await get_settings()
    
    # Test that memory property has all the expected attributes
    memory = settings.memory
    
    # Check that all attributes exist
    assert hasattr(memory, 'employee_max_messages')
    assert hasattr(memory, 'customer_max_messages')
    assert hasattr(memory, 'summary_threshold')
    
    # Check that values match the main settings
    assert memory.employee_max_messages == settings.employee_max_messages
    assert memory.customer_max_messages == settings.customer_max_messages
    assert memory.summary_threshold == settings.summary_threshold
    
    # Check that legacy memory attributes still exist
    assert hasattr(memory, 'max_messages')
    assert hasattr(memory, 'summary_interval')
    assert hasattr(memory, 'auto_summarize')
    assert hasattr(memory, 'context_window_size')
    
    print("✅ Memory property integration validated successfully")
    return True


async def test_configuration_types():
    """Test that configuration values have the correct types."""
    settings = await get_settings()
    
    # Test types
    assert isinstance(settings.employee_max_messages, int)
    assert isinstance(settings.customer_max_messages, int)
    assert isinstance(settings.summary_threshold, int)
    
    # Test that values are reasonable
    assert settings.employee_max_messages > 0
    assert settings.customer_max_messages > 0
    assert settings.summary_threshold > 0
    
    # Test that customer limit is typically higher than employee limit (default behavior)
    assert settings.customer_max_messages >= settings.employee_max_messages
    
    print("✅ Configuration types and ranges validated successfully")
    return True


if __name__ == "__main__":
    async def run_tests():
        print("🧪 Testing configurable message limits configuration...")
        
        try:
            await test_default_message_limits()
            await test_environment_variable_override()
            test_configuration_documentation()
            test_environment_variable_names()
            await test_memory_property_integration()
            await test_configuration_types()
            
            print("\n✅ All configurable message limit configuration tests passed!")
            print("📝 Summary:")
            print("   - Default values: EMPLOYEE_MAX_MESSAGES=12, CUSTOMER_MAX_MESSAGES=15, SUMMARY_THRESHOLD=10")
            print("   - Environment variable override: Working correctly")
            print("   - Memory property integration: Working correctly")
            print("   - Field documentation: Properly documented")
            print("   - Type validation: All integers with reasonable ranges")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
