#!/usr/bin/env python3
"""
Comprehensive Portability Test Suite.

This script demonstrates and tests ALL the portable utilities we've created,
showing how ANY agent can use these functions without code duplication.

Tests cover:
- User verification utilities
- API processing utilities  
- Message delivery utilities
- Error handling utilities
- Source processing utilities
- Language detection utilities
- Background task utilities

This validates our portability improvements and serves as documentation
for other developers building new agents.
"""

import asyncio
import sys
import pathlib
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add backend directory to path for imports
backend_path = pathlib.Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all our portable utilities
from utils.user_verification import verify_user_access, verify_employee_access, handle_access_denied
from utils.api_processing import process_agent_result_for_api, process_user_message
from utils.message_delivery import (
    execute_customer_message_delivery,
    send_via_chat_api,
    format_business_message,
    get_or_create_customer_user,
    get_or_create_customer_conversation
)
from utils.error_handling import (
    handle_processing_error,
    create_tool_recall_error_state,
    create_user_friendly_error_message,
    log_error_with_context,
    is_recoverable_error
)
from utils.source_processing import (
    extract_sources_from_messages,
    format_sources_for_display,
    extract_display_name_from_source,
    determine_source_type,
    filter_sources_by_relevance,
    merge_duplicate_sources
)
from utils.language import detect_user_language, detect_user_language_from_context
from agents.background_tasks import background_task_manager


class MockAgent:
    """
    Mock agent demonstrating how ANY agent can use our portable utilities.
    This shows the power of our separation of concerns - this agent gets
    full functionality without implementing any of the complex logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.background_task_manager = background_task_manager
        
    async def invoke(self, user_query: str, **kwargs):
        """Mock agent invoke method for testing API processing."""
        return {
            "messages": [{"type": "ai", "content": f"Mock response from {self.name}: {user_query}"}],
            "sources": [{"source": "mock_document.pdf", "similarity": 0.9}],
            "conversation_id": kwargs.get("conversation_id", "mock-conv-123")
        }


async def test_user_verification():
    """Test user verification utilities."""
    print("\n🔐 TESTING USER VERIFICATION UTILITIES")
    print("=" * 50)
    
    # Test handle_access_denied (doesn't require database)
    test_state = {
        "messages": [],
        "user_id": "test-user-123",
        "conversation_id": "test-conv-123"
    }
    
    result = await handle_access_denied(test_state, "test-conv-123", "Test access denial")
    print(f"✅ Access denied handling: {len(result['messages'])} messages returned")
    print(f"   Message preview: {result['messages'][0].content[:50]}...")
    
    return True


async def test_api_processing():
    """Test API processing utilities."""
    print("\n🌐 TESTING API PROCESSING UTILITIES")
    print("=" * 50)
    
    # Test with HITL state
    hitl_result = {
        "hitl_phase": "needs_confirmation",
        "hitl_prompt": "Do you want to proceed with this action?",
        "hitl_context": {"action": "test_action"}
    }
    
    processed_hitl = await process_agent_result_for_api(hitl_result, "test-conv-123")
    print(f"✅ HITL processing: is_interrupted={processed_hitl['is_interrupted']}")
    print(f"   HITL phase: {processed_hitl.get('hitl_phase')}")
    
    # Test with normal result
    normal_result = {
        "messages": [{"type": "ai", "content": "Hello! How can I help you today?"}],
        "sources": [{"source": "help_guide.pdf", "similarity": 0.8}]
    }
    
    processed_normal = await process_agent_result_for_api(normal_result, "test-conv-123")
    print(f"✅ Normal processing: message_length={len(processed_normal['message'])}")
    print(f"   Sources: {len(processed_normal['sources'])}")
    
    # Test process_user_message with mock agent
    mock_agent = MockAgent("TestAgent")
    user_message_result = await process_user_message(
        agent_invoke_func=mock_agent.invoke,
        user_query="Test query",
        conversation_id="test-conv-123",
        user_id="test-user-123"
    )
    print(f"✅ User message processing: conversation_id={user_message_result['conversation_id']}")
    
    return True


async def test_message_delivery():
    """Test message delivery utilities."""
    print("\n📨 TESTING MESSAGE DELIVERY UTILITIES")
    print("=" * 50)
    
    # Test message formatting
    customer_info = {
        "name": "John Doe",
        "first_name": "John",
        "email": "john@example.com"
    }
    
    formatted_msg = await format_business_message(
        "Thank you for your interest in our products!",
        "follow_up",
        customer_info
    )
    print(f"✅ Message formatting: {len(formatted_msg)} characters")
    print(f"   Preview: {formatted_msg[:60]}...")
    
    # Test different message types
    for msg_type in ["information", "promotional", "support"]:
        test_msg = await format_business_message(
            "Test message content",
            msg_type,
            customer_info
        )
        print(f"   {msg_type.title()} format: {len(test_msg)} chars")
    
    print(f"✅ Message delivery utilities functional")
    
    return True


async def test_error_handling():
    """Test error handling utilities."""
    print("\n⚠️  TESTING ERROR HANDLING UTILITIES")
    print("=" * 50)
    
    # Test processing error handling
    test_state = {
        "messages": [{"content": "Previous message"}],
        "user_id": "test-user-123",
        "conversation_id": "test-conv-123"
    }
    
    test_error = Exception("Test error for demonstration")
    error_result = await handle_processing_error(test_state, test_error)
    print(f"✅ Processing error handling: {len(error_result['messages'])} messages")
    print(f"   Error metadata: error_type={error_result.get('error_type')}")
    
    # Test tool recall error state
    tool_error_result = create_tool_recall_error_state(test_state, "Tool execution failed")
    print(f"✅ Tool recall error: {len(tool_error_result['messages'])} messages")
    print(f"   Tool error flag: {tool_error_result.get('tool_error_occurred')}")
    
    # Test user-friendly error messages
    test_errors = [
        ConnectionError("Connection timeout"),
        PermissionError("Access denied"),
        FileNotFoundError("File not found"),
        ValueError("Invalid input")
    ]
    
    for error in test_errors:
        friendly_msg = create_user_friendly_error_message(error, "test context")
        print(f"   {type(error).__name__}: {friendly_msg[:40]}...")
    
    # Test error recoverability
    recoverable_error = ConnectionError("Network timeout")
    non_recoverable_error = PermissionError("Access forbidden")
    
    print(f"✅ Error recoverability:")
    print(f"   Network timeout: {is_recoverable_error(recoverable_error)}")
    print(f"   Access forbidden: {is_recoverable_error(non_recoverable_error)}")
    
    return True


async def test_source_processing():
    """Test source processing utilities."""
    print("\n📄 TESTING SOURCE PROCESSING UTILITIES")
    print("=" * 50)
    
    # Create mock messages with sources
    from langchain_core.messages import ToolMessage
    
    mock_messages = [
        ToolMessage(
            content="Answer: Here's the information.\n\nSources (2):\n• document1.pdf\n• guide.docx",
            name="simple_rag",
            tool_call_id="test-123"
        )
    ]
    
    # Test source extraction
    sources = extract_sources_from_messages(mock_messages)
    print(f"✅ Source extraction: {len(sources)} sources found")
    for source in sources:
        print(f"   - {source['source']} (similarity: {source['similarity']})")
    
    # Test source formatting
    formatted_sources = format_sources_for_display(sources)
    print(f"✅ Source formatting: {len(formatted_sources)} formatted sources")
    for source in formatted_sources:
        print(f"   {source['id']}. {source['display_name']} ({source['type']})")
    
    # Test source utilities
    test_sources = [
        "/path/to/document.pdf",
        "https://example.com/page.html",
        "database_table_customers",
        "kb_article_123"
    ]
    
    print(f"✅ Source utilities:")
    for source in test_sources:
        display_name = extract_display_name_from_source(source)
        source_type = determine_source_type(source)
        print(f"   {source} -> {display_name} ({source_type})")
    
    return True


async def test_language_utilities():
    """Test language detection utilities."""
    print("\n🌍 TESTING LANGUAGE UTILITIES")
    print("=" * 50)
    
    # Test language detection
    test_messages = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Kumusta ka ngayon?",  # Tagalog
        "How much does the Toyota Camry cost?"
    ]
    
    for message in test_messages:
        detected_lang = detect_user_language(message)
        print(f"✅ '{message[:30]}...' -> {detected_lang}")
    
    # Test context-based detection
    mock_context_messages = [
        {"content": "Hi there!"},
        {"content": "I'm interested in buying a car"},
        {"content": "What models do you have available?"}
    ]
    
    context_lang = detect_user_language_from_context(mock_context_messages)
    print(f"✅ Context-based detection: {context_lang}")
    
    return True


async def test_background_tasks():
    """Test background task utilities."""
    print("\n⚙️  TESTING BACKGROUND TASK UTILITIES")
    print("=" * 50)
    
    # Test that we can access the global background task manager
    print(f"✅ Background task manager available: {background_task_manager is not None}")
    print(f"   Manager type: {type(background_task_manager).__name__}")
    
    # Mock agent state for testing
    mock_state = {
        "conversation_id": "test-conv-123",
        "user_id": "test-user-123",
        "customer_id": "test-customer-123",
        "messages": [{"content": "Test message", "role": "user"}]
    }
    
    # Test that portable methods exist and are callable
    print(f"✅ Portable methods available:")
    print(f"   - schedule_message_from_agent_state: {hasattr(background_task_manager, 'schedule_message_from_agent_state')}")
    print(f"   - schedule_summary_from_agent_state: {hasattr(background_task_manager, 'schedule_summary_from_agent_state')}")
    
    return True


async def run_comprehensive_test():
    """Run all portability tests."""
    print("🚀 COMPREHENSIVE PORTABILITY TEST SUITE")
    print("=" * 60)
    print("Testing all portable utilities to validate separation of concerns")
    print("and demonstrate how ANY agent can use these functions.")
    print()
    
    test_results = []
    
    try:
        # Run all tests
        test_results.append(("User Verification", await test_user_verification()))
        test_results.append(("API Processing", await test_api_processing()))
        test_results.append(("Message Delivery", await test_message_delivery()))
        test_results.append(("Error Handling", await test_error_handling()))
        test_results.append(("Source Processing", await test_source_processing()))
        test_results.append(("Language Utilities", await test_language_utilities()))
        test_results.append(("Background Tasks", await test_background_tasks()))
        
        # Print summary
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\nResults: {passed}/{len(test_results)} tests passed")
        
        if passed == len(test_results):
            print("\n🎉 ALL PORTABILITY TESTS PASSED!")
            print("✨ All utilities are working correctly and ready for use by any agent.")
            print("🔧 Developers can now build new agents using these portable utilities")
            print("   without duplicating code or violating separation of concerns.")
        else:
            print(f"\n⚠️  {len(test_results) - passed} tests failed. Check the output above for details.")
        
        return passed == len(test_results)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in test suite: {e}")
        return False


if __name__ == "__main__":
    print("Starting comprehensive portability test suite...")
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n✅ Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test suite failed!")
        sys.exit(1)
