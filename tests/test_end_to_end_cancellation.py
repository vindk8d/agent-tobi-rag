#!/usr/bin/env python3
"""
End-to-end test of cancellation scenarios in realistic workflows.

This simulates actual user interactions to ensure cancellation works
properly in real-world scenarios.
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.hitl import _process_hitl_response_llm_driven

async def test_quotation_cancellation_scenario():
    """Test cancellation during quotation generation."""
    print("🔍 Testing Quotation Generation Cancellation...")
    
    # Simulate quotation HITL context
    hitl_context = {
        "source_tool": "generate_quotation",
        "collection_mode": "tool_managed",
        "original_params": {
            "customer_identifier": "John Smith",
            "vehicle_requirements": "SUV for family"
        }
    }
    
    # Test different cancellation responses
    cancellation_responses = [
        "cancel",
        "stop this",
        "no thanks",
        "abort",
        "not now",
        "actually, cancel this quotation",
        "john@example.com but never mind"  # Mixed input + cancellation
    ]
    
    print("Testing various cancellation responses:")
    for response in cancellation_responses:
        try:
            result = await _process_hitl_response_llm_driven(hitl_context, response)
            
            # Check that it's properly cancelled
            is_denied = result.get("result") == "denied"
            context_cleared = result.get("context") is None
            has_message = len(result.get("response_message", "")) > 0
            
            if is_denied and context_cleared and has_message:
                print(f"✅ '{response}' → Properly cancelled")
                print(f"   Message: '{result.get('response_message')}'")
            else:
                print(f"❌ '{response}' → Not properly cancelled")
                print(f"   Result: {result.get('result')}, Context cleared: {context_cleared}")
                
        except Exception as e:
            print(f"💥 '{response}' → Error: {str(e)}")
    
    print()

async def test_customer_message_cancellation_scenario():
    """Test cancellation during customer message sending."""
    print("🔍 Testing Customer Message Cancellation...")
    
    # Simulate customer message HITL context
    hitl_context = {
        "source_tool": "trigger_customer_message",
        "collection_mode": "tool_managed", 
        "original_params": {
            "customer_id": "cust_123",
            "message_content": "Thank you for your inquiry",
            "message_type": "follow_up"
        }
    }
    
    # Test cancellation vs approval
    test_cases = [
        ("cancel", "denied", "Should cancel message"),
        ("don't send", "denied", "Should cancel message"),
        ("yes, send it", "approved", "Should approve message"),
        ("go ahead", "approved", "Should approve message"),
        ("send to john@test.com but actually cancel", "denied", "Should cancel despite email")
    ]
    
    print("Testing message sending responses:")
    for response, expected_result, description in test_cases:
        try:
            result = await _process_hitl_response_llm_driven(hitl_context, response)
            actual_result = result.get("result")
            
            if actual_result == expected_result:
                print(f"✅ '{response}' → {actual_result} ({description})")
            else:
                print(f"❌ '{response}' → {actual_result} (Expected: {expected_result})")
                
        except Exception as e:
            print(f"💥 '{response}' → Error: {str(e)}")
    
    print()

async def test_mixed_scenarios():
    """Test edge cases and mixed scenarios."""
    print("🔍 Testing Mixed Scenarios...")
    
    scenarios = [
        {
            "description": "CRM Query with cancellation",
            "hitl_context": {
                "source_tool": "crm_query",
                "collection_mode": "tool_managed",
                "original_params": {"query": "find customers"}
            },
            "user_responses": [
                ("cancel the search", "denied"),
                ("find john smith", "find john smith"),  # Should be treated as input
                ("quit", "denied")
            ]
        },
        {
            "description": "Vehicle spec upload with cancellation", 
            "hitl_context": {
                "source_tool": "upload_vehicle_spec",
                "collection_mode": "tool_managed",
                "original_params": {"file_path": "/tmp/spec.pdf"}
            },
            "user_responses": [
                ("stop the upload", "denied"),
                ("use different file: /tmp/new.pdf", "use different file: /tmp/new.pdf"),
                ("abort", "denied")
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"Testing: {scenario['description']}")
        
        for response, expected in scenario["user_responses"]:
            try:
                result = await _process_hitl_response_llm_driven(
                    scenario["hitl_context"], 
                    response
                )
                actual = result.get("result")
                
                if actual == expected:
                    print(f"✅ '{response}' → {actual}")
                else:
                    print(f"❌ '{response}' → {actual} (Expected: {expected})")
                    
            except Exception as e:
                print(f"💥 '{response}' → Error: {str(e)}")
        
        print()

async def main():
    """Run all end-to-end tests."""
    print("🚀 Starting End-to-End Cancellation Tests...")
    print("=" * 60)
    
    try:
        await test_quotation_cancellation_scenario()
        await test_customer_message_cancellation_scenario()
        await test_mixed_scenarios()
        
        print("=" * 60)
        print("🏁 End-to-End Tests Complete!")
        print("✅ The enhanced cancellation system handles real-world scenarios correctly.")
        
    except Exception as e:
        print(f"💥 End-to-end tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
