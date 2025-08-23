#!/usr/bin/env python3
"""
Test the routing fix to ensure tool execution stops on cancellation.

This test verifies that the _is_tool_managed_collection_needed function
correctly stops tool execution when user cancels.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent

def test_routing_logic():
    """Test the routing logic for different HITL phases."""
    
    agent = UnifiedToolCallingRAGAgent()
    
    print("🔍 Testing Routing Logic for Tool Execution...")
    print("=" * 50)
    
    # Test cases: (hitl_phase, should_continue_expected)
    test_cases = [
        # User approved - should continue
        {
            "hitl_phase": "approved",
            "hitl_context": {
                "source_tool": "generate_quotation",
                "collection_mode": "tool_managed",
                "original_params": {"customer_id": "123"}
            },
            "expected": True,
            "description": "User approved quotation"
        },
        
        # User denied - should NOT continue
        {
            "hitl_phase": "denied", 
            "hitl_context": {
                "source_tool": "generate_quotation",
                "collection_mode": "tool_managed",
                "original_params": {"customer_id": "123"}
            },
            "expected": False,
            "description": "User denied quotation"
        },
        
        # User denied customer message - should NOT continue
        {
            "hitl_phase": "denied",
            "hitl_context": {
                "source_tool": "trigger_customer_message",
                "collection_mode": "tool_managed",
                "original_params": {"customer_id": "456", "message": "test"}
            },
            "expected": False,
            "description": "User denied customer message"
        },
        
        # No HITL context - should NOT continue
        {
            "hitl_phase": "approved",
            "hitl_context": None,
            "expected": False,
            "description": "No HITL context"
        },
        
        # Wrong collection mode - should NOT continue
        {
            "hitl_phase": "approved",
            "hitl_context": {
                "source_tool": "generate_quotation",
                "collection_mode": "manual",  # Not "tool_managed"
                "original_params": {"customer_id": "123"}
            },
            "expected": False,
            "description": "Wrong collection mode"
        },
        
        # Pending phase - should NOT continue
        {
            "hitl_phase": "awaiting_response",
            "hitl_context": {
                "source_tool": "generate_quotation", 
                "collection_mode": "tool_managed",
                "original_params": {"customer_id": "123"}
            },
            "expected": False,
            "description": "Still awaiting response"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        # Create state dict
        state = {
            "hitl_phase": test_case["hitl_phase"],
            "messages": []
        }
        
        try:
            # Test the routing logic
            result = agent._is_tool_managed_collection_needed(
                test_case["hitl_context"], 
                state
            )
            
            expected = test_case["expected"]
            passed = result == expected
            
            if passed:
                passed_tests += 1
                print(f"✅ Test {i}: {test_case['description']}")
                print(f"   Phase: {test_case['hitl_phase']} → Continue: {result} (Expected: {expected})")
            else:
                print(f"❌ Test {i}: {test_case['description']}")
                print(f"   Phase: {test_case['hitl_phase']} → Continue: {result} (Expected: {expected})")
                print(f"   FAILED: Expected {expected}, got {result}")
            
        except Exception as e:
            print(f"💥 Test {i}: {test_case['description']}")
            print(f"   ERROR: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 ROUTING TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {total_tests - passed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All routing tests passed! Tool execution will stop correctly on cancellation.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} routing tests failed.")
        return False

if __name__ == "__main__":
    success = test_routing_logic()
    sys.exit(0 if success else 1)
