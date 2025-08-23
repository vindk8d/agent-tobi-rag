#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced cancellation detection system.

This script tests various cancellation scenarios to ensure the agent:
1. Detects cancellation intent correctly
2. Stops tool execution immediately 
3. Provides appropriate cancellation messages
4. Handles edge cases properly

Run with: python tests/test_enhanced_cancellation_system.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.hitl import (
    _interpret_user_intent_with_llm,
    _generate_context_appropriate_cancellation_message,
    _process_hitl_response_llm_driven
)

class CancellationTestSuite:
    """Comprehensive test suite for cancellation detection."""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test_result(self, test_name: str, expected: str, actual: str, passed: bool, details: str = ""):
        """Log the result of a test."""
        result = {
            "test_name": test_name,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        self.total_tests += 1
        
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.failed_tests += 1
            print(f"❌ {test_name}: FAILED")
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual}")
            if details:
                print(f"   Details: {details}")
    
    async def test_basic_cancellation_detection(self):
        """Test basic cancellation word detection."""
        print("\n🔍 Testing Basic Cancellation Detection...")
        
        test_cases = [
            # Clear cancellation cases
            ("cancel", "denial"),
            ("stop", "denial"),
            ("no", "denial"),
            ("abort", "denial"),
            ("quit", "denial"),
            ("exit", "denial"),
            ("not now", "denial"),
            ("never mind", "denial"),
            ("forget it", "denial"),
            
            # Clear approval cases
            ("yes", "approval"),
            ("send it", "approval"),
            ("go ahead", "approval"),
            ("proceed", "approval"),
            ("do it", "approval"),
            ("approve", "approval"),
            
            # Input cases
            ("john@example.com", "input"),
            ("Customer ABC", "input"),
            ("option 2", "input"),
            ("tomorrow", "input"),
        ]
        
        hitl_context = {"source_tool": "generate_quotation"}
        
        for user_input, expected_intent in test_cases:
            try:
                actual_intent = await _interpret_user_intent_with_llm(user_input, hitl_context)
                passed = actual_intent == expected_intent
                self.log_test_result(
                    f"Basic Detection: '{user_input}'",
                    expected_intent,
                    actual_intent,
                    passed
                )
            except Exception as e:
                self.log_test_result(
                    f"Basic Detection: '{user_input}'",
                    expected_intent,
                    f"ERROR: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    async def test_mixed_input_cancellation(self):
        """Test cases where user provides input but also wants to cancel."""
        print("\n🔍 Testing Mixed Input + Cancellation...")
        
        test_cases = [
            # These should be detected as cancellation despite containing input
            ("john@example.com but cancel", "denial"),
            ("Customer ABC, actually never mind", "denial"),
            ("send to john@test.com but stop", "denial"),
            ("actually, cancel this", "denial"),
            ("forget it, don't send", "denial"),
            
            # These should be detected as input (no cancellation intent)
            ("john@example.com for the quotation", "input"),
            ("Customer ABC needs this urgently", "input"),
            ("send to john@test.com please", "input"),
        ]
        
        hitl_context = {"source_tool": "trigger_customer_message"}
        
        for user_input, expected_intent in test_cases:
            try:
                actual_intent = await _interpret_user_intent_with_llm(user_input, hitl_context)
                passed = actual_intent == expected_intent
                self.log_test_result(
                    f"Mixed Input: '{user_input}'",
                    expected_intent,
                    actual_intent,
                    passed
                )
            except Exception as e:
                self.log_test_result(
                    f"Mixed Input: '{user_input}'",
                    expected_intent,
                    f"ERROR: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    async def test_context_appropriate_messages(self):
        """Test that cancellation messages are appropriate for different tools."""
        print("\n🔍 Testing Context-Appropriate Cancellation Messages...")
        
        tool_contexts = [
            {"source_tool": "generate_quotation"},
            {"source_tool": "trigger_customer_message"},
            {"source_tool": "crm_query"},
            {"source_tool": "upload_vehicle_spec"},
        ]
        
        cancellation_inputs = ["cancel", "stop", "no thanks", "abort"]
        
        for hitl_context in tool_contexts:
            for user_input in cancellation_inputs:
                try:
                    message = await _generate_context_appropriate_cancellation_message(hitl_context, user_input)
                    
                    # Check that message is not empty and contains appropriate keywords
                    tool_name = hitl_context["source_tool"]
                    passed = (
                        len(message) > 10 and  # Not too short
                        ("cancelled" in message.lower() or "canceled" in message.lower()) and  # Confirms cancellation
                        len(message) < 200  # Not too long
                    )
                    
                    self.log_test_result(
                        f"Context Message ({tool_name}): '{user_input}'",
                        "Appropriate cancellation message",
                        f"'{message}'",
                        passed,
                        f"Length: {len(message)} chars"
                    )
                except Exception as e:
                    self.log_test_result(
                        f"Context Message ({tool_name}): '{user_input}'",
                        "Appropriate cancellation message",
                        f"ERROR: {str(e)}",
                        False,
                        f"Exception occurred: {str(e)}"
                    )
    
    async def test_full_hitl_response_processing(self):
        """Test the complete HITL response processing with cancellation."""
        print("\n🔍 Testing Full HITL Response Processing...")
        
        test_scenarios = [
            # Approval scenario
            {
                "user_input": "yes, send it",
                "hitl_context": {"source_tool": "trigger_customer_message", "collection_mode": "tool_managed"},
                "expected_result": "approved",
                "expected_context_preserved": True
            },
            
            # Cancellation scenario
            {
                "user_input": "cancel",
                "hitl_context": {"source_tool": "generate_quotation", "collection_mode": "tool_managed"},
                "expected_result": "denied",
                "expected_context_preserved": False
            },
            
            # Input scenario
            {
                "user_input": "john@example.com",
                "hitl_context": {"source_tool": "crm_query", "collection_mode": "tool_managed"},
                "expected_result": "john@example.com",
                "expected_context_preserved": True
            },
            
            # Mixed cancellation scenario
            {
                "user_input": "send to john@test.com but actually cancel",
                "hitl_context": {"source_tool": "trigger_customer_message", "collection_mode": "tool_managed"},
                "expected_result": "denied",
                "expected_context_preserved": False
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            try:
                result = await _process_hitl_response_llm_driven(
                    scenario["hitl_context"], 
                    scenario["user_input"]
                )
                
                # Check result
                actual_result = result.get("result")
                expected_result = scenario["expected_result"]
                result_correct = actual_result == expected_result
                
                # Check context preservation
                context_preserved = result.get("context") is not None
                expected_context_preserved = scenario["expected_context_preserved"]
                context_correct = context_preserved == expected_context_preserved
                
                # Check response message exists
                response_message = result.get("response_message", "")
                message_exists = len(response_message) > 0
                
                overall_passed = result_correct and context_correct and message_exists
                
                details = f"Result: {actual_result}, Context preserved: {context_preserved}, Message: '{response_message}'"
                
                self.log_test_result(
                    f"Full Processing #{i+1}: '{scenario['user_input']}'",
                    f"Result: {expected_result}, Context: {expected_context_preserved}",
                    details,
                    overall_passed
                )
                
            except Exception as e:
                self.log_test_result(
                    f"Full Processing #{i+1}: '{scenario['user_input']}'",
                    f"Result: {scenario['expected_result']}",
                    f"ERROR: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    async def test_edge_cases(self):
        """Test edge cases and potential failure scenarios."""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            # Empty/minimal input
            ("", "input"),
            (" ", "input"),
            ("?", "input"),
            
            # Ambiguous cases
            ("maybe", "input"),
            ("I don't know", "input"),
            ("hmm", "input"),
            
            # Multiple intents in one message
            ("yes but actually no", "denial"),  # Should lean towards the final intent
            ("cancel... wait, actually send it", "approval"),  # Should lean towards the final intent
        ]
        
        hitl_context = {"source_tool": "generate_quotation"}
        
        for user_input, expected_intent in edge_cases:
            try:
                actual_intent = await _interpret_user_intent_with_llm(user_input, hitl_context)
                passed = actual_intent == expected_intent
                self.log_test_result(
                    f"Edge Case: '{user_input}'",
                    expected_intent,
                    actual_intent,
                    passed
                )
            except Exception as e:
                self.log_test_result(
                    f"Edge Case: '{user_input}'",
                    expected_intent,
                    f"ERROR: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    async def run_all_tests(self):
        """Run all test suites."""
        print("🚀 Starting Enhanced Cancellation System Tests...")
        print("=" * 60)
        
        # Run all test suites
        await self.test_basic_cancellation_detection()
        await self.test_mixed_input_cancellation()
        await self.test_context_appropriate_messages()
        await self.test_full_hitl_response_processing()
        await self.test_edge_cases()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        # Save detailed results
        results_file = f"tests/cancellation_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": self.total_tests,
                    "passed_tests": self.passed_tests,
                    "failed_tests": self.failed_tests,
                    "success_rate": (self.passed_tests/self.total_tests)*100
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return success/failure
        return self.failed_tests == 0


async def main():
    """Main test runner."""
    test_suite = CancellationTestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n🎉 All tests passed! The enhanced cancellation system is working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️  {test_suite.failed_tests} tests failed. Please review the results above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
