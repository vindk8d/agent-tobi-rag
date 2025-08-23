#!/usr/bin/env python3
"""
Comprehensive end-to-end HITL flow tests with enhanced cancellation.

This test suite verifies the complete flow:
1. User query triggers tool
2. Tool requests HITL confirmation  
3. User responds (approval/denial/input)
4. System handles response correctly
5. Tool execution continues or stops appropriately
6. User gets appropriate messages

Tests are fail-fast - they stop immediately on first failure for debugging.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.hitl import (
    _process_hitl_response_llm_driven,
    request_approval,
    HITLInteractionError
)

class FailFastHITLTester:
    """Fail-fast end-to-end HITL flow tester."""
    
    def __init__(self):
        self.agent = UnifiedToolCallingRAGAgent()
        self.test_results = []
        self.current_test = ""
        
    def log_step(self, step: str, status: str, details: str = ""):
        """Log a test step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {status} {step}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": self.current_test,
            "step": step,
            "status": status,
            "details": details,
            "timestamp": timestamp
        })
    
    def fail_test(self, step: str, error: str, expected: str = "", actual: str = ""):
        """Fail the current test and stop execution."""
        self.log_step(step, "❌ FAILED", error)
        if expected and actual:
            print(f"    Expected: {expected}")
            print(f"    Actual: {actual}")
        
        print(f"\n💥 TEST FAILED: {self.current_test}")
        print(f"💥 FAILED STEP: {step}")
        print(f"💥 ERROR: {error}")
        print("\n🛑 STOPPING TESTS FOR DEBUGGING")
        
        # Save results for debugging
        self.save_debug_results()
        return False
    
    def pass_step(self, step: str, details: str = ""):
        """Mark a step as passed."""
        self.log_step(step, "✅ PASSED", details)
        return True
    
    def save_debug_results(self):
        """Save test results for debugging."""
        results_file = f"tests/hitl_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"🔍 Debug results saved to: {results_file}")
    
    async def test_quotation_hitl_flow_with_cancellation(self):
        """Test complete quotation generation HITL flow with cancellation."""
        self.current_test = "Quotation HITL Flow with Cancellation"
        print(f"\n🧪 Testing: {self.current_test}")
        print("=" * 60)
        
        # Step 1: Simulate HITL request from quotation tool
        if not self.pass_step("Step 1: Create quotation HITL context"):
            return False
        
        hitl_context = {
            "source_tool": "generate_quotation",
            "collection_mode": "tool_managed",
            "original_params": {
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Family SUV, budget $50k",
                "quotation_validity_days": 30
            }
        }
        
        # Step 2: Generate HITL prompt (simulate what the tool would do)
        if not self.pass_step("Step 2: Generate HITL prompt"):
            return False
        
        try:
            hitl_prompt = request_approval(
                prompt="**Quotation Generation Request**\n\nCustomer: John Smith\nRequirements: Family SUV, budget $50k\nValidity: 30 days\n\nShould I generate this quotation?",
                context=hitl_context,
                approve_text="generate",
                deny_text="cancel"
            )
            
            if not hitl_prompt or not hitl_prompt.startswith("HITL_REQUIRED"):
                return self.fail_test("Step 2", "HITL prompt not generated correctly", "HITL_REQUIRED:...", hitl_prompt)
                
        except Exception as e:
            return self.fail_test("Step 2", f"Exception generating HITL prompt: {str(e)}")
        
        # Step 3: Test cancellation response processing
        if not self.pass_step("Step 3: Process cancellation response"):
            return False
        
        cancellation_responses = ["cancel", "stop", "no thanks", "abort this"]
        
        for response in cancellation_responses:
            try:
                result = await _process_hitl_response_llm_driven(hitl_context, response)
                
                # Verify cancellation was processed correctly
                if result.get("result") != "denied":
                    return self.fail_test(
                        f"Step 3 - Response '{response}'", 
                        "Cancellation not detected as denial",
                        "denied",
                        result.get("result")
                    )
                
                if result.get("context") is not None:
                    return self.fail_test(
                        f"Step 3 - Response '{response}'",
                        "Context not cleared on cancellation",
                        "None",
                        str(result.get("context"))
                    )
                
                response_message = result.get("response_message", "")
                if len(response_message) < 10:
                    return self.fail_test(
                        f"Step 3 - Response '{response}'",
                        "No appropriate cancellation message generated",
                        "Contextual cancellation message",
                        response_message
                    )
                
                if "quotation" not in response_message.lower():
                    return self.fail_test(
                        f"Step 3 - Response '{response}'",
                        "Cancellation message not contextual to quotation",
                        "Message mentioning 'quotation'",
                        response_message
                    )
                
                self.pass_step(f"Step 3 - '{response}'", f"Correctly cancelled: '{response_message}'")
                
            except Exception as e:
                return self.fail_test(f"Step 3 - Response '{response}'", f"Exception: {str(e)}")
        
        # Step 4: Test routing logic stops tool execution
        if not self.pass_step("Step 4: Verify routing stops tool execution on denial"):
            return False
        
        try:
            # Simulate state after denial
            denied_state = {
                "hitl_phase": "denied",
                "hitl_context": hitl_context,
                "messages": []
            }
            
            should_continue = self.agent._is_tool_managed_collection_needed(hitl_context, denied_state)
            
            if should_continue:
                return self.fail_test(
                    "Step 4",
                    "Tool execution continues after denial",
                    "False (should stop)",
                    "True (continues)"
                )
            
            self.pass_step("Step 4", "Tool execution correctly stopped after denial")
            
        except Exception as e:
            return self.fail_test("Step 4", f"Exception testing routing: {str(e)}")
        
        # Step 5: Test approval flow still works
        if not self.pass_step("Step 5: Verify approval flow still works"):
            return False
        
        try:
            approval_result = await _process_hitl_response_llm_driven(hitl_context, "yes, generate the quotation")
            
            if approval_result.get("result") != "approved":
                return self.fail_test(
                    "Step 5",
                    "Approval not detected correctly",
                    "approved",
                    approval_result.get("result")
                )
            
            if approval_result.get("context") is None:
                return self.fail_test(
                    "Step 5", 
                    "Context cleared on approval (should be preserved)",
                    "Preserved context",
                    "None"
                )
            
            # Test routing allows continuation on approval
            approved_state = {
                "hitl_phase": "approved",
                "hitl_context": hitl_context,
                "messages": []
            }
            
            should_continue = self.agent._is_tool_managed_collection_needed(hitl_context, approved_state)
            
            if not should_continue:
                return self.fail_test(
                    "Step 5",
                    "Tool execution stopped after approval",
                    "True (should continue)",
                    "False (stopped)"
                )
            
            self.pass_step("Step 5", "Approval flow works correctly")
            
        except Exception as e:
            return self.fail_test("Step 5", f"Exception testing approval: {str(e)}")
        
        print(f"\n✅ {self.current_test} - ALL STEPS PASSED!")
        return True
    
    async def test_customer_message_hitl_flow_with_cancellation(self):
        """Test complete customer message HITL flow with cancellation."""
        self.current_test = "Customer Message HITL Flow with Cancellation"
        print(f"\n🧪 Testing: {self.current_test}")
        print("=" * 60)
        
        # Step 1: Create customer message HITL context
        if not self.pass_step("Step 1: Create customer message HITL context"):
            return False
        
        hitl_context = {
            "source_tool": "trigger_customer_message",
            "collection_mode": "tool_managed",
            "original_params": {
                "customer_id": "cust_123",
                "message_content": "Thank you for your inquiry about our SUV models.",
                "message_type": "follow_up"
            }
        }
        
        # Step 2: Test cancellation responses
        if not self.pass_step("Step 2: Test customer message cancellation"):
            return False
        
        cancellation_responses = [
            "don't send",
            "cancel the message", 
            "stop",
            "john@test.com but actually don't send"  # Mixed input + cancellation
        ]
        
        for response in cancellation_responses:
            try:
                result = await _process_hitl_response_llm_driven(hitl_context, response)
                
                if result.get("result") != "denied":
                    return self.fail_test(
                        f"Step 2 - '{response}'",
                        "Customer message cancellation not detected",
                        "denied",
                        result.get("result")
                    )
                
                response_message = result.get("response_message", "")
                if "message" not in response_message.lower():
                    return self.fail_test(
                        f"Step 2 - '{response}'",
                        "Cancellation message not contextual to customer message",
                        "Message mentioning 'message'",
                        response_message
                    )
                
                self.pass_step(f"Step 2 - '{response}'", f"Correctly cancelled: '{response_message}'")
                
            except Exception as e:
                return self.fail_test(f"Step 2 - '{response}'", f"Exception: {str(e)}")
        
        # Step 3: Test approval still works
        if not self.pass_step("Step 3: Test customer message approval"):
            return False
        
        try:
            approval_result = await _process_hitl_response_llm_driven(hitl_context, "yes, send the message")
            
            if approval_result.get("result") != "approved":
                return self.fail_test(
                    "Step 3",
                    "Customer message approval not detected",
                    "approved", 
                    approval_result.get("result")
                )
            
            self.pass_step("Step 3", "Customer message approval works correctly")
            
        except Exception as e:
            return self.fail_test("Step 3", f"Exception testing approval: {str(e)}")
        
        print(f"\n✅ {self.current_test} - ALL STEPS PASSED!")
        return True
    
    async def test_crm_query_hitl_flow_with_cancellation(self):
        """Test CRM query HITL flow with cancellation."""
        self.current_test = "CRM Query HITL Flow with Cancellation"
        print(f"\n🧪 Testing: {self.current_test}")
        print("=" * 60)
        
        # Step 1: Create CRM query HITL context
        if not self.pass_step("Step 1: Create CRM query HITL context"):
            return False
        
        hitl_context = {
            "source_tool": "crm_query",
            "collection_mode": "tool_managed",
            "original_params": {
                "query": "Find all customers interested in SUVs",
                "filters": {"budget_min": 40000}
            }
        }
        
        # Step 2: Test cancellation
        if not self.pass_step("Step 2: Test CRM query cancellation"):
            return False
        
        try:
            result = await _process_hitl_response_llm_driven(hitl_context, "cancel the search")
            
            if result.get("result") != "denied":
                return self.fail_test(
                    "Step 2",
                    "CRM query cancellation not detected",
                    "denied",
                    result.get("result")
                )
            
            response_message = result.get("response_message", "")
            if len(response_message) < 10:
                return self.fail_test(
                    "Step 2",
                    "No cancellation message generated for CRM query",
                    "Contextual message",
                    response_message
                )
            
            self.pass_step("Step 2", f"CRM query cancelled: '{response_message}'")
            
        except Exception as e:
            return self.fail_test("Step 2", f"Exception: {str(e)}")
        
        # Step 3: Test input handling (user provides search criteria)
        if not self.pass_step("Step 3: Test CRM query input handling"):
            return False
        
        try:
            result = await _process_hitl_response_llm_driven(hitl_context, "search for customers named John")
            
            if result.get("result") != "search for customers named John":
                return self.fail_test(
                    "Step 3",
                    "CRM query input not handled correctly",
                    "search for customers named John",
                    result.get("result")
                )
            
            if result.get("context") is None:
                return self.fail_test(
                    "Step 3",
                    "Context cleared on input (should be preserved)",
                    "Preserved context",
                    "None"
                )
            
            self.pass_step("Step 3", "CRM query input handled correctly")
            
        except Exception as e:
            return self.fail_test("Step 3", f"Exception: {str(e)}")
        
        print(f"\n✅ {self.current_test} - ALL STEPS PASSED!")
        return True
    
    async def test_hitl_state_management(self):
        """Test HITL state management throughout the flow."""
        self.current_test = "HITL State Management"
        print(f"\n🧪 Testing: {self.current_test}")
        print("=" * 60)
        
        # Step 1: Test state transitions
        if not self.pass_step("Step 1: Test HITL state transitions"):
            return False
        
        hitl_context = {
            "source_tool": "generate_quotation",
            "collection_mode": "tool_managed",
            "original_params": {"test": "data"}
        }
        
        # Test denial clears state
        try:
            denial_result = await _process_hitl_response_llm_driven(hitl_context, "cancel")
            
            if denial_result.get("context") is not None:
                return self.fail_test(
                    "Step 1 - Denial",
                    "HITL context not cleared on denial",
                    "None",
                    str(denial_result.get("context"))
                )
            
            self.pass_step("Step 1 - Denial", "HITL context correctly cleared on denial")
            
        except Exception as e:
            return self.fail_test("Step 1 - Denial", f"Exception: {str(e)}")
        
        # Test approval preserves state
        try:
            approval_result = await _process_hitl_response_llm_driven(hitl_context, "yes, proceed")
            
            if approval_result.get("context") is None:
                return self.fail_test(
                    "Step 1 - Approval",
                    "HITL context cleared on approval (should be preserved)",
                    "Preserved context",
                    "None"
                )
            
            preserved_context = approval_result.get("context")
            if preserved_context.get("source_tool") != "generate_quotation":
                return self.fail_test(
                    "Step 1 - Approval",
                    "HITL context corrupted on approval",
                    "generate_quotation",
                    preserved_context.get("source_tool")
                )
            
            self.pass_step("Step 1 - Approval", "HITL context correctly preserved on approval")
            
        except Exception as e:
            return self.fail_test("Step 1 - Approval", f"Exception: {str(e)}")
        
        print(f"\n✅ {self.current_test} - ALL STEPS PASSED!")
        return True
    
    async def run_all_tests(self):
        """Run all end-to-end tests in fail-fast mode."""
        print("🚀 Starting Comprehensive End-to-End HITL Tests")
        print("🛑 FAIL-FAST MODE: Tests will stop on first failure")
        print("=" * 80)
        
        tests = [
            self.test_quotation_hitl_flow_with_cancellation,
            self.test_customer_message_hitl_flow_with_cancellation,
            self.test_crm_query_hitl_flow_with_cancellation,
            self.test_hitl_state_management
        ]
        
        for i, test_func in enumerate(tests, 1):
            success = await test_func()
            if not success:
                print(f"\n💥 TESTS STOPPED AT TEST {i}")
                print("🔧 Please debug and fix the issue, then re-run tests")
                return False
        
        print("\n" + "=" * 80)
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print("✅ HITL loop implementation preserved")
        print("✅ Enhanced cancellation system working")
        print("✅ Tool execution routing correct")
        print("✅ State management working")
        print("=" * 80)
        
        return True

async def main():
    """Main test runner."""
    tester = FailFastHITLTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n🏆 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
            sys.exit(0)
        else:
            print("\n🛑 TESTS FAILED - DEBUGGING REQUIRED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        tester.save_debug_results()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
