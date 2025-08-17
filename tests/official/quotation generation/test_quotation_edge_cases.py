#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing for Quotation Generation System

Tests critical edge cases that could break the HITL quotation flow:
- Invalid customer data
- Malformed vehicle specifications  
- HITL flow interruptions
- System state corruption
- Concurrent usage patterns
"""

import asyncio
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class QuotationEdgeCaseTests:
    """Comprehensive edge case testing for quotation generation."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
        self.errors_found = []
        
        # Use a valid employee for all tests
        self.test_employee = {
            "name": "Alex Thompson",
            "user_id": "54394d40-ad35-4b5b-a392-1ae7c9329d11"
        }

    def send_message(self, message: str, user_id: str, conversation_id: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Send a message to the chat API and return the response."""
        try:
            payload = {
                "message": message,
                "user_id": user_id
            }
            
            if conversation_id:
                payload["conversation_id"] = conversation_id
                
            response = requests.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("message", ""),
                    "is_interrupted": data.get("is_interrupted", False),
                    "conversation_id": data.get("conversation_id", "")
                }
            else:
                return {
                    "success": False,
                    "response": f"HTTP {response.status_code}: {response.text}",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "response": str(e),
                "error": "Exception"
            }

    def analyze_response_type(self, response: str) -> str:
        """Analyze the type of response received."""
        response_lower = response.lower()
        
        if "critical information needed" in response_lower or "essential information" in response_lower:
            return "missing_info_request"
        elif "quotation ready for generation" in response_lower and "approve" in response_lower:
            return "approval_request"
        elif "quotation has been generated" in response_lower or "pdf" in response_lower:
            return "quotation_generated"
        elif "unable to generate" in response_lower or "cannot proceed" in response_lower:
            return "error_response"
        else:
            return "unknown"

    # =============================================================================
    # EDGE CASE TEST CATEGORIES
    # =============================================================================

    async def test_non_existent_customer(self) -> Tuple[bool, str]:
        """Test quotation generation for a customer not in the database."""
        print("\n🧪 EDGE CASE: Non-existent Customer")
        
        try:
            # Step 1: Start quotation
            print("   📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result["success"]:
                print(f"      ❌ Step 1 Error: {result['error']}")
                return False, f"Step 1 failed: {result['error']}"
            
            conversation_id = result.get("conversation_id")
            print(f"      📋 Conversation ID: {conversation_id}")
            
            response_type = self.analyze_response_type(result["response"])
            if response_type != "missing_info_request":
                print(f"      ❌ Step 1 Error: Expected missing_info_request, got {response_type}")
                print(f"         Response: {result['response'][:200]}...")
                return False, f"Step 1 failed: Expected missing_info_request, got {response_type}"
            
            print("      ✅ Step 1 Success: Correctly requested missing information")
            
            # Step 2: Provide non-existent customer
            print("   👤 STEP 2: Provide Non-existent Customer Name")
            result = self.send_message("Nonexistent Customer", self.test_employee["user_id"], conversation_id)
            if not result["success"]:
                print(f"      ❌ Step 2 Error: {result['error']}")
                return False, f"Step 2 failed: {result['error']}"
            
            # Should ask for contact information since customer not found
            response_lower = result["response"].lower()
            if "contact" not in response_lower and "email" not in response_lower and "phone" not in response_lower:
                print(f"      ❌ Step 2 Error: System should ask for contact info for unknown customer")
                print(f"         Response: {result['response'][:300]}...")
                return False, "System should ask for contact info for unknown customer"
            
            print("      ✅ Step 2 Success: Correctly asked for contact info for unknown customer")
            return True, "Non-existent customer handled correctly"
            
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    async def test_malformed_vehicle_specs(self) -> Tuple[bool, str]:
        """Test with malformed vehicle specifications - FAIL-FAST on first issue."""
        print("\n🧪 EDGE CASE: Malformed Vehicle Specifications")
        
        # Test the most problematic case first (random text)
        malformed_spec = "xyz123"  # Random text that makes no sense
        
        try:
            # Step 1: Start quotation
            print("   📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result["success"]:
                print(f"      ❌ Step 1 Error: {result['error']}")
                return False, f"Step 1 failed: {result['error']}"
            
            conversation_id = result.get("conversation_id")
            print(f"      📋 Conversation ID: {conversation_id}")
            print("      ✅ Step 1 Success: Quotation request initiated")
            
            # Step 2: Provide valid customer
            print("   👤 STEP 2: Provide Valid Customer Name")
            result = self.send_message("Alice Johnson", self.test_employee["user_id"], conversation_id)
            if not result["success"]:
                print(f"      ❌ Step 2 Error: {result['error']}")
                return False, f"Step 2 failed: {result['error']}"
            
            print("      ✅ Step 2 Success: Customer recognized")
            
            # Step 3: Provide malformed vehicle spec
            print(f"   🚗 STEP 3: Provide Malformed Vehicle Spec: '{malformed_spec}'")
            result = self.send_message(malformed_spec, self.test_employee["user_id"], conversation_id, timeout=30)
            
            if not result["success"]:
                print(f"      ❌ Step 3 Error: {result['error']}")
                return False, f"Step 3 failed: {result['error']}"
            
            # System should handle gracefully by generating a new HITL request
            # Check if it's asking for more information (which is the correct behavior)
            response_type = self.analyze_response_type(result["response"])
            response_lower = result["response"].lower()
            
            # The system should either:
            # 1. Generate a new missing_info_request (asking for proper vehicle info)
            # 2. Ask for clarification
            # 3. Request specific vehicle details
            handled_gracefully = (
                response_type == "missing_info_request" or
                "clarification" in response_lower or
                "specify" in response_lower or
                "vehicle" in response_lower or
                "make" in response_lower or
                "model" in response_lower or
                "information" in response_lower or
                "essential" in response_lower or
                "critical" in response_lower
            )
            
            if handled_gracefully:
                print(f"      ✅ Step 3 Success: Malformed input handled gracefully")
                print(f"         Response Type: {response_type}")
                print(f"         Response: {result['response'][:200]}...")
                return True, f"Malformed vehicle spec '{malformed_spec}' handled gracefully"
            else:
                print(f"      ❌ Step 3 Error: System did not handle malformed input gracefully")
                print(f"         Response Type: {response_type}")
                print(f"         Response: {result['response'][:300]}...")
                return False, f"System did not handle malformed vehicle spec gracefully"
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    async def test_hitl_flow_interruptions(self) -> Tuple[bool, str]:
        """Test HITL flow interruption with invalid response."""
        print("\n🧪 EDGE CASE: HITL Flow Interruptions")
        
        try:
            # Step 1: Start quotation
            print("   📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result["success"]:
                print(f"      ❌ Step 1 Error: {result['error']}")
                return False, f"Step 1 failed: {result['error']}"
            
            conversation_id = result.get("conversation_id")
            print(f"      📋 Conversation ID: {conversation_id}")
            print("      ✅ Step 1 Success: Quotation request initiated")
            
            # Step 2: Provide completely invalid response to HITL prompt
            print("   ❌ STEP 2: Provide Invalid Response to HITL")
            invalid_response = "blah blah this is completely invalid nonsense xyz123"
            result = self.send_message(invalid_response, self.test_employee["user_id"], conversation_id)
            
            if not result["success"]:
                print(f"      ❌ Step 2 Error: {result['error']}")
                return False, f"Step 2 failed: {result['error']}"
            
            # System should handle gracefully - either:
            # 1. Ask for clarification
            # 2. Request proper information
            # 3. Show helpful error message
            # It should NOT crash or get stuck
            
            response_lower = result["response"].lower()
            handled_gracefully = (
                "clarification" in response_lower or
                "understand" in response_lower or
                "specify" in response_lower or
                "information" in response_lower or
                "help" in response_lower or
                "customer" in response_lower or
                "vehicle" in response_lower
            )
            
            if handled_gracefully:
                print(f"      ✅ Step 2 Success: Invalid response handled gracefully")
                print(f"         Response: {result['response'][:200]}...")
                return True, "Invalid HITL response handled gracefully"
            else:
                print(f"      ❌ Step 2 Error: System did not handle invalid response gracefully")
                print(f"         Response: {result['response'][:300]}...")
                return False, "System did not handle invalid HITL response gracefully"
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    async def test_approval_variations(self) -> Tuple[bool, str]:
        """Test non-standard approval response format."""
        print("\n🧪 EDGE CASE: Approval Variations")
        
        try:
            # Step 1: Start quotation
            print("   📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result["success"]:
                print(f"      ❌ Step 1 Error: {result['error']}")
                return False, f"Step 1 failed: {result['error']}"
            
            conversation_id = result.get("conversation_id")
            print("      ✅ Step 1 Success: Quotation request initiated")
            
            # Step 2: Provide customer
            print("   👤 STEP 2: Provide Customer Name")
            result = self.send_message("Alice Johnson", self.test_employee["user_id"], conversation_id)
            if not result["success"]:
                print(f"      ❌ Step 2 Error: {result['error']}")
                return False, f"Step 2 failed: {result['error']}"
            
            print("      ✅ Step 2 Success: Customer recognized")
            
            # Step 3: Provide vehicle
            print("   🚗 STEP 3: Provide Vehicle Information")
            result = self.send_message("Honda Civic", self.test_employee["user_id"], conversation_id)
            if not result["success"]:
                print(f"      ❌ Step 3 Error: {result['error']}")
                return False, f"Step 3 failed: {result['error']}"
            
            response_type = self.analyze_response_type(result["response"])
            if response_type != "approval_request":
                print(f"      ❌ Step 3 Error: Expected approval_request, got {response_type}")
                return False, f"Step 3 failed: Expected approval_request, got {response_type}"
            
            print("      ✅ Step 3 Success: Approval requested")
            
            # Step 4: Test non-standard approval format
            print("   ✅ STEP 4: Test Non-standard Approval ('yes please')")
            non_standard_approval = "yes please"  # Not the standard "Approve"
            result = self.send_message(non_standard_approval, self.test_employee["user_id"], conversation_id, timeout=45)
            
            if not result["success"]:
                print(f"      ❌ Step 4 Error: {result['error']}")
                return False, f"Step 4 failed: {result['error']}"
            
            # Should generate quotation successfully
            response_type = self.analyze_response_type(result["response"])
            if response_type == "quotation_generated":
                print(f"      ✅ Step 4 Success: Non-standard approval '{non_standard_approval}' worked")
                return True, f"Non-standard approval '{non_standard_approval}' handled correctly"
            else:
                print(f"      ❌ Step 4 Error: Non-standard approval not recognized")
                print(f"         Response: {result['response'][:300]}...")
                return False, f"Non-standard approval '{non_standard_approval}' not recognized"
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    async def test_concurrent_quotations(self) -> Tuple[bool, str]:
        """Test basic concurrent session handling."""
        print("\n🧪 EDGE CASE: Concurrent Quotation Sessions")
        
        try:
            # Step 1: Start first quotation session
            print("   📝 STEP 1: Start First Quotation Session")
            result1 = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result1["success"]:
                print(f"      ❌ Step 1 Error: {result1['error']}")
                return False, f"Step 1 failed: {result1['error']}"
            
            conversation_id1 = result1.get("conversation_id")
            print(f"      📋 First Conversation ID: {conversation_id1}")
            print("      ✅ Step 1 Success: First session started")
            
            # Step 2: Start second quotation session (without finishing first)
            print("   📝 STEP 2: Start Second Quotation Session")
            result2 = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result2["success"]:
                print(f"      ❌ Step 2 Error: {result2['error']}")
                return False, f"Step 2 failed: {result2['error']}"
            
            conversation_id2 = result2.get("conversation_id")
            print(f"      📋 Second Conversation ID: {conversation_id2}")
            
            # Verify they are different conversations
            if conversation_id1 == conversation_id2:
                print(f"      ❌ Step 2 Error: Same conversation ID used for both sessions")
                return False, "Same conversation ID used for concurrent sessions"
            
            print("      ✅ Step 2 Success: Second session started with different conversation ID")
            
            # Step 3: Continue with first session
            print("   👤 STEP 3: Continue First Session")
            result = self.send_message("Alice Johnson", self.test_employee["user_id"], conversation_id1)
            if not result["success"]:
                print(f"      ❌ Step 3 Error: {result['error']}")
                return False, f"Step 3 failed: {result['error']}"
            
            print("      ✅ Step 3 Success: First session continued successfully")
            return True, "Concurrent sessions handled with separate conversation IDs"
            
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    async def test_database_edge_cases(self) -> Tuple[bool, str]:
        """Test customer with special characters in name."""
        print("\n🧪 EDGE CASE: Database Edge Cases")
        
        try:
            # Step 1: Start quotation
            print("   📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", self.test_employee["user_id"])
            if not result["success"]:
                print(f"      ❌ Step 1 Error: {result['error']}")
                return False, f"Step 1 failed: {result['error']}"
            
            conversation_id = result.get("conversation_id")
            print("      ✅ Step 1 Success: Quotation request initiated")
            
            # Step 2: Test customer with special characters
            print("   👤 STEP 2: Provide Customer with Special Characters")
            special_customer = "O'Connor-Smith"  # Apostrophe and hyphen
            result = self.send_message(special_customer, self.test_employee["user_id"], conversation_id)
            
            if not result["success"]:
                print(f"      ❌ Step 2 Error: {result['error']}")
                return False, f"Step 2 failed: {result['error']}"
            
            # Should handle gracefully (either find customer or ask for contact info)
            response_lower = result["response"].lower()
            handled_gracefully = (
                "contact" in response_lower or
                "email" in response_lower or
                "phone" in response_lower or
                "vehicle" in response_lower or
                "information" in response_lower
            )
            
            if handled_gracefully:
                print(f"      ✅ Step 2 Success: Special character customer handled gracefully")
                print(f"         Response: {result['response'][:200]}...")
                return True, f"Customer with special characters '{special_customer}' handled correctly"
            else:
                print(f"      ❌ Step 2 Error: Special character customer not handled gracefully")
                print(f"         Response: {result['response'][:300]}...")
                return False, f"Customer with special characters not handled gracefully"
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return False, f"Exception: {str(e)}"

    # =============================================================================
    # MAIN TEST RUNNER
    # =============================================================================

    async def run_all_edge_case_tests(self) -> Dict[str, Any]:
        """Run all edge case tests with FAIL-FAST mode."""
        print("🚀 Starting Comprehensive Edge Case Testing for Quotation Generation")
        print(f"Base URL: {self.base_url}")
        print(f"Test Employee: {self.test_employee['name']}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print("🛑 FAIL-FAST MODE: Test will stop at first edge case failure for immediate debugging")
        
        # Define all test cases
        test_cases = [
            ("Non-existent Customer", self.test_non_existent_customer),
            ("Malformed Vehicle Specs", self.test_malformed_vehicle_specs),
            ("HITL Flow Interruptions", self.test_hitl_flow_interruptions),
            ("Approval Variations", self.test_approval_variations),
            ("Concurrent Sessions", self.test_concurrent_quotations),
            ("Database Edge Cases", self.test_database_edge_cases),
        ]
        
        results = []
        successful_tests = 0
        
        for i, (test_name, test_func) in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"🧪 EDGE CASE TEST {i}/{len(test_cases)}: {test_name}")
            print(f"{'='*80}")
            
            try:
                success, message = await test_func()
                
                results.append({
                    "test_name": test_name,
                    "success": success,
                    "message": message
                })
                
                if success:
                    successful_tests += 1
                    print(f"   🎉 SUCCESS: {message}")
                else:
                    print(f"   ❌ FAILED: {message}")
                    self.errors_found.append(f"{test_name}: {message}")
                    print(f"\n🛑 STOPPING TESTS: Edge case '{test_name}' failed")
                    print(f"   Error: {message}")
                    print(f"   This allows immediate debugging of the specific issue")
                    break  # FAIL-FAST: Stop at first failure
                
                # Delay between test categories
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"Exception during {test_name}: {str(e)}"
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "message": error_msg
                })
                self.errors_found.append(error_msg)
                print(f"   ❌ EXCEPTION: {error_msg}")
                print(f"\n🛑 STOPPING TESTS: Exception in '{test_name}'")
                print(f"   Exception: {error_msg}")
                break  # FAIL-FAST: Stop at first exception
        
        # Generate final report
        success_rate = (successful_tests / len(test_cases)) * 100
        overall_success = successful_tests == len(test_cases)
        
        final_report = {
            "test_type": "Quotation Generation Edge Cases",
            "total_test_categories": len(test_cases),
            "successful_categories": successful_tests,
            "success_rate": success_rate,
            "overall_success": overall_success,
            "test_results": results,
            "errors_found": self.errors_found,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n{'='*80}")
        print("📊 FINAL EDGE CASE TEST REPORT")
        print(f"{'='*80}")
        print(f"Total Test Categories: {len(test_cases)}")
        print(f"Successful Categories: {successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        if self.errors_found:
            print(f"\n❌ ERRORS FOUND ({len(self.errors_found)}):")
            for i, error in enumerate(self.errors_found, 1):
                print(f"  {i}. {error}")
        
        print(f"\n📄 Test Category Results:")
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  • {result['test_name']}: {status}")
        
        # Save detailed report
        report_filename = f"quotation_edge_case_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print(f"{'='*80}")
        
        return final_report


async def main():
    """Run the edge case tests."""
    tester = QuotationEdgeCaseTests()
    await tester.run_all_edge_case_tests()


if __name__ == "__main__":
    asyncio.run(main())
