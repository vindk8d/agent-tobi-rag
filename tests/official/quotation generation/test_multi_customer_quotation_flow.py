#!/usr/bin/env python3
"""
Multi-Customer End-to-End Quotation Generation Test

Tests quotation generation flow with 3 different database customers:
1. Alice Johnson - Individual customer
2. Bob Smith - Personal customer  
3. Carol Thompson - Logistics company customer

This ensures our system works consistently across different customer profiles.
"""

import asyncio
import json
import requests
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

class MultiCustomerQuotationTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
        self.errors_found = []
        
        # Test scenarios: Employees generating quotations for different database customers
        self.test_scenarios = [
            {
                "employee_name": "Alex Thompson",
                "user_id": "54394d40-ad35-4b5b-a392-1ae7c9329d11",  # Employee user ID
                "customer_name": "Alice Johnson",
                "customer_email": "alice.johnson@email.com",
                "customer_phone": "555-1111",
                "vehicle": "Honda Civic",
                "description": "Employee Alex generating quotation for customer Alice Johnson"
            },
            {
                "employee_name": "Amanda White",
                "user_id": "f7f1d012-0423-4add-a42c-6da8c2bae08b",  # Employee user ID
                "customer_name": "Bob Smith",
                "customer_email": "bob.smith@personal.com",
                "customer_phone": "555-2222", 
                "vehicle": "Ford F-150",
                "description": "Employee Amanda generating quotation for customer Bob Smith"
            },
            {
                "employee_name": "Carlos Rodriguez",
                "user_id": "492dbd86-ddc9-4d52-9208-b578e0fccc93",  # Employee user ID
                "customer_name": "Carol Thompson",
                "customer_email": "carol.thompson@logistics.com",
                "customer_phone": "555-3333",
                "vehicle": "Toyota Hiace",
                "description": "Employee Carlos generating quotation for customer Carol Thompson"
            }
        ]

    def send_message(self, message: str, user_id: str, conversation_id: str = None, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """Send a message to the chat API and return the response."""
        try:
            payload = {
                "message": message,
                "user_id": user_id
            }
            
            # CRITICAL: Include conversation_id to maintain conversation continuity
            if conversation_id:
                payload["conversation_id"] = conversation_id
                
            response = requests.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload,
                timeout=60  # Increased timeout for PDF generation
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
                "response": f"Request Exception: {str(e)}",
                "error": str(e)
            }

    def analyze_response_type(self, response: str) -> str:
        """Analyze the response to determine its type."""
        response_lower = response.lower()
        
        if "critical information needed" in response_lower or "need the following" in response_lower:
            return "missing_info_request"
        elif "quotation ready for generation" in response_lower:
            return "approval_request"
        elif "professional vehicle quotation" in response_lower or "quotation" in response_lower:
            return "quotation_generated"
        elif "error" in response_lower or "unable" in response_lower:
            return "error"
        else:
            return "unknown"

    def log_step(self, step: str, request: str, response: str, success: bool, notes: str = ""):
        """Log a test step result."""
        self.test_results.append({
            "step": step,
            "request": request,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "success": success,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        })

    async def test_customer_quotation_flow(self, scenario: Dict[str, Any]) -> Tuple[bool, str]:
        """Test the complete quotation flow for a single scenario."""
        print(f"\n🧪 Testing Scenario: {scenario['description']}")
        print(f"   Employee: {scenario['employee_name']} (User ID: {scenario['user_id']})")
        print(f"   Customer: {scenario['customer_name']}")
        print(f"   Expected Vehicle: {scenario['vehicle']}")
        
        scenario_results = []
        conversation_id = None
        
        try:
            # Step 1: Initial quotation request
            print(f"\n📝 STEP 1: Initial Quotation Request")
            result = self.send_message("generate a quotation", scenario['user_id'])
            
            if not result["success"]:
                return False, f"Step 1 failed: {result.get('error', 'Unknown error')}"
            
            conversation_id = result.get("conversation_id")
            print(f"   📋 Conversation ID: {conversation_id}")
            
            response_type = self.analyze_response_type(result["response"])
            
            if response_type != "missing_info_request":
                print(f"   ❌ Step 1 Error Details:")
                print(f"      Expected: missing_info_request")
                print(f"      Got: {response_type}")
                print(f"      Response: {result['response'][:500]}...")
                return False, f"Step 1 failed: Expected missing_info_request, got {response_type}"
            
            print(f"   ✅ Step 1 Success: Correctly requested missing information")
            
            # Step 2: Provide customer name (should be recognized from database)
            print(f"\n👤 STEP 2: Provide Customer Name")
            result = self.send_message(scenario['customer_name'], scenario['user_id'], conversation_id)
            
            if not result["success"]:
                return False, f"Step 2 failed: {result.get('error', 'Unknown error')}"
            
            response_type = self.analyze_response_type(result["response"])
            
            # Should ask for vehicle info since customer is recognized
            if response_type != "missing_info_request":
                print(f"   ❌ Step 2 Error Details:")
                print(f"      Expected: missing_info_request")
                print(f"      Got: {response_type}")
                print(f"      Response: {result['response'][:500]}...")
                return False, f"Step 2 failed: Expected missing_info_request, got {response_type}"
            
            # Check if customer was recognized (should not ask for contact info)
            if "contact info" in result["response"].lower():
                return False, f"Step 2 failed: Customer not recognized from database - asking for contact info"
            
            print(f"   ✅ Step 2 Success: Customer recognized from database")
            
            # Step 3: Provide vehicle information
            print(f"\n🚗 STEP 3: Provide Vehicle Information")
            result = self.send_message(scenario['vehicle'], scenario['user_id'], conversation_id)
            
            if not result["success"]:
                return False, f"Step 3 failed: {result.get('error', 'Unknown error')}"
            
            response_type = self.analyze_response_type(result["response"])
            
            # Should show approval request since all info is complete
            if response_type != "approval_request":
                print(f"   ❌ Step 3 Error Details:")
                print(f"      Expected: approval_request")
                print(f"      Got: {response_type}")
                print(f"      Response: {result['response'][:500]}...")
                return False, f"Step 3 failed: Expected approval_request, got {response_type}"
            
            # Verify customer info is preserved
            if scenario['customer_name'].lower() not in result["response"].lower():
                print(f"   ❌ Step 3 Error: Customer name '{scenario['customer_name']}' not found in response")
                print(f"      Response: {result['response'][:500]}...")
                return False, f"Step 3 failed: Customer name not preserved in approval request"
            
            # Check if vehicle info is preserved (more flexible matching)
            vehicle_parts = scenario['vehicle'].lower().split()
            vehicle_found = all(part in result["response"].lower() for part in vehicle_parts)
            
            if not vehicle_found:
                print(f"   ❌ Step 3 Error: Vehicle parts {vehicle_parts} not all found in response")
                print(f"      Response: {result['response'][:500]}...")
                return False, f"Step 3 failed: Vehicle info not preserved in approval request"
            
            print(f"   ✅ Step 3 Success: All information gathered, approval requested")
            
            # Step 4: Final approval
            print(f"\n✅ STEP 4: Final Approval")
            result = self.send_message("Approve", scenario['user_id'], conversation_id)
            
            if not result["success"]:
                return False, f"Step 4 failed: {result.get('error', 'Unknown error')}"
            
            response_type = self.analyze_response_type(result["response"])
            
            if response_type != "quotation_generated":
                return False, f"Step 4 failed: Expected quotation_generated, got {response_type}"
            
            # Verify quotation contains customer and vehicle info
            if scenario['customer_name'].lower() not in result["response"].lower():
                return False, f"Step 4 failed: Customer name not in final quotation"
            
            print(f"   ✅ Step 4 Success: Quotation generated successfully")
            
            return True, f"All steps completed successfully for {scenario['customer_name']}"
            
        except Exception as e:
            return False, f"Exception during test: {str(e)}"

    async def run_all_customer_tests(self) -> Dict[str, Any]:
        """Run quotation tests for all customers with FAIL-FAST mode."""
        print("🚀 Starting Multi-Customer Quotation Generation Tests")
        print(f"Base URL: {self.base_url}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print(f"Testing {len(self.test_scenarios)} employee-customer scenarios from database")
        print("🛑 FAIL-FAST MODE: Test will stop at first scenario failure for immediate debugging")
        
        results = []
        successful_tests = 0
        
        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"\n{'='*80}")
            print(f"🧪 SCENARIO TEST {i}/{len(self.test_scenarios)}")
            print(f"{'='*80}")
            
            success, message = await self.test_customer_quotation_flow(scenario)
            
            results.append({
                "employee": scenario['employee_name'],
                "customer": scenario['customer_name'],
                "user_id": scenario['user_id'],
                "description": scenario['description'],
                "success": success,
                "message": message
            })
            
            if success:
                successful_tests += 1
                print(f"   🎉 SUCCESS: {message}")
            else:
                print(f"   ❌ FAILED: {message}")
                self.errors_found.append(f"{scenario['employee_name']} -> {scenario['customer_name']}: {message}")
                print(f"\n🛑 STOPPING TESTS: Scenario {scenario['description']} failed")
                print(f"   Error: {message}")
                print(f"   This allows immediate debugging of the specific issue")
                break  # FAIL-FAST: Stop at first failure
            
            # Small delay between tests
            time.sleep(2)
        
        # Generate final report
        success_rate = (successful_tests / len(self.test_scenarios)) * 100
        overall_success = successful_tests == len(self.test_scenarios)
        
        final_report = {
            "test_type": "Multi-Customer Quotation Generation",
            "total_scenarios": len(self.test_scenarios),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "overall_success": overall_success,
            "scenario_results": results,
            "errors_found": self.errors_found,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n{'='*80}")
        print("📊 FINAL MULTI-CUSTOMER TEST REPORT")
        print(f"{'='*80}")
        print(f"Total Scenarios Tested: {len(self.test_scenarios)}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        if self.errors_found:
            print(f"\n❌ ERRORS FOUND ({len(self.errors_found)}):")
            for i, error in enumerate(self.errors_found, 1):
                print(f"  {i}. {error}")
        
        print(f"\n📄 Scenario Test Results:")
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  • {result['employee']} → {result['customer']}: {status}")
        
        # Save detailed report
        report_filename = f"multi_customer_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print(f"{'='*80}")
        
        return final_report

async def main():
    """Run the multi-customer quotation tests."""
    test = MultiCustomerQuotationTest()
    await test.run_all_customer_tests()

if __name__ == "__main__":
    asyncio.run(main())
