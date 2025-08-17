#!/usr/bin/env python3
"""
End-to-End Quotation Generation Test

This test runs through a complete conversation to generate a quotation,
providing responses step by step, evaluating logs, and correcting errors
until we get a generated quotation with a link.

Test Flow:
1. Start quotation request
2. Provide customer name
3. Provide vehicle information
4. Handle any missing information requests
5. Approve final quotation
6. Verify PDF generation and link

Each step includes:
- Request execution
- Log analysis
- Error detection and correction
- Retry mechanism
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

class QuotationE2ETest:
    """End-to-end quotation generation test with comprehensive logging and error handling."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_id = None
        self.test_results = []
        self.errors_found = []
        self.conversation_log = []
        # Use Christopher Lee's employee user ID from the database
        self.user_id = "1177c9a3-d6b3-4b5a-8e6f-483ca5dd6069"
        
    def log_step(self, step: str, request: str, response: str, success: bool, notes: str = ""):
        """Log each test step with detailed information."""
        step_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "request": request,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "success": success,
            "notes": notes,
            "conversation_id": self.conversation_id
        }
        self.test_results.append(step_data)
        self.conversation_log.append(f"[{step}] {request} → {response[:100]}...")
        
        print(f"\n{'='*60}")
        print(f"STEP: {step}")
        print(f"REQUEST: {request}")
        print(f"RESPONSE: {response[:200]}...")
        print(f"SUCCESS: {success}")
        if notes:
            print(f"NOTES: {notes}")
        print(f"{'='*60}")
    
    def send_message(self, message: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """Send a message and return the response with analysis."""
        try:
            payload = {
                "message": message,
                "user_id": self.user_id,
                "include_sources": True
            }
            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract conversation_id if not set
                if not self.conversation_id and "conversation_id" in data:
                    self.conversation_id = data["conversation_id"]
                
                response_text = data.get("message", data.get("response", ""))
                interrupted = data.get("is_interrupted", data.get("interrupted", False))
                
                # Check for expected keywords
                keywords_found = []
                if expected_keywords:
                    for keyword in expected_keywords:
                        if keyword.lower() in response_text.lower():
                            keywords_found.append(keyword)
                
                return {
                    "success": True,
                    "response": response_text,
                    "interrupted": interrupted,
                    "conversation_id": self.conversation_id,
                    "keywords_found": keywords_found,
                    "raw_data": data
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ API Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response": f"API Error: {error_msg}",
                    "interrupted": False
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Request Exception: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "response": f"Request Exception: {error_msg}",
                "interrupted": False
            }
    
    def analyze_response_type(self, response: str) -> str:
        """Analyze the type of response received."""
        response_lower = response.lower()
        
        if "critical information needed" in response_lower:
            return "missing_info_request"
        elif "quotation ready for generation" in response_lower:
            return "approval_request"
        elif "pdf" in response_lower and ("generated" in response_lower or "link" in response_lower):
            return "quotation_generated"
        elif "error" in response_lower:
            return "error"
        elif "vehicle make" in response_lower and "vehicle model" in response_lower:
            return "vehicle_info_request"
        elif ("need" in response_lower and "information" in response_lower) or \
             ("provide" in response_lower and "details" in response_lower) or \
             ("customer information" in response_lower and "vehicle" in response_lower):
            return "missing_info_request"
        else:
            return "unknown"
    
    def extract_missing_info(self, response: str) -> List[str]:
        """Extract what information is still missing from the response."""
        missing = []
        response_lower = response.lower()
        
        if "vehicle make" in response_lower:
            missing.append("vehicle_make")
        if "vehicle model" in response_lower:
            missing.append("vehicle_model")
        if "customer name" in response_lower:
            missing.append("customer_name")
        if "email" in response_lower or "contact" in response_lower:
            missing.append("contact_info")
            
        return missing
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete end-to-end quotation test."""
        print("🚀 Starting End-to-End Quotation Generation Test")
        print(f"Base URL: {self.base_url}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print("🛑 FAIL-FAST MODE: Test will stop at first failure for immediate debugging")
        
        try:
            # Step 1: Initial quotation request
            success = await self.step_1_initial_request()
            if not success:
                print("🛑 STOPPING TEST: Step 1 failed")
                return self.generate_final_report()
            
            # Step 2: Provide customer information
            success = await self.step_2_customer_info()
            if not success:
                print("🛑 STOPPING TEST: Step 2 failed")
                return self.generate_final_report()
            
            # Step 3: Provide vehicle information
            success = await self.step_3_vehicle_info()
            if not success:
                print("🛑 STOPPING TEST: Step 3 failed")
                return self.generate_final_report()
            
            # Step 4: Provide contact information if needed
            success = await self.step_4_provide_contact_info()
            if not success:
                print("🛑 STOPPING TEST: Step 4 failed")
                return self.generate_final_report()
            
            # Step 5: Final approval
            success = await self.step_5_final_approval()
            if not success:
                print("🛑 STOPPING TEST: Step 5 failed")
                return self.generate_final_report()
            
            # Step 6: Verify quotation generation
            await self.step_6_verify_quotation()
            
        except Exception as e:
            self.errors_found.append(f"Test execution error: {str(e)}")
            print(f"❌ Test failed with error: {e}")
        
        # Generate final report
        return self.generate_final_report()
    
    async def step_1_initial_request(self) -> bool:
        """Step 1: Send initial quotation request."""
        print("\n📝 STEP 1: Initial Quotation Request")
        
        result = self.send_message(
            "generate a quotation",
            expected_keywords=["customer", "information", "needed"]
        )
        
        response_type = self.analyze_response_type(result["response"])
        success = result["success"] and response_type in ["missing_info_request", "vehicle_info_request"]
        
        self.log_step(
            "1_initial_request",
            "generate a quotation",
            result["response"],
            success,
            f"Response type: {response_type}"
        )
        
        if not success:
            self.errors_found.append(f"Step 1 failed: Expected missing info request, got {response_type}")
            print(f"❌ Step 1 Error Details:")
            print(f"   - API Success: {result['success']}")
            print(f"   - Response Type: {response_type}")
            print(f"   - Response: {result['response'][:200]}...")
            if 'error' in result:
                print(f"   - Error: {result['error']}")
        
        return success
    
    async def step_2_customer_info(self) -> bool:
        """Step 2: Provide customer information."""
        print("\n👤 STEP 2: Provide Customer Information")
        
        result = self.send_message(
            "John Martinez",
            expected_keywords=["vehicle", "make", "model"]
        )
        
        response_type = self.analyze_response_type(result["response"])
        success = result["success"] and response_type in ["missing_info_request", "vehicle_info_request"]
        
        self.log_step(
            "2_customer_info",
            "John Martinez",
            result["response"],
            success,
            f"Response type: {response_type}"
        )
        
        if not success:
            self.errors_found.append(f"Step 2 failed: Expected vehicle info request, got {response_type}")
            print(f"❌ Step 2 Error Details:")
            print(f"   - API Success: {result['success']}")
            print(f"   - Response Type: {response_type}")
            print(f"   - Response: {result['response'][:200]}...")
            if 'error' in result:
                print(f"   - Error: {result['error']}")
        
        return success
    
    async def step_3_vehicle_info(self) -> bool:
        """Step 3: Provide vehicle information."""
        print("\n🚗 STEP 3: Provide Vehicle Information")
        
        result = self.send_message(
            "Toyota Camry",
            expected_keywords=["quotation ready", "approve", "toyota", "camry", "contact info"]
        )
        
        response_type = self.analyze_response_type(result["response"])
        # Success if we get approval request OR if we're asked for contact info (progress!)
        success = result["success"] and response_type in ["approval_request", "quotation_generated", "missing_info_request"]
        
        # Check if context is preserved (should mention Toyota/Camry and remember customer name)
        response_lower = result["response"].lower()
        has_toyota = "toyota" in response_lower
        has_camry = "camry" in response_lower or "make: toyota" in response_lower
        remembers_customer = "john martinez" in response_lower or "john" in response_lower
        not_asking_customer_name = "customer name" not in response_lower
        context_preserved = (has_toyota or has_camry) and (remembers_customer or not_asking_customer_name)
        
        self.log_step(
            "3_vehicle_info",
            "Toyota Camry",
            result["response"],
            success and context_preserved,
            f"Response type: {response_type}, Context preserved: {context_preserved}"
        )
        
        if not (success and context_preserved):
            missing_info = self.extract_missing_info(result["response"])
            if not context_preserved:
                self.errors_found.append("CRITICAL BUG: Context not preserved - system forgot previous information")
            else:
                self.errors_found.append(f"Step 3 failed: Still missing {missing_info}, response type: {response_type}")
            print(f"❌ Step 3 Error Details:")
            print(f"   - API Success: {result['success']}")
            print(f"   - Response Type: {response_type}")
            print(f"   - Context Preserved: {context_preserved}")
            print(f"   - Missing Info: {missing_info}")
            print(f"   - Response: {result['response'][:200]}...")
            if 'error' in result:
                print(f"   - Error: {result['error']}")
        
        return success and context_preserved
    
    async def step_4_provide_contact_info(self) -> bool:
        """Step 4: Provide contact information if needed."""
        print("\n📞 STEP 4: Provide Contact Information")
        
        # Check if the last response asked for contact info
        if self.test_results:
            last_result = self.test_results[-1]
            if "contact info" in last_result["response"].lower():
                # Provide contact information
                result = self.send_message(
                    "john.martinez@email.com, (555) 123-4567",
                    expected_keywords=["quotation ready", "approve", "generation"]
                )
                
                response_type = self.analyze_response_type(result["response"])
                success = result["success"] and response_type in ["approval_request", "quotation_generated"]
                
                self.log_step(
                    "4_contact_info",
                    "john.martinez@email.com, (555) 123-4567",
                    result["response"],
                    success,
                    f"Response type: {response_type}"
                )
                
                if not success:
                    print(f"❌ Step 4 Error Details:")
                    print(f"   - API Success: {result['success']}")
                    print(f"   - Response Type: {response_type}")
                    print(f"   - Response: {result['response'][:200]}...")
                    if 'error' in result:
                        print(f"   - Error: {result['error']}")
                
                return success
        
        # If no contact info needed, consider it successful
        return True
    
    async def step_5_final_approval(self) -> bool:
        """Step 5: Provide final approval."""
        print("\n✅ STEP 5: Final Approval")
        
        result = self.send_message(
            "Approve",
            expected_keywords=["pdf", "generated", "quotation", "link"]
        )
        
        response_type = self.analyze_response_type(result["response"])
        success = result["success"] and response_type == "quotation_generated"
        
        self.log_step(
            "5_final_approval",
            "Approve",
            result["response"],
            success,
            f"Response type: {response_type}"
        )
        
        if not success:
            if response_type == "vehicle_info_request":
                self.errors_found.append("CRITICAL BUG: System reverted to asking for vehicle info after approval")
            else:
                self.errors_found.append(f"Step 5 failed: Expected quotation generation, got {response_type}")
            print(f"❌ Step 5 Error Details:")
            print(f"   - API Success: {result['success']}")
            print(f"   - Response Type: {response_type}")
            print(f"   - Response: {result['response'][:200]}...")
            if 'error' in result:
                print(f"   - Error: {result['error']}")
        
        return success
    
    async def step_6_verify_quotation(self):
        """Step 6: Verify quotation was generated with link."""
        print("\n📄 STEP 6: Verify Quotation Generation")
        
        # Check if the last response contains a PDF link
        if self.test_results:
            last_result = self.test_results[-1]
            response = last_result["response"]
            
            has_pdf_link = "pdf" in response.lower() and ("http" in response.lower() or "link" in response.lower())
            has_quotation_number = "quotation" in response.lower() and ("number" in response.lower() or "q" in response.lower())
            
            success = has_pdf_link or has_quotation_number
            
            self.log_step(
                "6_verify_quotation",
                "Verification check",
                f"PDF link found: {has_pdf_link}, Quotation number found: {has_quotation_number}",
                success,
                "Final verification of quotation generation"
            )
            
            if not success:
                self.errors_found.append("Step 6 failed: No PDF link or quotation number found in final response")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        total_steps = len(self.test_results)
        successful_steps = sum(1 for result in self.test_results if result["success"])
        
        report = {
            "test_summary": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "success_rate": f"{(successful_steps/total_steps)*100:.1f}%" if total_steps > 0 else "0%",
                "overall_success": len(self.errors_found) == 0,
                "conversation_id": self.conversation_id,
                "test_duration": "N/A",  # Could be calculated
                "timestamp": datetime.now().isoformat()
            },
            "errors_found": self.errors_found,
            "step_results": self.test_results,
            "conversation_log": self.conversation_log,
            "recommendations": self.generate_recommendations()
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 FINAL TEST REPORT")
        print(f"{'='*80}")
        print(f"Total Steps: {total_steps}")
        print(f"Successful Steps: {successful_steps}")
        print(f"Success Rate: {report['test_summary']['success_rate']}")
        print(f"Overall Success: {'✅ PASS' if report['test_summary']['overall_success'] else '❌ FAIL'}")
        print(f"Conversation ID: {self.conversation_id}")
        
        if self.errors_found:
            print(f"\n❌ ERRORS FOUND ({len(self.errors_found)}):")
            for i, error in enumerate(self.errors_found, 1):
                print(f"  {i}. {error}")
        
        if report["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print(f"{'='*80}")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze errors and provide specific recommendations
        for error in self.errors_found:
            if "not processing Toyota Camry" in error:
                recommendations.append("Fix HITL parameter passing - user_response not being processed correctly")
            elif "reverted to asking for vehicle info" in error:
                recommendations.append("Fix context preservation - system losing previous HITL cycle information")
            elif "missing info request" in error:
                recommendations.append("Check initial tool routing - may not be calling generate_quotation correctly")
        
        # Add general recommendations
        if len(self.errors_found) > 0:
            recommendations.append("Review Docker logs for detailed error analysis")
            recommendations.append("Test individual HITL cycles in isolation")
            recommendations.append("Verify deterministic boolean logic is working correctly")
        
        return list(set(recommendations))  # Remove duplicates

# Test execution
async def main():
    """Run the complete end-to-end test."""
    test = QuotationE2ETest()
    
    try:
        report = await test.run_complete_test()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quotation_e2e_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {filename}")
        
        return report
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main())
