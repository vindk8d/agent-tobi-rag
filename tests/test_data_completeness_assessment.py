#!/usr/bin/env python3
"""
Data Completeness Assessment End-to-End Test

Tests the agent's ability to detect complete data and skip HITL when all required
information is provided upfront by an employee for a customer quotation.

Expected Behavior:
- Employee provides complete customer and vehicle information
- Agent detects data completeness (no HITL triggered)
- Quotation is generated directly
- PDF is created and uploaded
- Confirmation message with link is returned
"""

import requests
import json
import time
import uuid
from datetime import datetime
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Configuration
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/message"
EMPLOYEE_USER_ID = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # John Smith (employee)

class DataCompletenessTestRunner:
    """Test runner for data completeness assessment."""
    
    def __init__(self):
        self.conversation_id = f"completeness-test-{int(time.time())}"
        self.start_time = datetime.now()
        self.test_results = {
            "test_name": "Data Completeness Assessment E2E",
            "start_time": self.start_time.isoformat(),
            "conversation_id": self.conversation_id,
            "steps": [],
            "errors": [],
            "hitl_triggered": False,
            "quotation_created": False,
            "pdf_generated": False,
            "pdf_url": None,
            "success": False
        }
        
    def log_step(self, step_name, details, success=True):
        """Log a test step with timestamp."""
        step = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "success": success
        }
        self.test_results["steps"].append(step)
        
        status_icon = "✅" if success else "❌"
        print(f"\n{status_icon} STEP: {step_name}")
        print(f"⏰ TIME: {step['timestamp']}")
        print(f"📋 DETAILS: {details}")
        
        if not success:
            self.test_results["errors"].append({
                "step": step_name,
                "details": details,
                "timestamp": step['timestamp']
            })
    
    def send_message(self, message, expect_hitl=False):
        """Send a message to the chat API and return the response."""
        payload = {
            "message": message,
            "conversation_id": self.conversation_id,
            "user_id": EMPLOYEE_USER_ID
        }
        
        try:
            self.log_step("API_REQUEST", f"Sending message: {message[:100]}...")
            
            response = requests.post(
                CHAT_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minute timeout for complex operations
            )
            
            if response.status_code != 200:
                self.log_step("API_ERROR", f"HTTP {response.status_code}: {response.text}", success=False)
                return None
            
            response_data = response.json()
            self.log_step("API_RESPONSE", f"Response received (length: {len(str(response_data))})")
            
            # Check for HITL indicators
            response_text = response_data.get("response", "").lower()
            
            # HITL detection patterns
            hitl_patterns = [
                "please provide",
                "need more information",
                "missing information",
                "could you clarify",
                "additional details needed",
                "to complete",
                "please specify"
            ]
            
            hitl_detected = any(pattern in response_text for pattern in hitl_patterns)
            
            if hitl_detected and not expect_hitl:
                self.test_results["hitl_triggered"] = True
                self.log_step("UNEXPECTED_HITL", f"HITL triggered when not expected: {response_text[:200]}...", success=False)
            elif not hitl_detected and expect_hitl:
                self.log_step("MISSING_HITL", f"HITL not triggered when expected: {response_text[:200]}...", success=False)
            elif hitl_detected and expect_hitl:
                self.test_results["hitl_triggered"] = True
                self.log_step("EXPECTED_HITL", "HITL triggered as expected")
            else:
                self.log_step("NO_HITL", "No HITL triggered as expected")
            
            # Check for quotation creation
            if "quotation generated" in response_text or "pdf" in response_text:
                self.test_results["quotation_created"] = True
                self.log_step("QUOTATION_CREATED", "Quotation creation detected in response")
            
            # Check for PDF generation
            if "pdf" in response_text and ("http" in response_text or "link" in response_text):
                self.test_results["pdf_generated"] = True
                # Extract PDF URL if present
                import re
                url_pattern = r'https?://[^\s)]+\.pdf[^\s)]*'
                pdf_urls = re.findall(url_pattern, response_text)
                if pdf_urls:
                    self.test_results["pdf_url"] = pdf_urls[0]
                    self.log_step("PDF_URL_FOUND", f"PDF URL extracted: {pdf_urls[0]}")
            
            return response_data
            
        except requests.exceptions.Timeout:
            self.log_step("TIMEOUT_ERROR", "Request timed out after 120 seconds", success=False)
            return None
        except Exception as e:
            self.log_step("REQUEST_ERROR", f"Error sending request: {str(e)}", success=False)
            return None
    
    def run_complete_data_test(self):
        """Run the main test with complete data provided upfront."""
        
        print("\n" + "="*80)
        print("🧪 DATA COMPLETENESS ASSESSMENT TEST")
        print("="*80)
        print(f"📊 Conversation ID: {self.conversation_id}")
        print(f"👤 Employee ID: {EMPLOYEE_USER_ID}")
        print(f"🎯 Goal: Test complete data detection (no HITL expected)")
        print("="*80)
        
        # STEP 1: Send complete quotation request with all required data
        complete_request = """
        Hi Tobi, I need to generate a quotation for one of our customers with the following complete information:

        **Customer Information:**
        - Name: Maria Santos
        - Email: maria.santos@email.com
        - Phone: +63-917-123-4567
        - Company: Santos Trading Corp

        **Vehicle Requirements:**
        - Make: Toyota
        - Model: Camry
        - Year: 2024
        - Type: Sedan
        - Color: Pearl White
        - Quantity: 1 unit

        **Purchase Details:**
        - Budget Range: PHP 1,800,000 - PHP 2,200,000
        - Financing: Bank loan preferred
        - Trade-in: No trade-in
        - Payment Method: Financing + down payment

        **Timeline & Delivery:**
        - Urgency: Standard (within 2 weeks)
        - Decision Timeline: This week
        - Delivery Preference: Pickup from dealership
        - Delivery Timeline: End of month

        Please generate the quotation immediately since all information is complete.
        """
        
        self.log_step("SENDING_COMPLETE_REQUEST", "Sending request with all required data")
        
        response = self.send_message(complete_request, expect_hitl=False)
        
        if not response:
            self.log_step("TEST_FAILED", "Failed to get response from API", success=False)
            return self.finalize_test()
        
        # Analyze the response
        response_text = response.get("response", "")
        
        # Check if the response indicates successful completion
        success_indicators = [
            "quotation generated successfully",
            "quotation has been created",
            "pdf generated",
            "quotation is ready",
            "successfully created"
        ]
        
        success_found = any(indicator in response_text.lower() for indicator in success_indicators)
        
        if success_found:
            self.log_step("SUCCESS_DETECTED", "Success indicators found in response")
            self.test_results["success"] = True
        else:
            self.log_step("NO_SUCCESS_INDICATOR", f"No clear success indicator found. Response: {response_text[:300]}...", success=False)
        
        # Additional checks for completeness assessment
        if not self.test_results["hitl_triggered"]:
            self.log_step("COMPLETENESS_ASSESSMENT_PASSED", "✅ Data completeness correctly detected - no HITL triggered")
        else:
            self.log_step("COMPLETENESS_ASSESSMENT_FAILED", "❌ Data completeness not detected - HITL was triggered", success=False)
        
        return self.finalize_test()
    
    def finalize_test(self):
        """Finalize test results and generate report."""
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        # Overall success criteria
        overall_success = (
            not self.test_results["hitl_triggered"] and  # No HITL should be triggered
            self.test_results["quotation_created"] and   # Quotation should be created
            len(self.test_results["errors"]) == 0        # No errors should occur
        )
        
        self.test_results["overall_success"] = overall_success
        
        # Generate test report
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"🎯 Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"⏱️  Duration: {self.test_results['duration_seconds']:.2f} seconds")
        print(f"🔄 HITL Triggered: {'❌ Yes (unexpected)' if self.test_results['hitl_triggered'] else '✅ No (expected)'}")
        print(f"📄 Quotation Created: {'✅ Yes' if self.test_results['quotation_created'] else '❌ No'}")
        print(f"📎 PDF Generated: {'✅ Yes' if self.test_results['pdf_generated'] else '❌ No'}")
        
        if self.test_results["pdf_url"]:
            print(f"🔗 PDF URL: {self.test_results['pdf_url']}")
        
        print(f"❌ Errors: {len(self.test_results['errors'])}")
        
        if self.test_results["errors"]:
            print("\n🚨 ERROR DETAILS:")
            for error in self.test_results["errors"]:
                print(f"  • {error['step']}: {error['details']}")
        
        print("\n📋 STEP SUMMARY:")
        for step in self.test_results["steps"]:
            status = "✅" if step.get("success", True) else "❌"
            print(f"  {status} {step['step']}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data_completeness_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results file: {e}")
        
        return overall_success

def main():
    """Main test execution function."""
    print("🚀 Starting Data Completeness Assessment Test...")
    
    # Check if backend is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Backend health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}: {e}")
        print("💡 Make sure Docker containers are running: docker-compose up -d")
        return False
    
    print("✅ Backend is running")
    
    # Run the test
    tester = DataCompletenessTestRunner()
    success = tester.run_complete_data_test()
    
    if success:
        print("\n🎉 DATA COMPLETENESS TEST PASSED!")
        print("✅ Agent correctly detected complete data and generated quotation without HITL")
    else:
        print("\n💥 DATA COMPLETENESS TEST FAILED!")
        print("❌ Check the errors above and Docker logs for more details")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

