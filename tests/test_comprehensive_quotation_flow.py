#!/usr/bin/env python3
"""
Comprehensive End-to-End Quotation Generation Flow Test

This test tracks the complete quotation generation process from initial request
to PDF generation, with detailed logging and database monitoring.
"""

import requests
import json
import time
import uuid
from datetime import datetime, timedelta
import subprocess
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/message"
EMPLOYEE_USER_ID = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # John Smith

class ComprehensiveQuotationFlowTester:
    """Comprehensive quotation flow tester with detailed monitoring."""
    
    def __init__(self):
        self.conversation_id = f"comprehensive-test-{int(time.time())}"
        self.start_time = datetime.now()
        self.test_results = {
            "start_time": self.start_time.isoformat(),
            "conversation_id": self.conversation_id,
            "steps": [],
            "errors": [],
            "quotation_created": False,
            "pdf_generated": False,
            "pdf_url": None
        }
        
    def log_step(self, step_name, details):
        """Log a test step with timestamp."""
        step = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results["steps"].append(step)
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"TIME: {step['timestamp']}")
        print(f"DETAILS: {details}")
        print('='*60)
        
    def get_docker_logs(self, lines=50):
        """Get recent Docker logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), "agent-tobi-rag-backend-1"],
                capture_output=True,
                text=True
            )
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
        except Exception as e:
            return f"Error getting logs: {str(e)}"
    
    def send_message(self, message, step_name):
        """Send a message to the API and log the response."""
        self.log_step(step_name, f"Sending message: {message[:100]}...")
        
        payload = {
            'message': message,
            'conversation_id': self.conversation_id,
            'user_id': EMPLOYEE_USER_ID,
            'include_sources': True
        }
        
        try:
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
            result = response.json()
            
            step_result = {
                "status_code": response.status_code,
                "response_preview": result.get('message', '')[:300],
                "is_interrupted": result.get('is_interrupted', False),
                "full_response": result
            }
            
            self.test_results["steps"][-1]["api_response"] = step_result
            
            print(f"Status: {response.status_code}")
            print(f"Interrupted: {result.get('is_interrupted', False)}")
            print(f"Response Preview: {result.get('message', '')[:200]}...")
            
            # Check for quotation-related content
            message_content = result.get('message', '').lower()
            if 'quotation' in message_content and ('generated' in message_content or 'created' in message_content):
                self.test_results["quotation_created"] = True
                print("🎉 QUOTATION CREATION DETECTED!")
                
            # Check for PDF links
            if 'http' in result.get('message', ''):
                import re
                urls = re.findall(r'http[s]?://[^\s]+', result.get('message', ''))
                if urls:
                    self.test_results["pdf_generated"] = True
                    self.test_results["pdf_url"] = urls[0]
                    print(f"🎉 PDF URL FOUND: {urls[0]}")
            
            return result
            
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            self.test_results["errors"].append({
                "step": step_name,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            print(f"❌ ERROR: {error_msg}")
            return None
    
    def check_docker_logs_for_quotation(self):
        """Check Docker logs for quotation-related activity."""
        self.log_step("Docker Log Analysis", "Checking for quotation generation activity")
        
        logs = self.get_docker_logs(100)
        
        # Look for quotation-related log entries
        quotation_keywords = [
            "quotation", "pdf", "generate_quotation", "storage", 
            "intelligent quotation", "completeness", "pricing"
        ]
        
        relevant_logs = []
        for line in logs.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in quotation_keywords):
                relevant_logs.append(line)
        
        self.test_results["steps"][-1]["docker_logs"] = relevant_logs
        
        print("Recent quotation-related logs:")
        for log in relevant_logs[-10:]:  # Show last 10 relevant logs
            print(f"  {log}")
            
        return relevant_logs
    
    def run_comprehensive_test(self):
        """Run the complete quotation generation test."""
        print("🚀 Starting Comprehensive Quotation Generation Test")
        print(f"Conversation ID: {self.conversation_id}")
        print(f"Employee User ID: {EMPLOYEE_USER_ID}")
        
        # Step 1: Initial quotation request
        initial_message = """Hi! I need to generate a quotation for our customer Robert Brown. Here are the details:

Customer Information:
- Name: Robert Brown
- Email: robert.brown@email.com  
- Phone: +63 912 345 6789
- Company: Personal Purchase

Vehicle Requirements:
- Toyota Camry 2024 sedan
- Silver color preferred
- Standard package
- 30 days validity period

Please generate a complete quotation with PDF document."""

        response1 = self.send_message(initial_message, "Initial Quotation Request")
        
        if not response1:
            return self.test_results
            
        # Check Docker logs after initial request
        self.check_docker_logs_for_quotation()
        
        # Step 2: Handle any HITL interruption
        if response1.get('is_interrupted', False):
            self.log_step("HITL Detected", "System is requesting human approval")
            
            # Approve the action
            approval_response = self.send_message("Yes, go ahead and generate the quotation", "HITL Approval")
            
            if approval_response:
                self.check_docker_logs_for_quotation()
        
        # Step 3: Follow up if no PDF link yet
        if not self.test_results["pdf_generated"]:
            self.log_step("Follow-up Request", "Requesting PDF link if not provided")
            
            follow_up = self.send_message(
                "Can you provide the PDF download link for the quotation?", 
                "PDF Link Request"
            )
            
            if follow_up:
                self.check_docker_logs_for_quotation()
        
        # Step 4: Final status check
        self.log_step("Final Status Check", "Checking final test results")
        
        print(f"\n{'='*80}")
        print("FINAL TEST RESULTS")
        print('='*80)
        print(f"Quotation Created: {self.test_results['quotation_created']}")
        print(f"PDF Generated: {self.test_results['pdf_generated']}")
        print(f"PDF URL: {self.test_results.get('pdf_url', 'None')}")
        print(f"Total Steps: {len(self.test_results['steps'])}")
        print(f"Errors: {len(self.test_results['errors'])}")
        
        if self.test_results["errors"]:
            print("\nERRORS ENCOUNTERED:")
            for error in self.test_results["errors"]:
                print(f"  {error['step']}: {error['error']}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_quotation_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
        return self.test_results

def main():
    """Run the comprehensive quotation flow test."""
    tester = ComprehensiveQuotationFlowTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if results["quotation_created"] and results["pdf_generated"]:
        print("\n🎉 SUCCESS: Complete quotation generation flow working!")
        sys.exit(0)
    elif results["quotation_created"]:
        print("\n⚠️  PARTIAL SUCCESS: Quotation created but PDF generation incomplete")
        sys.exit(1)
    else:
        print("\n❌ FAILURE: Quotation generation not working")
        sys.exit(2)

if __name__ == "__main__":
    main()
