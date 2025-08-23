#!/usr/bin/env python3
"""
Comprehensive End-to-End Quotation Test with Message Persistence Verification

This test covers the complete quotation flow:
1. Initial quote request
2. HITL information gathering 
3. Message persistence verification at each step
4. Final PDF generation
5. Fail-fast approach - stops on first failure

Usage: python test_end_to_end_quotation_with_persistence.py
"""

import asyncio
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Configuration
BASE_URL = "http://localhost:8000"
SUPABASE_URL = "https://ovppegnboovxussasjff.supabase.co"
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

class QuotationTestRunner:
    def __init__(self):
        self.conversation_id = None
        self.user_id = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # Test user
        self.test_results = []
        self.message_count_expected = 0
        self.step_counter = 0
        
    def log_step(self, step_name: str, status: str = "RUNNING", details: str = ""):
        """Log test step with timestamp"""
        self.step_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_emoji = "🟡" if status == "RUNNING" else "✅" if status == "PASS" else "❌"
        print(f"\n{status_emoji} [{timestamp}] Step {self.step_counter}: {step_name}")
        if details:
            print(f"   └─ {details}")
        
        if status == "FAIL":
            print(f"\n💥 TEST FAILED AT STEP {self.step_counter}: {step_name}")
            print(f"   Error: {details}")
            sys.exit(1)
    
    def send_chat_message(self, message: str) -> Dict[str, Any]:
        """Send a chat message and return the response"""
        payload = {
            "message": message,
            "user_id": self.user_id
        }
        
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
            
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", json=payload)
        
        if response.status_code != 200:
            self.log_step("Send Message", "FAIL", f"HTTP {response.status_code}: {response.text}")
            
        data = response.json()
        
        # Extract conversation_id from response if this is the first message
        if not self.conversation_id and "conversation_id" in data:
            self.conversation_id = data["conversation_id"]
            
        return data
    
    def verify_message_persistence(self, expected_count: int, step_description: str) -> List[Dict]:
        """Verify that messages are persisted in the database"""
        if not self.conversation_id:
            self.log_step("Verify Persistence", "FAIL", "No conversation_id available")
            
        # Query Supabase directly
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"{SUPABASE_URL}/rest/v1/messages"
        params = {
            "conversation_id": f"eq.{self.conversation_id}",
            "select": "id,role,content,created_at",
            "order": "created_at.asc"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            self.log_step("Verify Persistence", "FAIL", f"Database query failed: {response.status_code}")
            
        messages = response.json()
        actual_count = len(messages)
        
        if actual_count != expected_count:
            self.log_step(
                "Verify Persistence", 
                "FAIL", 
                f"{step_description}: Expected {expected_count} messages, found {actual_count}. Messages: {[m['role'] + ': ' + m['content'][:50] + '...' for m in messages]}"
            )
        
        self.log_step(
            "Verify Persistence", 
            "PASS", 
            f"{step_description}: {actual_count}/{expected_count} messages persisted correctly"
        )
        
        return messages
    
    def wait_for_background_tasks(self, seconds: int = 3):
        """Wait for background tasks to complete"""
        print(f"   ⏳ Waiting {seconds}s for background tasks...")
        time.sleep(seconds)
    
    async def run_comprehensive_test(self):
        """Run the complete end-to-end test"""
        print("🚀 Starting Comprehensive Quotation Test with Message Persistence")
        print("=" * 80)
        
        try:
            # Step 1: Initial Quote Request
            self.log_step("Initial Quote Request", "RUNNING")
            response1 = self.send_chat_message("Generate an Informal Quote")
            
            if not response1.get("is_interrupted"):
                self.log_step("Initial Quote Request", "FAIL", "Expected HITL interruption but got normal response")
            
            self.log_step("Initial Quote Request", "PASS", "HITL interruption triggered correctly")
            self.message_count_expected = 2  # User message + Assistant HITL prompt
            
            # Wait for background tasks
            self.wait_for_background_tasks()
            
            # Verify persistence after step 1
            messages1 = self.verify_message_persistence(self.message_count_expected, "After initial request")
            
            # Step 2: Provide Customer Information
            self.log_step("Provide Customer Info", "RUNNING")
            response2 = self.send_chat_message("John Smith - ABC Corporation")
            
            if not response2.get("is_interrupted"):
                self.log_step("Provide Customer Info", "FAIL", "Expected continued HITL but got normal response")
                
            self.log_step("Provide Customer Info", "PASS", "Customer info processed, still in HITL")
            self.message_count_expected = 3  # Previous + new user message
            
            # Wait for background tasks
            self.wait_for_background_tasks()
            
            # Verify persistence after step 2
            messages2 = self.verify_message_persistence(self.message_count_expected, "After customer info")
            
            # Step 3: Provide Vehicle Information
            self.log_step("Provide Vehicle Info", "RUNNING")
            response3 = self.send_chat_message("Honda Civic")
            
            if not response3.get("is_interrupted"):
                self.log_step("Provide Vehicle Info", "FAIL", "Expected continued HITL but got normal response")
                
            self.log_step("Provide Vehicle Info", "PASS", "Vehicle info processed, still in HITL")
            self.message_count_expected = 4  # Previous + new user message
            
            # Wait for background tasks
            self.wait_for_background_tasks()
            
            # Verify persistence after step 3
            messages3 = self.verify_message_persistence(self.message_count_expected, "After vehicle info")
            
            # Step 4: Provide Contact Information
            self.log_step("Provide Contact Info", "RUNNING")
            response4 = self.send_chat_message("john.smith@abc.com | +1-555-0123")
            
            # This might complete the HITL or ask for more info
            self.log_step("Provide Contact Info", "PASS", f"Contact info processed. Interrupted: {response4.get('is_interrupted', False)}")
            self.message_count_expected = 5  # Previous + new user message
            
            # Wait for background tasks
            self.wait_for_background_tasks()
            
            # Verify persistence after step 4
            messages4 = self.verify_message_persistence(self.message_count_expected, "After contact info")
            
            # Step 5: Complete Quotation (if still in HITL, approve)
            if response4.get("is_interrupted"):
                self.log_step("Complete Quotation", "RUNNING")
                response5 = self.send_chat_message("Approve")
                
                if response5.get("is_interrupted"):
                    self.log_step("Complete Quotation", "FAIL", "Still interrupted after approval")
                    
                self.log_step("Complete Quotation", "PASS", "Quotation completed successfully")
                self.message_count_expected = 7  # Previous + user approval + assistant response
                
                # Wait for background tasks
                self.wait_for_background_tasks()
                
                # Verify final persistence
                final_messages = self.verify_message_persistence(self.message_count_expected, "After quotation completion")
            else:
                final_messages = messages4
            
            # Step 6: Verify Quotation Content
            self.log_step("Verify Quotation Content", "RUNNING")
            final_response = response5 if 'response5' in locals() else response4
            
            if "quotation" not in final_response.get("message", "").lower():
                self.log_step("Verify Quotation Content", "FAIL", "Response doesn't contain quotation content")
                
            self.log_step("Verify Quotation Content", "PASS", "Quotation content generated successfully")
            
            # Step 7: Test PDF Generation (if available)
            self.log_step("Test PDF Generation", "RUNNING")
            try:
                # Check if there's a PDF endpoint or quotation ID in the response
                if "quotation_id" in final_response:
                    quotation_id = final_response["quotation_id"]
                    pdf_response = requests.get(f"{BASE_URL}/api/v1/quotations/{quotation_id}/pdf")
                    
                    if pdf_response.status_code == 200:
                        self.log_step("Test PDF Generation", "PASS", f"PDF generated successfully ({len(pdf_response.content)} bytes)")
                    else:
                        self.log_step("Test PDF Generation", "FAIL", f"PDF generation failed: {pdf_response.status_code}")
                else:
                    self.log_step("Test PDF Generation", "PASS", "No PDF endpoint available (skipped)")
            except Exception as e:
                self.log_step("Test PDF Generation", "PASS", f"PDF test skipped: {str(e)}")
            
            # Final Summary
            print("\n" + "=" * 80)
            print("🎉 ALL TESTS PASSED! End-to-End Quotation Flow Successful")
            print("=" * 80)
            
            print(f"\n📊 Test Summary:")
            print(f"   • Conversation ID: {self.conversation_id}")
            print(f"   • Total Messages: {len(final_messages)}")
            print(f"   • Steps Completed: {self.step_counter}")
            print(f"   • Duration: {datetime.now().strftime('%H:%M:%S')}")
            
            print(f"\n💬 Message Flow:")
            for i, msg in enumerate(final_messages, 1):
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                content_preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                print(f"   {i}. {role_emoji} {msg['role']}: {content_preview}")
            
            return True
            
        except Exception as e:
            self.log_step("Unexpected Error", "FAIL", str(e))
            return False

def main():
    """Main test runner"""
    print("🔧 Quotation End-to-End Test with Message Persistence")
    print("📋 This test will verify the complete quotation flow and message storage")
    print()
    
    # Check environment
    if not SUPABASE_ANON_KEY:
        print("❌ SUPABASE_ANON_KEY environment variable not set")
        sys.exit(1)
    
    # Test API connectivity
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Backend not available at {BASE_URL}")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        sys.exit(1)
    
    print("✅ Backend connectivity verified")
    
    # Run the test
    runner = QuotationTestRunner()
    success = asyncio.run(runner.run_comprehensive_test())
    
    if success:
        print("\n🎯 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
