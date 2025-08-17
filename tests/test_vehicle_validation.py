#!/usr/bin/env python3
"""
Test Vehicle Validation in Quotation Generation

This test verifies that the system properly validates vehicles against the database
and prevents quotations for non-existent vehicles.
"""

import requests
import time
from datetime import datetime


class VehicleValidationTest:
    """Test vehicle validation in quotation generation."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        # Use a valid employee for testing
        self.employee_user_id = "54394d40-ad35-4b5b-a392-1ae7c9329d11"  # Alex Thompson
    
    def send_message(self, message: str, conversation_id: str = None, timeout: int = 30) -> dict:
        """Send a message to the chat API."""
        try:
            payload = {
                "message": message,
                "user_id": self.employee_user_id
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
                    "conversation_id": data.get("conversation_id", "")
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_invalid_vehicle_blocked(self) -> bool:
        """Test that invalid vehicles are blocked from quotation generation."""
        print("🧪 Testing Invalid Vehicle Validation")
        print("=" * 60)
        
        try:
            # Step 1: Start quotation
            print("📝 Step 1: Initial quotation request")
            result = self.send_message("generate a quotation")
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            
            conversation_id = result["conversation_id"]
            print(f"✅ Started quotation (ID: {conversation_id})")
            
            # Step 2: Provide customer name
            print("\n👤 Step 2: Provide customer name")
            result = self.send_message("John Test Customer", conversation_id)
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            print("✅ Customer name provided")
            
            # Step 3: Provide INVALID vehicle (Jaguar Sedan - not in database)
            print("\n🚗 Step 3: Provide INVALID vehicle (Jaguar Sedan)")
            result = self.send_message("Jaguar Sedan", conversation_id)
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            
            # Check if system properly blocked the invalid vehicle
            response = result["response"].lower()
            
            if "not in our current inventory" in response or "available vehicle options" in response:
                print("✅ SUCCESS: System correctly blocked invalid vehicle!")
                print(f"📄 Response: {result['response'][:200]}...")
                return True
            elif "quotation ready" in response or "approve" in response:
                print("❌ FAILURE: System allowed invalid vehicle to proceed to quotation!")
                print(f"📄 Response: {result['response'][:200]}...")
                return False
            else:
                print("⚠️  UNCLEAR: System response unclear - manual review needed")
                print(f"📄 Response: {result['response'][:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def test_valid_vehicle_allowed(self) -> bool:
        """Test that valid vehicles are allowed for quotation generation."""
        print("\n🧪 Testing Valid Vehicle Acceptance")
        print("=" * 60)
        
        try:
            # Step 1: Start quotation
            print("📝 Step 1: Initial quotation request")
            result = self.send_message("generate a quotation")
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            
            conversation_id = result["conversation_id"]
            print(f"✅ Started quotation (ID: {conversation_id})")
            
            # Step 2: Provide customer name
            print("\n👤 Step 2: Provide customer name")
            result = self.send_message("Jane Valid Customer", conversation_id)
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            print("✅ Customer name provided")
            
            # Step 3: Provide VALID vehicle (Toyota Camry - exists in database)
            print("\n🚗 Step 3: Provide VALID vehicle (Toyota Camry)")
            result = self.send_message("Toyota Camry", conversation_id)
            if not result["success"]:
                print(f"❌ Failed: {result['error']}")
                return False
            
            # Check if system allows the valid vehicle to proceed
            response = result["response"].lower()
            
            if "contact info" in response or "email" in response or "phone" in response:
                print("✅ SUCCESS: System correctly accepted valid vehicle and requested contact info!")
                print(f"📄 Response: {result['response'][:200]}...")
                return True
            elif "not in our current inventory" in response:
                print("❌ FAILURE: System incorrectly blocked valid vehicle!")
                print(f"📄 Response: {result['response'][:200]}...")
                return False
            else:
                print("⚠️  UNCLEAR: System response unclear - manual review needed")
                print(f"📄 Response: {result['response'][:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def run_all_tests(self):
        """Run all vehicle validation tests."""
        print("🚀 Vehicle Validation Test Suite")
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Employee: Alex Thompson (ID: {self.employee_user_id})")
        print("=" * 80)
        
        results = []
        
        # Test 1: Invalid vehicle should be blocked
        test1_result = self.test_invalid_vehicle_blocked()
        results.append(("Invalid Vehicle Blocked", test1_result))
        
        # Small delay between tests
        time.sleep(2)
        
        # Test 2: Valid vehicle should be allowed
        test2_result = self.test_valid_vehicle_allowed()
        results.append(("Valid Vehicle Allowed", test2_result))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 VEHICLE VALIDATION TEST RESULTS")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  • {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Vehicle validation is working correctly!")
        else:
            print("⚠️  SOME TESTS FAILED - Vehicle validation needs attention!")
        
        return passed == total


if __name__ == "__main__":
    tester = VehicleValidationTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)
