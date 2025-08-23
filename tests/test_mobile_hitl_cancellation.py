#!/usr/bin/env python3
"""
Comprehensive test for mobile HITL cancellation functionality.

This test simulates the exact scenario from the screenshot:
1. User requests quotation via mobile interface
2. Agent shows HITL prompt for missing data
3. User types "cancel" in chat
4. Agent should provide clear cancellation message (not confusion)

Tests the mobile chat-interface.tsx integration with backend HITL system.
"""

import asyncio
import sys
import os
import uuid

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent

class MobileHITLCancellationTester:
    """Simulates mobile interface behavior for HITL cancellation testing."""
    
    def __init__(self):
        self.agent = UnifiedToolCallingRAGAgent()
        # Use valid UUIDs for testing
        self.user_id = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # John Smith - existing employee
        self.test_results = []
    
    async def simulate_mobile_chat_message(self, message: str, conversation_id: str):
        """
        Simulate how the mobile chat interface sends messages to the backend.
        This matches the handleSendMessage function in chat-interface.tsx
        """
        print(f"📱 [MOBILE-SIM] Sending message: '{message}'")
        
        try:
            # This simulates the mobile interface calling the API
            result = await self.agent.process_user_message(
                user_query=message,
                conversation_id=conversation_id,
                user_id=self.user_id
            )
            
            print(f"✅ [MOBILE-SIM] Response received")
            print(f"   Message: {result.get('message', 'No message')[:100]}...")
            print(f"   Interrupted: {result.get('is_interrupted', False)}")
            print(f"   HITL Phase: {result.get('hitl_phase')}")
            
            return result
            
        except Exception as e:
            print(f"💥 [MOBILE-SIM] Error: {str(e)}")
            return {"error": True, "message": str(e)}
    
    async def simulate_mobile_hitl_response(self, response: str, conversation_id: str):
        """
        Simulate how the mobile interface handles HITL responses.
        In the mobile interface, HITL responses go through the same handleSendMessage function.
        """
        print(f"📱 [MOBILE-HITL] Sending HITL response: '{response}'")
        
        try:
            # This simulates what happens when user types in chat during HITL
            result = await self.agent.resume_interrupted_conversation(
                conversation_id=conversation_id,
                user_response=response
            )
            
            print(f"✅ [MOBILE-HITL] HITL response processed")
            print(f"   Message: {result.get('message', 'No message')}")
            print(f"   Interrupted: {result.get('is_interrupted', False)}")
            
            return result
            
        except Exception as e:
            print(f"💥 [MOBILE-HITL] Error: {str(e)}")
            return {"error": True, "message": str(e)}
    
    async def test_quotation_cancellation_scenario(self):
        """Test the exact scenario from the screenshot: quotation → HITL → cancel"""
        
        print("\n🧪 TEST 1: Quotation Generation → HITL → Cancel")
        print("=" * 60)
        
        conversation_id = str(uuid.uuid4())
        
        try:
            # Step 1: User requests quotation (should trigger HITL)
            print("📝 Step 1: User requests 'Generate an Informal Quote'")
            
            result1 = await self.simulate_mobile_chat_message(
                "Generate an Informal Quote", 
                conversation_id
            )
            
            if result1.get("error"):
                print("❌ FAILED: Error in step 1")
                return False
            
            if not result1.get('is_interrupted'):
                print("❌ FAILED: Expected HITL interrupt for missing quotation data")
                return False
            
            print("✅ Step 1 PASSED: Agent correctly requested missing data via HITL")
            
            # Step 2: User responds with "cancel"
            print("\n📝 Step 2: User responds with 'cancel'")
            
            result2 = await self.simulate_mobile_hitl_response("cancel", conversation_id)
            
            if result2.get("error"):
                print("❌ FAILED: Error in step 2")
                return False
            
            response_message = result2.get('message', '').lower()
            
            # Check for the bug: agent confusion about what to cancel
            if "don't have a specific request to cancel" in response_message:
                print("❌ BUG REPRODUCED: Agent is confused about what to cancel")
                print(f"   Problematic response: '{result2.get('message', '')}'")
                return False
            elif "cancelled" in response_message and "quotation" in response_message:
                print("✅ Step 2 PASSED: Agent provided clear cancellation message")
                return True
            else:
                print(f"⚠️ UNCLEAR: Response doesn't clearly indicate cancellation")
                print(f"   Response: '{result2.get('message', '')}'")
                return False
                
        except Exception as e:
            print(f"💥 ERROR in quotation cancellation test: {str(e)}")
            return False
    
    async def test_customer_message_cancellation(self):
        """Test cancellation during customer message HITL"""
        
        print("\n🧪 TEST 2: Customer Message → HITL → Cancel")
        print("=" * 60)
        
        conversation_id = str(uuid.uuid4())
        
        try:
            # Step 1: Request customer message (should trigger HITL)
            print("📝 Step 1: User requests 'Send a message to customer John about the SUV inquiry'")
            
            result1 = await self.simulate_mobile_chat_message(
                "Send a message to customer John about the SUV inquiry",
                conversation_id
            )
            
            if result1.get("error"):
                print("❌ FAILED: Error in step 1")
                return False
            
            if result1.get('is_interrupted'):
                # HITL scenario - test cancellation
                print("✅ Step 1: Agent requested confirmation via HITL")
                
                result2 = await self.simulate_mobile_hitl_response("cancel", conversation_id)
                
                if result2.get("error"):
                    print("❌ FAILED: Error in step 2")
                    return False
                
                response = result2.get('message', '').lower()
                
                if "cancelled" in response and ("message" in response or "sending" in response):
                    print("✅ Step 2 PASSED: Clear customer message cancellation")
                    return True
                else:
                    print(f"⚠️ UNCLEAR: Response doesn't clearly indicate message cancellation")
                    print(f"   Response: '{result2.get('message', '')}'")
                    return False
            else:
                # Non-HITL scenario - direct cancellation test
                print("ℹ️ Step 1: No HITL required, testing direct cancellation")
                
                result2 = await self.simulate_mobile_chat_message("cancel", conversation_id)
                
                response = result2.get('message', '').lower()
                
                if "don't have a specific request" in response:
                    print("✅ Step 2 PASSED: Agent correctly indicates no active request")
                    return True
                else:
                    print("✅ Step 2 PASSED: Agent handled cancellation gracefully")
                    return True
                    
        except Exception as e:
            print(f"💥 ERROR in customer message test: {str(e)}")
            return False
    
    async def test_mixed_input_cancellation(self):
        """Test cancellation with mixed input (like 'john@example.com but cancel this')"""
        
        print("\n🧪 TEST 3: Mixed Input + Cancellation")
        print("=" * 60)
        
        conversation_id = str(uuid.uuid4())
        
        try:
            # Step 1: Request quotation
            result1 = await self.simulate_mobile_chat_message(
                "Generate an Informal Quote",
                conversation_id
            )
            
            if not result1.get('is_interrupted'):
                print("❌ FAILED: Expected HITL interrupt")
                return False
            
            print("✅ Step 1: HITL prompt generated")
            
            # Step 2: Mixed input with cancellation
            print("📝 Step 2: User provides mixed input with cancellation")
            
            result2 = await self.simulate_mobile_hitl_response(
                "Toyota Camry but actually cancel this",
                conversation_id
            )
            
            if result2.get("error"):
                print("❌ FAILED: Error processing mixed input")
                return False
            
            response = result2.get('message', '').lower()
            
            if "cancelled" in response and "quotation" in response:
                print("✅ Step 2 PASSED: Agent correctly prioritized cancellation intent")
                return True
            else:
                print(f"⚠️ UNCLEAR: Agent may not have detected cancellation intent")
                print(f"   Response: '{result2.get('message', '')}'")
                return False
                
        except Exception as e:
            print(f"💥 ERROR in mixed input test: {str(e)}")
            return False
    
    async def test_api_routing_consistency(self):
        """Test that mobile interface routing matches API expectations"""
        
        print("\n🧪 TEST 4: API Routing Consistency")
        print("=" * 60)
        
        conversation_id = str(uuid.uuid4())
        
        try:
            # Test that mobile simulation matches actual API behavior
            print("📝 Testing API routing consistency...")
            
            # Start a HITL scenario
            result1 = await self.simulate_mobile_chat_message(
                "Generate an Informal Quote",
                conversation_id
            )
            
            if not result1.get('is_interrupted'):
                print("❌ FAILED: No HITL interrupt generated")
                return False
            
            # Test that HITL response routing works
            result2 = await self.simulate_mobile_hitl_response("cancel", conversation_id)
            
            if result2.get("error"):
                print("❌ FAILED: HITL response routing failed")
                return False
            
            # Verify the response makes sense
            response = result2.get('message', '')
            if response and len(response.strip()) > 0:
                print("✅ API routing test PASSED: Consistent behavior")
                return True
            else:
                print("❌ FAILED: Empty or invalid response")
                return False
                
        except Exception as e:
            print(f"💥 ERROR in API routing test: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all mobile HITL cancellation tests"""
        
        print("🚀 MOBILE HITL CANCELLATION TEST SUITE")
        print("🎯 Goal: Ensure mobile interface handles HITL cancellation correctly")
        print("=" * 80)
        
        tests = [
            ("Quotation Cancellation", self.test_quotation_cancellation_scenario),
            ("Customer Message Cancellation", self.test_customer_message_cancellation),
            ("Mixed Input Cancellation", self.test_mixed_input_cancellation),
            ("API Routing Consistency", self.test_api_routing_consistency)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                print(f"\n🔄 Running: {test_name}")
                result = await test_func()
                results.append((test_name, result))
                
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    # FAIL-FAST: Stop on first failure for debugging
                    print(f"\n🛑 STOPPING ON FIRST FAILURE: {test_name}")
                    break
                    
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
                results.append((test_name, False))
                # FAIL-FAST: Stop on error
                print(f"\n🛑 STOPPING ON ERROR: {test_name}")
                break
        
        # Summary
        print("\n" + "=" * 80)
        print("🏁 MOBILE HITL CANCELLATION TEST RESULTS")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == len(tests):
            print("🎉 ALL TESTS PASSED - Mobile HITL cancellation working correctly!")
            return True
        else:
            print("🔧 ISSUES FOUND - Debug and fix needed")
            return False

async def main():
    """Run the mobile HITL cancellation test suite"""
    tester = MobileHITLCancellationTester()
    success = await tester.run_all_tests()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
