#!/usr/bin/env python3
"""
Test to reproduce the HITL cancellation bug seen in the screenshot.

This test simulates the exact scenario:
1. User requests quotation
2. Agent asks for missing data (HITL prompt)
3. User says "cancel" 
4. Agent should provide clear cancellation message, not confusion
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent

async def test_hitl_cancellation_bug():
    """Reproduce the exact HITL cancellation bug from the screenshot."""
    
    print("🧪 Testing HITL Cancellation Bug Reproduction")
    print("=" * 60)
    
    agent = UnifiedToolCallingRAGAgent()
    conversation_id = "test_hitl_cancel_bug"
    user_id = "test_user"
    
    try:
        # Step 1: User requests quotation (should trigger HITL for missing data)
        print("📝 Step 1: User requests 'Generate an Informal Quote'")
        
        result1 = await agent.process_user_message(
            user_query="Generate an Informal Quote",
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        print(f"✅ Agent Response: {result1.get('message', 'No message')[:100]}...")
        print(f"✅ Interrupted: {result1.get('is_interrupted', False)}")
        print(f"✅ HITL Phase: {result1.get('hitl_phase')}")
        
        if not result1.get('is_interrupted'):
            print("❌ FAILED: Expected agent to interrupt for missing quotation data")
            return False
        
        # Step 2: User says "cancel" in response to HITL prompt
        print("\n📝 Step 2: User responds with 'cancel'")
        
        result2 = await agent.resume_interrupted_conversation(
            conversation_id=conversation_id,
            user_response="cancel"
        )
        
        print(f"✅ Agent Response: {result2.get('message', 'No message')}")
        print(f"✅ Interrupted: {result2.get('is_interrupted', False)}")
        
        response_message = result2.get('message', '')
        
        # Check if the response is appropriate
        if "don't have a specific request to cancel" in response_message:
            print("❌ BUG REPRODUCED: Agent is confused about what to cancel")
            print(f"❌ Problematic response: '{response_message}'")
            return False
        elif "cancelled" in response_message.lower() and "quotation" in response_message.lower():
            print("✅ GOOD: Agent provided clear cancellation message")
            return True
        else:
            print(f"⚠️ UNCLEAR: Response doesn't clearly indicate cancellation: '{response_message}'")
            return False
            
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_cancellation_scenarios():
    """Test various cancellation scenarios to ensure robustness."""
    
    print("\n🧪 Testing Multiple Cancellation Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Quotation with typo cancellation",
            "initial_request": "Generate an Informal Quote", 
            "cancellation": "Cancell",  # Typo like in the logs
            "expected_keywords": ["cancelled", "quotation"]
        },
        {
            "name": "Customer message cancellation",
            "initial_request": "Send a message to customer John about the SUV inquiry",
            "cancellation": "cancel",
            "expected_keywords": ["cancelled", "message"]
        },
        {
            "name": "Mixed input cancellation",
            "initial_request": "Generate an Informal Quote",
            "cancellation": "john@example.com but cancel this",
            "expected_keywords": ["cancelled", "quotation"]
        }
    ]
    
    all_passed = True
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        
        agent = UnifiedToolCallingRAGAgent()
        conversation_id = f"test_scenario_{i}"
        user_id = "test_user"
        
        try:
            # Initial request
            result1 = await agent.process_user_message(
                user_query=scenario["initial_request"],
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            if result1.get('is_interrupted'):
                # HITL scenario - test cancellation
                result2 = await agent.resume_interrupted_conversation(
                    conversation_id=conversation_id,
                    user_response=scenario["cancellation"]
                )
                
                response = result2.get('message', '').lower()
                
                # Check if response contains expected keywords
                has_keywords = all(keyword in response for keyword in scenario["expected_keywords"])
                
                if has_keywords and "don't have a specific request" not in response:
                    print(f"✅ PASSED: Clear cancellation message")
                else:
                    print(f"❌ FAILED: Unclear cancellation response")
                    print(f"   Response: '{result2.get('message', '')}'")
                    all_passed = False
            else:
                # Non-HITL scenario - agent should handle cancellation gracefully
                result2 = await agent.process_user_message(
                    user_query=scenario["cancellation"],
                    conversation_id=conversation_id,
                    user_id=user_id
                )
                
                response = result2.get('message', '').lower()
                
                if "don't have a specific request" in response:
                    print(f"⚠️ INFO: Agent correctly indicates no active request to cancel")
                else:
                    print(f"✅ PASSED: Agent handled cancellation gracefully")
                    
        except Exception as e:
            print(f"💥 ERROR in scenario {i}: {str(e)}")
            all_passed = False
    
    return all_passed

async def main():
    """Run all cancellation bug tests."""
    
    print("🚀 HITL Cancellation Bug Investigation")
    print("🎯 Goal: Reproduce and understand the cancellation confusion bug")
    print("=" * 80)
    
    # Test 1: Reproduce the exact bug
    bug_reproduced = await test_hitl_cancellation_bug()
    
    # Test 2: Test multiple scenarios
    scenarios_passed = await test_multiple_cancellation_scenarios()
    
    print("\n" + "=" * 80)
    print("🏁 INVESTIGATION RESULTS")
    print("=" * 80)
    
    if bug_reproduced:
        print("✅ Bug reproduction test: PASSED (no bug found)")
    else:
        print("❌ Bug reproduction test: FAILED (bug reproduced)")
    
    if scenarios_passed:
        print("✅ Multiple scenarios test: PASSED")
    else:
        print("❌ Multiple scenarios test: FAILED")
    
    if bug_reproduced and scenarios_passed:
        print("\n🎉 All tests passed - cancellation system working correctly!")
        return True
    else:
        print("\n🔧 Issues found - debugging needed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
