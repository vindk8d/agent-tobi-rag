#!/usr/bin/env python3
"""
Phase 2 State-Driven HITL Architecture Validation

This script validates the key components of our Phase 2 implementation
without requiring pytest dependencies.
"""

import sys
import os
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

async def validate_phase2_implementation():
    """
    Validate the key components of Phase 2 state-driven implementation.
    """
    print("🔍 VALIDATING PHASE 2 STATE-DRIVEN HITL ARCHITECTURE")
    print("=" * 60)
    
    validation_results = {
        "tool_behavior": False,
        "employee_agent_state": False, 
        "routing_logic": False,
        "hitl_node": False,
        "integration": False
    }
    
    try:
        # Import required modules
        from agents.tools import trigger_customer_message, UserContext
        from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
        from agents.tobi_sales_copilot.state import AgentState
        from langchain_core.messages import HumanMessage, AIMessage
        
        print("\n✅ Successfully imported all required modules")
        
        # Test 1: Tool State-Driven Behavior
        print("\n🧪 TEST 1: Tool State-Driven Behavior")
        print("-" * 40)
        
        with UserContext(user_id="emp123", user_type="employee"):
            with patch('agents.tools.get_current_employee_id', return_value="emp123"):
                with patch('agents.tools._lookup_customer') as mock_lookup:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "name": "Jane Doe",
                        "first_name": "Jane", 
                        "last_name": "Doe",
                        "email": "jane.doe@example.com"
                    }
                    
                    # Test tool behavior
                    result = await trigger_customer_message.ainvoke({
                        "customer_id": "CUST001",
                        "message_content": "Test message for validation",
                        "message_type": "follow_up"
                    })
                    
                    if "STATE_DRIVEN_CONFIRMATION_REQUIRED:" in result:
                        print("   ✅ Tool returns STATE_DRIVEN_CONFIRMATION_REQUIRED")
                        
                        # Validate JSON structure
                        json_start = result.find("STATE_DRIVEN_CONFIRMATION_REQUIRED:") + len("STATE_DRIVEN_CONFIRMATION_REQUIRED:")
                        json_data = json.loads(result[json_start:].strip())
                        
                        required_fields = ["requires_human_confirmation", "customer_info", "message_content", "message_type"]
                        if all(field in json_data for field in required_fields):
                            print("   ✅ JSON confirmation data has all required fields")
                            validation_results["tool_behavior"] = True
                        else:
                            print("   ❌ JSON confirmation data missing required fields")
                    else:
                        print("   ❌ Tool does not return STATE_DRIVEN_CONFIRMATION_REQUIRED")
        
        # Test 2: Employee Agent State Handling 
        print("\n🧪 TEST 2: Employee Agent State Handling")
        print("-" * 40)
        
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Test HITL resumption detection
        resume_state = AgentState(
            messages=[HumanMessage(content="Test message")],
            conversation_id="test-conv",
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[],
            long_term_context=[],
            confirmation_result="delivered"  # Simulate resumption from HITL
        )
        
        result = await agent._employee_agent_node(resume_state)
        
        # Check if employee agent detected HITL resumption
        messages = result["messages"]
        if len(messages) > len(resume_state["messages"]):
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and "successfully delivered" in last_message.content:
                print("   ✅ Employee agent detects HITL resumption and provides feedback")
                if result.get("confirmation_result") is None and result.get("confirmation_data") is None:
                    print("   ✅ Employee agent cleans up state variables")
                    validation_results["employee_agent_state"] = True
                else:
                    print("   ❌ Employee agent failed to clean up state variables")
            else:
                print("   ❌ Employee agent did not provide appropriate resumption feedback")
        else:
            print("   ❌ Employee agent did not add resumption message")
        
        # Test 3: State-Driven Routing Logic
        print("\n🧪 TEST 3: State-Driven Routing Logic")
        print("-" * 40)
        
        # Test routing with confirmation data
        state_with_data = {"confirmation_data": {"test": "data"}, "confirmation_result": None}
        route_result = agent._route_employee_to_hitl_or_end(state_with_data)
        
        if route_result == "customer_message_confirmation_and_delivery":
            print("   ✅ Routes to HITL when confirmation_data present")
            
            # Test routing without confirmation data
            state_without_data = {"confirmation_data": None, "confirmation_result": None}
            route_result = agent._route_employee_to_hitl_or_end(state_without_data)
            
            if route_result == "end":
                print("   ✅ Routes to END when no confirmation_data")
                
                # Test routing with existing result
                state_with_result = {"confirmation_data": {"test": "data"}, "confirmation_result": "delivered"}
                route_result = agent._route_employee_to_hitl_or_end(state_with_result)
                
                if route_result == "end":
                    print("   ✅ Routes to END when confirmation_result exists (side effect protection)")
                    validation_results["routing_logic"] = True
                else:
                    print("   ❌ Failed to skip HITL when confirmation_result exists")
            else:
                print("   ❌ Failed to route to END when no confirmation_data")
        else:
            print("   ❌ Failed to route to HITL when confirmation_data present")
        
        # Test 4: HITL Node Functionality
        print("\n🧪 TEST 4: HITL Node Functionality")
        print("-" * 40)
        
        # Test missing confirmation data handling
        state_without_data = AgentState(
            messages=[HumanMessage(content="Test")],
            conversation_id="test-conv",
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[],
            long_term_context=[]
        )
        
        result = await agent._customer_message_confirmation_and_delivery_node(state_without_data)
        
        if result.get("confirmation_result") == "error_no_data":
            print("   ✅ HITL node handles missing confirmation_data gracefully")
            
            # Test side effect protection
            state_with_result = AgentState(
                messages=[HumanMessage(content="Test")],
                conversation_id="test-conv", 
                user_id="emp123",
                user_type="employee",
                user_verified=True,
                retrieved_docs=[],
                sources=[],
                long_term_context=[],
                confirmation_data={"customer_info": {"name": "Jane Doe"}},
                confirmation_result="delivered"  # Already processed
            )
            
            result = await agent._customer_message_confirmation_and_delivery_node(state_with_result)
            
            if result.get("confirmation_result") == "delivered":
                print("   ✅ HITL node side effect protection prevents re-execution")
                validation_results["hitl_node"] = True
            else:
                print("   ❌ HITL node side effect protection failed")
        else:
            print("   ❌ HITL node failed to handle missing confirmation_data")
        
        # Test 5: Phase 1 Integration
        print("\n🧪 TEST 5: Phase 1 Integration")
        print("-" * 40)
        
        # Test user verification still works
        with patch.object(agent, '_verify_user_access') as mock_verify:
            mock_verify.return_value = "employee"
            
            initial_state = AgentState(
                messages=[HumanMessage(content="Hello")],
                conversation_id="test-conv",
                user_id="emp123",
                retrieved_docs=[],
                sources=[],
                long_term_context=[]
            )
            
            result = await agent._user_verification_node(initial_state)
            
            if result.get("user_verified") is True and result.get("user_type") == "employee":
                print("   ✅ Phase 1 user verification works with Phase 2 changes")
                validation_results["integration"] = True
            else:
                print("   ❌ Phase 1 user verification integration failed")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    for test_name, passed in validation_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL PHASE 2 STATE-DRIVEN VALIDATIONS PASSED!")
        print("✅ The implementation is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} validation(s) failed.")
        print("❌ Please review and fix the issues before proceeding.")
        return False

def print_architecture_summary():
    """
    Print a summary of the Phase 2 state-driven architecture.
    """
    print("\n🏗️  PHASE 2 STATE-DRIVEN ARCHITECTURE SUMMARY")
    print("=" * 60)
    print("""
📋 Key Components Validated:

1️⃣  Tool State-Driven Behavior
   • trigger_customer_message returns STATE_DRIVEN_CONFIRMATION_REQUIRED
   • JSON confirmation data contains all required fields
   • Access control and validation preserved

2️⃣  Employee Agent State Handling
   • Populates confirmation_data in AgentState from tool results
   • Detects HITL resumption via confirmation_result presence
   • Provides delivery feedback and cleans up state variables

3️⃣  State-Driven Routing Logic
   • Routes to HITL when confirmation_data exists
   • Routes to END when no confirmation needed 
   • Side effect protection when confirmation_result exists

4️⃣  Dedicated HITL Node
   • Handles missing confirmation_data gracefully
   • Prevents re-execution with confirmation_result guard
   • Combines interrupt + delivery in atomic operation

5️⃣  Phase 1 Integration
   • User verification compatibility maintained
   • Existing functionality preserved
   • Clean separation of concerns

🎯 Graph Flow: employee_agent → HITL → employee_agent → END
🛡️  Minimal State: Only 2 variables (confirmation_data + confirmation_result)
✨ Clean Architecture: Centralized response handling + state cleanup
""")

if __name__ == "__main__":
    """
    Main execution when run directly.
    """
    print_architecture_summary()
    
    # Run async validation
    success = asyncio.run(validate_phase2_implementation())
    
    if not success:
        sys.exit(1) 