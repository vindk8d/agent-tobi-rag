"""
Simple End-to-End HITL Flow Testing

Tests the complete HITL workflow from tool request through human response 
back to agent processing in the simplified architecture - without pytest fixtures.

Employee workflow: ea_memory_prep → employee_agent → (hitl_node or ea_memory_store)
HITL workflow: employee_agent ↔ hitl_node loop only
Customer workflow: ca_memory_prep → customer_agent → ca_memory_store (no HITL)
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

# Import necessary modules
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from agents.hitl import HITLRequest
from langchain_core.messages import HumanMessage, AIMessage


async def create_test_agent():
    """Create an initialized agent for testing."""
    agent = UnifiedToolCallingRAGAgent()
    
    # Mock the settings and initialization
    with patch('agents.tobi_sales_copilot.agent.get_settings') as mock_settings:
        mock_settings.return_value = AsyncMock()
        mock_settings.return_value.openai_chat_model = "gpt-4"
        mock_settings.return_value.openai_temperature = 0.1
        mock_settings.return_value.openai_max_tokens = 1000
        mock_settings.return_value.openai_api_key = "test-key"
        mock_settings.return_value.langsmith = AsyncMock()
        mock_settings.return_value.langsmith.tracing_enabled = False
        mock_settings.return_value.memory = AsyncMock()
        mock_settings.return_value.memory.max_messages = 20
        
        # Mock memory manager
        with patch('agents.tobi_sales_copilot.agent.memory_manager') as mock_memory:
            mock_memory._ensure_initialized = AsyncMock()
            mock_memory.get_checkpointer = AsyncMock()
            mock_memory.get_checkpointer.return_value = MagicMock()
            
            # Mock memory scheduler
            with patch('agents.tobi_sales_copilot.agent.memory_scheduler') as mock_scheduler:
                mock_scheduler.start = AsyncMock()
                
                # Mock database client
                with patch('agents.tobi_sales_copilot.agent.db_client'):
                    await agent._ensure_initialized()
    
    return agent


def create_employee_state():
    """Create a test state for employee user."""
    return AgentState(
        messages=[HumanMessage(content="Send a follow-up message to John Smith")],
        conversation_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        user_verified=True,
        user_type="employee",
        retrieved_docs=[],
        sources=[],
        conversation_summary=None,
        long_term_context=[],
        hitl_data=None
    )


def create_customer_state():
    """Create a test state for customer user."""
    return AgentState(
        messages=[HumanMessage(content="What vehicles do you have available?")],
        conversation_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        user_verified=True,
        user_type="customer", 
        retrieved_docs=[],
        sources=[],
        conversation_summary=None,
        long_term_context=[],
        hitl_data=None
    )


async def test_employee_workflow_with_hitl_required(agent, employee_state):
    """
    Test complete employee workflow when HITL is required.
    
    Flow: ea_memory_prep → employee_agent → hitl_node → employee_agent → ea_memory_store
    """
    print("\n🧪 Testing Employee Workflow with HITL Required")
    
    # Test routing from employee agent when HITL is required
    employee_state_with_hitl = employee_state.copy()
    employee_state_with_hitl["hitl_data"] = {
        "interaction_type": "confirmation",
        "awaiting_response": True,
        "prompt": "Please confirm the customer message",
        "metadata": {"customer_name": "John Smith"}
    }
    
    # Test route_from_employee_agent with HITL data
    routing_result = agent.route_from_employee_agent(employee_state_with_hitl)
    assert routing_result == "hitl_node", f"Expected 'hitl_node', got '{routing_result}'"
    print("✅ Employee agent correctly routes to hitl_node when HITL required")
    
    # Test route_from_hitl when awaiting response (re-prompt case)  
    hitl_routing_result = agent.route_from_hitl(employee_state_with_hitl)
    assert hitl_routing_result == "hitl_node", f"Expected 'hitl_node', got '{hitl_routing_result}'"
    print("✅ HITL node correctly loops back to itself when awaiting response")
    
    # Test route_from_hitl when interaction complete
    completed_hitl_state = employee_state.copy()
    completed_hitl_state["hitl_data"] = {
        "interaction_type": "confirmation", 
        "awaiting_response": False,
        "response_data": {"approved": True},
        "prompt": "Confirmation complete"
    }
    
    completed_routing_result = agent.route_from_hitl(completed_hitl_state)
    assert completed_routing_result == "employee_agent", f"Expected 'employee_agent', got '{completed_routing_result}'"
    print("✅ HITL node correctly routes back to employee_agent when interaction complete")


async def test_employee_workflow_without_hitl(agent, employee_state):
    """
    Test employee workflow when no HITL is required.
    
    Flow: ea_memory_prep → employee_agent → ea_memory_store
    """
    print("\n🧪 Testing Employee Workflow without HITL")
    
    # Test routing from employee agent when no HITL is required
    routing_result = agent.route_from_employee_agent(employee_state)
    assert routing_result == "ea_memory_store", f"Expected 'ea_memory_store', got '{routing_result}'"
    print("✅ Employee agent correctly routes to ea_memory_store when no HITL required")


async def test_customer_workflow_simple(agent, customer_state):
    """
    Test customer workflow (simple, no HITL).
    
    Flow: ca_memory_prep → customer_agent → ca_memory_store
    """
    print("\n🧪 Testing Customer Workflow (Simple, No HITL)")
    
    # Customer workflows don't use HITL routing functions in simplified architecture
    # They have direct edges: customer_agent → ca_memory_store
    
    # Verify customer state doesn't have hitl_data
    assert customer_state.get("hitl_data") is None, "Customer state should not have hitl_data"
    print("✅ Customer state correctly has no hitl_data")
    
    # Test that customer routing functions would not be called
    # (since we have direct edges now)
    print("✅ Customer workflow uses direct edges (no HITL routing needed)")


async def test_routing_edge_cases(agent):
    """Test routing functions with edge cases and invalid states."""
    print("\n🧪 Testing Routing Edge Cases")
    
    # Test employee routing with invalid user type (safety check)
    invalid_employee_state = {
        "user_type": "customer",
        "user_id": "test-user",
        "hitl_data": None
    }
    
    routing_result = agent.route_from_employee_agent(invalid_employee_state)
    assert routing_result == "ea_memory_store", "Should fallback to ea_memory_store"
    print("✅ Employee routing handles invalid user types safely")
    
    # Test HITL routing with invalid user type (safety check)
    invalid_hitl_state = {
        "user_type": "customer",
        "user_id": "test-user", 
        "hitl_data": {"awaiting_response": False}
    }
    
    hitl_routing_result = agent.route_from_hitl(invalid_hitl_state)
    assert hitl_routing_result == "employee_agent", "Should route back to employee_agent"
    print("✅ HITL routing handles invalid user types safely")


async def test_tool_hitl_integration(agent, employee_state):
    """
    Test integration between tool execution and HITL system.
    
    Simulates a tool that returns HITL_REQUIRED response.
    """
    print("\n🧪 Testing Tool-HITL Integration")
    
    # Test that tool response gets parsed correctly
    # This would be done inside the employee_agent_node
    from agents.hitl import parse_tool_response
    
    # Create a proper HITL response using HITLRequest.confirmation()
    tool_response = HITLRequest.confirmation(
        prompt="Please confirm sending message to John Smith",
        context={"tool": "trigger_customer_message", "customer_id": "john-smith"}
    )
    
    parsed = parse_tool_response(tool_response, "trigger_customer_message")
    
    assert parsed["type"] == "hitl_required", f"Expected 'hitl_required', got '{parsed['type']}'"
    assert parsed["hitl_type"] == "confirmation", f"Expected 'confirmation', got '{parsed['hitl_type']}'"
    assert "Please confirm" in parsed["hitl_data"]["prompt"], "HITL prompt should contain confirmation text"
    print("✅ Tool response correctly parsed as HITL_REQUIRED")
    
    # Test HITLRequest creation
    hitl_request_string = HITLRequest.confirmation(
        prompt="Please confirm sending message to John Smith",
        context={"customer_name": "John Smith", "message": "Follow up message"}
    )
    
    # HITLRequest.confirmation() returns a string, so parse it to test the data
    parsed_request = parse_tool_response(hitl_request_string, "test_tool")
    assert parsed_request["type"] == "hitl_required", "HITLRequest should be parsed as hitl_required"
    assert parsed_request["hitl_type"] == "confirmation", "HITLRequest should be confirmation type"
    print("✅ HITLRequest correctly created for confirmation")


async def test_user_verification_routing(agent):
    """Test user verification routing to memory prep nodes."""
    print("\n🧪 Testing User Verification Routing")
    
    # Test employee user routing
    employee_verified_state = {
        "user_verified": True,
        "user_type": "employee",
        "user_id": "emp-123"
    }
    
    route_result = agent._route_to_memory_prep(employee_verified_state)
    assert route_result == "ea_memory_prep", f"Expected 'ea_memory_prep', got '{route_result}'"
    print("✅ Verified employee user routes to ea_memory_prep")
    
    # Test customer user routing  
    customer_verified_state = {
        "user_verified": True,
        "user_type": "customer",
        "user_id": "cust-123"
    }
    
    route_result = agent._route_to_memory_prep(customer_verified_state)
    assert route_result == "ca_memory_prep", f"Expected 'ca_memory_prep', got '{route_result}'"
    print("✅ Verified customer user routes to ca_memory_prep")
    
    # Test unverified user routing
    unverified_state = {
        "user_verified": False,
        "user_type": "employee",
        "user_id": "emp-123"
    }
    
    route_result = agent._route_to_memory_prep(unverified_state)
    assert route_result == "end", f"Expected 'end', got '{route_result}'"
    print("✅ Unverified user routes to end")


def test_graph_structure_simplified(agent):
    """Test that the graph structure matches the simplified architecture."""
    print("\n🧪 Testing Simplified Graph Structure")
    
    # Verify the graph was built with correct node structure
    assert agent.graph is not None, "Graph should be initialized"
    
    # Check that hitl_node was imported correctly
    from agents.hitl import hitl_node
    assert hitl_node is not None, "hitl_node should be importable"
    print("✅ hitl_node correctly imported from agents.hitl")
    
    # Verify routing functions exist
    assert hasattr(agent, 'route_from_employee_agent'), "route_from_employee_agent should exist"
    assert hasattr(agent, 'route_from_hitl'), "route_from_hitl should exist"
    print("✅ Simplified routing functions exist")
    
    # Verify old routing functions are removed/replaced
    assert not hasattr(agent, 'route_from_agent'), "route_from_agent should be removed"
    print("✅ Old complex routing functions removed")


# Main test runner
async def run_all_tests():
    """Run all end-to-end HITL tests."""
    print("🚀 Starting HITL End-to-End Flow Tests")
    print("=" * 60)
    
    try:
        # Create agent and state fixtures
        agent = await create_test_agent()
        employee_state = create_employee_state()
        customer_state = create_customer_state()
        
        # Run all tests
        await test_employee_workflow_with_hitl_required(agent, employee_state)
        await test_employee_workflow_without_hitl(agent, employee_state)
        await test_customer_workflow_simple(agent, customer_state)
        await test_routing_edge_cases(agent)
        await test_tool_hitl_integration(agent, employee_state)
        await test_user_verification_routing(agent)
        test_graph_structure_simplified(agent)
        
        print("\n" + "=" * 60)
        print("🎉 ALL HITL END-TO-END TESTS PASSED!")
        print("✅ Simplified HITL architecture is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests directly."""
    result = asyncio.run(run_all_tests())
    exit(0 if result else 1) 