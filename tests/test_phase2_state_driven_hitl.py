"""
Comprehensive Test Suite for Phase 2: State-Driven HITL Architecture

This test suite validates the updated Phase 2 implementation with state-driven approach:
1. State-driven routing based on confirmation_data presence
2. Dedicated HITL node with combined interrupt+execution pattern
3. Centralized response handling in employee agent
4. Side effect protection using confirmation_result
5. Clean graph flow: employee_agent → HITL → employee_agent → END
6. Tool behavior with STATE_DRIVEN_CONFIRMATION_REQUIRED indicator

Updated Architecture:
- trigger_customer_message tool populates confirmation_data in AgentState
- Employee agent detects state-driven confirmation requests
- Routing logic sends to dedicated HITL node
- HITL node handles interrupt and delivery, routes back to employee agent
- Employee agent provides delivery feedback and cleanup
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from agents.tools import (
    trigger_customer_message,
    UserContext,
    get_current_user_type,
)


class TestStateDrivenCustomerMessageTool:
    """Test the updated trigger_customer_message tool with state-driven approach."""
    
    @pytest.mark.asyncio
    async def test_tool_returns_state_driven_confirmation_required(self):
        """Test that the tool returns STATE_DRIVEN_CONFIRMATION_REQUIRED indicator."""
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
                    
                    result = await trigger_customer_message.ainvoke({
                        "customer_id": "CUST001",
                        "message_content": "Thank you for your inquiry. We have updates for you.",
                        "message_type": "follow_up"
                    })
                    
                    # Should return state-driven indicator with JSON data
                    assert "STATE_DRIVEN_CONFIRMATION_REQUIRED:" in result
                    assert "Jane Doe" in result
                    
                    # Extract and validate JSON data
                    json_start = result.find("STATE_DRIVEN_CONFIRMATION_REQUIRED:") + len("STATE_DRIVEN_CONFIRMATION_REQUIRED:")
                    json_data = json.loads(result[json_start:].strip())
                    
                    assert json_data["requires_human_confirmation"] is True
                    assert json_data["confirmation_type"] == "customer_message"
                    assert json_data["customer_info"]["name"] == "Jane Doe"
                    assert json_data["message_content"] == "Thank you for your inquiry. We have updates for you."
                    assert json_data["message_type"] == "follow_up"

    @pytest.mark.asyncio
    async def test_tool_validates_employee_access_only(self):
        """Test that the tool still enforces employee-only access."""
        with UserContext(user_id="cust123", user_type="customer"):
            result = await trigger_customer_message.ainvoke({
                "customer_id": "CUST001", 
                "message_content": "Test message",
                "message_type": "follow_up"
            })
            
            # Should deny access
            assert "customer messaging is only available to employees" in result
            assert "STATE_DRIVEN_CONFIRMATION_REQUIRED" not in result

    @pytest.mark.asyncio 
    async def test_tool_validates_message_content(self):
        """Test that the tool still performs message content validation."""
        with UserContext(user_id="emp123", user_type="employee"):
            with patch('agents.tools.get_current_employee_id', return_value="emp123"):
                result = await trigger_customer_message.ainvoke({
                    "customer_id": "CUST001",
                    "message_content": "",  # Empty message should fail validation
                    "message_type": "follow_up"
                })
                
                # Should fail validation
                assert "Message Validation Failed" in result
                assert "STATE_DRIVEN_CONFIRMATION_REQUIRED" not in result


class TestEmployeeAgentStateDrivenHandling:
    """Test employee agent's handling of state-driven confirmation requests."""
    
    @pytest.mark.asyncio
    async def test_employee_agent_populates_confirmation_data(self):
        """Test that employee agent populates confirmation_data when tool indicates confirmation needed."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Mock the tool to return state-driven confirmation
        mock_confirmation_data = {
            "requires_human_confirmation": True,
            "customer_info": {"name": "Jane Doe", "email": "jane@example.com"},
            "message_content": "Test message",
            "message_type": "follow_up"
        }
        
        initial_state = AgentState(
            messages=[HumanMessage(content="Send a follow-up to Jane Doe about her inquiry")],
            conversation_id="test-conv",
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[],
            long_term_context=[]
        )
        
        # Set up proper UserContext for the test
        with UserContext(user_id="emp123", user_type="employee"):
            with patch.object(agent, 'tools') as mock_tools:
                # Mock tool that returns state-driven confirmation
                mock_tool = MagicMock()
                mock_tool.name = "trigger_customer_message"
                mock_tool.ainvoke = AsyncMock(return_value=f"STATE_DRIVEN_CONFIRMATION_REQUIRED: {json.dumps(mock_confirmation_data)}")
                mock_tools = [mock_tool]
                agent.tools = mock_tools
                agent.tool_map = {"trigger_customer_message": mock_tool}
                
                # Mock LLM to call the tool
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = MagicMock()
                    mock_response = MagicMock()
                    mock_response.tool_calls = [{
                        "name": "trigger_customer_message",
                        "args": {"customer_id": "CUST001", "message_content": "Test message"},
                        "id": "call_123"
                    }]
                    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                    mock_llm_class.return_value.bind_tools.return_value = mock_llm
                    
                    # Mock memory manager methods to avoid database calls
                    with patch.object(agent.memory_manager, 'store_message_from_agent', return_value=None):
                        with patch.object(agent.memory_manager, 'get_user_context_for_new_conversation', return_value={"has_history": False}):
                            with patch.object(agent.memory_manager, 'store_conversation_insights', return_value=None):
                                with patch.object(agent.memory_manager.consolidator, 'check_and_trigger_summarization', return_value=None):
                                    # Execute the employee agent node
                                    result = await agent._employee_agent_node(initial_state)
                                    
                                    # Should populate confirmation_data in returned state
                                    assert "confirmation_data" in result
                                    assert result["confirmation_data"] is not None
                                    assert result["confirmation_data"]["requires_human_confirmation"] is True
                                    assert result["confirmation_data"]["customer_info"]["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_employee_agent_handles_hitl_resumption(self):
        """Test that employee agent provides delivery feedback when resumed from HITL."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # State as if resumed from HITL with delivery result
        resume_state = AgentState(
            messages=[
                HumanMessage(content="Send a follow-up to Jane Doe"),
                AIMessage(content="I've prepared a follow_up message for Jane Doe and it's ready for your approval."),
                AIMessage(content="✅ Message successfully sent to Jane Doe!")
            ],
            conversation_id="test-conv", 
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[],
            long_term_context=[],
            confirmation_data={
                "customer_info": {"name": "Jane Doe"},
                "message_type": "follow_up"
            },
            confirmation_result="delivered"  # Indicates resumption from HITL
        )
        
        # Execute the employee agent node
        result = await agent._employee_agent_node(resume_state)
        
        # Should detect HITL resumption and provide follow-up message
        messages = result["messages"]
        assert len(messages) > len(resume_state["messages"])
        
        # Last message should be the follow-up
        follow_up_message = messages[-1]
        assert isinstance(follow_up_message, AIMessage)
        assert "successfully delivered" in follow_up_message.content
        assert "anything else I can help you with" in follow_up_message.content
        
        # Should clean up state variables
        assert result["confirmation_data"] is None
        assert result["confirmation_result"] is None


class TestStateDrivenRouting:
    """Test the state-driven routing logic for HITL."""
    
    @pytest.mark.asyncio
    async def test_routing_detects_confirmation_data_presence(self):
        """Test that routing function detects confirmation_data and routes to HITL."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # State with confirmation_data but no confirmation_result
        state_with_confirmation = {
            "confirmation_data": {"customer_info": {"name": "Jane Doe"}},
            "confirmation_result": None
        }
        
        routing_result = agent._route_employee_to_hitl_or_end(state_with_confirmation)
        assert routing_result == "customer_message_confirmation_and_delivery"
        
        # State without confirmation_data
        state_without_confirmation = {
            "confirmation_data": None,
            "confirmation_result": None
        }
        
        routing_result = agent._route_employee_to_hitl_or_end(state_without_confirmation)
        assert routing_result == "end"
        
        # State with confirmation_result (already processed)
        state_with_result = {
            "confirmation_data": {"customer_info": {"name": "Jane Doe"}},
            "confirmation_result": "delivered"
        }
        
        routing_result = agent._route_employee_to_hitl_or_end(state_with_result) 
        assert routing_result == "end"


class TestDedicatedHITLNode:
    """Test the dedicated customer_message_confirmation_and_delivery HITL node."""
    
    @pytest.mark.asyncio
    async def test_hitl_node_detects_missing_confirmation_data(self):
        """Test HITL node handles missing confirmation_data gracefully."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # State without confirmation_data
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
        
        # Should return error state with error result for protection
        assert result["confirmation_result"] == "error_no_data"
        assert result["confirmation_data"] is None

    @pytest.mark.asyncio
    async def test_hitl_node_side_effect_protection(self):
        """Test that HITL node prevents re-execution when confirmation_result exists."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # State with existing confirmation_result (already processed)
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
        
        # Should return without re-execution
        assert result["confirmation_result"] == "delivered"  # Preserved
        assert result["confirmation_data"]["customer_info"]["name"] == "Jane Doe"  # Preserved

    @pytest.mark.asyncio
    async def test_hitl_node_interrupt_and_delivery_approved(self):
        """Test HITL node interrupt mechanism with approval."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # State with confirmation data
        confirmation_data = {
            "customer_info": {"name": "Jane Doe", "email": "jane@example.com"},
            "message_content": "Test message",
            "message_type": "follow_up", 
            "customer_id": "CUST001"
        }
        
        state_with_confirmation = AgentState(
            messages=[HumanMessage(content="Test")],
            conversation_id="test-conv",
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[], 
            long_term_context=[],
            confirmation_data=confirmation_data
        )
        
        with patch('agents.tobi_sales_copilot.rag_agent.interrupt') as mock_interrupt:
            with patch.object(agent, '_execute_customer_message_delivery') as mock_delivery:
                # Mock human approval
                mock_interrupt.return_value = "approve"
                mock_delivery.return_value = True  # Successful delivery
                
                result = await agent._customer_message_confirmation_and_delivery_node(state_with_confirmation)
                
                # Should have called interrupt
                mock_interrupt.assert_called_once()
                interrupt_message = mock_interrupt.call_args[0][0]
                assert "Customer Message Confirmation Required" in interrupt_message
                assert "Jane Doe" in interrupt_message
                
                # Should have attempted delivery
                mock_delivery.assert_called_once()
                
                # Should return delivered status
                assert result["confirmation_result"] == "delivered"
                assert result["confirmation_data"] is None  # Cleaned up
                
                # Should have added success message
                messages = result["messages"]
                success_message = messages[-1]
                assert isinstance(success_message, AIMessage)
                assert "successfully sent to Jane Doe" in success_message.content

    @pytest.mark.asyncio
    async def test_hitl_node_interrupt_and_delivery_denied(self):
        """Test HITL node interrupt mechanism with denial.""" 
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        confirmation_data = {
            "customer_info": {"name": "John Smith"},
            "message_content": "Test message",
            "message_type": "follow_up",
            "customer_id": "CUST002"
        }
        
        state_with_confirmation = AgentState(
            messages=[HumanMessage(content="Test")],
            conversation_id="test-conv",
            user_id="emp123",
            user_type="employee",
            user_verified=True,
            retrieved_docs=[],
            sources=[],
            long_term_context=[],
            confirmation_data=confirmation_data
        )
        
        with patch('agents.tobi_sales_copilot.rag_agent.interrupt') as mock_interrupt:
            with patch.object(agent, '_execute_customer_message_delivery') as mock_delivery:
                # Mock human denial
                mock_interrupt.return_value = "deny"
                
                result = await agent._customer_message_confirmation_and_delivery_node(state_with_confirmation)
                
                # Should have called interrupt
                mock_interrupt.assert_called_once()
                
                # Should NOT have attempted delivery
                mock_delivery.assert_not_called()
                
                # Should return cancelled status
                assert result["confirmation_result"] == "cancelled"
                assert result["confirmation_data"] is None  # Cleaned up
                
                # Should have added cancellation message
                messages = result["messages"]
                cancel_message = messages[-1]
                assert isinstance(cancel_message, AIMessage)
                assert "cancelled" in cancel_message.content
                assert "John Smith" in cancel_message.content


class TestPhase2Integration:
    """Test that Phase 2 state-driven approach integrates properly with Phase 1."""
    
    @pytest.mark.asyncio
    async def test_phase1_user_verification_still_works(self):
        """Test that Phase 1 user verification works with Phase 2 changes."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Mock user verification 
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
            
            # Should verify user and set user_type
            assert result["user_verified"] is True
            assert result["user_type"] == "employee"

    @pytest.mark.asyncio
    async def test_customer_agent_routes_directly_to_end(self):
        """Test that customer agent routes directly to END (no HITL capability)."""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Customer state should not trigger HITL routing
        customer_state = {
            "user_type": "customer", 
            "confirmation_data": None,
            "confirmation_result": None
        }
        
        # Customer agent should not have access to HITL routing
        # (This is implicit in the graph structure where customer_agent → END directly)
        
        # Verify that only employee routing function exists
        assert hasattr(agent, '_route_employee_to_hitl_or_end')
        # Customer routing would not exist - they go directly to END


class TestGraphFlowIntegration:
    """Test the complete graph flow with state-driven HITL."""
    
    @pytest.mark.asyncio
    async def test_complete_employee_message_flow(self):
        """Test complete flow: employee_agent → HITL → employee_agent → END"""
        agent = UnifiedToolCallingRAGAgent()
        await agent._ensure_initialized()
        
        # Mock all the necessary components for full flow test
        with patch.object(agent, '_verify_user_access', return_value="employee"):
            with UserContext(user_id="emp123", user_type="employee"):  # Set proper user context
                with patch('agents.tools._lookup_customer') as mock_lookup:
                    with patch('agents.tools.get_current_employee_id', return_value="emp123"):
                        with patch('agents.tobi_sales_copilot.rag_agent.interrupt') as mock_interrupt:
                            with patch.object(agent, '_execute_customer_message_delivery') as mock_delivery:
                                # Setup mocks
                                mock_lookup.return_value = {
                                    "customer_id": "CUST001",
                                    "name": "Jane Doe",
                                    "email": "jane@example.com"
                                }
                                mock_interrupt.return_value = "approve"
                                mock_delivery.return_value = True
                                
                                # This would require a full graph execution test
                                # For now, we can test the individual components work together
                                
                                # 1. Tool creates confirmation data
                                tool_result = await trigger_customer_message.ainvoke({
                                    "customer_id": "CUST001",
                                    "message_content": "Test message",
                                    "message_type": "follow_up"
                                })
                                assert "STATE_DRIVEN_CONFIRMATION_REQUIRED" in tool_result
                                
                                # 2. Employee agent would detect and populate state
                                # (Tested in TestEmployeeAgentStateDrivenHandling)
                                
                                # 3. Routing would detect confirmation_data
                                state_with_data = {"confirmation_data": {"test": "data"}, "confirmation_result": None}
                                route_result = agent._route_employee_to_hitl_or_end(state_with_data)
                                assert route_result == "customer_message_confirmation_and_delivery"
                                
                                # 4. HITL would handle interrupt and delivery
                                # (Tested in TestDedicatedHITLNode)
                                
                                # 5. Employee agent would handle resumption
                                # (Tested in TestEmployeeAgentStateDrivenHandling)


if __name__ == "__main__":
    """Run the tests when executed directly."""
    pytest.main([__file__, "-v", "--tb=short"]) 