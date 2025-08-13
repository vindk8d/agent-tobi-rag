#!/usr/bin/env python3
"""
Test HITL compatibility with simplified AgentState.

This test verifies that our streamlined AgentState (Task 3.1-3.8) maintains
full compatibility with the HITL system's 3-field architecture.

Tests cover:
1. HITL state field preservation (hitl_phase, hitl_prompt, hitl_context)
2. HITL routing logic with simplified state
3. State transitions during HITL workflows
4. Serialization/deserialization of HITL fields
"""

import sys
import os
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.tobi_sales_copilot.state import AgentState
from backend.agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from langchain_core.messages import HumanMessage, AIMessage


class TestHITLStateCompatibility:
    """Test HITL compatibility with simplified AgentState."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = UnifiedToolCallingRAGAgent()
        
        # Sample simplified AgentState with HITL fields
        self.sample_state = {
            "messages": [
                HumanMessage(content="Test message"),
                AIMessage(content="Test response")
            ],
            "conversation_id": "test-conv-123",
            "user_id": "test-user-456", 
            "employee_id": "emp-789",
            "customer_id": None,
            "conversation_summary": None,
            # HITL fields - the core of our test
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None
        }

    def test_hitl_fields_preserved_in_state(self):
        """Test that HITL fields are correctly defined in simplified AgentState."""
        from backend.agents.tobi_sales_copilot.state import AgentState
        
        # Verify HITL fields are present in AgentState TypedDict
        state_annotations = AgentState.__annotations__
        
        assert "hitl_phase" in state_annotations
        assert "hitl_prompt" in state_annotations  
        assert "hitl_context" in state_annotations
        
        # Verify they are Optional types (can be None)
        assert str(state_annotations["hitl_phase"]) == "typing.Optional[str]"
        assert str(state_annotations["hitl_prompt"]) == "typing.Optional[str]"  
        assert str(state_annotations["hitl_context"]) == "typing.Optional[typing.Dict[str, typing.Any]]"

    def test_route_from_employee_agent_no_hitl(self):
        """Test routing when no HITL is needed."""
        state = self.sample_state.copy()
        
        # No HITL fields set - should route to "end"
        result = self.agent.route_from_employee_agent(state)
        
        assert result == "end"

    def test_route_from_employee_agent_hitl_needed(self):
        """Test routing when HITL interaction is needed."""
        state = self.sample_state.copy()
        
        # Set HITL fields to simulate tool requesting confirmation
        state.update({
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Send this message to customer?",
            "hitl_context": {
                "source_tool": "trigger_customer_message",
                "customer_id": "cust-123",
                "message_content": "Hello from support team"
            }
        })
        
        result = self.agent.route_from_employee_agent(state)
        
        assert result == "hitl_node"

    def test_route_from_employee_agent_hitl_approved(self):
        """Test routing when HITL action was approved."""
        state = self.sample_state.copy()
        
        # Set HITL phase to approved (prompt cleared after user responds)
        state.update({
            "hitl_phase": "approved",
            "hitl_prompt": None,  # Prompt is cleared after user response
            "hitl_context": {
                "source_tool": "trigger_customer_message",
                "customer_id": "cust-123",
                "message_content": "Hello from support team"
            }
        })
        
        result = self.agent.route_from_employee_agent(state)
        
        # Should route back to employee_agent for execution or end
        assert result in ["employee_agent", "end"]

    def test_route_from_employee_agent_hitl_denied(self):
        """Test routing when HITL action was denied."""
        state = self.sample_state.copy()
        
        # Set HITL phase to denied (prompt cleared after user responds)
        state.update({
            "hitl_phase": "denied",
            "hitl_prompt": None,  # Prompt is cleared after user response
            "hitl_context": {
                "source_tool": "trigger_customer_message",
                "customer_id": "cust-123",
                "message_content": "Hello from support team"
            }
        })
        
        result = self.agent.route_from_employee_agent(state)
        
        # Should route back to employee_agent or end
        assert result in ["employee_agent", "end"]

    def test_hitl_context_json_serializable(self):
        """Test that HITL context can be JSON serialized."""
        import json
        
        hitl_context = {
            "source_tool": "trigger_customer_message",
            "customer_id": "cust-123",
            "message_content": "Hello from support team",
            "metadata": {
                "timestamp": "2024-01-15T10:30:00Z",
                "priority": "normal"
            }
        }
        
        # Should serialize without errors
        serialized = json.dumps(hitl_context)
        deserialized = json.loads(serialized)
        
        assert deserialized == hitl_context

    def test_state_with_hitl_fields_serializable(self):
        """Test that AgentState with HITL fields can be serialized."""
        import json
        from backend.core.utils.serialization import safe_json_dumps
        
        state = {
            "conversation_id": "test-conv-123",
            "user_id": "test-user-456",
            "employee_id": "emp-789",
            "customer_id": None,
            "conversation_summary": "Previous conversation about product inquiry",
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Send this message to customer?",
            "hitl_context": {
                "source_tool": "trigger_customer_message",
                "customer_id": "cust-123"
            }
        }
        
        # Should serialize without errors using our utility
        serialized = safe_json_dumps(state)
        deserialized = json.loads(serialized)
        
        # Verify HITL fields are preserved
        assert deserialized["hitl_phase"] == "needs_prompt"
        assert deserialized["hitl_prompt"] == "Send this message to customer?"
        assert deserialized["hitl_context"]["source_tool"] == "trigger_customer_message"

    def test_hitl_fields_optional_none_values(self):
        """Test that HITL fields can be None without issues."""
        state = self.sample_state.copy()
        
        # All HITL fields should be None initially
        assert state["hitl_phase"] is None
        assert state["hitl_prompt"] is None
        assert state["hitl_context"] is None
        
        # Should still route correctly
        result = self.agent.route_from_employee_agent(state)
        assert result == "end"

    def test_hitl_workflow_state_transitions(self):
        """Test complete HITL workflow state transitions."""
        state = self.sample_state.copy()
        
        # 1. Initial state - no HITL
        assert self.agent.route_from_employee_agent(state) == "end"
        
        # 2. Tool sets HITL fields
        state.update({
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Approve this quotation?",
            "hitl_context": {"source_tool": "generate_quotation", "quotation_id": "q-123"}
        })
        assert self.agent.route_from_employee_agent(state) == "hitl_node"
        
        # 3. User approves (simulated) - prompt is cleared after user response
        state.update({
            "hitl_phase": "approved",
            "hitl_prompt": None  # Prompt is cleared after user responds
        })
        result = self.agent.route_from_employee_agent(state)
        assert result in ["employee_agent", "end"]
        
        # 4. After execution, HITL fields should be cleared (would happen in actual agent)
        state.update({
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None
        })
        assert self.agent.route_from_employee_agent(state) == "end"

    @patch('backend.agents.tobi_sales_copilot.agent.logger')
    def test_hitl_routing_logging(self, mock_logger):
        """Test that HITL routing includes proper logging."""
        state = self.sample_state.copy()
        state.update({
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Test prompt",
            "hitl_context": {"source_tool": "test_tool"}
        })
        
        self.agent.route_from_employee_agent(state)
        
        # Verify logging calls were made
        assert mock_logger.info.called
        
        # Check that HITL state is logged
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        hitl_log_found = any("3-field state" in log for log in log_calls)
        assert hitl_log_found

    def test_hitl_compatibility_with_removed_fields(self):
        """Test that HITL works without the removed bloated state fields."""
        state = self.sample_state.copy()
        
        # Verify removed fields are not present (from Task 3.1)
        assert "retrieved_docs" not in state
        assert "sources" not in state  
        assert "long_term_context" not in state
        
        # HITL should still work with simplified state
        state.update({
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Test with simplified state",
            "hitl_context": {"source_tool": "test_tool"}
        })
        
        result = self.agent.route_from_employee_agent(state)
        assert result == "hitl_node"


if __name__ == "__main__":
    # Run specific test
    test = TestHITLStateCompatibility()
    test.setup_method()
    
    print("🧪 Testing HITL State Compatibility...")
    
    try:
        # Run key tests
        test.test_hitl_fields_preserved_in_state()
        print("✅ HITL fields preserved in AgentState")
        
        test.test_route_from_employee_agent_no_hitl()
        print("✅ Routing works when no HITL needed")
        
        test.test_route_from_employee_agent_hitl_needed()
        print("✅ Routing works when HITL needed")
        
        test.test_hitl_context_json_serializable()
        print("✅ HITL context is JSON serializable")
        
        test.test_hitl_workflow_state_transitions()
        print("✅ HITL workflow state transitions work")
        
        test.test_hitl_compatibility_with_removed_fields()
        print("✅ HITL works without removed bloated fields")
        
        print("\n🎉 All HITL compatibility tests passed!")
        print("✅ Simplified AgentState maintains full HITL functionality")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
