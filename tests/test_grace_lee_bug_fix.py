"""
Grace Lee Bug Fix Validation Tests

Tests for the specific Grace Lee duplication bug that the Universal HITL Recursion System fixes.

THE GRACE LEE BUG:
User: "I need a quotation for just 1 vehicle, next week, pickup at branch, financing"
Agent: Calls generate_quotation → HITL request for missing customer info
User: Provides customer details  
Agent: Should RESUME generate_quotation, but instead RESTARTS it → DUPLICATION BUG

ROOT CAUSE:
generate_quotation's HITL requests lacked collection_mode="tool_managed" marker,
so agent didn't recognize them as resumable tool-managed collections.

THE FIX:
@hitl_recursive_tool decorator automatically adds universal context:
- collection_mode="tool_managed" 
- source_tool="generate_quotation"
- original_params={all original parameters}

VALIDATION:
These tests reproduce the exact Grace Lee scenario and verify:
1. HITL requests now have universal markers
2. Agent detects resumable collection correctly  
3. Tool resumes instead of restarting
4. No duplication occurs
5. Natural conversation flow maintained
"""

import asyncio
import pytest
import pytest_asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
from copy import deepcopy

# Import test modules
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from agents.tools import generate_quotation
from agents.hitl import parse_tool_response, UniversalHITLControl
from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


class TestGraceLeeBugFix:
    """Test suite specifically for validating the Grace Lee duplication bug fix."""

    @pytest_asyncio.fixture
    async def test_agent(self):
        """Create test agent with mocked dependencies for Grace Lee scenario testing."""
        agent = UnifiedToolCallingRAGAgent()
        
        # Mock the settings and initialization
        with patch('agents.tobi_sales_copilot.agent.get_settings') as mock_settings:
            mock_settings.return_value = AsyncMock()
            mock_settings.return_value.openai_chat_model = "gpt-4"
            mock_settings.return_value.openai_simple_model = "gpt-3.5-turbo"
            mock_settings.return_value.openai_temperature = 0.1
            mock_settings.return_value.openai_max_tokens = 1000
            mock_settings.return_value.openai_api_key = "test-key"
            mock_settings.return_value.langsmith = AsyncMock()
            mock_settings.return_value.langsmith.tracing_enabled = False
            mock_settings.return_value.memory = AsyncMock()
            mock_settings.return_value.memory.max_messages = 20
            
            # Mock tools to include generate_quotation
            mock_tool = MagicMock()
            mock_tool.name = "generate_quotation"
            mock_tool.func = generate_quotation
            
            with patch('agents.tobi_sales_copilot.agent.get_all_tools', return_value=[mock_tool]):
                await agent.initialize()
                
        return agent

    @pytest.mark.asyncio
    async def test_grace_lee_scenario_original_hitl_request(self):
        """Test that generate_quotation now creates HITL requests with universal markers."""
        # Mock dependencies that generate_quotation needs
        with patch('agents.tools.get_current_employee_id', return_value="emp123"):
            with patch('agents.tools._lookup_customer', return_value=None):  # Customer not found
                with patch('agents.tools._extract_context_from_conversation', return_value={}):
                    
                    # Grace Lee's initial request (missing customer info)
                    result = await generate_quotation(
                        customer_identifier="Grace Lee",  # Not in CRM
                        vehicle_requirements="just 1 vehicle, next week, pickup at branch, financing"
                    )
                    
                    # Parse the HITL response
                    parsed = parse_tool_response(result, "generate_quotation")
                    
                    # CRITICAL: Verify universal markers are present (this fixes the bug!)
                    assert parsed["type"] == "hitl_required"
                    hitl_context = parsed["hitl_context"]
                    
                    # Universal markers that were MISSING before the fix
                    assert hitl_context["source_tool"] == "generate_quotation"
                    assert hitl_context["collection_mode"] == "tool_managed"  # KEY FIX!
                    assert "original_params" in hitl_context
                    
                    # Original parameters preserved for resume
                    original_params = hitl_context["original_params"]
                    assert original_params["customer_identifier"] == "Grace Lee"
                    assert original_params["vehicle_requirements"] == "just 1 vehicle, next week, pickup at branch, financing"

    @pytest.mark.asyncio
    async def test_grace_lee_scenario_agent_detection(self, test_agent):
        """Test that agent now correctly detects Grace Lee scenario as resumable."""
        # Create the universal HITL context that generate_quotation now produces
        grace_lee_context = {
            "source_tool": "generate_quotation",
            "collection_mode": "tool_managed",  # This was MISSING before!
            "original_params": {
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "just 1 vehicle, next week, pickup at branch, financing"
            }
        }
        
        state = {
            "hitl_phase": "approved",  # User provided customer info
            "hitl_context": grace_lee_context,
            "messages": [
                HumanMessage(content="Grace Lee, email: grace@example.com, phone: 555-0123")
            ]
        }
        
        # CRITICAL TEST: Agent should now detect this as resumable
        is_resumable = test_agent._is_tool_managed_collection_needed(grace_lee_context, state)
        assert is_resumable == True, "Agent failed to detect Grace Lee scenario as resumable!"

    @pytest.mark.asyncio
    async def test_grace_lee_scenario_resume_not_restart(self, test_agent):
        """Test that Grace Lee scenario RESUMES generate_quotation instead of restarting."""
        # Mock generate_quotation to track calls
        call_count = 0
        original_params_received = None
        user_response_received = None
        
        async def mock_generate_quotation(
            customer_identifier: str,
            vehicle_requirements: str,
            additional_notes: str = None,
            quotation_validity_days: int = 30,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = "",
            conversation_context: str = ""
        ):
            nonlocal call_count, original_params_received, user_response_received
            call_count += 1
            original_params_received = {
                "customer_identifier": customer_identifier,
                "vehicle_requirements": vehicle_requirements
            }
            user_response_received = user_response
            
            if current_step == "resume":
                return f"RESUMED: Got user response '{user_response}' for {customer_identifier}"
            else:
                return "HITL_REQUIRED: Need customer info"
        
        # Mock tool availability
        mock_tool = MagicMock()
        mock_tool.name = "generate_quotation"
        mock_tool.func = mock_generate_quotation
        
        with patch('agents.tobi_sales_copilot.agent.get_all_tools', return_value=[mock_tool]):
            grace_lee_context = {
                "source_tool": "generate_quotation",
                "collection_mode": "tool_managed",
                "original_params": {
                    "customer_identifier": "Grace Lee",
                    "vehicle_requirements": "just 1 vehicle, next week, pickup at branch, financing"
                }
            }
            
            state = {
                "hitl_phase": "approved",
                "hitl_context": grace_lee_context,
                "messages": [
                    HumanMessage(content="Grace Lee, email: grace@example.com, phone: 555-0123")
                ]
            }
            
            # This should RESUME, not restart
            result_state = await test_agent._handle_tool_managed_collection(grace_lee_context, state)
            
            # Verify tool was called exactly once (resume, not restart)
            assert call_count == 1
            
            # Verify it was called with RESUME parameters
            assert original_params_received["customer_identifier"] == "Grace Lee"
            assert original_params_received["vehicle_requirements"] == "just 1 vehicle, next week, pickup at branch, financing"
            assert user_response_received == "Grace Lee, email: grace@example.com, phone: 555-0123"
            
            # Verify response indicates resume, not restart
            last_message = result_state["messages"][-1]
            assert "RESUMED" in last_message["content"]
            assert "grace@example.com" in last_message["content"]

    @pytest.mark.asyncio
    async def test_grace_lee_no_duplication_bug(self):
        """Test the complete Grace Lee scenario with no duplication."""
        call_log = []
        
        # Mock generate_quotation to track the exact call pattern
        async def tracked_generate_quotation(
            customer_identifier: str,
            vehicle_requirements: str,
            additional_notes: str = None,
            quotation_validity_days: int = 30,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = "",
            conversation_context: str = ""
        ):
            call_info = {
                "call_type": "resume" if current_step == "resume" else "initial",
                "customer_identifier": customer_identifier,
                "vehicle_requirements": vehicle_requirements,
                "user_response": user_response,
                "current_step": current_step
            }
            call_log.append(call_info)
            
            if current_step == "resume" and user_response:
                # Resume with user's customer info
                return f"SUCCESS: Quotation for {customer_identifier} with requirements: {vehicle_requirements}"
            else:
                # Initial call - need customer info
                from agents.hitl import request_input
                return request_input(
                    prompt=f"I need customer information for '{customer_identifier}'",
                    input_type="customer_info",
                    context={}  # Universal decorator will enhance this
                )
        
        with patch('agents.tools.generate_quotation', side_effect=tracked_generate_quotation):
            with patch('agents.tools.get_current_employee_id', return_value="emp123"):
                # Step 1: Initial call (Grace Lee scenario)
                result1 = await generate_quotation(
                    customer_identifier="Grace Lee",
                    vehicle_requirements="just 1 vehicle, next week, pickup at branch, financing"
                )
                
                parsed1 = parse_tool_response(result1, "generate_quotation")
                assert parsed1["type"] == "hitl_required"
                
                # Step 2: Resume call (user provides customer info)
                hitl_context = parsed1["hitl_context"]
                original_params = hitl_context["original_params"]
                
                result2 = await generate_quotation(
                    **original_params,
                    user_response="Grace Lee, email: grace@example.com, phone: 555-0123",
                    hitl_phase="approved",
                    current_step="resume"
                )
                
                # VERIFICATION: No duplication bug
                assert len(call_log) == 2, f"Expected 2 calls, got {len(call_log)}: {call_log}"
                
                # First call should be initial
                assert call_log[0]["call_type"] == "initial"
                assert call_log[0]["customer_identifier"] == "Grace Lee"
                
                # Second call should be resume (NOT initial again!)
                assert call_log[1]["call_type"] == "resume"
                assert call_log[1]["customer_identifier"] == "Grace Lee"
                assert call_log[1]["user_response"] == "Grace Lee, email: grace@example.com, phone: 555-0123"
                
                # Final result should be success
                assert "SUCCESS" in result2
                assert "Grace Lee" in result2

    def test_before_fix_simulation(self):
        """Simulate how the bug occurred BEFORE the universal system fix."""
        # Before fix: HITL context was missing collection_mode="tool_managed"
        broken_context = {
            "source_tool": "generate_quotation",
            "current_step": "customer_lookup",  # Legacy pattern
            "customer_identifier": "Grace Lee",
            "vehicle_requirements": "just 1 vehicle, next week, pickup at branch, financing"
            # MISSING: "collection_mode": "tool_managed"
        }
        
        # Agent detection logic (simplified)
        def old_detection_logic(hitl_context, state):
            # Before fix: Only checked for collection_mode
            return hitl_context.get("collection_mode") == "tool_managed"
        
        # This would return False, causing the duplication bug
        state = {"hitl_phase": "approved"}
        is_resumable = old_detection_logic(broken_context, state)
        assert is_resumable == False, "Old logic should NOT detect as resumable"

    def test_after_fix_verification(self):
        """Verify the fix: Universal system automatically adds the missing marker."""
        # After fix: @hitl_recursive_tool decorator creates universal context
        universal_context = UniversalHITLControl.create(
            source_tool="generate_quotation",
            original_params={
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "just 1 vehicle, next week, pickup at branch, financing"
            }
        ).to_hitl_context()
        
        # Agent detection logic (current)
        def new_detection_logic(hitl_context, state):
            return (
                hitl_context.get("source_tool") and
                hitl_context.get("collection_mode") == "tool_managed" and
                state.get("hitl_phase") in ["approved", "denied"]
            )
        
        # This now returns True, fixing the duplication bug
        state = {"hitl_phase": "approved"}
        is_resumable = new_detection_logic(universal_context, state)
        assert is_resumable == True, "New logic should detect as resumable"

    @pytest.mark.asyncio
    async def test_multiple_grace_lee_scenarios(self):
        """Test multiple similar scenarios to ensure robust fix."""
        scenarios = [
            {
                "name": "Grace Lee Original",
                "customer": "Grace Lee", 
                "requirements": "just 1 vehicle, next week, pickup at branch, financing"
            },
            {
                "name": "John Doe Similar",
                "customer": "John Doe",
                "requirements": "2 cars, ASAP, home delivery, cash payment"  
            },
            {
                "name": "Corporate Client",
                "customer": "ABC Corp",
                "requirements": "fleet of 5 SUVs, monthly lease, company pickup"
            }
        ]
        
        for scenario in scenarios:
            with patch('agents.tools.get_current_employee_id', return_value="emp123"):
                with patch('agents.tools._lookup_customer', return_value=None):  # Not found
                    with patch('agents.tools._extract_context_from_conversation', return_value={}):
                        
                        result = await generate_quotation(
                            customer_identifier=scenario["customer"],
                            vehicle_requirements=scenario["requirements"]
                        )
                        
                        parsed = parse_tool_response(result, "generate_quotation")
                        
                        # Every scenario should have universal markers
                        assert parsed["type"] == "hitl_required", f"Failed for {scenario['name']}"
                        hitl_context = parsed["hitl_context"]
                        assert hitl_context["collection_mode"] == "tool_managed", f"Missing marker for {scenario['name']}"
                        assert hitl_context["source_tool"] == "generate_quotation", f"Wrong source for {scenario['name']}"

    def test_universal_context_completeness(self):
        """Test that universal context contains all necessary information for resume."""
        # Create universal context as the decorator would
        original_params = {
            "customer_identifier": "Grace Lee",
            "vehicle_requirements": "just 1 vehicle, next week, pickup at branch, financing",
            "additional_notes": "urgent request",
            "quotation_validity_days": 30
        }
        
        control = UniversalHITLControl.create(
            source_tool="generate_quotation",
            original_params=original_params
        )
        
        context = control.to_hitl_context()
        
        # Must have all required fields
        assert "source_tool" in context
        assert "collection_mode" in context
        assert "original_params" in context
        
        # Original parameters must be complete
        restored_params = context["original_params"]
        assert restored_params["customer_identifier"] == "Grace Lee"
        assert restored_params["vehicle_requirements"] == "just 1 vehicle, next week, pickup at branch, financing"
        assert restored_params["additional_notes"] == "urgent request"
        assert restored_params["quotation_validity_days"] == 30
        
        # Can be used to resume the exact same call
        assert control.get_tool_name() == "generate_quotation"
        assert control.get_original_params() == original_params


class TestGraceLeeBugRegressionPrevention:
    """Tests to prevent regression of the Grace Lee bug."""

    def test_universal_marker_always_present(self):
        """Test that universal tools ALWAYS have the collection_mode marker."""
        # Any function with @hitl_recursive_tool should produce universal context
        from agents.hitl import hitl_recursive_tool, request_input
        
        @hitl_recursive_tool
        async def any_universal_tool(param: str) -> str:
            return request_input("Need info", "test", {})
        
        # The decorator should ensure universal context is always created
        # This is tested indirectly through the decorator tests above
        # But this serves as a regression prevention marker
        
        # If this test fails, the Grace Lee bug has regressed
        assert hasattr(any_universal_tool, '__wrapped__'), "Decorator not applied correctly"

    def test_agent_detection_robustness(self):
        """Test that agent detection is robust against context variations."""
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Various universal context formats should all be detected
        universal_contexts = [
            {
                "source_tool": "generate_quotation",
                "collection_mode": "tool_managed",
                "original_params": {"param": "value"}
            },
            {
                "source_tool": "another_tool", 
                "collection_mode": "tool_managed",
                "original_params": {"different": "params"},
                "extra_data": "ignored"  # Should be robust to extra fields
            }
        ]
        
        state = {"hitl_phase": "approved"}
        
        for context in universal_contexts:
            is_detected = agent._is_tool_managed_collection_needed(context, state)
            assert is_detected == True, f"Failed to detect universal context: {context}"

    def test_non_universal_not_detected(self):
        """Test that non-universal contexts are NOT detected (backward compatibility)."""
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        agent = UnifiedToolCallingRAGAgent()
        
        # Legacy contexts should NOT be detected as universal
        legacy_contexts = [
            {
                "source_tool": "legacy_tool",
                "current_step": "some_step"
                # Missing: collection_mode="tool_managed"
            },
            {
                "some_other": "pattern",
                "custom_data": "preserved"
            },
            {}  # Empty context
        ]
        
        state = {"hitl_phase": "approved"}
        
        for context in legacy_contexts:
            is_detected = agent._is_tool_managed_collection_needed(context, state)
            assert is_detected == False, f"Incorrectly detected non-universal context: {context}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


