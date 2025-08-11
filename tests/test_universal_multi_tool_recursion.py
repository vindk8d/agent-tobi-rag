"""
Universal Multi-Tool Recursion Tests

Tests for multiple tools using the Universal HITL Recursion System simultaneously.
This validates that the universal system works across different tools and scenarios.

TESTS COVER:
1. Multiple universal tools working independently
2. Universal tools mixed with legacy tools (backward compatibility)
3. Different HITL request types (input, approval, selection) with universal system
4. Tool-to-tool interactions with universal context
5. Complex multi-step scenarios across different tools
6. Performance and reliability of universal system at scale

This extends our existing test foundation with universal system validation
while ensuring backward compatibility with existing HITL patterns.
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

from agents.hitl import (
    hitl_recursive_tool, 
    request_input, 
    request_approval, 
    request_selection,
    parse_tool_response,
    UniversalHITLControl
)
from agents.tools import generate_quotation, collect_sales_requirements
from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field


class TestUniversalMultiToolScenarios:
    """Test universal system with multiple different tools."""

    @pytest_asyncio.fixture
    async def test_agent(self):
        """Create test agent with mocked dependencies."""
        agent = UnifiedToolCallingRAGAgent()
        
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
            
            with patch('agents.tobi_sales_copilot.agent.get_all_tools', return_value=[]):
                await agent.initialize()
                
        return agent

    def create_universal_tools(self):
        """Create various universal tools for testing."""
        
        @hitl_recursive_tool
        async def universal_customer_lookup(
            customer_name: str,
            search_criteria: str = "",
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            if current_step == "resume" and user_response:
                return f"Found customer: {customer_name} with info: {user_response}"
            return request_input(
                prompt=f"Need more details to find customer '{customer_name}'",
                input_type="customer_details",
                context={}
            )

        @hitl_recursive_tool
        async def universal_vehicle_configurator(
            vehicle_type: str,
            budget: str = "",
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            if current_step == "resume" and user_response:
                return f"Configured {vehicle_type} with preferences: {user_response}"
            return request_selection(
                prompt=f"Please select options for your {vehicle_type}",
                options=["Basic", "Premium", "Luxury"],
                context={}
            )

        @hitl_recursive_tool
        async def universal_financing_calculator(
            vehicle_price: float,
            down_payment: float = 0,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            if current_step == "resume" and user_response:
                return f"Financing calculated for ${vehicle_price} with terms: {user_response}"
            return request_approval(
                prompt=f"Approve financing terms for ${vehicle_price} vehicle?",
                context={}
            )

        return {
            "customer_lookup": universal_customer_lookup,
            "vehicle_configurator": universal_vehicle_configurator,
            "financing_calculator": universal_financing_calculator
        }

    @pytest.mark.asyncio
    async def test_multiple_universal_tools_independent(self):
        """Test multiple universal tools working independently."""
        tools = self.create_universal_tools()
        
        # Test each tool creates proper universal context
        results = {}
        
        # Customer lookup
        result1 = await tools["customer_lookup"]("John Doe", "phone number")
        parsed1 = parse_tool_response(result1, "customer_lookup")
        assert parsed1["type"] == "hitl_required"
        assert parsed1["hitl_context"]["collection_mode"] == "tool_managed"
        assert parsed1["hitl_context"]["source_tool"] == "customer_lookup"
        results["customer"] = parsed1
        
        # Vehicle configurator
        result2 = await tools["vehicle_configurator"]("SUV", "$50000")
        parsed2 = parse_tool_response(result2, "vehicle_configurator")
        assert parsed2["type"] == "hitl_required"
        assert parsed2["hitl_context"]["collection_mode"] == "tool_managed"
        assert parsed2["hitl_context"]["source_tool"] == "vehicle_configurator"
        results["vehicle"] = parsed2
        
        # Financing calculator
        result3 = await tools["financing_calculator"](45000.0, 5000.0)
        parsed3 = parse_tool_response(result3, "financing_calculator")
        assert parsed3["type"] == "hitl_required"
        assert parsed3["hitl_context"]["collection_mode"] == "tool_managed"
        assert parsed3["hitl_context"]["source_tool"] == "financing_calculator"
        results["financing"] = parsed3
        
        # Verify each has unique, complete context
        contexts = [r["hitl_context"] for r in results.values()]
        assert len(set(c["source_tool"] for c in contexts)) == 3  # All different tools
        assert all(c["collection_mode"] == "tool_managed" for c in contexts)  # All universal

    @pytest.mark.asyncio
    async def test_universal_tool_resume_flows(self):
        """Test resuming different universal tools with user responses."""
        tools = self.create_universal_tools()
        
        # Resume customer lookup
        result1 = await tools["customer_lookup"](
            "John Doe", 
            "phone number",
            user_response="Phone: 555-0123, Email: john@example.com",
            hitl_phase="approved",
            current_step="resume"
        )
        assert "Found customer: John Doe" in result1
        assert "555-0123" in result1
        
        # Resume vehicle configurator
        result2 = await tools["vehicle_configurator"](
            "SUV",
            "$50000",
            user_response="Premium",
            hitl_phase="approved", 
            current_step="resume"
        )
        assert "Configured SUV" in result2
        assert "Premium" in result2
        
        # Resume financing calculator
        result3 = await tools["financing_calculator"](
            45000.0,
            5000.0,
            user_response="approved",
            hitl_phase="approved",
            current_step="resume"
        )
        assert "Financing calculated" in result3
        assert "45000" in result3

    @pytest.mark.asyncio
    async def test_universal_and_legacy_tools_coexistence(self, test_agent):
        """Test universal tools working alongside existing legacy tools."""
        # Create a legacy tool (no decorator)
        async def legacy_tool(param: str) -> str:
            return request_input(
                prompt="Legacy tool needs info",
                input_type="legacy_input",
                context={
                    "source_tool": "legacy_tool",
                    "current_step": "legacy_step",  # Legacy pattern
                    "param": param
                    # Missing: collection_mode="tool_managed"
                }
            )
        
        # Create universal tool
        @hitl_recursive_tool
        async def universal_tool(param: str) -> str:
            return request_input(
                prompt="Universal tool needs info",
                input_type="universal_input",
                context={}
            )
        
        # Test legacy tool (should work as before)
        legacy_result = await legacy_tool("legacy_param")
        legacy_parsed = parse_tool_response(legacy_result, "legacy_tool")
        assert legacy_parsed["type"] == "hitl_required"
        legacy_context = legacy_parsed["hitl_context"]
        assert legacy_context.get("collection_mode") != "tool_managed"  # Not universal
        assert legacy_context.get("current_step") == "legacy_step"  # Legacy pattern
        
        # Test universal tool
        universal_result = await universal_tool("universal_param")
        universal_parsed = parse_tool_response(universal_result, "universal_tool")
        assert universal_parsed["type"] == "hitl_required"
        universal_context = universal_parsed["hitl_context"]
        assert universal_context["collection_mode"] == "tool_managed"  # Universal
        assert universal_context["source_tool"] == "universal_tool"
        
        # Agent should detect only universal tool as resumable
        state = {"hitl_phase": "approved"}
        legacy_resumable = test_agent._is_tool_managed_collection_needed(legacy_context, state)
        universal_resumable = test_agent._is_tool_managed_collection_needed(universal_context, state)
        
        assert legacy_resumable == False  # Legacy not detected
        assert universal_resumable == True  # Universal detected

    @pytest.mark.asyncio
    async def test_complex_multi_step_universal_scenario(self):
        """Test complex scenario with multiple universal tools in sequence."""
        call_log = []
        
        @hitl_recursive_tool
        async def step1_customer_info(
            inquiry: str,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            call_log.append(f"step1: {current_step or 'initial'}")
            if current_step == "resume" and user_response:
                return f"STEP1_COMPLETE: Customer info gathered: {user_response}"
            return request_input("What's your customer information?", "customer", {})
        
        @hitl_recursive_tool
        async def step2_vehicle_selection(
            customer_info: str,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            call_log.append(f"step2: {current_step or 'initial'}")
            if current_step == "resume" and user_response:
                return f"STEP2_COMPLETE: Vehicle selected: {user_response}"
            return request_selection("Choose vehicle type", ["Sedan", "SUV", "Truck"], {})
        
        @hitl_recursive_tool
        async def step3_finalize_order(
            customer_info: str,
            vehicle_choice: str,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            call_log.append(f"step3: {current_step or 'initial'}")
            if current_step == "resume" and user_response:
                return f"ORDER_COMPLETE: {customer_info} ordered {vehicle_choice}"
            return request_approval("Confirm your order?", {})
        
        # Step 1: Initial call
        result1 = await step1_customer_info("I want to buy a car")
        parsed1 = parse_tool_response(result1, "step1_customer_info")
        assert parsed1["hitl_context"]["collection_mode"] == "tool_managed"
        
        # Step 1: Resume
        step1_params = parsed1["hitl_context"]["original_params"]
        result1_resume = await step1_customer_info(
            **step1_params,
            user_response="John Doe, john@example.com",
            hitl_phase="approved",
            current_step="resume"
        )
        assert "STEP1_COMPLETE" in result1_resume
        
        # Step 2: Initial call
        result2 = await step2_vehicle_selection("John Doe info")
        parsed2 = parse_tool_response(result2, "step2_vehicle_selection")
        assert parsed2["hitl_context"]["collection_mode"] == "tool_managed"
        
        # Step 2: Resume
        step2_params = parsed2["hitl_context"]["original_params"]
        result2_resume = await step2_vehicle_selection(
            **step2_params,
            user_response="SUV",
            hitl_phase="approved", 
            current_step="resume"
        )
        assert "STEP2_COMPLETE" in result2_resume
        
        # Step 3: Initial call
        result3 = await step3_finalize_order("John Doe info", "SUV")
        parsed3 = parse_tool_response(result3, "step3_finalize_order")
        assert parsed3["hitl_context"]["collection_mode"] == "tool_managed"
        
        # Step 3: Resume
        step3_params = parsed3["hitl_context"]["original_params"]
        result3_resume = await step3_finalize_order(
            **step3_params,
            user_response="approved",
            hitl_phase="approved",
            current_step="resume"
        )
        assert "ORDER_COMPLETE" in result3_resume
        
        # Verify proper call sequence
        expected_calls = ["step1: initial", "step1: resume", "step2: initial", "step2: resume", "step3: initial", "step3: resume"]
        assert call_log == expected_calls

    @pytest.mark.asyncio
    async def test_agent_handling_multiple_universal_tools(self, test_agent):
        """Test agent correctly handles multiple universal tools in sequence."""
        # Mock multiple universal tools
        tools_registry = {}
        
        @hitl_recursive_tool
        async def universal_tool_a(param: str, user_response: str = "", current_step: str = "", **kwargs) -> str:
            if current_step == "resume":
                return f"ToolA completed with: {user_response}"
            return request_input("ToolA needs info", "tool_a", {})
        
        @hitl_recursive_tool
        async def universal_tool_b(param: str, user_response: str = "", current_step: str = "", **kwargs) -> str:
            if current_step == "resume":
                return f"ToolB completed with: {user_response}"
            return request_input("ToolB needs info", "tool_b", {})
        
        tools_registry["universal_tool_a"] = universal_tool_a
        tools_registry["universal_tool_b"] = universal_tool_b
        
        # Mock tool availability
        mock_tools = []
        for name, func in tools_registry.items():
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.func = func
            mock_tools.append(mock_tool)
        
        with patch('agents.tobi_sales_copilot.agent.get_all_tools', return_value=mock_tools):
            # Test Tool A flow
            tool_a_context = {
                "source_tool": "universal_tool_a",
                "collection_mode": "tool_managed",
                "original_params": {"param": "test_a"}
            }
            
            state_a = {
                "hitl_phase": "approved",
                "hitl_context": tool_a_context,
                "messages": [HumanMessage(content="Tool A response")]
            }
            
            result_a = await test_agent._handle_tool_managed_collection(tool_a_context, state_a)
            last_message_a = result_a["messages"][-1]
            assert "ToolA completed" in last_message_a["content"]
            assert "Tool A response" in last_message_a["content"]
            
            # Test Tool B flow
            tool_b_context = {
                "source_tool": "universal_tool_b", 
                "collection_mode": "tool_managed",
                "original_params": {"param": "test_b"}
            }
            
            state_b = {
                "hitl_phase": "approved",
                "hitl_context": tool_b_context,
                "messages": [HumanMessage(content="Tool B response")]
            }
            
            result_b = await test_agent._handle_tool_managed_collection(tool_b_context, state_b)
            last_message_b = result_b["messages"][-1]
            assert "ToolB completed" in last_message_b["content"]
            assert "Tool B response" in last_message_b["content"]

    def test_universal_context_isolation(self):
        """Test that universal contexts from different tools don't interfere."""
        # Create contexts for different tools
        context_a = UniversalHITLControl.create(
            source_tool="tool_a",
            original_params={"param_a": "value_a"}
        ).to_hitl_context()
        
        context_b = UniversalHITLControl.create(
            source_tool="tool_b", 
            original_params={"param_b": "value_b"}
        ).to_hitl_context()
        
        # Contexts should be independent
        assert context_a["source_tool"] != context_b["source_tool"]
        assert context_a["original_params"] != context_b["original_params"]
        
        # But both should be universal
        assert UniversalHITLControl.is_universal_context(context_a) == True
        assert UniversalHITLControl.is_universal_context(context_b) == True
        
        # Extracting params should give correct results
        control_a = UniversalHITLControl.from_hitl_context(context_a)
        control_b = UniversalHITLControl.from_hitl_context(context_b)
        
        assert control_a.get_tool_name() == "tool_a"
        assert control_b.get_tool_name() == "tool_b"
        assert control_a.get_original_params()["param_a"] == "value_a"
        assert control_b.get_original_params()["param_b"] == "value_b"


class TestUniversalSystemCompatibility:
    """Test compatibility of universal system with existing HITL patterns."""

    @pytest.mark.asyncio
    async def test_existing_collect_sales_requirements_still_works(self):
        """Test that existing collect_sales_requirements tool still works unchanged."""
        # This tool already uses collection_mode="tool_managed" pattern
        with patch('agents.tools._extract_conversation_context', return_value={}):
            result = await collect_sales_requirements("I need a car")
            
            parsed = parse_tool_response(result, "collect_sales_requirements")
            assert parsed["type"] == "hitl_required"
            
            # Should have the collection_mode marker (existing pattern)
            hitl_context = parsed["hitl_context"]
            assert hitl_context.get("collection_mode") == "tool_managed"

    @pytest.mark.asyncio
    async def test_existing_generate_quotation_now_universal(self):
        """Test that generate_quotation is now universal after our migration."""
        # Mock dependencies
        with patch('agents.tools.get_current_employee_id', return_value="emp123"):
            with patch('agents.tools._lookup_customer', return_value=None):
                with patch('agents.tools._extract_context_from_conversation', return_value={}):
                    
                    result = await generate_quotation(
                        customer_identifier="Test Customer",
                        vehicle_requirements="test requirements"
                    )
                    
                    parsed = parse_tool_response(result, "generate_quotation")
                    assert parsed["type"] == "hitl_required"
                    
                    # Should now have universal markers (after our migration)
                    hitl_context = parsed["hitl_context"]
                    assert hitl_context["collection_mode"] == "tool_managed"
                    assert hitl_context["source_tool"] == "generate_quotation"
                    assert "original_params" in hitl_context

    def test_hitl_request_tools_unchanged(self):
        """Test that core HITL request tools work unchanged with universal system."""
        # request_input should work the same
        result1 = request_input("Test prompt", "test_type", {})
        assert "HITL_REQUIRED" in result1
        
        # request_approval should work the same
        result2 = request_approval("Test approval", {})
        assert "HITL_REQUIRED" in result2
        
        # request_selection should work the same
        result3 = request_selection("Test selection", ["A", "B"], {})
        assert "HITL_REQUIRED" in result3
        
        # All should be parseable
        parsed1 = parse_tool_response(result1, "test_tool")
        parsed2 = parse_tool_response(result2, "test_tool")
        parsed3 = parse_tool_response(result3, "test_tool")
        
        assert all(p["type"] == "hitl_required" for p in [parsed1, parsed2, parsed3])


class TestUniversalSystemPerformance:
    """Test performance characteristics of universal system with multiple tools."""

    @pytest.mark.asyncio
    async def test_multiple_tools_performance(self):
        """Test that universal system performs well with multiple tools."""
        import time
        
        # Create multiple universal tools
        tools = []
        for i in range(10):
            @hitl_recursive_tool
            async def universal_tool(param: str, tool_id: int = i) -> str:
                return request_input(f"Tool {tool_id} needs info", "test", {})
            tools.append(universal_tool)
        
        # Time multiple tool calls
        start_time = time.time()
        results = []
        for i, tool in enumerate(tools):
            result = await tool(f"param_{i}")
            results.append(result)
        end_time = time.time()
        
        # Should complete quickly
        total_time = end_time - start_time
        assert total_time < 1.0, f"Universal tools too slow: {total_time:.2f}s for 10 tools"
        
        # All should have universal context
        for i, result in enumerate(results):
            parsed = parse_tool_response(result, f"tool_{i}")
            assert parsed["hitl_context"]["collection_mode"] == "tool_managed"

    def test_universal_context_memory_efficiency(self):
        """Test memory efficiency of universal contexts."""
        import sys
        
        # Create many universal contexts
        contexts = []
        for i in range(100):
            control = UniversalHITLControl.create(
                source_tool=f"tool_{i}",
                original_params={f"param_{j}": f"value_{j}" for j in range(5)}
            )
            contexts.append(control.to_hitl_context())
        
        # Calculate total memory usage
        total_size = sum(sys.getsizeof(ctx) for ctx in contexts)
        avg_size = total_size / len(contexts)
        
        # Should be memory efficient
        assert avg_size < 2000, f"Universal contexts too large: {avg_size:.0f} bytes average"
        assert total_size < 200000, f"Total memory usage too high: {total_size} bytes for 100 contexts"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
