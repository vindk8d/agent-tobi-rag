"""
Universal HITL Recursion System Tests

Tests for the ultra-minimal universal HITL system (Task 14.0) that solves the Grace Lee bug:

1. UniversalHITLControl - Thin wrapper class for universal context
2. @hitl_recursive_tool decorator - Automatic universal capability injection
3. Universal context auto-generation - Eliminates manual configuration
4. Agent integration - Universal detection and handling
5. Backward compatibility - Existing tools continue working

This validates the revolutionary universal system that makes any tool HITL-recursive
with just a single decorator, eliminating the Grace Lee duplication bug forever.
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
    UniversalHITLControl, 
    hitl_recursive_tool, 
    request_input, 
    request_approval, 
    request_selection,
    parse_tool_response
)
from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field


class TestUniversalHITLControl:
    """Test the ultra-minimal UniversalHITLControl wrapper class."""

    def test_create_universal_control(self):
        """Test creating UniversalHITLControl with minimal parameters."""
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param1": "value1", "param2": "value2"}
        )
        
        assert control.source_tool == "test_tool"
        assert control.collection_mode == "tool_managed"
        assert control.original_params == {"param1": "value1", "param2": "value2"}

    def test_from_hitl_context(self):
        """Test creating UniversalHITLControl from existing hitl_context."""
        hitl_context = {
            "source_tool": "generate_quotation",
            "collection_mode": "tool_managed", 
            "original_params": {
                "customer_identifier": "John Doe",
                "vehicle_requirements": "SUV, red, 2024"
            },
            "other_data": "preserved"
        }
        
        control = UniversalHITLControl.from_hitl_context(hitl_context)
        
        assert control.source_tool == "generate_quotation"
        assert control.collection_mode == "tool_managed"
        assert control.original_params["customer_identifier"] == "John Doe"
        assert control.original_params["vehicle_requirements"] == "SUV, red, 2024"

    def test_to_hitl_context(self):
        """Test converting UniversalHITLControl back to hitl_context."""
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param1": "value1"}
        )
        
        context = control.to_hitl_context()
        
        assert context["source_tool"] == "test_tool"
        assert context["collection_mode"] == "tool_managed"
        assert context["original_params"] == {"param1": "value1"}

    def test_is_universal_detection(self):
        """Test detecting universal HITL context."""
        # Universal context
        universal_context = {
            "source_tool": "test_tool",
            "collection_mode": "tool_managed",
            "original_params": {"param": "value"}
        }
        assert UniversalHITLControl.is_universal_context(universal_context) == True
        
        # Non-universal context
        legacy_context = {
            "source_tool": "test_tool",
            "current_step": "step1",
            "some_data": "value"
        }
        assert UniversalHITLControl.is_universal_context(legacy_context) == False
        
        # Empty context
        assert UniversalHITLControl.is_universal_context({}) == False

    def test_get_tool_name(self):
        """Test extracting tool name from universal context."""
        control = UniversalHITLControl.create(
            source_tool="generate_quotation",
            original_params={}
        )
        assert control.get_tool_name() == "generate_quotation"

    def test_get_original_params(self):
        """Test extracting original parameters."""
        params = {"customer": "John", "vehicle": "SUV"}
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params=params
        )
        assert control.get_original_params() == params

    def test_update_params(self):
        """Test updating original parameters."""
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param1": "value1"}
        )
        
        control.update_params(param2="value2", param1="updated")
        
        assert control.original_params["param1"] == "updated"
        assert control.original_params["param2"] == "value2"

    def test_repr(self):
        """Test string representation."""
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param": "value"}
        )
        
        repr_str = repr(control)
        assert "test_tool" in repr_str
        assert "param" in repr_str


class TestHitlRecursiveToolDecorator:
    """Test the @hitl_recursive_tool decorator functionality."""

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @hitl_recursive_tool
        async def test_function(param1: str, param2: int = 10) -> str:
            """Test function docstring."""
            return f"result: {param1}, {param2}"
        
        assert test_function.__name__ == "test_function"
        assert "Test function docstring" in test_function.__doc__
        # Function signature should be preserved

    @pytest.mark.asyncio
    async def test_decorator_normal_execution(self):
        """Test decorator doesn't interfere with normal function execution."""
        @hitl_recursive_tool
        async def test_function(param1: str, param2: int = 10) -> str:
            return f"result: {param1}, {param2}"
        
        result = await test_function("test", 20)
        assert result == "result: test, 20"

    @pytest.mark.asyncio
    async def test_decorator_hitl_request_enhancement(self):
        """Test decorator automatically enhances HITL requests with universal context."""
        @hitl_recursive_tool
        async def test_function(param1: str, param2: int = 10) -> str:
            # Simulate a HITL request
            return request_input(
                prompt="Need more info",
                input_type="test_input",
                context={}  # Empty context - decorator should enhance
            )
        
        result = await test_function("test_value", 25)
        
        # Parse the HITL response
        parsed = parse_tool_response(result, "test_function")
        assert parsed["type"] == "hitl_required"
        
        # Verify universal context was injected
        hitl_context = parsed["hitl_context"]
        assert hitl_context["source_tool"] == "test_function"
        assert hitl_context["collection_mode"] == "tool_managed"
        assert hitl_context["original_params"]["param1"] == "test_value"
        assert hitl_context["original_params"]["param2"] == 25

    @pytest.mark.asyncio
    async def test_decorator_resume_detection(self):
        """Test decorator detects resume calls and handles them appropriately."""
        call_count = 0
        
        @hitl_recursive_tool
        async def test_function(
            param1: str, 
            param2: int = 10,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            nonlocal call_count
            call_count += 1
            
            if current_step == "resume" and user_response:
                return f"resumed with: {user_response}, original: {param1}"
            else:
                return request_input(
                    prompt="Need info",
                    input_type="test",
                    context={}
                )
        
        # First call - should trigger HITL
        result1 = await test_function("original_value")
        parsed1 = parse_tool_response(result1, "test_function")
        assert parsed1["type"] == "hitl_required"
        assert call_count == 1
        
        # Resume call - should handle user response
        result2 = await test_function(
            "original_value", 
            user_response="user provided info",
            hitl_phase="approved",
            current_step="resume"
        )
        assert "resumed with: user provided info" in result2
        assert "original: original_value" in result2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_multiple_hitl_requests(self):
        """Test decorator works with functions that make multiple HITL requests."""
        step_count = 0
        
        @hitl_recursive_tool
        async def multi_step_function(
            param1: str,
            user_response: str = "",
            current_step: str = ""
        ) -> str:
            nonlocal step_count
            
            if current_step == "resume" and user_response:
                step_count += 1
                if step_count < 2:
                    # Need more information
                    return request_input(
                        prompt=f"Step {step_count + 1}: Need more info (got: {user_response})",
                        input_type="multi_step",
                        context={}
                    )
                else:
                    # Collection complete
                    return f"Complete: {param1}, responses: {user_response}"
            else:
                # First call
                return request_input(
                    prompt="Step 1: Initial info needed",
                    input_type="multi_step",
                    context={}
                )
        
        # First call
        result1 = await multi_step_function("test_param")
        parsed1 = parse_tool_response(result1, "multi_step_function")
        assert "Step 1" in parsed1["hitl_prompt"]
        
        # First resume
        result2 = await multi_step_function(
            "test_param",
            user_response="first response",
            current_step="resume"
        )
        parsed2 = parse_tool_response(result2, "multi_step_function")
        assert "Step 2" in parsed2["hitl_prompt"]
        assert "got: first response" in parsed2["hitl_prompt"]
        
        # Second resume - should complete
        result3 = await multi_step_function(
            "test_param",
            user_response="second response", 
            current_step="resume"
        )
        assert "Complete:" in result3
        assert "test_param" in result3


class TestUniversalSystemIntegration:
    """Test integration of universal system with existing HITL infrastructure."""

    @pytest_asyncio.fixture
    async def test_agent(self):
        """Create test agent with mocked dependencies."""
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
            
            # Mock tools
            with patch('agents.tobi_sales_copilot.agent.get_all_tools') as mock_tools:
                mock_tools.return_value = []
                await agent.initialize()
                
        return agent

    @pytest.mark.asyncio
    async def test_universal_tool_detection(self, test_agent):
        """Test agent detects universal tool-managed collections."""
        # Create universal HITL context
        universal_context = {
            "source_tool": "test_universal_tool",
            "collection_mode": "tool_managed",
            "original_params": {
                "param1": "value1",
                "param2": "value2"
            }
        }
        
        # Test detection
        state = {
            "hitl_phase": "approved",
            "hitl_context": universal_context
        }
        
        is_needed = test_agent._is_tool_managed_collection_needed(universal_context, state)
        assert is_needed == True

    @pytest.mark.asyncio
    async def test_universal_tool_handling(self, test_agent):
        """Test agent handles universal tool re-calling correctly."""
        # Mock the universal tool
        @hitl_recursive_tool
        async def mock_universal_tool(
            param1: str, 
            param2: str,
            user_response: str = "",
            hitl_phase: str = "",
            current_step: str = ""
        ) -> str:
            if current_step == "resume":
                return f"Universal tool resumed: {param1}, {param2}, response: {user_response}"
            return request_input("Need info", "test", {})
        
        # Mock tool availability
        mock_tool = MagicMock()
        mock_tool.name = "mock_universal_tool"
        mock_tool.func = mock_universal_tool
        
        with patch('agents.tobi_sales_copilot.agent.get_all_tools', return_value=[mock_tool]):
            universal_context = {
                "source_tool": "mock_universal_tool",
                "collection_mode": "tool_managed", 
                "original_params": {
                    "param1": "original_value1",
                    "param2": "original_value2"
                }
            }
            
            state = {
                "hitl_phase": "approved",
                "hitl_context": universal_context,
                "messages": [
                    HumanMessage(content="user provided response")
                ]
            }
            
            result_state = await test_agent._handle_tool_managed_collection(universal_context, state)
            
            # Should contain the tool's response
            assert len(result_state["messages"]) > len(state["messages"])
            # Check that the tool was called with universal parameters
            last_message = result_state["messages"][-1]
            assert "Universal tool resumed" in last_message["content"]
            assert "original_value1" in last_message["content"]
            assert "user provided response" in last_message["content"]

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, test_agent):
        """Test that non-universal tools continue working alongside universal tools."""
        # Non-universal context (legacy pattern)
        legacy_context = {
            "source_tool": "legacy_tool",
            "current_step": "some_step",
            "legacy_data": "preserved"
        }
        
        state = {
            "hitl_phase": "approved",
            "hitl_context": legacy_context
        }
        
        # Should NOT be detected as universal
        is_needed = test_agent._is_tool_managed_collection_needed(legacy_context, state)
        assert is_needed == False  # Because no collection_mode="tool_managed"

    def test_universal_context_structure(self):
        """Test that universal context has the expected minimal structure."""
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param": "value"}
        )
        
        context = control.to_hitl_context()
        
        # Must have exactly these fields for universal detection
        required_fields = {"source_tool", "collection_mode", "original_params"}
        assert set(context.keys()) == required_fields
        
        # Values must be correct
        assert context["source_tool"] == "test_tool"
        assert context["collection_mode"] == "tool_managed"
        assert context["original_params"] == {"param": "value"}


class TestUniversalSystemPerformance:
    """Test performance characteristics of the universal system."""

    @pytest.mark.asyncio
    async def test_minimal_overhead(self):
        """Test that universal decorator adds minimal overhead."""
        import time
        
        # Baseline function
        async def baseline_function(param1: str, param2: int = 10) -> str:
            return f"result: {param1}, {param2}"
        
        # Universal function
        @hitl_recursive_tool
        async def universal_function(param1: str, param2: int = 10) -> str:
            return f"result: {param1}, {param2}"
        
        # Time baseline
        start_time = time.time()
        for _ in range(100):
            await baseline_function("test", 10)
        baseline_time = time.time() - start_time
        
        # Time universal
        start_time = time.time()
        for _ in range(100):
            await universal_function("test", 10)
        universal_time = time.time() - start_time
        
        # Universal should have minimal overhead (< 50% increase)
        overhead_ratio = universal_time / baseline_time
        assert overhead_ratio < 1.5, f"Universal system overhead too high: {overhead_ratio:.2f}x"

    def test_memory_efficiency(self):
        """Test that UniversalHITLControl is memory efficient."""
        import sys
        
        # Create control object
        control = UniversalHITLControl.create(
            source_tool="test_tool",
            original_params={"param1": "value1", "param2": "value2"}
        )
        
        # Should be very small
        size = sys.getsizeof(control)
        assert size < 1000, f"UniversalHITLControl too large: {size} bytes"
        
        # Converting to context should also be efficient
        context = control.to_hitl_context()
        context_size = sys.getsizeof(context)
        assert context_size < 1000, f"Universal context too large: {context_size} bytes"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
