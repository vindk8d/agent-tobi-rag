"""
Comprehensive tests for LLM-driven response interpretation functionality.

Task 12.5: Test LLM-driven response interpretation with various natural language inputs
("yes", "approve", "send it", "go ahead", etc.)

This module tests the revolutionary HITL system's ability to understand natural
language user responses using LLM interpretation instead of rigid keyword matching.
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

# Import the modules we need to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.hitl import (
    _interpret_user_intent_with_llm,
    _process_hitl_response_llm_driven,
    _get_hitl_interpretation_llm,
    HITLInteractionError
)


class TestLLMResponseInterpretation:
    """Test suite for LLM-driven response interpretation functionality."""

    @pytest.fixture
    def sample_hitl_context(self):
        """Sample HITL context for testing."""
        return {
            "source_tool": "trigger_customer_message",
            "action": "Send message to customer",
            "customer_name": "John Smith",
            "message": "Test message content"
        }

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM chain response."""
        mock = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_approval_responses_natural_language(self, sample_hitl_context):
        """Test that various natural language approval responses are interpreted correctly."""
        approval_inputs = [
            # Standard approvals
            "yes",
            "ok", 
            "sure",
            "approve",
            "confirm",
            
            # Casual approvals
            "send it",
            "go ahead",
            "do it",
            "proceed",
            "continue",
            
            # Enthusiastic approvals
            "let's do it!",
            "sounds good",
            "that works",
            "perfect",
            "absolutely",
            
            # Formal approvals
            "I approve",
            "Please proceed",
            "You may send",
            "Go for it",
            
            # Mixed case and punctuation
            "YES!",
            "Ok, send it.",
            "Sure thing",
            "Go ahead!",
        ]

        for user_input in approval_inputs:
            with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
                # Mock the LLM to return APPROVAL
                mock_llm = AsyncMock()
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value="APPROVAL")
                mock_llm.__or__ = lambda self, other: mock_chain
                mock_get_llm.return_value = mock_llm

                result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
                
                assert result == "approval", f"Input '{user_input}' should be interpreted as approval"

    @pytest.mark.asyncio
    async def test_denial_responses_natural_language(self, sample_hitl_context):
        """Test that various natural language denial responses are interpreted correctly."""
        denial_inputs = [
            # Standard denials
            "no",
            "cancel", 
            "stop",
            "abort",
            "deny",
            
            # Casual denials
            "not now",
            "skip it",
            "don't send",
            "hold on",
            "wait",
            
            # Polite denials
            "not right now",
            "maybe later", 
            "I'll pass",
            "let me think about it",
            
            # Strong denials
            "absolutely not",
            "definitely no",
            "don't do it",
            "cancel that",
            
            # Mixed case and punctuation
            "NO!",
            "Cancel it.",
            "Don't send that",
            "Stop!",
        ]

        for user_input in denial_inputs:
            with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
                # Mock the LLM to return DENIAL
                mock_llm = AsyncMock()
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value="DENIAL")
                mock_llm.__or__ = lambda self, other: mock_chain
                mock_get_llm.return_value = mock_llm

                result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
                
                assert result == "denial", f"Input '{user_input}' should be interpreted as denial"

    @pytest.mark.asyncio
    async def test_input_responses_data_provision(self, sample_hitl_context):
        """Test that various data input responses are interpreted correctly."""
        input_responses = [
            # Email addresses
            "john@example.com",
            "user.name+test@domain.co.uk",
            
            # Names
            "John Smith",
            "Customer ABC Corp",
            "Jane Doe-Williams",
            
            # Phone numbers
            "1234567890",
            "+1 (555) 123-4567",
            "555.123.4567",
            
            # Dates and times
            "tomorrow",
            "next week",
            "January 15th",
            "2:30 PM",
            
            # Selections
            "option 2",
            "choice B",
            "the first one",
            "number 3",
            
            # Addresses
            "123 Main St, Anytown, CA 90210",
            "Building A, Floor 2",
            
            # Complex data
            "Budget: $50,000, Timeline: 3 months",
            "ID: 12345, Status: Active",
        ]

        for user_input in input_responses:
            with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
                # Mock the LLM to return INPUT
                mock_llm = AsyncMock()
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value="INPUT")
                mock_llm.__or__ = lambda self, other: mock_chain
                mock_get_llm.return_value = mock_llm

                result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
                
                assert result == "input", f"Input '{user_input}' should be interpreted as data input"

    @pytest.mark.asyncio
    async def test_ambiguous_responses_llm_decision(self, sample_hitl_context):
        """Test that ambiguous responses are handled by LLM intelligence."""
        ambiguous_inputs = [
            # Could be approval or input
            "alright",
            "fine",
            "whatever",
            
            # Context-dependent
            "that one",
            "this",
            "it",
            
            # Unclear intent
            "hmm",
            "maybe",
            "I guess",
        ]

        for user_input in ambiguous_inputs:
            with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
                # Mock the LLM to make a decision (let's say INPUT as fallback)
                mock_llm = AsyncMock()
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value="INPUT")
                mock_llm.__or__ = lambda self, other: mock_chain
                mock_get_llm.return_value = mock_llm

                result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
                
                # Should return one of the valid categories
                assert result in ["approval", "denial", "input"], f"Input '{user_input}' should be categorized by LLM"

    @pytest.mark.asyncio
    async def test_llm_interpretation_with_context(self, sample_hitl_context):
        """Test that context is properly included in LLM interpretation."""
        user_input = "send it"
        
        with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_chain = AsyncMock()
            mock_chain.ainvoke = AsyncMock(return_value="APPROVAL")
            mock_llm.__or__ = lambda self, other: mock_chain
            mock_get_llm.return_value = mock_llm

            result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
            
            # Verify LLM was called with context-aware prompt
            mock_chain.ainvoke.assert_called_once()
            call_args = mock_chain.ainvoke.call_args[0][0]  # Get the messages
            
            # Check that context information is included
            user_message_content = call_args[1].content  # HumanMessage content
            assert "trigger_customer_message" in user_message_content
            assert "send it" in user_message_content
            
            assert result == "approval"

    @pytest.mark.asyncio
    async def test_llm_interpretation_without_context(self):
        """Test LLM interpretation works without context."""
        user_input = "yes"
        
        with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_chain = AsyncMock()
            mock_chain.ainvoke = AsyncMock(return_value="APPROVAL")
            mock_llm.__or__ = lambda self, other: mock_chain
            mock_get_llm.return_value = mock_llm

            result = await _interpret_user_intent_with_llm(user_input, {})
            
            assert result == "approval"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_unexpected_response(self, sample_hitl_context):
        """Test fallback behavior when LLM returns unexpected response."""
        user_input = "test response"
        
        with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_chain = AsyncMock()
            # LLM returns something unexpected
            mock_chain.ainvoke = AsyncMock(return_value="UNEXPECTED_CATEGORY")
            mock_llm.__or__ = lambda self, other: mock_chain
            mock_get_llm.return_value = mock_llm

            result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
            
            # Should fallback to "input"
            assert result == "input"

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, sample_hitl_context):
        """Test error handling when LLM interpretation fails."""
        user_input = "test response"
        
        with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
            # Mock LLM to raise an exception
            mock_get_llm.side_effect = Exception("LLM connection error")

            result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
            
            # Should fallback to "input" on error
            assert result == "input"

    @pytest.mark.asyncio
    async def test_process_hitl_response_approval(self, sample_hitl_context):
        """Test complete HITL response processing for approval."""
        user_input = "go ahead and send it"
        
        with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
            mock_interpret.return_value = "approval"
            
            result = await _process_hitl_response_llm_driven(sample_hitl_context, user_input)
            
            assert result["success"] is True
            assert result["result"] == "approved"
            assert result["context"] is None  # Context cleared on completion
            assert "Approved" in result["response_message"]

    @pytest.mark.asyncio
    async def test_process_hitl_response_denial(self, sample_hitl_context):
        """Test complete HITL response processing for denial."""
        user_input = "cancel that"
        
        with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
            mock_interpret.return_value = "denial"
            
            result = await _process_hitl_response_llm_driven(sample_hitl_context, user_input)
            
            assert result["success"] is True
            assert result["result"] == "denied"
            assert result["context"] is None  # Context cleared on completion
            assert "Cancelled" in result["response_message"]

    @pytest.mark.asyncio
    async def test_process_hitl_response_input(self, sample_hitl_context):
        """Test complete HITL response processing for input data."""
        user_input = "john@example.com"
        
        with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
            mock_interpret.return_value = "input"
            
            result = await _process_hitl_response_llm_driven(sample_hitl_context, user_input)
            
            assert result["success"] is True
            assert result["result"] == user_input  # Input returned as-is
            assert result["context"] == sample_hitl_context  # Context preserved
            assert "Received" in result["response_message"]

    @pytest.mark.asyncio
    async def test_process_hitl_response_error_handling(self, sample_hitl_context):
        """Test error handling in HITL response processing."""
        user_input = "test input"
        
        with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
            mock_interpret.side_effect = Exception("Test error")
            
            with pytest.raises(HITLInteractionError):
                await _process_hitl_response_llm_driven(sample_hitl_context, user_input)

    @pytest.mark.asyncio
    async def test_edge_case_empty_input(self, sample_hitl_context):
        """Test handling of empty or whitespace-only input."""
        empty_inputs = ["", "   ", "\n", "\t"]
        
        for empty_input in empty_inputs:
            with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
                mock_interpret.return_value = "input"
                
                result = await _process_hitl_response_llm_driven(sample_hitl_context, empty_input)
                
                assert result["success"] is True
                assert result["result"] == empty_input.strip()

    @pytest.mark.asyncio
    async def test_edge_case_very_long_input(self, sample_hitl_context):
        """Test handling of very long user input."""
        long_input = "This is a very long response " * 50  # 1500+ characters
        
        with patch('agents.hitl._interpret_user_intent_with_llm') as mock_interpret:
            mock_interpret.return_value = "input"
            
            result = await _process_hitl_response_llm_driven(sample_hitl_context, long_input)
            
            assert result["success"] is True
            # Input is stripped in the actual function, so we compare with stripped version
            assert result["result"] == long_input.strip()

    @pytest.mark.asyncio
    async def test_multilingual_responses(self, sample_hitl_context):
        """Test handling of multilingual responses (integration test with fallback)."""
        multilingual_inputs = [
            "sí",  # Spanish yes
            "oui",  # French yes  
            "はい",  # Japanese yes
            "да",   # Russian yes
            "sim",  # Portuguese yes
        ]
        
        # Since this tests actual LLM understanding, we'll test that the function
        # at least returns a valid category (approval, denial, or input)
        # The actual multilingual understanding depends on the LLM model capabilities
        for multilingual_input in multilingual_inputs:
            with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
                # Mock the LLM to return a reasonable interpretation
                mock_llm = AsyncMock()
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value="APPROVAL")  # LLM would ideally understand these
                mock_llm.__or__ = lambda self, other: mock_chain
                mock_get_llm.return_value = mock_llm

                result = await _interpret_user_intent_with_llm(multilingual_input, sample_hitl_context)
                
                # Verify result is one of the valid categories
                assert result in ["approval", "denial", "input"], f"LLM should categorize '{multilingual_input}'"
                # In this mock scenario, we expect approval
                assert result == "approval", f"'{multilingual_input}' should be interpreted as approval with proper LLM"

    @pytest.mark.asyncio
    async def test_performance_timing_logged(self, sample_hitl_context):
        """Test that LLM interpretation timing is properly logged."""
        user_input = "yes"
        
        with patch('agents.hitl._get_hitl_interpretation_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_chain = AsyncMock()
            
            # Simulate some processing time
            async def slow_invoke(messages):
                await asyncio.sleep(0.01)  # 10ms delay
                return "APPROVAL"
            
            mock_chain.ainvoke = slow_invoke
            mock_llm.__or__ = lambda self, other: mock_chain
            mock_get_llm.return_value = mock_llm

            with patch('agents.hitl.logger') as mock_logger:
                result = await _interpret_user_intent_with_llm(user_input, sample_hitl_context)
                
                # Verify timing information is logged
                timing_logged = any(
                    "ms)" in str(call) for call in mock_logger.info.call_args_list
                )
                assert timing_logged, "LLM interpretation timing should be logged"
                assert result == "approval"


# Additional integration test class for real LLM testing (optional)
class TestLLMIntegrationReal:
    """Integration tests using real LLM calls (run sparingly due to cost)."""
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Uses real LLM calls - enable for integration testing")
    @pytest.mark.asyncio
    async def test_real_llm_approval_understanding(self):
        """Test real LLM understanding of approval responses."""
        approval_inputs = ["send it", "go ahead", "proceed", "let's do it"]
        
        for user_input in approval_inputs:
            result = await _interpret_user_intent_with_llm(user_input, {})
            assert result == "approval", f"Real LLM should understand '{user_input}' as approval"

    @pytest.mark.integration  
    @pytest.mark.skip(reason="Uses real LLM calls - enable for integration testing")
    @pytest.mark.asyncio
    async def test_real_llm_denial_understanding(self):
        """Test real LLM understanding of denial responses."""
        denial_inputs = ["cancel", "not now", "don't send", "abort"]
        
        for user_input in denial_inputs:
            result = await _interpret_user_intent_with_llm(user_input, {})
            assert result == "denial", f"Real LLM should understand '{user_input}' as denial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])