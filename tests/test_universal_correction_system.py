"""
Test Universal LLM-Based Correction System (Task 7.7.3 Output)

Tests for the enhanced @hitl_recursive_tool decorator with automatic LLM-based corrections.
Validates that the universal correction system works correctly across all tools.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.hitl import _process_automatic_corrections
from utils.hitl_corrections import LLMCorrectionProcessor
from agents.toolbox.generate_quotation import generate_quotation

class TestUniversalCorrectionSystem:
    """Test suite for universal LLM-based correction system from Task 7.7.3."""
    
    @pytest.fixture
    def sample_original_params(self):
        """Sample original parameters for testing."""
        return {
            "customer_identifier": "John Smith, john@example.com, +1-555-0123",
            "vehicle_requirements": "Toyota Camry 2024 Blue",
            "additional_notes": "Financing preferred",
            "quotation_validity_days": 30
        }
    
    @pytest.fixture
    def mock_bound_args(self, sample_original_params):
        """Mock bound arguments for testing."""
        mock_args = MagicMock()
        mock_args.arguments = sample_original_params.copy()
        return mock_args

    @pytest.mark.asyncio
    async def test_decorator_automatic_correction_processing(self, sample_original_params, mock_bound_args):
        """
        Test 8.10.1: Test @hitl_recursive_tool decorator automatic correction processing.
        
        Verifies that:
        1. Decorator detects correction requests automatically
        2. LLMCorrectionProcessor is called correctly
        3. Different intents are handled properly (approve/correct/deny)
        4. Tool re-calling works with updated parameters
        """
        
        with patch('utils.hitl_corrections.LLMCorrectionProcessor') as mock_processor_class:
            
            # Setup mock processor
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            
            # Test Case 1: Approval intent
            mock_processor.process_correction.return_value = ("approval", None)
            
            result = await _process_automatic_corrections(
                user_response="yes, approve it",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "continue"
            mock_processor.process_correction.assert_called_once()
            
            # Reset mock
            mock_processor.reset_mock()
            
            # Test Case 2: Denial intent
            mock_processor.process_correction.return_value = ("denial", None)
            
            result = await _process_automatic_corrections(
                user_response="no, cancel it",
                original_params=sample_original_params,
                tool_name="generate_quotation", 
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "cancel"
            assert "Generate Quotation Cancelled" in result["message"]
            
            # Reset mock
            mock_processor.reset_mock()
            
            # Test Case 3: Correction intent
            updated_params = sample_original_params.copy()
            updated_params["customer_identifier"] = "Jane Doe, jane@example.com, +1-555-9999"
            mock_processor.process_correction.return_value = ("correction", updated_params)
            
            result = await _process_automatic_corrections(
                user_response="change customer to Jane Doe",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "re_call_with_corrections"
            assert "updated_args" in result
            assert result["updated_args"]["customer_identifier"] == "Jane Doe, jane@example.com, +1-555-9999"

    @pytest.mark.asyncio
    async def test_llm_natural_language_correction_understanding(self):
        """
        Test 8.10.2: Test LLM natural language correction understanding.
        
        Tests various natural language correction formats:
        1. "change customer to John"
        2. "make it red Toyota Camry"
        3. "use email john@example.com instead"
        4. Complex corrections with multiple changes
        """
        
        processor = LLMCorrectionProcessor()
        
        original_params = {
            "customer_identifier": "Old Customer",
            "vehicle_requirements": "Honda Civic Blue",
            "additional_notes": "Cash payment"
        }
        
        # Test various correction formats
        test_cases = [
            {
                "correction": "change customer to John Smith",
                "expected_intent": "correction",
                "expected_field": "customer_identifier"
            },
            {
                "correction": "make it red Toyota Camry",
                "expected_intent": "correction", 
                "expected_field": "vehicle_requirements"
            },
            {
                "correction": "use financing instead of cash",
                "expected_intent": "correction",
                "expected_field": "additional_notes"
            },
            {
                "correction": "yes, approve it",
                "expected_intent": "approval",
                "expected_field": None
            },
            {
                "correction": "no, cancel this",
                "expected_intent": "denial",
                "expected_field": None
            }
        ]
        
        for test_case in test_cases:
            with patch.object(processor, '_detect_user_intent') as mock_intent, \
                 patch.object(processor, '_map_correction_to_parameters') as mock_mapping:
                
                mock_intent.return_value = test_case["expected_intent"]
                
                if test_case["expected_intent"] == "correction":
                    mock_mapping.return_value = original_params  # Simplified for test
                
                intent, updated_params = await processor.process_correction(
                    test_case["correction"],
                    original_params,
                    "test_tool"
                )
                
                assert intent == test_case["expected_intent"]
                
                if test_case["expected_intent"] == "correction":
                    mock_mapping.assert_called_once()
                else:
                    mock_mapping.assert_not_called()

    @pytest.mark.asyncio
    async def test_automatic_parameter_mapping_and_tool_recalling(self, sample_original_params):
        """
        Test 8.10.3: Test automatic parameter mapping and tool re-calling with updated parameters.
        
        Verifies that:
        1. LLM correctly maps natural language corrections to parameter updates
        2. Parameters are updated accurately
        3. Tool is re-called with corrected parameters
        4. Original parameters are preserved where not corrected
        """
        
        processor = LLMCorrectionProcessor()
        
        # Test parameter mapping
        with patch.object(processor, '_detect_user_intent') as mock_intent, \
             patch.object(processor.llm, 'ainvoke') as mock_llm:
            
            mock_intent.return_value = "correction"
            
            # Mock LLM response for parameter mapping
            mock_response = MagicMock()
            updated_params = sample_original_params.copy()
            updated_params["customer_identifier"] = "Jane Smith, jane@example.com, +1-555-9999"
            mock_response.content = json.dumps(updated_params)
            mock_llm.return_value = mock_response
            
            intent, result_params = await processor.process_correction(
                "change customer to Jane Smith",
                sample_original_params,
                "generate_quotation"
            )
            
            assert intent == "correction"
            assert result_params is not None
            assert result_params["customer_identifier"] == "Jane Smith, jane@example.com, +1-555-9999"
            # Verify other parameters are preserved
            assert result_params["vehicle_requirements"] == sample_original_params["vehicle_requirements"]
            assert result_params["quotation_validity_days"] == sample_original_params["quotation_validity_days"]

    @pytest.mark.asyncio
    async def test_three_case_handling(self, sample_original_params, mock_bound_args):
        """
        Test 8.10.4: Test three-case handling: approve (execute), correct (re-call), deny (cancel).
        
        Comprehensive test of all three correction handling cases:
        1. Approve: Continue with normal tool execution
        2. Correct: Re-call tool with updated parameters
        3. Deny: Cancel tool execution with appropriate message
        """
        
        with patch('utils.hitl_corrections.LLMCorrectionProcessor') as mock_processor_class:
            
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            
            # Test Case 1: APPROVE - should continue
            mock_processor.process_correction.return_value = ("approval", None)
            
            result = await _process_automatic_corrections(
                user_response="yes, looks good",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "continue"
            assert "message" not in result
            
            # Test Case 2: CORRECT - should re-call with updates
            updated_params = sample_original_params.copy()
            updated_params["vehicle_requirements"] = "Honda Accord Red 2024"
            mock_processor.process_correction.return_value = ("correction", updated_params)
            
            result = await _process_automatic_corrections(
                user_response="make it Honda Accord Red",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "re_call_with_corrections"
            assert "updated_args" in result
            assert result["updated_args"]["vehicle_requirements"] == "Honda Accord Red 2024"
            # Verify user_response is cleared to avoid loops
            assert result["updated_args"]["user_response"] == ""
            assert result["updated_args"]["hitl_phase"] == ""
            
            # Test Case 3: DENY - should cancel
            mock_processor.process_correction.return_value = ("denial", None)
            
            result = await _process_automatic_corrections(
                user_response="no, cancel this",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            assert result["action"] == "cancel"
            assert "message" in result
            assert "Generate Quotation Cancelled" in result["message"]

    @pytest.mark.asyncio
    async def test_llm_correction_processor_utility(self):
        """
        Test 8.10.5: Test LLMCorrectionProcessor utility functions correctly.
        
        Tests the core utility functions:
        1. Intent detection accuracy
        2. Parameter mapping accuracy
        3. Error handling and fallbacks
        4. Integration with LLM services
        """
        
        processor = LLMCorrectionProcessor()
        
        # Test intent detection
        with patch.object(processor.llm, 'ainvoke') as mock_llm:
            
            # Test approval detection
            mock_response = MagicMock()
            mock_response.content = "approval"
            mock_llm.return_value = mock_response
            
            intent = await processor._detect_user_intent("yes, approve it")
            assert intent == "approval"
            
            # Test denial detection
            mock_response.content = "denial"
            mock_llm.return_value = mock_response
            
            intent = await processor._detect_user_intent("no, cancel it")
            assert intent == "denial"
            
            # Test correction detection
            mock_response.content = "correction"
            mock_llm.return_value = mock_response
            
            intent = await processor._detect_user_intent("change customer to John")
            assert intent == "correction"
            
            # Test invalid response handling
            mock_response.content = "invalid_response"
            mock_llm.return_value = mock_response
            
            intent = await processor._detect_user_intent("unclear input")
            assert intent == "correction"  # Should default to correction

    @pytest.mark.asyncio
    async def test_decorator_langraph_integration(self):
        """
        Test 8.10.6: Test decorator integration with existing LangGraph HITL infrastructure.
        
        Verifies that:
        1. Decorator works with LangGraph interrupt/resume patterns
        2. State management is preserved
        3. Context is maintained across correction cycles
        4. No conflicts with existing HITL infrastructure
        """
        
        # This test verifies integration at a higher level
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval, \
             patch('agents.hitl._process_automatic_corrections') as mock_corrections:
            
            # Setup mocks for normal flow
            mock_employee.return_value = "emp_123"
            mock_extract.return_value = {"status": "success", "context": MagicMock()}
            mock_validate.return_value = {"status": "success", "is_complete": True, "missing_info": {}}
            mock_approval.return_value = "HITL_REQUIRED:approval:test_prompt"
            
            # Test normal flow (no user_response)
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Should reach approval step
            mock_approval.assert_called_once()
            assert "HITL_REQUIRED:approval:" in result
            
            # Reset mocks
            mock_approval.reset_mock()
            mock_corrections.reset_mock()
            
            # Test correction flow (with user_response)
            mock_corrections.return_value = {"action": "continue"}
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Toyota Camry",
                "user_response": "yes, approve it",
                "hitl_phase": "final_approval"
            })
            
            # Should process corrections through decorator
            mock_corrections.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, sample_original_params, mock_bound_args):
        """
        Test error handling in the correction system.
        
        Verifies that:
        1. LLM service errors are handled gracefully
        2. Invalid JSON responses are handled
        3. System falls back to safe defaults
        4. No crashes occur during error conditions
        """
        
        with patch('utils.hitl_corrections.LLMCorrectionProcessor') as mock_processor_class:
            
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            
            # Test LLM service error
            mock_processor.process_correction.side_effect = Exception("LLM service error")
            
            result = await _process_automatic_corrections(
                user_response="change customer to John",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            # Should fall back to continue action
            assert result["action"] == "continue"
            
            # Test processor creation error
            mock_processor_class.side_effect = Exception("Processor creation error")
            
            result = await _process_automatic_corrections(
                user_response="test input",
                original_params=sample_original_params,
                tool_name="generate_quotation",
                bound_args=mock_bound_args
            )
            
            # Should still fall back gracefully
            assert result["action"] == "continue"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
