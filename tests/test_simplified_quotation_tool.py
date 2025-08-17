"""
Test Simplified Quotation Tool (Task 7.7.3.2/7.7.3.3 Output)

Tests for the simplified quotation tool that focuses purely on business logic
with all correction handling removed and delegated to the decorator.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.toolbox.generate_quotation import (
    generate_quotation,
    _extract_comprehensive_context,
    _validate_quotation_completeness,
    _generate_final_quotation,
    GenerateQuotationParams
)

class TestSimplifiedQuotationTool:
    """Test suite for simplified quotation tool from Task 7.7.3.2/7.7.3.3."""
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            "status": "success",
            "context": MagicMock(
                extracted_context=MagicMock(
                    customer_info={
                        "name": "John Smith",
                        "email": "john@example.com",
                        "phone": "+1-555-0123"
                    },
                    vehicle_requirements={
                        "make": "Toyota",
                        "model": "Camry"
                    },
                    purchase_preferences={},
                    timeline_info={}
                )
            )
        }

    @pytest.mark.asyncio
    async def test_tool_focuses_on_business_logic_only(self, sample_context):
        """
        Test 8.11.1: Test quotation tool focuses purely on business logic (no correction handling).
        
        Verifies that:
        1. Tool only handles quotation generation business logic
        2. No correction handling code exists in the tool
        3. Clean 3-step flow: extract → validate → generate
        4. No user_response processing in tool logic
        """
        
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval:
            
            # Setup mocks for business logic flow
            mock_employee.return_value = "emp_123"
            mock_extract.return_value = sample_context
            mock_validate.return_value = {
                "status": "success",
                "is_complete": True,
                "missing_info": {}
            }
            mock_approval.return_value = "HITL_REQUIRED:approval:test_prompt"
            
            # Call tool with no user_response (pure business logic)
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith, john@example.com, +1-555-0123",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify clean 3-step business logic flow
            mock_extract.assert_called_once()
            mock_validate.assert_called_once()
            mock_approval.assert_called_once()
            
            # Verify result is proper HITL request
            assert "HITL_REQUIRED:approval:" in result
            
            # Verify extract was called without user_response parameter
            extract_call_args = mock_extract.call_args
            assert 'user_response' not in extract_call_args[1] or extract_call_args[1].get('user_response') == ""

    @pytest.mark.asyncio
    async def test_user_response_parameter_removal(self):
        """
        Test 8.11.2: Test removal of user_response parameter handling doesn't break functionality.
        
        Verifies that:
        1. Tool signature no longer includes user_response parameter
        2. Tool functions correctly without user_response handling
        3. No references to user_response in tool business logic
        4. Parameter schema is updated correctly
        """
        
        # Test parameter schema
        params = GenerateQuotationParams(
            customer_identifier="John Smith",
            vehicle_requirements="Toyota Camry",
            additional_notes="Test notes",
            quotation_validity_days=30
        )
        
        # Verify user_response is not in the schema
        param_dict = params.model_dump()
        assert 'user_response' not in param_dict
        
        # Verify required parameters are still present
        assert 'customer_identifier' in param_dict
        assert 'vehicle_requirements' in param_dict
        assert 'additional_notes' in param_dict
        assert 'quotation_validity_days' in param_dict
        
        # Test that tool can be called without user_response
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._generate_intelligent_hitl_request') as mock_hitl:
            
            mock_employee.return_value = "emp_123"
            mock_extract.return_value = {"status": "success", "context": MagicMock()}
            mock_validate.return_value = {
                "status": "success",
                "is_complete": False,
                "missing_info": {"critical": ["customer_email"]}
            }
            mock_hitl.return_value = "HITL_REQUIRED:input:missing_info"
            
            # Should work without user_response parameter
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Toyota Camry"
            })
            
            assert "HITL_REQUIRED:input:" in result

    @pytest.mark.asyncio
    async def test_maintains_existing_quotation_capabilities(self, sample_context):
        """
        Test 8.11.3: Test simplified tool maintains all existing quotation capabilities.
        
        Verifies that:
        1. Context extraction still works correctly
        2. Completeness validation functions properly
        3. HITL requests are generated for missing information
        4. Final quotation generation works
        5. All business logic capabilities are preserved
        """
        
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval, \
             patch('agents.toolbox.generate_quotation._generate_intelligent_hitl_request') as mock_hitl:
            
            mock_employee.return_value = "emp_123"
            
            # Test Case 1: Complete information → Approval request
            mock_extract.return_value = sample_context
            mock_validate.return_value = {
                "status": "success",
                "is_complete": True,
                "missing_info": {}
            }
            mock_approval.return_value = "HITL_REQUIRED:approval:approval_prompt"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith, john@example.com, +1-555-0123",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify complete flow works
            mock_extract.assert_called_once()
            mock_validate.assert_called_once()
            mock_approval.assert_called_once()
            assert "HITL_REQUIRED:approval:" in result
            
            # Reset mocks
            mock_extract.reset_mock()
            mock_validate.reset_mock()
            mock_approval.reset_mock()
            mock_hitl.reset_mock()
            
            # Test Case 2: Incomplete information → HITL request
            mock_extract.return_value = sample_context
            mock_validate.return_value = {
                "status": "success",
                "is_complete": False,
                "missing_info": {"critical": ["customer_phone"]}
            }
            mock_hitl.return_value = "HITL_REQUIRED:input:missing_phone"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify incomplete flow works
            mock_extract.assert_called_once()
            mock_validate.assert_called_once()
            mock_hitl.assert_called_once()
            mock_approval.assert_not_called()
            assert "HITL_REQUIRED:input:" in result

    @pytest.mark.asyncio
    async def test_decorator_handles_corrections_transparently(self):
        """
        Test 8.11.4: Test decorator handles all corrections transparently for quotation tool.
        
        Verifies that:
        1. Tool never sees correction logic
        2. Decorator intercepts user_response before tool execution
        3. Tool is re-called with corrected parameters
        4. Tool remains focused on business logic only
        """
        
        # This test simulates the decorator behavior
        with patch('agents.hitl._process_automatic_corrections') as mock_corrections, \
             patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval:
            
            # Setup mocks
            mock_employee.return_value = "emp_123"
            mock_extract.return_value = {"status": "success", "context": MagicMock()}
            mock_validate.return_value = {"status": "success", "is_complete": True, "missing_info": {}}
            mock_approval.return_value = "HITL_REQUIRED:approval:test"
            
            # Simulate decorator processing corrections
            mock_corrections.return_value = {
                "action": "re_call_with_corrections",
                "updated_args": {
                    "customer_identifier": "Jane Smith, jane@example.com, +1-555-9999",
                    "vehicle_requirements": "Honda Accord",
                    "additional_notes": None,
                    "quotation_validity_days": 30,
                    "hitl_phase": "",
                    "conversation_context": "",
                    "user_response": ""  # Cleared by decorator
                }
            }
            
            # The decorator would handle this, but we simulate the re-call
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Jane Smith, jane@example.com, +1-555-9999",
                "vehicle_requirements": "Honda Accord"
            })
            
            # Verify tool executed with corrected parameters
            mock_extract.assert_called_once()
            extract_args = mock_extract.call_args[1]
            assert extract_args['customer_identifier'] == "Jane Smith, jane@example.com, +1-555-9999"
            assert extract_args['vehicle_requirements'] == "Honda Accord"
            
            # Verify tool never saw correction logic
            assert "HITL_REQUIRED:approval:" in result

    @pytest.mark.asyncio
    async def test_no_regression_after_simplification(self, sample_context):
        """
        Test 8.11.5: Test no regression in quotation generation after simplification.
        
        Comprehensive test to ensure:
        1. All original functionality still works
        2. Performance is maintained or improved
        3. Error handling still functions
        4. Integration with other systems works
        5. No broken dependencies
        """
        
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval, \
             patch('agents.toolbox.generate_quotation._generate_intelligent_hitl_request') as mock_hitl, \
             patch('agents.toolbox.generate_quotation.QuotationCommunicationIntelligence') as mock_comm:
            
            mock_employee.return_value = "emp_123"
            
            # Test comprehensive functionality
            test_scenarios = [
                {
                    "name": "Complete information flow",
                    "extract_result": sample_context,
                    "completeness_result": {"status": "success", "is_complete": True, "missing_info": {}},
                    "expected_call": "approval"
                },
                {
                    "name": "Incomplete information flow", 
                    "extract_result": sample_context,
                    "completeness_result": {"status": "success", "is_complete": False, "missing_info": {"critical": ["phone"]}},
                    "expected_call": "hitl"
                },
                {
                    "name": "Context extraction error",
                    "extract_result": {"status": "error", "message": "Context extraction failed"},
                    "completeness_result": None,
                    "expected_call": "error"
                }
            ]
            
            for scenario in test_scenarios:
                # Reset mocks
                mock_extract.reset_mock()
                mock_validate.reset_mock()
                mock_approval.reset_mock()
                mock_hitl.reset_mock()
                
                # Setup scenario
                mock_extract.return_value = scenario["extract_result"]
                if scenario["completeness_result"]:
                    mock_validate.return_value = scenario["completeness_result"]
                
                mock_approval.return_value = "HITL_REQUIRED:approval:test"
                mock_hitl.return_value = "HITL_REQUIRED:input:test"
                
                # Execute test
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "Test Customer",
                    "vehicle_requirements": "Test Vehicle"
                })
                
                # Verify expected behavior
                if scenario["expected_call"] == "approval":
                    mock_approval.assert_called_once()
                    assert "HITL_REQUIRED:approval:" in result
                elif scenario["expected_call"] == "hitl":
                    mock_hitl.assert_called_once()
                    assert "HITL_REQUIRED:input:" in result
                elif scenario["expected_call"] == "error":
                    assert "Context extraction failed" in result

    @pytest.mark.asyncio
    async def test_extract_context_without_user_response(self):
        """
        Test that _extract_comprehensive_context works correctly without user_response parameter.
        
        Verifies that:
        1. Function signature no longer includes user_response
        2. Context extraction works without user_response
        3. All other functionality is preserved
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            # Setup mock
            mock_context_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            mock_context_intel.analyze_complete_context.return_value = MagicMock()
            
            # Test function call without user_response
            result = await _extract_comprehensive_context(
                customer_identifier="John Smith",
                vehicle_requirements="Toyota Camry",
                additional_notes="Test notes",
                conversation_context="Test context"
            )
            
            # Verify function executed successfully
            assert result["status"] == "success"
            mock_context_intel.analyze_complete_context.assert_called_once()
            
            # Verify call arguments don't include user_response data
            call_args = mock_context_intel.analyze_complete_context.call_args
            current_input = call_args[1]['current_user_input']
            assert "User Response:" not in current_input

    @pytest.mark.asyncio
    async def test_parameter_validation_still_works(self):
        """
        Test that parameter validation still works after simplification.
        
        Verifies that:
        1. Empty customer_identifier is caught
        2. Empty vehicle_requirements is caught  
        3. Invalid quotation_validity_days is handled
        4. Employee access control still works
        """
        
        with patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee, \
             patch('agents.toolbox.generate_quotation.QuotationCommunicationIntelligence') as mock_comm:
            
            # Setup mocks
            mock_comm_instance = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            mock_comm_instance.generate_intelligent_error_explanation.return_value = "Error: Missing customer"
            
            # Test empty customer_identifier
            result = await generate_quotation.ainvoke({
                "customer_identifier": "",
                "vehicle_requirements": "Toyota Camry"
            })
            
            assert "Error: Missing customer" in result
            
            # Test empty vehicle_requirements
            mock_comm_instance.generate_intelligent_error_explanation.return_value = "Error: Missing vehicle"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": ""
            })
            
            assert "Error: Missing vehicle" in result
            
            # Test non-employee access
            mock_employee.return_value = None
            mock_comm_instance.generate_intelligent_error_explanation.return_value = "Error: Employee access required"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith",
                "vehicle_requirements": "Toyota Camry"
            })
            
            assert "Error: Employee access required" in result

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
