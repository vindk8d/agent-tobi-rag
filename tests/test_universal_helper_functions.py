"""
Test Universal Helper Functions Still Work (Existing Functions)

Tests to ensure that all existing helper functions continue to work correctly
after the simplification and enhancement changes.
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
    _extract_comprehensive_context,
    _validate_quotation_completeness,
    _generate_intelligent_hitl_request,
    _generate_final_quotation,
    _create_quotation_preview
)

class TestUniversalHelperFunctions:
    """Test suite for universal helper functions from existing functionality."""
    
    @pytest.fixture
    def sample_context_result(self):
        """Sample context analysis result for testing."""
        return MagicMock(
            customer_info={
                "name": "John Smith",
                "email": "john@example.com",
                "phone": "+1-555-0123",
                "company": "ABC Corp",
                "address": "123 Main St"
            },
            vehicle_requirements={
                "make": "Toyota",
                "model": "Camry",
                "year": "2024",
                "color": "Blue",
                "type": "Sedan"
            },
            purchase_preferences={
                "financing": "Cash",
                "trade_in": "None",
                "payment_method": "Bank transfer"
            },
            timeline_info={
                "urgency": "Within 2 weeks",
                "decision_timeline": "This month"
            }
        )

    @pytest.mark.asyncio
    async def test_extract_comprehensive_context_without_user_response(self):
        """
        Test 8.12.1: Test _extract_comprehensive_context() works without user_response parameter.
        
        Verifies that:
        1. Function works correctly without user_response parameter
        2. Context extraction logic is preserved
        3. All other parameters are processed correctly
        4. QuotationContextIntelligence integration works
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            # Setup mock
            mock_context_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            mock_context_intel.analyze_complete_context.return_value = MagicMock()
            
            # Test function call without user_response
            result = await _extract_comprehensive_context(
                customer_identifier="John Smith, john@example.com, +1-555-0123",
                vehicle_requirements="Toyota Camry 2024 Blue Sedan",
                additional_notes="Financing preferred",
                conversation_context="Previous conversation about vehicle needs"
            )
            
            # Verify successful execution
            assert result["status"] == "success"
            assert "context" in result
            
            # Verify QuotationContextIntelligence was called correctly
            mock_context_intel.analyze_complete_context.assert_called_once()
            call_args = mock_context_intel.analyze_complete_context.call_args[1]
            
            # Verify all parameters are included
            assert call_args["customer_identifier"] == "John Smith, john@example.com, +1-555-0123"
            assert call_args["vehicle_requirements"] == "Toyota Camry 2024 Blue Sedan"
            assert call_args["additional_notes"] == "Financing preferred"
            assert call_args["conversation_context"] == "Previous conversation about vehicle needs"
            
            # Verify current_user_input is properly formatted without user_response
            current_input = call_args["current_user_input"]
            assert "Customer: John Smith, john@example.com, +1-555-0123" in current_input
            assert "Vehicle: Toyota Camry 2024 Blue Sedan" in current_input
            assert "Notes: Financing preferred" in current_input
            assert "User Response:" not in current_input  # Should not include user_response section

    @pytest.mark.asyncio
    async def test_validate_quotation_completeness_functions_correctly(self, sample_context_result):
        """
        Test 8.12.2: Test _validate_quotation_completeness() functions correctly.
        
        Verifies that:
        1. Completeness validation logic works correctly
        2. Required fields are properly identified
        3. Missing information is correctly detected
        4. Completeness status is accurate
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            mock_completeness_intel = AsyncMock()
            mock_intelligence.return_value = mock_completeness_intel
            
            # Test Case 1: Complete information
            mock_completeness_intel.assess_quotation_completeness.return_value = {
                "is_complete": True,
                "missing_info": {},
                "completeness_score": 95,
                "required_fields_status": "complete"
            }
            
            result = await _validate_quotation_completeness(sample_context_result)
            
            assert result["status"] == "success"
            assert result["is_complete"] is True
            assert result["missing_info"] == {}
            
            # Verify intelligence was called correctly
            mock_completeness_intel.assess_quotation_completeness.assert_called_once_with(sample_context_result)
            
            # Reset mock
            mock_completeness_intel.reset_mock()
            
            # Test Case 2: Incomplete information
            mock_completeness_intel.assess_quotation_completeness.return_value = {
                "is_complete": False,
                "missing_info": {
                    "critical": ["customer_phone", "vehicle_model"],
                    "optional": ["customer_address"]
                },
                "completeness_score": 60,
                "required_fields_status": "incomplete"
            }
            
            result = await _validate_quotation_completeness(sample_context_result)
            
            assert result["status"] == "success"
            assert result["is_complete"] is False
            assert "critical" in result["missing_info"]
            assert "customer_phone" in result["missing_info"]["critical"]
            assert "vehicle_model" in result["missing_info"]["critical"]

    @pytest.mark.asyncio
    async def test_generate_intelligent_hitl_request_works(self, sample_context_result):
        """
        Test 8.12.3: Test _generate_intelligent_hitl_request() still works for missing info.
        
        Verifies that:
        1. HITL request generation works correctly
        2. Missing information is properly communicated
        3. Context is preserved for resume
        4. Request format is correct
        """
        
        missing_info = {
            "critical": ["customer_phone", "vehicle_model"],
            "optional": ["customer_address", "vehicle_year"]
        }
        
        with patch('agents.toolbox.generate_quotation.QuotationCommunicationIntelligence') as mock_comm_intel, \
             patch('agents.hitl.request_input') as mock_request_input:
            
            # Setup mocks
            mock_comm_instance = AsyncMock()
            mock_comm_intel.return_value = mock_comm_instance
            mock_comm_instance.generate_intelligent_missing_info_request.return_value = "Please provide missing customer phone and vehicle model."
            mock_request_input.return_value = "HITL_REQUIRED:input:missing_info_context"
            
            result = await _generate_intelligent_hitl_request(
                extracted_context=sample_context_result,
                missing_info=missing_info,
                customer_identifier="John Smith",
                vehicle_requirements="Toyota",
                additional_notes="Test notes",
                conversation_context="Test context",
                quotation_validity_days=30
            )
            
            # Verify HITL request format
            assert result.startswith("HITL_REQUIRED:input:")
            
            # Verify communication intelligence was used
            mock_comm_instance.generate_intelligent_missing_info_request.assert_called_once()
            
            # Verify request_input was called with proper context
            mock_request_input.assert_called_once()
            call_args = mock_request_input.call_args[1]
            
            assert "prompt" in call_args
            assert "context" in call_args
            
            # Verify context preservation
            context = call_args["context"]
            assert context["tool"] == "generate_quotation"
            assert context["step"] == "missing_info"
            assert "extracted_context" in context
            assert context["missing_info"] == missing_info
            assert context["customer_identifier"] == "John Smith"
            assert context["vehicle_requirements"] == "Toyota"

    @pytest.mark.asyncio
    async def test_generate_final_quotation_produces_correct_pdf(self, sample_context_result):
        """
        Test 8.12.4: Test _generate_final_quotation() produces correct PDF output.
        
        Verifies that:
        1. PDF generation process works correctly
        2. Context data is properly used
        3. Storage and signed URL generation works
        4. Success message is properly formatted
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationGenerator') as mock_generator, \
             patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee:
            
            # Setup mocks
            mock_employee.return_value = "emp_123"
            mock_gen_instance = AsyncMock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_quotation_pdf.return_value = {
                "status": "success",
                "pdf_url": "https://storage.example.com/quotation_123.pdf",
                "quotation_id": "QUO-2024-001",
                "message": "Quotation generated successfully"
            }
            
            result = await _generate_final_quotation(
                extracted_context=sample_context_result,
                customer_identifier="John Smith, john@example.com, +1-555-0123",
                vehicle_requirements="Toyota Camry 2024",
                additional_notes="Cash payment preferred",
                conversation_context="Customer interested in family sedan",
                quotation_validity_days=30
            )
            
            # Verify PDF generation was called
            mock_gen_instance.generate_quotation_pdf.assert_called_once()
            
            # Verify call parameters
            call_args = mock_gen_instance.generate_quotation_pdf.call_args[1]
            assert call_args["extracted_context"] == sample_context_result
            assert call_args["employee_id"] == "emp_123"
            assert call_args["quotation_validity_days"] == 30
            
            # Verify result format
            assert "✅ **Quotation Generated Successfully!**" in result
            assert "https://storage.example.com/quotation_123.pdf" in result
            assert "QUO-2024-001" in result

    @pytest.mark.asyncio
    async def test_create_quotation_preview_displays_correctly(self, sample_context_result):
        """
        Test 8.12.5: Test _create_quotation_preview() displays required/optional info correctly.
        
        Verifies that:
        1. Preview generation works correctly
        2. Required and optional sections are properly formatted
        3. Visual distinction is maintained
        4. All available information is displayed
        """
        
        preview = await _create_quotation_preview(
            extracted_context=sample_context_result,
            quotation_validity_days=30
        )
        
        # Verify basic structure
        assert isinstance(preview, str)
        assert len(preview) > 0
        
        # Verify required information section
        assert "🔹 **REQUIRED INFORMATION**" in preview
        assert "**Customer Contact:**" in preview
        assert "John Smith" in preview
        assert "john@example.com" in preview
        assert "+1-555-0123" in preview
        assert "**Vehicle Basics:**" in preview
        assert "Toyota" in preview
        assert "Camry" in preview
        
        # Verify optional information section
        assert "🔸 **OPTIONAL INFORMATION**" in preview
        assert "**Customer Details:**" in preview
        assert "ABC Corp" in preview
        assert "123 Main St" in preview
        assert "**Vehicle Details:**" in preview
        assert "2024" in preview
        assert "Blue" in preview
        assert "Sedan" in preview
        assert "**Purchase Preferences:**" in preview
        assert "Cash" in preview
        assert "**Timeline Information:**" in preview
        assert "Within 2 weeks" in preview
        
        # Verify quotation validity
        assert "**Quotation Validity:** 30 days" in preview
        
        # Verify visual separators
        assert "─" * 50 in preview

    @pytest.mark.asyncio
    async def test_helper_functions_error_handling(self):
        """
        Test error handling in helper functions.
        
        Verifies that:
        1. Functions handle errors gracefully
        2. Appropriate error messages are returned
        3. No crashes occur during error conditions
        4. Error context is preserved
        """
        
        # Test _extract_comprehensive_context error handling
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            mock_context_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            mock_context_intel.analyze_complete_context.side_effect = Exception("Context analysis failed")
            
            result = await _extract_comprehensive_context(
                customer_identifier="Test Customer",
                vehicle_requirements="Test Vehicle"
            )
            
            assert result["status"] == "error"
            assert "Context analysis failed" in result["message"]
        
        # Test _validate_quotation_completeness error handling
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            mock_completeness_intel = AsyncMock()
            mock_intelligence.return_value = mock_completeness_intel
            mock_completeness_intel.assess_quotation_completeness.side_effect = Exception("Completeness check failed")
            
            result = await _validate_quotation_completeness(MagicMock())
            
            assert result["status"] == "error"
            assert "Completeness check failed" in result["message"]

    @pytest.mark.asyncio
    async def test_helper_functions_integration(self, sample_context_result):
        """
        Test integration between helper functions.
        
        Verifies that:
        1. Functions work together correctly
        2. Data flows properly between functions
        3. Context is preserved across function calls
        4. No data corruption occurs
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            # Setup mocks for integration test
            mock_context_intel = AsyncMock()
            mock_completeness_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            
            # Mock context extraction
            mock_context_intel.analyze_complete_context.return_value = sample_context_result
            
            # Test context extraction
            context_result = await _extract_comprehensive_context(
                customer_identifier="John Smith, john@example.com, +1-555-0123",
                vehicle_requirements="Toyota Camry 2024"
            )
            
            assert context_result["status"] == "success"
            extracted_context = context_result["context"]
            
            # Test completeness validation with extracted context
            mock_intelligence.return_value = mock_completeness_intel
            mock_completeness_intel.assess_quotation_completeness.return_value = {
                "is_complete": True,
                "missing_info": {},
                "completeness_score": 95
            }
            
            completeness_result = await _validate_quotation_completeness(extracted_context)
            
            assert completeness_result["status"] == "success"
            assert completeness_result["is_complete"] is True
            
            # Test preview generation with extracted context
            preview = await _create_quotation_preview(
                extracted_context=extracted_context,
                quotation_validity_days=30
            )
            
            assert "John Smith" in preview
            assert "Toyota" in preview
            assert "Camry" in preview

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
