"""
Simple Test for Universal Helper Functions

This test focuses on verifying that the helper functions work correctly
without complex mocking that doesn't match the actual implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.toolbox.generate_quotation import (
    _create_quotation_preview,
    _extract_comprehensive_context,
    _validate_quotation_completeness
)

class TestSimpleHelperFunctions:
    """Simple test suite for universal helper functions."""

    @pytest.mark.asyncio
    async def test_create_quotation_preview_basic_functionality(self):
        """
        Test that _create_quotation_preview works with basic mock data.
        
        This test verifies:
        1. Function can be called without errors
        2. Returns a string with expected structure
        3. Contains required and optional information sections
        """
        
        # Create a mock context with the correct structure
        mock_context = MagicMock()
        mock_extracted_context = MagicMock()
        mock_context.extracted_context = mock_extracted_context
        
        # Mock the customer_info as a dictionary
        mock_extracted_context.customer_info = {
            "name": "John Smith",
            "email": "john@example.com", 
            "phone": "+1-555-0123",
            "company": "ABC Corp",
            "address": "123 Main St"
        }
        
        # Mock the vehicle_requirements as a dictionary
        mock_extracted_context.vehicle_requirements = {
            "make": "Toyota",
            "model": "Camry",
            "year": "2024",
            "color": "Blue",
            "type": "Sedan"
        }
        
        # Mock other sections
        mock_extracted_context.purchase_preferences = {
            "financing": "Cash",
            "payment_method": "Bank transfer"
        }
        
        mock_extracted_context.timeline_info = {
            "urgency": "Within 2 weeks"
        }
        
        # Test the preview function
        preview = await _create_quotation_preview(
            extracted_context=mock_context,
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
        
        # Verify quotation validity
        assert "**Quotation Validity:** 30 days" in preview
        
        # Verify visual separators
        assert "─" * 50 in preview

    @pytest.mark.asyncio
    async def test_create_quotation_preview_with_missing_optional_data(self):
        """
        Test preview generation when some optional information is missing.
        """
        
        # Create context with minimal required data
        mock_context = MagicMock()
        mock_extracted_context = MagicMock()
        mock_context.extracted_context = mock_extracted_context
        
        mock_extracted_context.customer_info = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+1-555-9999"
            # Missing company and address
        }
        mock_extracted_context.vehicle_requirements = {
            "make": "Honda",
            "model": "Civic"
            # Missing year, color, type
        }
        mock_extracted_context.purchase_preferences = {}  # Empty
        mock_extracted_context.timeline_info = {}  # Empty
        
        preview = await _create_quotation_preview(
            extracted_context=mock_context,
            quotation_validity_days=45
        )
        
        # Verify required information is still displayed
        assert "🔹 **REQUIRED INFORMATION**" in preview
        assert "Jane Doe" in preview
        assert "jane@example.com" in preview
        assert "Honda" in preview
        assert "Civic" in preview
        
        # Verify optional section exists but may be minimal
        assert "🔸 **OPTIONAL INFORMATION**" in preview
        
        # Verify custom validity period
        assert "**Quotation Validity:** 45 days" in preview

    @pytest.mark.asyncio
    async def test_extract_comprehensive_context_basic_call(self):
        """
        Test that _extract_comprehensive_context can be called without user_response.
        
        This test verifies:
        1. Function can be called with basic parameters
        2. Doesn't require user_response parameter
        3. Returns expected structure
        """
        
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            # Setup mock to return a simple success response
            mock_context_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            
            # Mock the analyze_complete_context method
            mock_result = MagicMock()
            mock_context_intel.analyze_complete_context.return_value = mock_result
            
            # Test function call without user_response
            result = await _extract_comprehensive_context(
                customer_identifier="John Smith, john@example.com, +1-555-0123",
                vehicle_requirements="Toyota Camry 2024",
                additional_notes="Test notes",
                conversation_context="Test context"
            )
            
            # Verify function executed successfully
            assert result["status"] == "success"
            assert "context" in result
            
            # Verify QuotationContextIntelligence was called
            mock_context_intel.analyze_complete_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_quotation_completeness_basic_call(self):
        """
        Test that _validate_quotation_completeness can be called.
        
        This test verifies:
        1. Function can be called with mock context
        2. Returns expected structure
        3. Handles completeness assessment
        """
        
        # Create a mock context with the expected structure
        mock_context = MagicMock()
        
        # Mock the completeness assessment attributes
        mock_completeness_assessment = MagicMock()
        mock_completeness_assessment.quotation_ready = True
        mock_context.completeness_assessment = mock_completeness_assessment
        
        # Mock missing info analysis
        mock_missing_info = MagicMock()
        mock_missing_info.critical_missing = []
        mock_missing_info.optional_missing = []
        mock_context.missing_info_analysis = mock_missing_info
        
        # Mock business recommendations
        mock_context.business_recommendations = MagicMock()
        
        result = await _validate_quotation_completeness(mock_context)
        
        # Verify function executed successfully
        assert result["status"] == "success"
        assert "is_complete" in result
        assert "missing_info" in result
        
        # Verify the completeness result
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_helper_functions_error_handling(self):
        """
        Test that helper functions handle errors gracefully.
        """
        
        # Test _extract_comprehensive_context error handling
        with patch('agents.toolbox.generate_quotation.QuotationContextIntelligence') as mock_intelligence:
            
            mock_context_intel = AsyncMock()
            mock_intelligence.return_value = mock_context_intel
            mock_context_intel.analyze_complete_context.side_effect = Exception("Test error")
            
            result = await _extract_comprehensive_context(
                customer_identifier="Test Customer",
                vehicle_requirements="Test Vehicle"
            )
            
            # Should handle error gracefully
            assert result["status"] == "error"
            assert "Test error" in result["message"]

    @pytest.mark.asyncio
    async def test_preview_function_with_different_validity_periods(self):
        """
        Test preview function with different quotation validity periods.
        """
        
        mock_context = MagicMock()
        mock_extracted_context = MagicMock()
        mock_context.extracted_context = mock_extracted_context
        
        mock_extracted_context.customer_info = {"name": "Test Customer", "email": "test@example.com", "phone": "+1-555-0000"}
        mock_extracted_context.vehicle_requirements = {"make": "Test", "model": "Vehicle"}
        mock_extracted_context.purchase_preferences = {}
        mock_extracted_context.timeline_info = {}
        
        # Test different validity periods
        for days in [7, 15, 30, 60, 90]:
            preview = await _create_quotation_preview(
                extracted_context=mock_context,
                quotation_validity_days=days
            )
            
            assert f"**Quotation Validity:** {days} days" in preview

    @pytest.mark.asyncio
    async def test_functions_work_independently(self):
        """
        Test that helper functions can work independently without complex dependencies.
        """
        
        # Test that _create_quotation_preview works with minimal mock data
        simple_context = MagicMock()
        mock_extracted_context = MagicMock()
        simple_context.extracted_context = mock_extracted_context
        
        mock_extracted_context.customer_info = {"name": "Simple Test"}
        mock_extracted_context.vehicle_requirements = {"make": "Test"}
        mock_extracted_context.purchase_preferences = {}
        mock_extracted_context.timeline_info = {}
        
        preview = await _create_quotation_preview(simple_context, 30)
        
        # Should work without errors
        assert isinstance(preview, str)
        assert "Simple Test" in preview
        assert "Test" in preview
        assert "30 days" in preview

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
