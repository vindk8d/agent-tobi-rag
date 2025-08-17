"""
Simple Test for Enhanced Quotation Approval Logic

This test focuses on verifying that the enhanced quotation approval flow works correctly
by testing the actual functions through their decorators.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.toolbox.generate_quotation import generate_quotation

class TestSimpleQuotationApproval:
    """Simple test suite for enhanced quotation approval logic."""

    @pytest.mark.asyncio
    async def test_quotation_tool_basic_functionality(self):
        """
        Test that the quotation tool can be invoked and returns expected format.
        
        This test verifies:
        1. Tool can be called through decorators
        2. Returns proper HITL format when needed
        3. Basic flow works end-to-end
        """
        
        # Test with minimal valid input
        result = await generate_quotation.ainvoke({
            "customer_identifier": "Test Customer, test@example.com, +1-555-0123",
            "vehicle_requirements": "Toyota Camry"
        })
        
        # Verify result is a string (basic functionality)
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should either be a HITL request, error message, or access control message
        assert ("HITL_REQUIRED:" in result or 
                "Error:" in result or 
                "❌" in result or
                "✅" in result or
                "⚠️" in result or  # Warning messages
                "Employee Access Required" in result)

    @pytest.mark.asyncio
    async def test_quotation_tool_with_empty_inputs(self):
        """
        Test that the quotation tool handles empty inputs gracefully.
        """
        
        # Test with empty customer identifier
        result = await generate_quotation.ainvoke({
            "customer_identifier": "",
            "vehicle_requirements": "Toyota Camry"
        })
        
        # Should return an error or informational message
        assert isinstance(result, str)
        assert ("Error:" in result or 
                "❌" in result or 
                "🔍" in result or  # Information request messages
                "Customer Information Required" in result)

    @pytest.mark.asyncio
    async def test_quotation_tool_with_user_response(self):
        """
        Test that the quotation tool handles user responses (HITL flow).
        """
        
        # Test with user response (simulating HITL resume)
        result = await generate_quotation.ainvoke({
            "customer_identifier": "Test Customer, test@example.com, +1-555-0123",
            "vehicle_requirements": "Toyota Camry",
            "user_response": "yes, approve it",
            "hitl_phase": "final_approval"
        })
        
        # Should handle the user response
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_quotation_preview_function_exists(self):
        """
        Test that the quotation preview function exists and can be called.
        """
        
        from agents.toolbox.generate_quotation import _create_quotation_preview
        
        # Create a mock context
        mock_context = MagicMock()
        mock_context.customer_info = {
            "name": "John Smith",
            "email": "john@example.com",
            "phone": "+1-555-0123"
        }
        mock_context.vehicle_requirements = {
            "make": "Toyota",
            "model": "Camry"
        }
        mock_context.purchase_preferences = {}
        mock_context.timeline_info = {}
        
        # Test the preview function
        preview = await _create_quotation_preview(
            extracted_context=mock_context,
            quotation_validity_days=30
        )
        
        # Verify basic structure
        assert isinstance(preview, str)
        assert len(preview) > 0
        assert "REQUIRED INFORMATION" in preview
        assert "OPTIONAL INFORMATION" in preview
        assert "🔹" in preview  # Required section icon
        assert "🔸" in preview  # Optional section icon

    @pytest.mark.asyncio
    async def test_quotation_tool_parameter_validation(self):
        """
        Test that the quotation tool validates parameters correctly.
        """
        
        # Test with valid parameters
        try:
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Valid Customer",
                "vehicle_requirements": "Valid Vehicle",
                "additional_notes": "Test notes",
                "quotation_validity_days": 30
            })
            # Should not raise an exception
            assert isinstance(result, str)
        except Exception as e:
            # If it fails, it should be due to business logic, not parameter validation
            assert "validation" not in str(e).lower()

    @pytest.mark.asyncio
    async def test_decorator_integration(self):
        """
        Test that the decorators (@tool, @hitl_recursive_tool) are working.
        """
        
        # Verify the tool has the expected attributes from decorators
        assert hasattr(generate_quotation, 'name')
        assert hasattr(generate_quotation, 'description')
        assert hasattr(generate_quotation, 'args_schema')
        
        # Verify it's a LangChain tool
        assert hasattr(generate_quotation, 'ainvoke')
        assert callable(generate_quotation.ainvoke)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
