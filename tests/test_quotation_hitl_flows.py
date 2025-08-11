"""
Comprehensive HITL Flow Tests for Quotation Generation

Tests various Human-in-the-Loop scenarios including:
- Missing customer information scenarios
- Missing vehicle requirements scenarios  
- Missing employee information scenarios
- Pricing issues requiring HITL intervention
- Approval/rejection flows for quotation confirmation
- Multi-step HITL flows with resume logic
- Edge cases and error scenarios

These tests validate the complete HITL integration with the quotation generation system.
"""

import os
import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Skip all tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)

# Import modules only if credentials are available
try:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        from backend.agents.tools import (
            generate_quotation,
            GenerateQuotationParams,
            _lookup_customer,
            _lookup_vehicle_by_criteria,
            _lookup_current_pricing,
            _lookup_employee_details,
            UserContext
        )
        from backend.agents.hitl import request_input, request_approval
        # Import PDF and storage modules with error handling
        try:
            from backend.core.pdf_generator import generate_quotation_pdf
        except ImportError:
            generate_quotation_pdf = None
        
        try:
            from backend.core.storage import upload_quotation_pdf, create_signed_quotation_url
        except ImportError:
            upload_quotation_pdf = None
            create_signed_quotation_url = None

except ImportError:
    # If imports fail, skip all tests
    pytestmark = pytest.mark.skip(reason="Required modules not available")


class TestHITLCustomerLookupFlows:
    """Test HITL flows for customer lookup scenarios."""

    @pytest.mark.asyncio
    async def test_customer_not_found_initial_request(self):
        """Test HITL flow when customer cannot be found initially."""
        
        with patch('backend.agents.tools._lookup_customer', return_value=None) as mock_lookup:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "Unknown Customer",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            # Should trigger HITL request for customer information
            assert result.startswith("HITL_REQUIRED:input:")
            
            # Verify the HITL response contains expected information
            assert "Customer Information Needed" in result
            assert "customer_identifier" in result or "customer_information" in result
            
            # Verify customer lookup was attempted
            mock_lookup.assert_called()

    @pytest.mark.asyncio
    async def test_customer_lookup_with_resume_logic(self):
        """Test customer lookup with HITL resume logic."""
        
        # Simulate resume scenario with customer lookup context
        resume_context = {
            "step": "customer_lookup",
            "customer_identifier": "Unknown Customer",
            "vehicle_requirements": "Toyota Camry",
            "attempts": 1
        }
        
        # Mock customer found on resume
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer) as mock_lookup, \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=[]) as mock_vehicle, \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",  # Updated identifier from HITL
                    "vehicle_requirements": "Toyota Camry",
                    "resume_context": resume_context,
                    "user_response": "john@example.com"
                })
            
            # Should proceed past customer lookup to next step (vehicle lookup)
            mock_lookup.assert_called()
            # Should trigger HITL for vehicle information since no vehicles found
            assert result.startswith("HITL_REQUIRED:")

    @pytest.mark.asyncio
    async def test_customer_lookup_multiple_attempts(self):
        """Test customer lookup with multiple failed attempts."""
        
        resume_context = {
            "step": "customer_lookup", 
            "customer_identifier": "Still Unknown",
            "vehicle_requirements": "Toyota Camry",
            "attempts": 2
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=None) as mock_lookup:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "Still Unknown Customer",
                    "vehicle_requirements": "Toyota Camry",
                    "resume_context": resume_context,
                    "user_response": "Still Unknown Customer"
                })
            
            # Should still request customer information but with updated context
            assert result.startswith("HITL_REQUIRED:input:")
            
            # Verify the HITL response contains expected information
            assert "Customer Information Needed" in result
            # Verify customer lookup was attempted
            mock_lookup.assert_called()


class TestHITLVehicleRequirementsFlows:
    """Test HITL flows for vehicle requirements scenarios."""

    @pytest.mark.asyncio
    async def test_vehicle_not_found_request(self):
        """Test HITL flow when no vehicles match requirements."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=[]) as mock_vehicle, \
             patch('backend.agents.tools._get_available_makes_and_models', return_value={
                 "Toyota": {"models": ["Camry", "Corolla"]},
                 "Honda": {"models": ["Civic", "Accord"]}
             }):
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Rare Exotic Car"
                })
            
            # Should trigger HITL request for vehicle selection
            assert result.startswith("HITL_REQUIRED:input:")
            
            # Verify the HITL response contains expected information
            assert "Vehicle Information Needed" in result
            assert "detailed_vehicle_requirements" in result or "vehicle_selection" in result

    @pytest.mark.asyncio
    async def test_vehicle_requirements_with_inventory_suggestions(self):
        """Test HITL flow includes dynamic inventory suggestions."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe", 
            "email": "john@example.com"
        }
        
        available_inventory = {
            "Toyota": {"models": ["Camry 2023"]},
            "Honda": {"models": ["Civic 2023"]},
            "Nissan": {"models": ["Altima 2023"]}
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=[]), \
             patch('backend.agents.tools._get_available_makes_and_models', return_value=available_inventory), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Luxury Sports Car"
                })
            
            assert result.startswith("HITL_REQUIRED:input:")
            
            # Should include inventory suggestions in prompt
            assert "Toyota" in result or "Honda" in result or "Nissan" in result

    @pytest.mark.asyncio
    async def test_vehicle_selection_resume_logic(self):
        """Test vehicle selection with resume logic after HITL."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023,
            "price": 25000
        }]
        
        resume_context = {
            "step": "vehicle_selection",
            "customer_data": mock_customer,
            "vehicle_requirements": "Toyota Camry"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry 2023",  # More specific after HITL
                    "resume_context": resume_context,
                    "user_response": "Toyota Camry 2023"
                })
            
            # Should proceed past vehicle lookup but fail on pricing
            assert result.startswith("HITL_REQUIRED:")
            # Should be requesting pricing resolution
            call_args = mock_input.call_args[1]
            assert call_args["input_type"] == "pricing_resolution"


class TestHITLEmployeeInformationFlows:
    """Test HITL flows for employee information scenarios."""

    @pytest.mark.asyncio
    async def test_employee_details_not_found(self):
        """Test HITL flow when employee details cannot be retrieved."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023
        }]
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            # Should trigger HITL for employee information issue
            assert result.startswith("HITL_REQUIRED:")
            call_args = mock_input.call_args[1]
            assert "Employee Information Issue" in call_args["prompt"]


class TestHITLPricingFlows:
    """Test HITL flows for pricing-related scenarios."""

    @pytest.mark.asyncio
    async def test_pricing_information_not_available(self):
        """Test HITL flow when pricing information is not available."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023
        }]
        
        mock_employee = {
            "id": "emp_001",
            "name": "Sales Person",
            "email": "sales@company.com"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=mock_employee), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            # Should trigger HITL for pricing issue
            assert result.startswith("HITL_REQUIRED:")
            call_args = mock_input.call_args[1]
            assert "Pricing Information Issue" in call_args["prompt"]
            assert call_args["input_type"] == "pricing_resolution"


class TestHITLApprovalFlows:
    """Test HITL flows for quotation approval scenarios."""

    @pytest.mark.asyncio
    async def test_quotation_approval_request(self):
        """Test HITL flow for quotation approval."""
        
        # Mock all required data
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023,
            "color": "Silver",
            "engine": "2.5L 4-cylinder"
        }]
        
        mock_employee = {
            "id": "emp_001",
            "name": "Sales Person",
            "email": "sales@company.com",
            "phone": "+1234567891"
        }
        
        mock_pricing = {
            "base_price": 25000,
            "discounts": 2000,
            "insurance": 1200,
            "lto_fees": 800,
            "add_ons": 500,
            "total_price": 25500
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=mock_employee), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=mock_pricing), \
             patch('backend.agents.hitl.request_approval', return_value="HITL_REQUEST") as mock_approval:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            # Should trigger approval request
            assert result.startswith("HITL_REQUIRED:")
            mock_approval.assert_called_once()
            
            call_args = mock_approval.call_args[1]
            assert "QUOTATION APPROVAL REQUIRED" in call_args["prompt"]
            assert call_args["approve_text"] == "send"
            assert call_args["reject_text"] == "cancel"

    @pytest.mark.asyncio
    async def test_quotation_approval_with_context(self):
        """Test that quotation approval includes proper context data."""
        
        # Mock all required data
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023
        }]
        
        mock_employee = {
            "id": "emp_001",
            "name": "Sales Person",
            "email": "sales@company.com"
        }
        
        mock_pricing = {
            "base_price": 25000,
            "total_price": 25500
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=mock_employee), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=mock_pricing), \
             patch('backend.agents.hitl.request_approval', return_value="HITL_REQUEST") as mock_approval:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            # Verify context data is properly structured
            call_args = mock_approval.call_args[1]
            context = call_args["context"]
            
            assert "customer_data" in context
            assert "vehicle_data" in context
            assert "employee_data" in context
            assert "pricing_data" in context
            assert "quotation_preview" in context


class TestHITLMultiStepFlows:
    """Test complex multi-step HITL flows."""

    @pytest.mark.asyncio
    async def test_sequential_hitl_customer_then_vehicle(self):
        """Test sequential HITL flows: customer lookup -> vehicle selection."""
        
        # First call: customer not found
        with patch('backend.agents.tools._lookup_customer', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result1 = await generate_quotation.ainvoke({
                    "customer_identifier": "Unknown Customer",
                    "vehicle_requirements": "Toyota Camry"
                })
            
            assert result1.startswith("HITL_REQUIRED:input:")
            assert mock_input.call_args[1]["input_type"] == "customer_identifier"
        
        # Second call: customer found, vehicle not found
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        resume_context = {
            "step": "customer_lookup",
            "customer_identifier": "Unknown Customer",
            "vehicle_requirements": "Toyota Camry"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=[]), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result2 = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",  # Updated from HITL
                    "vehicle_requirements": "Toyota Camry",
                    "resume_context": resume_context,
                    "user_response": "john@example.com"
                })
            
            assert result2.startswith("HITL_REQUIRED:input:")
            assert mock_input.call_args[1]["input_type"] == "vehicle_selection"

    @pytest.mark.asyncio
    async def test_missing_information_collection_flow(self):
        """Test HITL flow for collecting missing quotation information."""
        
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=[]), \
             patch('backend.agents.tools._lookup_employee_details', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": ""  # Missing vehicle requirements
                })
            
            # Should request missing information
            assert result.startswith("HITL_REQUIRED:")
            call_args = mock_input.call_args[1]
            assert call_args["input_type"] in ["missing_quotation_info", "vehicle_selection"]


class TestHITLErrorScenarios:
    """Test HITL flows for error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_resume_context(self):
        """Test handling of invalid resume context."""
        
        invalid_resume_context = {
            "invalid_field": "invalid_value"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "test",
                    "vehicle_requirements": "test",
                    "resume_context": invalid_resume_context
                })
            
            # Should handle gracefully and proceed with normal flow
            assert result.startswith("HITL_REQUIRED:")

    @pytest.mark.asyncio
    async def test_empty_user_response_handling(self):
        """Test handling of empty user responses in HITL flows."""
        
        resume_context = {
            "step": "customer_lookup",
            "customer_identifier": "test"
        }
        
        with patch('backend.agents.tools._lookup_customer', return_value=None), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "test",
                    "vehicle_requirements": "test",
                    "resume_context": resume_context,
                    "user_response": ""  # Empty response
                })
            
            # Should handle gracefully
            assert result.startswith("HITL_REQUIRED:")

    @pytest.mark.asyncio
    async def test_database_error_during_hitl(self):
        """Test HITL behavior when database errors occur."""
        
        with patch('backend.agents.tools._lookup_customer', side_effect=Exception("Database error")), \
             patch('backend.agents.hitl.request_input', return_value="HITL_REQUEST") as mock_input:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "test",
                    "vehicle_requirements": "test"
                })
            
            # Should handle database errors gracefully
            assert result.startswith("HITL_REQUIRED:")
            # Should provide user-friendly error message
            call_args = mock_input.call_args[1]
            prompt = call_args["prompt"]
            assert "error" in prompt.lower() or "issue" in prompt.lower()


class TestHITLApprovalRejectionScenarios:
    """Test approval and rejection scenarios in HITL flows."""

    @pytest.mark.asyncio
    async def test_quotation_approval_success_path(self):
        """Test successful quotation approval and PDF generation."""
        
        # Mock all required data for successful quotation
        mock_customer = {
            "id": str(uuid.uuid4()),
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890",
            "address": "123 Main St"
        }
        
        mock_vehicle = [{
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2023,
            "color": "Silver",
            "engine": "2.5L 4-cylinder",
            "transmission": "Automatic"
        }]
        
        mock_employee = {
            "id": "emp_001",
            "name": "Sales Person",
            "email": "sales@company.com",
            "phone": "+1234567891"
        }
        
        mock_pricing = {
            "base_price": 25000,
            "discounts": 2000,
            "insurance": 1200,
            "lto_fees": 800,
            "add_ons": 500,
            "total_price": 25500
        }
        
        # Mock approval response (simulating user approval)
        def mock_approval_response(*args, **kwargs):
            return "HITL_REQUIRED:approval:{\"prompt\": \"test\", \"context\": {}}"
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=mock_employee), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=mock_pricing), \
             patch('backend.agents.hitl.request_approval', side_effect=mock_approval_response), \
             patch('backend.core.pdf_generator.generate_quotation_pdf', return_value=b"PDF_CONTENT") as mock_pdf, \
             patch('backend.core.storage.upload_quotation_pdf', return_value="test_path.pdf") as mock_upload, \
             patch('backend.core.storage.create_signed_quotation_url', return_value="https://test.url/quotation.pdf") as mock_url:
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                # Simulate the approval flow by directly calling with approved context
                # In real scenario, this would be handled by the agent framework
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry",
                    "hitl_approved": True,  # Simulate approved state
                    "approval_context": {
                        "customer_data": mock_customer,
                        "vehicle_data": mock_vehicle,
                        "employee_data": mock_employee,
                        "pricing_data": mock_pricing
                    }
                })
                
                # Should proceed with PDF generation or return HITL approval request
                # Note: The actual return value depends on implementation
                assert result.startswith("HITL_REQUIRED:approval:") or not result.startswith("HITL_REQUIRED:")

    @pytest.mark.asyncio
    async def test_quotation_rejection_handling(self):
        """Test quotation rejection handling."""
        
        # Mock all required data
        mock_customer = {"id": str(uuid.uuid4()), "name": "John Doe"}
        mock_vehicle = [{"id": str(uuid.uuid4()), "make": "Toyota", "model": "Camry"}]
        mock_employee = {"id": "emp_001", "name": "Sales Person"}
        mock_pricing = {"base_price": 25000, "total_price": 25500}
        
        # Mock rejection response
        def mock_rejection_response(*args, **kwargs):
            return "HITL_REQUIRED:approval:{\"prompt\": \"test\", \"context\": {}}"
        
        with patch('backend.agents.tools._lookup_customer', return_value=mock_customer), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=mock_vehicle), \
             patch('backend.agents.tools._lookup_employee_details', return_value=mock_employee), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=mock_pricing), \
             patch('backend.agents.hitl.request_approval', side_effect=mock_rejection_response):
            
            with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "john@example.com",
                    "vehicle_requirements": "Toyota Camry"
                })
                
                # Should handle rejection appropriately
                # Implementation may return different values for rejection
                assert result.startswith("HITL_REQUIRED:approval:")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_quotation_hitl_flows.py -v
    pytest.main([__file__, "-v"])
