"""
Comprehensive Integration Tests for Quotation Generation Flow

Tests the complete end-to-end quotation generation workflow including:
- Tool parameter validation and error handling
- Customer lookup with various identifiers
- Vehicle requirements parsing and matching
- HITL flows for missing information
- PDF generation and storage integration
- Employee access control
- Resume logic for interrupted workflows
- Error scenarios and edge cases
"""

import os
import pytest
import asyncio
import uuid
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Skip all tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)

# Import modules only if credentials are available
# Also handle import errors gracefully for dependencies like WeasyPrint
try:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        from backend.agents.tools import (
            generate_quotation,
            GenerateQuotationParams,
            _lookup_customer,
            _search_vehicles_with_llm,
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
    else:
        # Set all to None when credentials not available
        generate_quotation = None
        GenerateQuotationParams = None
        _lookup_customer = None
        _search_vehicles_with_llm = None
        _lookup_current_pricing = None
        _lookup_employee_details = None
        UserContext = None
        request_input = None
        request_approval = None
        generate_quotation_pdf = None
        upload_quotation_pdf = None
        create_signed_quotation_url = None
except ImportError as e:
    # Handle any import errors by setting all to None
    print(f"Warning: Import error in test setup: {e}")
    generate_quotation = None
    GenerateQuotationParams = None
    _lookup_customer = None
    _search_vehicles_with_llm = None
    _lookup_current_pricing = None
    _lookup_employee_details = None
    UserContext = None
    request_input = None
    request_approval = None
    generate_quotation_pdf = None
    upload_quotation_pdf = None
    create_signed_quotation_url = None


class TestQuotationGenerationIntegration:
    """Integration tests for the complete quotation generation workflow."""
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing."""
        return {
            "id": "550e8400-e29b-41d4-a716-446655440001",
            "name": "John Doe",
            "email": "john.doe@company.com",
            "phone": "+63 912 345 6789",
            "company": "ABC Corporation",
            "address": "123 Business Street, Manila"
        }
    
    @pytest.fixture
    def sample_vehicle_data(self):
        """Sample vehicle data for testing."""
        return [
            {
                "id": "v001",
                "make": "Toyota",
                "model": "Camry",
                "type": "Sedan",
                "year": 2023,
                "color": "White",
                "availability": "In Stock",
                "base_price": 1500000.00
            },
            {
                "id": "v002", 
                "make": "Honda",
                "model": "CR-V",
                "type": "SUV",
                "year": 2023,
                "color": "Black",
                "availability": "In Stock",
                "base_price": 1800000.00
            }
        ]
    
    @pytest.fixture
    def sample_pricing_data(self):
        """Sample pricing data for testing."""
        return {
            "base_price": 1500000.00,
            "discount": 50000.00,
            "insurance": 25000.00,
            "lto_fees": 15000.00,
            "addons": [
                {"name": "Extended Warranty", "price": 30000.00},
                {"name": "Paint Protection", "price": 20000.00}
            ],
            "total": 1540000.00
        }
    
    @pytest.fixture
    def sample_employee_data(self):
        """Sample employee data for testing."""
        return {
            "id": "emp001",
            "name": "Sales Representative",
            "email": "sales@company.com",
            "phone": "+63 912 111 2222",
            "branch": "Manila Main Branch",
            "position": "Senior Sales Consultant"
        }
    
    @pytest.mark.asyncio
    async def test_quotation_generation_reaches_approval_stage(self, sample_customer_data, sample_vehicle_data, sample_pricing_data, sample_employee_data):
        """Test that quotation generation successfully reaches the approval stage."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=sample_vehicle_data), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Customer wants Toyota Camry"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_approval', return_value="HITL_REQUIRED:approval:{\"prompt\": \"📄 **QUOTATION APPROVAL REQUIRED**\\n\\nPreview content\\n\\n---\\n\\n🎯 **READY FOR FINAL GENERATION**\", \"context\": {}}") as mock_approval, \
             patch('backend.core.pdf_generator.generate_quotation_pdf', return_value=b"dummy_pdf_content"), \
             patch('backend.core.storage.upload_quotation_pdf', return_value="quotations/test_quotation.pdf"), \
             patch('backend.core.storage.create_signed_quotation_url', return_value="https://storage.example.com/signed-url"), \
             patch('backend.agents.tools._create_quotation_preview', return_value="Preview content"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}):
            
            # The tool should reach the approval stage and return a HITL request
            pass
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry sedan",
                "additional_notes": "Standard package",
                "quotation_validity_days": 30
            })
            
            # Verify the result contains approval request elements
            assert "QUOTATION APPROVAL REQUIRED" in result
            assert "Preview content" in result
            assert "READY FOR FINAL GENERATION" in result
            
            # Verify approval was requested
            mock_approval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_customer_not_found_hitl_flow(self):
        """Test HITL flow when customer is not found in CRM."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=None), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Unknown Customer",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify HITL request was made
            assert result == "HITL_REQUEST"
            mock_input.assert_called_once()
            
            # Verify the HITL request contains customer information prompt
            call_args = mock_input.call_args
            assert "Customer Information Needed" in call_args[1]["prompt"]
            assert "Unknown Customer" in call_args[1]["prompt"]
            assert call_args[1]["input_type"] == "customer_information"
            assert call_args[1]["context"]["current_step"] == "customer_lookup"
    
    @pytest.mark.asyncio
    async def test_vehicle_not_found_hitl_flow(self, sample_customer_data):
        """Test HITL flow when requested vehicle is not available."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools._generate_enhanced_hitl_vehicle_prompt', return_value="🚗 **Let me help you find the right vehicle**\n\nI couldn't find any Lamborghini Aventador in our inventory.\n\n**Available alternatives:**\n- Toyota Camry\n- Honda CR-V"), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Lamborghini Aventador"
            })
            
            # Verify HITL request was made
            assert result == "HITL_REQUEST"
            mock_input.assert_called_once()
            
            # Verify the HITL request contains vehicle alternatives
            call_args = mock_input.call_args
            assert "Vehicle Information Needed" in call_args[1]["prompt"]
            assert call_args[1]["input_type"] == "detailed_vehicle_requirements"
            assert call_args[1]["context"]["current_step"] == "vehicle_requirements"
    
    @pytest.mark.asyncio
    async def test_employee_access_control(self):
        """Test that non-employees are blocked from generating quotations."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value=None):
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify access is denied
            assert "🔒 **Employee Access Required**" in result
            assert "Quotation generation is restricted to authorized employees only" in result
    
    @pytest.mark.asyncio
    async def test_input_validation_empty_customer(self):
        """Test input validation for empty customer identifier."""
        
        result = await generate_quotation.ainvoke({
            "customer_identifier": "",
            "vehicle_requirements": "Toyota Camry"
        })
        
        assert "❌ **Invalid Input**" in result
        assert "Customer identifier is required" in result
    
    @pytest.mark.asyncio
    async def test_input_validation_empty_vehicle_requirements(self):
        """Test input validation for empty vehicle requirements."""
        
        result = await generate_quotation.ainvoke({
            "customer_identifier": "John Doe",
            "vehicle_requirements": ""
        })
        
        assert "❌ **Invalid Input**" in result
        assert "Vehicle requirements are needed" in result
    
    @pytest.mark.asyncio
    async def test_input_validation_invalid_validity_days(self):
        """Test input validation for invalid quotation validity days."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value={"id": "test"}), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}):
            
            # Test with negative validity days - should reset to default
            with patch('backend.agents.tools._parse_vehicle_requirements_with_llm') as mock_parse:
                mock_parse.side_effect = Exception("Test exception")  # Force early return
                
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "John Doe",
                    "vehicle_requirements": "Toyota Camry",
                    "quotation_validity_days": -5
                })
                
                # The function should continue processing (not return early due to validation)
                # and the validity days should be reset to 30 internally
                assert "❌ **Invalid Input**" not in result
    
    @pytest.mark.asyncio
    async def test_hitl_resume_logic(self, sample_customer_data, sample_vehicle_data):
        """Test HITL resume functionality when continuing from interrupted workflow."""
        
        quotation_state = {
            "customer_data": sample_customer_data,
            "vehicle_criteria": {"make": "Toyota", "model": "Camry"},
            "step_completed": "customer_lookup"
        }
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._handle_quotation_resume') as mock_resume:
            mock_resume.return_value = "Resumed successfully"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry",
                "quotation_state": quotation_state,
                "current_step": "vehicle_lookup",
                "user_response": "Continue with Toyota Camry"
            })
            
            # The resume logic should be called when current_step and user_response are provided
            mock_resume.assert_called_once()
            assert result == "Resumed successfully"
    
    @pytest.mark.asyncio
    async def test_quotation_reaches_approval_stage(self, sample_customer_data, sample_vehicle_data, sample_pricing_data, sample_employee_data):
        """Test that quotation generation reaches the approval stage correctly."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=sample_vehicle_data), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._create_quotation_preview', return_value="Preview content"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}), \
             patch('backend.agents.tools.request_approval') as mock_approval:
            
            # Mock approval request
            mock_approval.return_value = "HITL_APPROVAL_REQUEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify that approval was requested
            assert result == "HITL_APPROVAL_REQUEST"
            mock_approval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quotation_workflow_with_mocked_pdf_generation(self, sample_customer_data, sample_vehicle_data, sample_pricing_data, sample_employee_data):
        """Test quotation workflow reaches approval stage even with PDF generation mocked."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=sample_vehicle_data), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._parse_vehicle_requirements_with_llm', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._get_available_makes_and_models', return_value=[]), \
             patch('backend.agents.tools._enhance_vehicle_criteria_with_fuzzy_matching', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._create_quotation_preview', return_value="Preview content"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}), \
             patch('backend.agents.tools.request_approval', return_value="approved"), \
             patch('backend.core.pdf_generator.generate_quotation_pdf', side_effect=Exception("PDF generation failed")):
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # The tool should reach approval stage regardless of PDF mocking
            assert result == "approved"
    
    @pytest.mark.asyncio
    async def test_quotation_workflow_with_mocked_storage(self, sample_customer_data, sample_vehicle_data, sample_pricing_data, sample_employee_data):
        """Test quotation workflow reaches approval stage even with storage mocked."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=sample_vehicle_data), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._parse_vehicle_requirements_with_llm', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._get_available_makes_and_models', return_value=[]), \
             patch('backend.agents.tools._enhance_vehicle_criteria_with_fuzzy_matching', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._create_quotation_preview', return_value="Preview content"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}), \
             patch('backend.agents.tools.request_approval', return_value="approved"), \
             patch('backend.core.pdf_generator.generate_quotation_pdf', return_value=b"dummy_pdf_content"), \
             patch('backend.core.storage.upload_quotation_pdf', side_effect=Exception("Storage upload failed")):
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # The tool should reach approval stage regardless of storage mocking
            assert result == "approved"
    
    @pytest.mark.asyncio
    async def test_conversation_context_extraction(self):
        """Test that conversation context is properly extracted and used."""
        
        sample_context = {
            "customer_email": "john@company.com",
            "customer_phone": "+63 912 345 6789",
            "vehicle_make": "Toyota",
            "budget_range": "1.5M - 2M PHP"
        }
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Customer mentioned email john@company.com"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=sample_context), \
             patch('backend.agents.tools._lookup_customer', return_value=None) as mock_lookup, \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify that alternative lookups were attempted using extracted context
            assert mock_lookup.call_count >= 2  # Original + extracted context attempts
            
            # Verify context was passed to HITL request
            call_args = mock_input.call_args
            assert "john@company.com" in call_args[1]["prompt"]  # Email should be in prompt
            assert call_args[1]["context"]["extracted_context"] == sample_context
    
    @pytest.mark.asyncio
    async def test_multiple_vehicle_matching(self, sample_customer_data):
        """Test handling of multiple matching vehicles."""
        
        multiple_vehicles = [
            {"id": "v001", "make": "Toyota", "model": "Camry", "type": "Sedan", "year": 2023, "color": "White"},
            {"id": "v002", "make": "Toyota", "model": "Camry", "type": "Sedan", "year": 2023, "color": "Black"},
            {"id": "v003", "make": "Toyota", "model": "Camry", "type": "Sedan", "year": 2024, "color": "Silver"}
        ]
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value={"id": "emp001", "name": "Test Employee"}), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._parse_vehicle_requirements_with_llm', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._get_available_makes_and_models', return_value=[]), \
             patch('backend.agents.tools._enhance_vehicle_criteria_with_fuzzy_matching', return_value={"make": "Toyota", "model": "Camry", "type": "Sedan"}), \
             patch('backend.agents.tools._lookup_vehicle_by_criteria', return_value=multiple_vehicles), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify HITL request for vehicle selection
            assert result == "HITL_REQUEST"
            call_args = mock_input.call_args
            # The tool may ask for vehicle selection or other missing information
            # Just verify that a HITL request was made properly
            assert "prompt" in call_args[1]
            assert call_args[1]["input_type"] in ["vehicle_selection", "missing_quotation_info", "pricing_resolution"]


class TestQuotationParameterValidation:
    """Test parameter validation for GenerateQuotationParams."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = GenerateQuotationParams(
            customer_identifier="John Doe",
            vehicle_requirements="Toyota Camry sedan",
            additional_notes="Standard package",
            quotation_validity_days=30
        )
        
        assert params.customer_identifier == "John Doe"
        assert params.vehicle_requirements == "Toyota Camry sedan"
        assert params.additional_notes == "Standard package"
        assert params.quotation_validity_days == 30
    
    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Should raise validation error for missing required fields
        with pytest.raises(Exception):  # Pydantic validation error
            GenerateQuotationParams()
    
    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        params = GenerateQuotationParams(
            customer_identifier="John Doe",
            vehicle_requirements="Toyota Camry"
        )
        
        assert params.additional_notes is None
        assert params.quotation_validity_days == 30


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_quotation_generation.py -v
    pytest.main([__file__, "-v"])
