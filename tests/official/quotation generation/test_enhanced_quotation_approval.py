"""
Test Enhanced Quotation Approval Logic (Task 7.6 Output)

Tests for the enhanced quotation approval flow with required and optional information display.
Validates that the approval step works correctly and integrates with existing HITL infrastructure.
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

from agents.toolbox.generate_quotation import (
    generate_quotation,
    _extract_comprehensive_context,
    _validate_quotation_completeness,
    _create_quotation_preview,
    _request_quotation_approval
)

class TestEnhancedQuotationApprovalLogic:
    """Test suite for enhanced quotation approval logic from Task 7.6."""
    
    @pytest.fixture
    def sample_complete_context(self):
        """Sample context with complete information for testing."""
        return {
            "status": "success",
            "context": MagicMock(
                extracted_context=MagicMock(
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
                        "decision_timeline": "This month",
                        "delivery_date": "End of month"
                    }
                )
            )
        }
    
    @pytest.fixture
    def sample_incomplete_context(self):
        """Sample context with incomplete information for testing."""
        return {
            "status": "success", 
            "context": MagicMock(
                extracted_context=MagicMock(
                    customer_info={
                        "name": "Jane Doe",
                        "email": "",  # Missing email
                        "phone": "",  # Missing phone
                    },
                    vehicle_requirements={
                        "make": "Honda",
                        "model": "",  # Missing model
                    },
                    purchase_preferences={},
                    timeline_info={}
                )
            )
        }

    @pytest.mark.asyncio
    async def test_approval_step_appears_only_when_complete(self, sample_complete_context, sample_incomplete_context):
        """
        Test 8.9.1: Test approval step appears only when completeness validation passes.
        
        This test verifies that:
        1. When information is complete, approval step is triggered
        2. When information is incomplete, HITL request for missing info is returned
        3. No approval step appears when completeness validation fails
        """
        
        # Mock the dependencies
        with patch('agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract, \
             patch('agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate, \
             patch('agents.toolbox.generate_quotation._request_quotation_approval') as mock_approval, \
             patch('agents.toolbox.generate_quotation._generate_intelligent_hitl_request') as mock_hitl, \
             patch('agents.toolbox.generate_quotation.get_current_employee_id') as mock_employee:
            
            # Setup mocks
            mock_employee.return_value = "emp_123"
            
            # Test Case 1: Complete information should trigger approval
            mock_extract.return_value = sample_complete_context
            mock_validate.return_value = {
                "status": "success",
                "is_complete": True,
                "missing_info": {}
            }
            mock_approval.return_value = "HITL_REQUIRED:approval:approval_prompt_here"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Smith, john@example.com, +1-555-0123",
                "vehicle_requirements": "Toyota Camry 2024 Blue Sedan"
            })
            
            # Verify approval was called for complete information
            mock_approval.assert_called_once()
            assert "HITL_REQUIRED:approval:" in result
            
            # Reset mocks
            mock_extract.reset_mock()
            mock_validate.reset_mock()
            mock_approval.reset_mock()
            mock_hitl.reset_mock()
            
            # Test Case 2: Incomplete information should NOT trigger approval
            mock_extract.return_value = sample_incomplete_context
            mock_validate.return_value = {
                "status": "success", 
                "is_complete": False,
                "missing_info": {
                    "critical": ["customer_email", "customer_phone", "vehicle_model"]
                }
            }
            mock_hitl.return_value = "HITL_REQUIRED:input:missing_info_prompt"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Jane Doe",
                "vehicle_requirements": "Honda"
            })
            
            # Verify approval was NOT called for incomplete information
            mock_approval.assert_not_called()
            mock_hitl.assert_called_once()
            assert "HITL_REQUIRED:input:" in result

    @pytest.mark.asyncio
    async def test_enhanced_quotation_preview_display(self, sample_complete_context):
        """
        Test 8.9.2: Test enhanced quotation preview with required vs optional information display.
        
        Verifies that the quotation preview correctly displays:
        1. Required information section (customer contact, vehicle basics)
        2. Optional information section (customer details, vehicle details, preferences)
        3. Clear visual distinction between required and optional sections
        """
        
        preview = await _create_quotation_preview(
            extracted_context=sample_complete_context["context"],
            quotation_validity_days=30
        )
        
        # Verify required information section is present
        assert "🔹 **REQUIRED INFORMATION**" in preview
        assert "**Customer Contact:**" in preview
        assert "John Smith" in preview
        assert "john@example.com" in preview
        assert "+1-555-0123" in preview
        assert "**Vehicle Basics:**" in preview
        assert "Toyota" in preview
        assert "Camry" in preview
        
        # Verify optional information section is present
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
        
        # Verify visual distinction (different icons and separators)
        assert "🔹" in preview  # Required section icon
        assert "🔸" in preview  # Optional section icon
        assert "─" in preview   # Visual separators

    @pytest.mark.asyncio
    async def test_visual_distinction_required_optional(self, sample_complete_context):
        """
        Test 8.9.3: Test visual distinction between required and optional sections works correctly.
        
        Ensures that the visual formatting clearly distinguishes between:
        1. Required information (🔹 icon, specific formatting)
        2. Optional information (🔸 icon, different formatting)
        3. Proper section separators
        """
        
        preview = await _create_quotation_preview(
            extracted_context=sample_complete_context["context"],
            quotation_validity_days=30
        )
        
        # Split preview into lines for detailed analysis
        lines = preview.split('\n')
        
        # Find required and optional section headers
        required_line = None
        optional_line = None
        
        for i, line in enumerate(lines):
            if "🔹 **REQUIRED INFORMATION**" in line:
                required_line = i
            elif "🔸 **OPTIONAL INFORMATION**" in line:
                optional_line = i
        
        # Verify both sections exist
        assert required_line is not None, "Required information section not found"
        assert optional_line is not None, "Optional information section not found"
        
        # Verify required section comes before optional section
        assert required_line < optional_line, "Required section should come before optional section"
        
        # Verify visual separators exist between sections
        separator_count = preview.count("─" * 50)
        assert separator_count >= 2, "Should have visual separators between sections"
        
        # Verify different icons are used
        assert "🔹" in preview, "Required section should use 🔹 icon"
        assert "🔸" in preview, "Optional section should use 🔸 icon"

    @pytest.mark.asyncio
    async def test_approval_prompt_comprehensive_display(self, sample_complete_context):
        """
        Test 8.9.4: Test approval prompt shows comprehensive customer and vehicle information.
        
        Validates that the approval prompt includes:
        1. Complete customer information (required and optional)
        2. Complete vehicle information (required and optional)
        3. Clear approval instructions
        4. Professional formatting
        """
        
        with patch('agents.toolbox.generate_quotation._create_quotation_preview') as mock_preview, \
             patch('agents.hitl.request_approval') as mock_request_approval:
            
            # Mock the preview generation
            mock_preview.return_value = "Sample preview with required and optional info"
            mock_request_approval.return_value = "HITL_REQUIRED:approval:test_prompt"
            
            result = await _request_quotation_approval(
                extracted_context=sample_complete_context["context"],
                quotation_validity_days=30
            )
            
            # Verify preview was generated
            mock_preview.assert_called_once()
            
            # Verify request_approval was called with proper structure
            mock_request_approval.assert_called_once()
            call_args = mock_request_approval.call_args
            
            # Check the approval prompt structure
            prompt = call_args[1]['prompt']
            assert "📄 **Quotation Ready for Generation**" in prompt
            assert "Sample preview with required and optional info" in prompt
            assert "Please review the information above and confirm:" in prompt
            assert "Is the customer information accurate?" in prompt
            assert "Are the vehicle requirements correct?" in prompt
            assert "✅ **Approve** to generate the PDF quotation" in prompt
            assert "❌ **Deny** to cancel" in prompt
            assert "🔄 **Correct** any information that needs to be updated" in prompt
            
            # Check context data
            context = call_args[1]['context']
            assert context['tool'] == 'generate_quotation'
            assert context['step'] == 'final_approval'
            assert 'extracted_context' in context
            assert context['quotation_validity_days'] == 30

    @pytest.mark.asyncio
    async def test_approval_flow_hitl_integration(self, sample_complete_context):
        """
        Test 8.9.5: Test approval flow integrates properly with existing HITL infrastructure.
        
        Ensures that:
        1. Approval requests use existing HITL infrastructure
        2. Context is properly preserved
        3. HITL state management works correctly
        4. Integration with LangGraph interrupt/resume patterns
        """
        
        with patch('agents.hitl.request_approval') as mock_request_approval:
            
            # Mock successful approval request
            mock_request_approval.return_value = "HITL_REQUIRED:approval:test_context"
            
            result = await _request_quotation_approval(
                extracted_context=sample_complete_context["context"],
                quotation_validity_days=30
            )
            
            # Verify HITL infrastructure is used
            mock_request_approval.assert_called_once()
            
            # Verify proper HITL format is returned
            assert result.startswith("HITL_REQUIRED:approval:")
            
            # Verify context preservation
            call_args = mock_request_approval.call_args
            context = call_args[1]['context']
            
            # Check that all necessary context is preserved for resume
            required_context_keys = ['tool', 'step', 'extracted_context', 'quotation_validity_days']
            for key in required_context_keys:
                assert key in context, f"Missing required context key: {key}"
            
            # Verify tool and step identification for proper routing
            assert context['tool'] == 'generate_quotation'
            assert context['step'] == 'final_approval'

    @pytest.mark.asyncio
    async def test_approval_with_partial_optional_data(self):
        """
        Test approval flow when some optional information is missing.
        
        Ensures that:
        1. Approval still works when optional info is missing
        2. Only available optional information is displayed
        3. Missing optional info doesn't prevent approval
        """
        
        # Create context with missing optional information
        partial_context = {
            "status": "success",
            "context": MagicMock(
                extracted_context=MagicMock(
                    customer_info={
                        "name": "Test Customer",
                        "email": "test@example.com",
                        "phone": "+1-555-0000",
                        # Missing company and address
                    },
                    vehicle_requirements={
                        "make": "Ford",
                        "model": "Focus",
                        # Missing year, color, type
                    },
                    purchase_preferences={},  # Empty
                    timeline_info={}  # Empty
                )
            )
        }
        
        preview = await _create_quotation_preview(
            extracted_context=partial_context["context"],
            quotation_validity_days=30
        )
        
        # Verify required information is still displayed
        assert "🔹 **REQUIRED INFORMATION**" in preview
        assert "Test Customer" in preview
        assert "test@example.com" in preview
        assert "Ford" in preview
        assert "Focus" in preview
        
        # Verify optional section exists but may be minimal
        assert "🔸 **OPTIONAL INFORMATION**" in preview
        
        # Verify preview is still properly formatted
        assert "**Quotation Validity:** 30 days" in preview

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
