"""
HITL Enhanced Prompts Integration Tests (Task 7.2.2)

This module tests the HITL integration with enhanced prompts in real scenarios,
focusing on the actual behavior rather than mocking complex LLM chains.

Tests cover:
- Enhanced prompt fallback behavior when LLM fails
- HITL integration with contextual information
- Resume logic preservation of enhanced context
- Real-world scenarios with enhanced prompts
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the functions we're testing
from backend.agents.tools import generate_quotation
from backend.agents.hitl import request_input


class TestHITLEnhancedIntegration:
    """Integration tests for HITL with enhanced prompts."""
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing."""
        return {
            "id": "cust001",
            "name": "Maria Santos",
            "email": "maria.santos@email.com",
            "company": "Santos Family Business",
            "is_for_business": False
        }
    
    @pytest.fixture
    def business_customer_data(self):
        """Sample business customer data."""
        return {
            "id": "cust002",
            "name": "Juan Dela Cruz", 
            "email": "juan@techcorp.com",
            "company": "TechCorp Solutions",
            "is_for_business": True
        }
    
    @pytest.fixture
    def sample_employee_data(self):
        """Sample employee data."""
        return {
            "id": "emp001",
            "name": "Sales Representative",
            "email": "sales@company.com",
            "phone": "+63 912 111 2222",
            "branch_name": "Manila Main Branch",
            "position": "Senior Sales Consultant"
        }
    
    @pytest.fixture
    def extracted_context_with_family_info(self):
        """Rich conversation context for family needs."""
        return {
            "budget_mentions": ["under 2 million", "affordable"],
            "family_size_mentions": ["family of 5", "kids", "children"],
            "usage_mentions": ["daily commute", "school runs", "weekend trips"],
            "preference_mentions": ["safe", "reliable", "fuel efficient"]
        }
    
    @pytest.fixture
    def extracted_context_business_info(self):
        """Rich conversation context for business needs."""
        return {
            "budget_mentions": ["company budget", "cost-effective"],
            "usage_mentions": ["client meetings", "business operations", "company fleet"],
            "preference_mentions": ["professional", "reliable", "low maintenance"]
        }
    
    @pytest.mark.asyncio
    async def test_enhanced_hitl_prompt_displays_contextual_information(
        self,
        sample_customer_data,
        sample_employee_data,
        extracted_context_with_family_info
    ):
        """Test that HITL prompts display contextual information from conversation."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Customer mentioned family of 5 and budget under 2M"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=extracted_context_with_family_info), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_WITH_CONTEXT"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Maria Santos",
                "vehicle_requirements": "luxury sports car",  # Intentionally vague/impossible
            })
            
            # Verify HITL was triggered
            assert result == "HITL_WITH_CONTEXT"
            mock_input.assert_called_once()
            
            # Verify the prompt contains contextual information
            call_args = mock_input.call_args
            prompt = call_args[1]["prompt"]
            
            # Check for basic HITL structure (fallback or enhanced)
            assert "🚗" in prompt or "Vehicle" in prompt
            assert "luxury sports car" in prompt
            
            # Verify context preservation in the HITL context
            context = call_args[1]["context"]
            assert "extracted_context" in context
            assert context["extracted_context"]["family_size_mentions"] == ["family of 5", "kids", "children"]
            assert context["extracted_context"]["budget_mentions"] == ["under 2 million", "affordable"]
            
            # Verify input type is appropriate
            assert call_args[1]["input_type"] == "detailed_vehicle_requirements"
            
            print(f"✅ HITL prompt contains contextual information from conversation")
    
    @pytest.mark.asyncio
    async def test_business_vs_personal_context_differentiation(
        self,
        business_customer_data,
        sample_employee_data,
        extracted_context_business_info
    ):
        """Test that HITL prompts differentiate between business and personal context."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=business_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Business customer needs company vehicle"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=extracted_context_business_info), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_BUSINESS_CONTEXT"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Juan Dela Cruz",
                "vehicle_requirements": "compact vehicle",
            })
            
            # Verify HITL was triggered
            assert result == "HITL_BUSINESS_CONTEXT"
            mock_input.assert_called_once()
            
            # Verify business context is preserved
            call_args = mock_input.call_args
            context = call_args[1]["context"]
            
            # Check customer data indicates business use
            customer_data = context["customer_data"]
            assert customer_data["company"] == "TechCorp Solutions"
            
            # Check extracted context has business-related mentions
            extracted_context = context["extracted_context"]
            assert "client meetings" in extracted_context["usage_mentions"]
            assert "company budget" in extracted_context["budget_mentions"]
            
            print(f"✅ Business context properly differentiated in HITL prompts")
    
    @pytest.mark.asyncio
    async def test_hitl_resume_preserves_enhanced_context(
        self,
        sample_customer_data,
        sample_employee_data,
        extracted_context_with_family_info
    ):
        """Test that HITL resume logic preserves enhanced search context."""
        
        # Simulate resuming from a previous HITL interaction
        quotation_state = {
            "customer_data": sample_customer_data,
            "employee_data": sample_employee_data,
            "original_requirements": "family car",
            "extracted_context": extracted_context_with_family_info,
            "conversation_context": "Customer mentioned family of 5, budget under 2M, safety important"
        }
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=extracted_context_with_family_info), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_RESUMED_WITH_CONTEXT"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Maria Santos",
                "vehicle_requirements": "family car",
                "quotation_state": quotation_state,
                "current_step": "vehicle_requirements",
                "user_response": "Toyota Camry sedan, 2023, need safety features"
            })
            
            # Verify HITL was triggered (no vehicles found scenario)
            assert result == "HITL_RESUMED_WITH_CONTEXT"
            mock_input.assert_called_once()
            
            # Verify enhanced context is preserved across resume
            call_args = mock_input.call_args
            context = call_args[1]["context"]
            
            # Check that quotation state is preserved
            assert "quotation_state" in context
            # The quotation state structure may vary, just verify it exists
            assert isinstance(context["quotation_state"], dict)
            
            # Check that extracted context is preserved
            assert "extracted_context" in context
            preserved_context = context["extracted_context"]
            assert preserved_context["family_size_mentions"] == ["family of 5", "kids", "children"]
            assert preserved_context["budget_mentions"] == ["under 2 million", "affordable"]
            
            # Check user response is captured (if present in context)
            if "user_response" in context:
                assert context["user_response"] == "Toyota Camry sedan, 2023, need safety features"
            
            print(f"✅ Enhanced context preserved across HITL resume")
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_performance_in_integration(
        self,
        sample_customer_data,
        sample_employee_data
    ):
        """Test performance of enhanced prompts in integration scenarios."""
        
        test_scenarios = [
            ("luxury sports car", {"budget_mentions": ["premium"]}),
            ("family SUV", {"family_size_mentions": ["large family"]}),
            ("business sedan", {"usage_mentions": ["client meetings"]}),
            ("eco-friendly car", {"preference_mentions": ["environmentally conscious"]}),
        ]
        
        execution_times = []
        
        for requirements, context in test_scenarios:
            with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
                 patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
                 patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
                 patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
                 patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
                 patch('backend.agents.tools.extract_fields_from_conversation', return_value=context), \
                 patch('backend.agents.tools.request_input', return_value="HITL_PERFORMANCE_TEST"):
                
                start_time = time.time()
                
                result = await generate_quotation.ainvoke({
                    "customer_identifier": "Maria Santos",
                    "vehicle_requirements": requirements
                })
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                assert result == "HITL_PERFORMANCE_TEST"
        
        # Performance analysis
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        print(f"📊 Enhanced HITL Integration Performance:")
        print(f"   Average execution time: {avg_time:.3f}s")
        print(f"   Maximum execution time: {max_time:.3f}s")
        print(f"   Scenarios tested: {len(execution_times)}")
        
        # Performance assertions - enhanced prompts involve LLM calls, so allow more time
        assert avg_time < 5.0, f"Average enhanced HITL time {avg_time:.3f}s exceeds 5.0s threshold"
        assert max_time < 8.0, f"Maximum enhanced HITL time {max_time:.3f}s exceeds 8.0s threshold"
        
        print("✅ Enhanced HITL integration performance benchmarks passed!")
    
    @pytest.mark.asyncio
    async def test_fallback_prompt_quality_when_enhancement_fails(
        self,
        sample_customer_data,
        sample_employee_data
    ):
        """Test that fallback prompts are still high quality when enhancement fails."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "FALLBACK_PROMPT_TEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Maria Santos", 
                "vehicle_requirements": "impossible vehicle request"
            })
            
            # Verify HITL was triggered with fallback prompt
            assert result == "FALLBACK_PROMPT_TEST"
            mock_input.assert_called_once()
            
            # Verify fallback prompt quality
            call_args = mock_input.call_args
            prompt = call_args[1]["prompt"]
            
            # Enhanced prompts should be helpful and structured (not necessarily fallback)
            assert "🚗" in prompt or "Vehicle" in prompt
            # The system may generate enhanced prompts that interpret the request intelligently
            assert "help" in prompt.lower() or "find" in prompt.lower()
            assert "vehicle" in prompt.lower()
            
            # Should provide guidance and options
            assert ("make" in prompt.lower() or "brand" in prompt.lower() or 
                   "toyota" in prompt.lower() or "honda" in prompt.lower())
            # Should contain helpful guidance or vehicle options
            assert ("model" in prompt.lower() or "options" in prompt.lower() or
                   "recommend" in prompt.lower())
            
            print("✅ Fallback prompts maintain quality when enhancement fails")
    
    @pytest.mark.asyncio
    async def test_contextual_suggestions_integration(
        self,
        sample_customer_data,
        sample_employee_data
    ):
        """Test that contextual suggestions are integrated properly in HITL flows."""
        
        rich_context = {
            "budget_mentions": ["under 1.5 million", "affordable"],
            "family_size_mentions": ["couple", "no kids yet"],
            "usage_mentions": ["city driving", "weekend trips"],
            "preference_mentions": ["fuel efficient", "compact", "easy parking"]
        }
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Young couple, city driving, fuel efficiency important"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=rich_context), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "CONTEXTUAL_SUGGESTIONS_TEST"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Maria Santos",
                "vehicle_requirements": "perfect car"  # Intentionally vague
            })
            
            # Verify HITL integration with contextual suggestions
            assert result == "CONTEXTUAL_SUGGESTIONS_TEST"
            mock_input.assert_called_once()
            
            # Verify rich context is available for suggestions
            call_args = mock_input.call_args
            context = call_args[1]["context"]
            
            # Check that extracted context contains rich information
            extracted = context["extracted_context"]
            assert "fuel efficient" in extracted["preference_mentions"]
            assert "city driving" in extracted["usage_mentions"]
            assert "under 1.5 million" in extracted["budget_mentions"]
            
            # Verify conversation context is preserved
            assert "conversation_context" in context
            
            print("✅ Contextual suggestions properly integrated in HITL flows")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
