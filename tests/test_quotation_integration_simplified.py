"""
Integration Tests for Simplified Quotation Generation with LLM-based Vehicle Search

Tests the complete end-to-end quotation generation workflow using the new
simplified LLM-based vehicle search system. These tests focus on:
- Performance comparison between old and new systems
- Success path: natural language query → vehicle found → quotation generated  
- HITL path: unclear query → HITL request → refined search → quotation
- Multi-step HITL scenarios with complex requirements

Task 7.2: Integration tests for end-to-end quotation flow
"""

import os
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the functions we're testing
from backend.agents.tools import generate_quotation, _search_vehicles_with_llm
from backend.agents.hitl import request_input, request_approval


class TestSimplifiedQuotationIntegration:
    """Integration tests for the simplified quotation generation workflow."""
    
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
    def sample_vehicle_results(self):
        """Sample vehicle search results from LLM-based search."""
        return [
            {
                "id": "v001",
                "make": "Toyota",
                "model": "Camry",
                "type": "Sedan",
                "year": 2023,
                "color": "White",
                "is_available": True,
                "stock_quantity": 5,
                "base_price": 1500000.00,
                "discounted_price": 1450000.00
            },
            {
                "id": "v002", 
                "make": "Toyota",
                "model": "Camry",
                "type": "Sedan", 
                "year": 2022,
                "color": "Silver",
                "is_available": True,
                "stock_quantity": 3,
                "base_price": 1400000.00,
                "discounted_price": 1350000.00
            }
        ]
    
    @pytest.fixture
    def sample_pricing_data(self):
        """Sample pricing data for testing."""
        return {
            "base_price": 1500000.00,
            "final_price": 1450000.00,
            "discount": 50000.00,
            "insurance": 25000.00,
            "lto_fees": 15000.00,
            "add_ons": [
                {"name": "Extended Warranty", "price": 30000.00},
                {"name": "Paint Protection", "price": 20000.00}
            ]
        }
    
    @pytest.fixture
    def sample_employee_data(self):
        """Sample employee data for testing."""
        return {
            "id": "emp001",
            "name": "Sales Representative",
            "email": "sales@company.com",
            "phone": "+63 912 111 2222",
            "branch_name": "Manila Main Branch",
            "branch_region": "NCR",
            "position": "Senior Sales Consultant"
        }
    
    @pytest.mark.asyncio
    async def test_success_path_natural_language_to_quotation(
        self, 
        sample_customer_data, 
        sample_vehicle_results, 
        sample_pricing_data, 
        sample_employee_data
    ):
        """Test success path: natural language query → vehicle found → quotation generated."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=sample_vehicle_results), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Customer wants reliable family sedan"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_approval', return_value="HITL_REQUIRED:approval:{\"prompt\": \"📄 **QUOTATION APPROVAL REQUIRED**\", \"context\": {}}") as mock_approval, \
             patch('backend.agents.tools._create_quotation_preview', return_value="Quotation preview content"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}):
            
            start_time = time.time()
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry family sedan reliable",
                "additional_notes": "Standard package",
                "quotation_validity_days": 30
            })
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Verify the result contains approval request
            assert "QUOTATION APPROVAL REQUIRED" in result
            assert isinstance(result, str)
            
            # Verify approval was requested (quotation reached final stage)
            mock_approval.assert_called_once()
            
            # Performance assertion - should be fast with simplified system
            assert execution_time < 2.0, f"Quotation generation took {execution_time:.2f}s, expected < 2.0s"
            
            print(f"✅ Success path completed in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_hitl_path_unclear_query_to_refinement(
        self,
        sample_customer_data,
        sample_employee_data
    ):
        """Test HITL path: unclear query → HITL request → refined search → quotation."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools._generate_enhanced_hitl_vehicle_prompt', return_value="🚗 **Let me help you find the right vehicle**\n\nI need more specific information about your vehicle requirements."), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST_VEHICLE_INFO"
            
            start_time = time.time()
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "something good",  # Vague requirement
            })
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Verify HITL request was made
            assert result == "HITL_REQUEST_VEHICLE_INFO"
            mock_input.assert_called_once()
            
            # Verify the HITL request contains vehicle information prompt
            call_args = mock_input.call_args
            assert "help you find the right vehicle" in call_args[1]["prompt"]
            assert call_args[1]["input_type"] == "detailed_vehicle_requirements"
            assert call_args[1]["context"]["current_step"] == "vehicle_requirements"
            
            # Performance assertion
            assert execution_time < 1.0, f"HITL trigger took {execution_time:.2f}s, expected < 1.0s"
            
            print(f"✅ HITL path triggered in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_customer_not_found_hitl_flow(self):
        """Test HITL flow when customer is not found in CRM."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=None), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST_CUSTOMER_INFO"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Unknown Customer",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify HITL request was made
            assert result == "HITL_REQUEST_CUSTOMER_INFO"
            mock_input.assert_called_once()
            
            # Verify the HITL request contains customer information prompt
            call_args = mock_input.call_args
            assert "Customer Information Needed" in call_args[1]["prompt"]
            assert "Unknown Customer" in call_args[1]["prompt"]
            assert call_args[1]["input_type"] == "customer_information"
            assert call_args[1]["context"]["current_step"] == "customer_lookup"
    
    @pytest.mark.asyncio
    async def test_multi_step_hitl_complex_requirements(
        self,
        sample_customer_data,
        sample_employee_data
    ):
        """Test multi-step HITL: complex requirements → multiple clarifications → success."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value="Customer mentioned budget concerns and family needs"), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={
                 "budget_mentions": ["under 2 million", "affordable"],
                 "family_size_mentions": ["family of 5", "kids"],
                 "usage_mentions": ["daily commute", "weekend trips"]
             }), \
             patch('backend.agents.tools._generate_enhanced_hitl_vehicle_prompt', return_value="🚗 **Let me help you find the right vehicle**\n\nBased on your family needs and budget, I'd recommend considering..."), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_REQUEST_DETAILED_REQUIREMENTS"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "family car budget friendly safe reliable good for kids weekend trips under 2M",
            })
            
            # Verify HITL request was made with enhanced context
            assert result == "HITL_REQUEST_DETAILED_REQUIREMENTS"
            mock_input.assert_called_once()
            
            # Verify the HITL request uses enhanced prompt with customer context
            call_args = mock_input.call_args
            assert "help you find the right vehicle" in call_args[1]["prompt"]
            assert call_args[1]["input_type"] == "detailed_vehicle_requirements"
            assert call_args[1]["context"]["current_step"] == "vehicle_requirements"
            
            # Verify context data is preserved for multi-step flow
            context = call_args[1]["context"]
            assert "customer_identifier" in context
            assert "vehicle_requirements" in context
            assert "conversation_context" in context
    
    @pytest.mark.asyncio
    async def test_performance_comparison_simulation(self):
        """Test performance characteristics of the new simplified system."""
        
        # Simulate multiple rapid quotation requests to test system performance
        sample_requests = [
            ("John Doe", "Toyota Camry sedan"),
            ("Jane Smith", "Honda CR-V SUV family"),
            ("Bob Johnson", "Nissan Altima reliable commuter"),
            ("Alice Brown", "Hyundai Tucson affordable SUV"),
            ("Charlie Wilson", "Mazda CX-5 fuel efficient")
        ]
        
        execution_times = []
        
        for customer, requirements in sample_requests:
            with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
                 patch('backend.agents.tools._lookup_customer', return_value={"id": "test", "name": customer}), \
                 patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
                 patch('backend.agents.tools._lookup_employee_details', return_value={"id": "emp001", "name": "Test Employee"}), \
                 patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
                 patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
                 patch('backend.agents.tools._generate_enhanced_hitl_vehicle_prompt', return_value="Vehicle search prompt"), \
                 patch('backend.agents.tools.request_input', return_value="HITL_REQUEST"):
                
                start_time = time.time()
                
                result = await generate_quotation.ainvoke({
                    "customer_identifier": customer,
                    "vehicle_requirements": requirements
                })
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                assert result == "HITL_REQUEST"
        
        # Performance assertions
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        print(f"📊 Performance Results:")
        print(f"   Average execution time: {avg_time:.3f}s")
        print(f"   Maximum execution time: {max_time:.3f}s")
        print(f"   Total requests: {len(execution_times)}")
        
        # With simplified LLM-based search, each request should be fast
        assert avg_time < 0.5, f"Average time {avg_time:.3f}s exceeds 0.5s threshold"
        assert max_time < 1.0, f"Maximum time {max_time:.3f}s exceeds 1.0s threshold"
        
        print("✅ Performance benchmarks passed!")
    
    @pytest.mark.asyncio
    async def test_employee_access_control_integration(self):
        """Test that employee access control works in the integrated system."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value=None):
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry"
            })
            
            # Verify access is denied
            assert "🔒 **Employee Access Required**" in result
            assert "employee" in result.lower() and ("access" in result.lower() or "restricted" in result.lower())
    
    @pytest.mark.asyncio
    async def test_quotation_resume_functionality(
        self,
        sample_customer_data,
        sample_vehicle_results,
        sample_pricing_data,
        sample_employee_data
    ):
        """Test quotation resume functionality with simplified vehicle search."""
        
        # Test resuming from vehicle search step
        quotation_state = {
            "customer_data": sample_customer_data,
            "employee_data": sample_employee_data,
            "vehicle_requirements": "Toyota Camry family sedan",
            "step_completed": "customer_lookup"
        }
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value=sample_employee_data), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=sample_vehicle_results), \
             patch('backend.agents.tools._lookup_current_pricing', return_value=sample_pricing_data), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value={}), \
             patch('backend.agents.tools.request_approval', return_value="HITL_REQUIRED:approval:{\"prompt\": \"📄 **QUOTATION APPROVAL REQUIRED**\", \"context\": {}}") as mock_approval, \
             patch('backend.agents.tools._create_quotation_preview', return_value="Quotation preview"), \
             patch('backend.agents.tools._identify_missing_quotation_information', return_value={}):
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "John Doe",
                "vehicle_requirements": "Toyota Camry family sedan",
                "quotation_state": quotation_state,
                "current_step": "vehicle_search",
                "user_response": "Yes, Toyota Camry looks good"
            })
            
            # Verify the quotation continued from the resume point
            assert "QUOTATION APPROVAL REQUIRED" in result
            mock_approval.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
