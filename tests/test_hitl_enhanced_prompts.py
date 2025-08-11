"""
Tests for HITL Integration with Enhanced Prompts (Task 7.2.2)

This module tests the enhanced HITL prompt generation system that uses LLM
intelligence to create personalized, contextual prompts when vehicle searches
don't return results or need clarification.

Tests cover:
- LLM interpretation display in HITL prompts
- Contextual suggestions based on search results  
- HITL resume logic with simplified search context
- Intelligent requirement merging
- Customer-specific personalization
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the functions we're testing
from backend.agents.tools import (
    _generate_enhanced_hitl_vehicle_prompt,
    _merge_vehicle_requirements_intelligently,
    generate_quotation
)
from backend.agents.hitl import request_input


class TestEnhancedHITLPrompts:
    """Tests for enhanced HITL prompt generation with LLM interpretation."""
    
    @pytest.fixture
    def sample_extracted_context(self):
        """Sample conversation context data."""
        return {
            "budget_mentions": ["under 2 million", "affordable", "budget-friendly"],
            "family_size_mentions": ["family of 5", "kids", "children"],
            "usage_mentions": ["daily commute", "weekend trips", "school runs"],
            "preference_mentions": ["reliable", "fuel efficient", "safe"]
        }
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for personalization."""
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
    def sample_inventory_suggestions(self):
        """Sample inventory suggestions string."""
        return """Available vehicles that might interest you:

**Toyota Family Vehicles:**
- Toyota Camry 2023 (₱1,850,000) - Reliable sedan, excellent fuel economy
- Toyota Vios 2023 (₱1,200,000) - Compact, perfect for city driving
- Toyota Innova 2023 (₱1,950,000) - 7-seater MPV, ideal for families

**Honda Options:**
- Honda City 2023 (₱1,300,000) - Stylish sedan with modern features
- Honda CR-V 2023 (₱2,100,000) - Premium SUV with safety features

**Budget-Friendly Options:**
- Mitsubishi Mirage 2023 (₱950,000) - Most affordable, great fuel efficiency
- Suzuki Swift 2023 (₱1,100,000) - Sporty hatchback, easy to park"""
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_with_llm_interpretation(
        self,
        sample_extracted_context,
        sample_customer_data,
        sample_inventory_suggestions
    ):
        """Test that LLM interpretation is displayed in HITL prompts."""
        
        # Mock the LLM chain to return a specific enhanced prompt
        expected_enhanced_prompt = """🚗 **Let me help you find the right vehicle**

I understand you're looking for a "luxury sports car" for your family needs. While we don't have luxury sports cars that are practical for families with children, I can see from our conversation that you prioritize safety, reliability, and comfort for your family of 5.

**Based on your requirements, here's what I found:**
You mentioned needing something for daily commutes and weekend trips with budget considerations under 2 million. A luxury sports car might not be the best fit for family use with kids.

**I'd recommend considering:**
- Toyota Camry 2023 - Reliable, safe, and comfortable for families
- Honda CR-V 2023 - SUV with excellent safety ratings and space
- Toyota Innova 2023 - Perfect for larger families, very reliable

**To find the perfect match, could you help me with:**
- What's most important: performance, family space, or fuel efficiency?
- Would you consider a premium sedan instead of a sports car?
- Any specific safety features that are must-haves?

**Available Options:**
Based on your family needs and budget, I've curated these options from our current inventory that balance performance with practicality."""
        
        with patch('backend.agents.tools.get_settings') as mock_settings, \
             patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
             patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
             patch('backend.agents.tools.StrOutputParser') as mock_parser:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
            mock_llm_class.return_value = MagicMock()
            mock_template.from_template.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            # Mock the chain to return our expected prompt
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = expected_enhanced_prompt
            mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            result = await _generate_enhanced_hitl_vehicle_prompt(
                vehicle_requirements="luxury sports car",
                extracted_context=sample_extracted_context,
                inventory_suggestions=sample_inventory_suggestions,
                customer_data=sample_customer_data
            )
            
            # Verify LLM interpretation is displayed
            assert "I understand you're looking for a \"luxury sports car\"" in result
            assert "for your family needs" in result
            assert "Based on your requirements, here's what I found:" in result
            
            # Verify contextual analysis
            assert "family of 5" in result or "families with children" in result
            assert "budget considerations under 2 million" in result or "budget" in result
            assert "daily commutes and weekend trips" in result
            
            # Verify intelligent suggestions
            assert "Toyota Camry" in result
            assert "Honda CR-V" in result
            assert "safety" in result.lower()
            
            # Verify the chain was called with proper context
            mock_chain.ainvoke.assert_called_once()
            call_args = mock_chain.ainvoke.call_args[0][0]
            assert call_args["vehicle_requirements"] == "luxury sports car"
            assert "family of 5" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_contextual_suggestions_based_on_customer_type(
        self,
        sample_extracted_context,
        business_customer_data,
        sample_inventory_suggestions
    ):
        """Test contextual suggestions based on customer type (business vs personal)."""
        
        business_enhanced_prompt = """🚗 **Let me help you find the right vehicle**

I see you're looking for a "compact car" for TechCorp Solutions business use. 

**Based on your business requirements, here's what I found:**
For business use, you'll want something professional, reliable, and cost-effective for your company operations.

**I'd recommend considering:**
- Toyota Vios 2023 - Professional appearance, excellent fuel economy, low maintenance
- Honda City 2023 - Modern features, good for client meetings
- Mitsubishi Mirage 2023 - Most cost-effective option for business fleet

**To find the perfect match, could you help me with:**
- Will this be used for client meetings or general business operations?
- Do you need specific business features like large trunk space?
- Is this for a company fleet or executive use?

**Available Options:**
For business use, I recommend focusing on total cost of ownership and professional appearance."""
        
        with patch('backend.agents.tools.get_settings') as mock_settings, \
             patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
             patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
             patch('backend.agents.tools.StrOutputParser') as mock_parser:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
            mock_llm_class.return_value = MagicMock()
            mock_template.from_template.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            # Mock the chain to return business-focused prompt
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = business_enhanced_prompt
            mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            result = await _generate_enhanced_hitl_vehicle_prompt(
                vehicle_requirements="compact car",
                extracted_context=sample_extracted_context,
                inventory_suggestions=sample_inventory_suggestions,
                customer_data=business_customer_data
            )
            
            # Verify business-specific contextualization
            assert "business use" in result.lower()
            assert "TechCorp Solutions" in result or "company" in result.lower()
            assert "professional" in result.lower()
            assert "cost-effective" in result.lower() or "cost of ownership" in result.lower()
            
            # Verify business-specific questions
            assert "client meetings" in result.lower()
            assert "company fleet" in result.lower() or "fleet" in result.lower()
            
            # Verify the customer context was passed correctly
            mock_chain.ainvoke.assert_called_once()
            call_args = mock_chain.ainvoke.call_args[0][0]
            assert "TechCorp Solutions" in call_args["customer_context"]
            assert "business customer" in call_args["customer_context"]
    
    @pytest.mark.asyncio
    async def test_intelligent_requirement_merging(self):
        """Test intelligent merging of original requirements with user clarifications."""
        
        test_cases = [
            {
                "original": "family car",
                "clarification": "Toyota Camry sedan, 2023, budget 1.5M",
                "expected_elements": ["Toyota Camry", "sedan", "2023", "1.5M", "family"]
            },
            {
                "original": "SUV for business",
                "clarification": "Honda CR-V, white or silver, automatic",
                "expected_elements": ["Honda CR-V", "SUV", "business", "white", "silver", "automatic"]
            },
            {
                "original": "affordable vehicle",
                "clarification": "under 1M budget, fuel efficient, manual transmission",
                "expected_elements": ["affordable", "under 1M", "fuel efficient", "manual"]
            }
        ]
        
        for test_case in test_cases:
            with patch('backend.agents.tools.get_settings') as mock_settings, \
                 patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
                 patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
                 patch('backend.agents.tools.StrOutputParser') as mock_parser:
                
                # Setup mocks
                mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
                mock_llm_class.return_value = MagicMock()
                mock_template.from_template.return_value = MagicMock()
                mock_parser.return_value = MagicMock()
                
                # Create expected merged result
                expected_merged = f"{test_case['original']} - {test_case['clarification']} merged intelligently"
                
                # Mock the chain to return merged requirements
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = expected_merged
                mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                result = await _merge_vehicle_requirements_intelligently(
                    original_requirements=test_case["original"],
                    user_clarification=test_case["clarification"],
                    extracted_context={}
                )
                
                # Verify the merge was called correctly
                mock_chain.ainvoke.assert_called_once()
                call_args = mock_chain.ainvoke.call_args[0][0]
                assert call_args["original_requirements"] == test_case["original"]
                assert call_args["user_clarification"] == test_case["clarification"]
                
                # Verify result contains expected elements
                assert result == expected_merged
                
                # Reset mocks for next iteration
                mock_chain.reset_mock()
    
    @pytest.mark.asyncio
    async def test_hitl_resume_logic_with_enhanced_context(
        self,
        sample_extracted_context,
        sample_customer_data
    ):
        """Test HITL resume logic preserves enhanced search context."""
        
        # Test resuming from a HITL vehicle requirements clarification
        quotation_state = {
            "customer_data": sample_customer_data,
            "original_requirements": "family car",
            "extracted_context": sample_extracted_context,
            "hitl_step": "vehicle_requirements_clarification"
        }
        
        enhanced_prompt = """🚗 **Let me help you find the right vehicle**

Thank you for the clarification! Now I understand you want a "Toyota Camry sedan 2023 budget 1.5M for family use with safety features".

**Based on your updated requirements, here's what I found:**
Your clarification helps me focus on exactly what you need - a reliable Toyota Camry that fits your family budget and safety priorities.

**I'd recommend considering:**
- Toyota Camry 2023 LE (₱1,450,000) - Fits your budget, excellent safety ratings
- Toyota Camry 2023 XLE (₱1,550,000) - Slightly over budget but premium features

**Available Options:**
I found 2 Toyota Camry models that match your requirements."""
        
        with patch('backend.agents.tools.get_current_employee_id', return_value="emp001"), \
             patch('backend.agents.tools._lookup_customer', return_value=sample_customer_data), \
             patch('backend.agents.tools._lookup_employee_details', return_value={"id": "emp001", "name": "Sales Rep"}), \
             patch('backend.agents.tools._merge_vehicle_requirements_intelligently', return_value="Toyota Camry sedan 2023 budget 1.5M for family use with safety features"), \
             patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \
             patch('backend.agents.tools._generate_enhanced_hitl_vehicle_prompt', return_value=enhanced_prompt), \
             patch('backend.agents.tools.get_recent_conversation_context', return_value=""), \
             patch('backend.agents.tools.extract_fields_from_conversation', return_value=sample_extracted_context), \
             patch('backend.agents.tools.request_input') as mock_input:
            
            mock_input.return_value = "HITL_ENHANCED_PROMPT"
            
            result = await generate_quotation.ainvoke({
                "customer_identifier": "Maria Santos",
                "vehicle_requirements": "family car",
                "quotation_state": quotation_state,
                "current_step": "vehicle_requirements",
                "user_response": "Toyota Camry sedan, 2023, budget 1.5M, need safety features"
            })
            
            # Verify HITL was triggered with enhanced prompt
            assert result == "HITL_ENHANCED_PROMPT"
            mock_input.assert_called_once()
            
            # Verify the enhanced prompt contains merged requirements
            call_args = mock_input.call_args
            assert "Thank you for the clarification" in call_args[1]["prompt"]
            assert "Toyota Camry sedan 2023" in call_args[1]["prompt"]
            assert "family use with safety features" in call_args[1]["prompt"]
            
            # Verify context preservation
            context = call_args[1]["context"]
            assert "extracted_context" in context
            assert context["extracted_context"]["family_size_mentions"] == ["family of 5", "kids", "children"]
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_error_handling(self):
        """Test enhanced prompt generation handles LLM errors gracefully."""
        
        with patch('backend.agents.tools.get_settings') as mock_settings, \
             patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
             patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
             patch('backend.agents.tools.StrOutputParser') as mock_parser:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
            mock_llm_class.return_value = MagicMock()
            mock_template.from_template.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            # Mock the chain to raise an exception
            mock_chain = AsyncMock()
            mock_chain.ainvoke.side_effect = Exception("LLM API error")
            mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            result = await _generate_enhanced_hitl_vehicle_prompt(
                vehicle_requirements="luxury car",
                extracted_context={},
                inventory_suggestions="No vehicles available",
                customer_data=None
            )
            
            # Verify fallback prompt is returned
            assert isinstance(result, str)
            assert "vehicle" in result.lower()
            assert "help" in result.lower()
            # Should contain basic fallback structure even when LLM fails
    
    @pytest.mark.asyncio 
    async def test_contextual_suggestions_with_empty_inventory(self):
        """Test enhanced prompts handle empty inventory gracefully."""
        
        empty_inventory_prompt = """🚗 **Let me help you find the right vehicle**

I understand you're looking for a "sports car" but unfortunately, we don't currently have any sports cars in our inventory.

**Based on your requirements, here's what I found:**
While we don't have sports cars available right now, I can help you find alternatives that might meet your performance and style preferences.

**I'd recommend considering:**
- Checking back in a few weeks for new arrivals
- Considering a sporty sedan that offers performance and practicality
- Looking at our pre-order options for upcoming sports models

**To find the perfect match, could you help me with:**
- What aspects of a sports car appeal to you most (performance, style, handling)?
- Would you consider a sporty sedan or coupe as an alternative?
- Are you flexible on timing if we can locate your preferred model?

**Available Options:**
Currently, our inventory is focused on family vehicles and business cars, but I can help you explore special order options."""
        
        with patch('backend.agents.tools.get_settings') as mock_settings, \
             patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
             patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
             patch('backend.agents.tools.StrOutputParser') as mock_parser:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
            mock_llm_class.return_value = MagicMock()
            mock_template.from_template.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            # Mock the chain to return empty inventory prompt
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = empty_inventory_prompt
            mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            result = await _generate_enhanced_hitl_vehicle_prompt(
                vehicle_requirements="sports car",
                extracted_context={},
                inventory_suggestions="No vehicles currently match your criteria.",
                customer_data=None
            )
            
            # Verify appropriate handling of empty inventory
            assert "don't currently have any sports cars" in result
            assert "alternatives" in result.lower()
            assert "special order" in result.lower() or "pre-order" in result.lower()
            assert "checking back" in result.lower()
            
            # Verify helpful questions are still asked
            assert "What aspects" in result or "would you consider" in result.lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_performance_benchmarks(self):
        """Test enhanced prompt generation performance."""
        
        # Test multiple rapid prompt generations to ensure performance
        test_scenarios = [
            ("family SUV", {"budget_mentions": ["under 2M"]}),
            ("business sedan", {"usage_mentions": ["client meetings"]}),
            ("compact car", {"preference_mentions": ["fuel efficient"]}),
            ("luxury vehicle", {"budget_mentions": ["premium budget"]}),
            ("pickup truck", {"usage_mentions": ["construction work"]})
        ]
        
        execution_times = []
        
        for requirements, context in test_scenarios:
            with patch('backend.agents.tools.get_settings') as mock_settings, \
                 patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
                 patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
                 patch('backend.agents.tools.StrOutputParser') as mock_parser:
                
                # Setup mocks for fast execution
                mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
                mock_llm_class.return_value = MagicMock()
                mock_template.from_template.return_value = MagicMock()
                mock_parser.return_value = MagicMock()
                
                # Mock fast chain response
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = f"Enhanced prompt for {requirements}"
                mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                import time
                start_time = time.time()
                
                result = await _generate_enhanced_hitl_vehicle_prompt(
                    vehicle_requirements=requirements,
                    extracted_context=context,
                    inventory_suggestions="Sample inventory",
                    customer_data=None
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                assert isinstance(result, str)
                assert requirements in result or "Enhanced prompt" in result
        
        # Performance assertions
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        print(f"📊 Enhanced Prompt Performance:")
        print(f"   Average generation time: {avg_time:.3f}s")
        print(f"   Maximum generation time: {max_time:.3f}s")
        print(f"   Total scenarios tested: {len(execution_times)}")
        
        # Enhanced prompts should be generated quickly even with LLM calls
        assert avg_time < 0.1, f"Average enhanced prompt time {avg_time:.3f}s exceeds 0.1s threshold"
        assert max_time < 0.2, f"Maximum enhanced prompt time {max_time:.3f}s exceeds 0.2s threshold"
        
        print("✅ Enhanced prompt performance benchmarks passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
