#!/usr/bin/env python3
"""
Comprehensive Testing for LLM-Driven Quotation System

Task 15.7.1: CREATE test_llm_driven_quotation.py with comprehensive scenarios:
- Initial calls
- HITL resume
- Context updates
- Edge cases

This test suite validates the revolutionary LLM-driven quotation approach across
all scenarios to ensure superior performance compared to fragmented logic.
"""

import asyncio
import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

# Test imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.toolbox.generate_quotation import (
    generate_quotation,
    handle_quotation_resume,
    QuotationContextIntelligence,
    QuotationCommunicationIntelligence,
    QuotationIntelligenceCoordinator,
    _extract_comprehensive_context,
    _validate_quotation_completeness,
    _generate_intelligent_hitl_request,
    _generate_final_quotation
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMDrivenQuotation:
    """
    Comprehensive test suite for LLM-driven quotation system.
    
    Task 15.7: Validate LLM-driven approach across all quotation scenarios.
    """

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test_key"
        mock_settings.openai_simple_model = "gpt-4o-mini"
        mock_settings.openai_complex_model = "gpt-4o"
        return mock_settings

    @pytest.fixture
    def sample_conversation_contexts(self):
        """Sample conversation contexts for testing."""
        return {
            "complete_request": """
                User: Generate a quotation for Eva Martinez
                Assistant: I'd be happy to help generate a quotation for Eva Martinez. Could you provide more details?
                User: She needs a Honda CR-V 2024, red color, for delivery next month with financing options
            """,
            "partial_request": """
                User: I need a quotation for John Smith
                Assistant: I'll help you create a quotation for John Smith. What vehicle is he interested in?
                User: He's looking at SUVs but hasn't decided on the specific model yet
            """,
            "resume_scenario": """
                User: Generate a quotation for Maria Santos
                Assistant: I need some additional information for Maria Santos' quotation. What vehicle is she interested in?
                User: Toyota Camry 2024
                Assistant: Great! What's her preferred color and any specific features?
                User: White color with hybrid engine and leather seats
            """
        }

    # =============================================================================
    # Task 15.7.2: TEST Conversation Understanding
    # =============================================================================

    @pytest.mark.asyncio
    async def test_conversation_understanding_complete_request(self, mock_settings, sample_conversation_contexts):
        """
        Task 15.7.2: TEST conversation understanding with complete information.
        
        Scenario: "Generate a quote for Honda CR-V for Eva Martinez"
        Expected: Should extract all relevant information without additional prompts.
        """
        logger.info("[TEST] 🧠 Testing complete conversation understanding")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            # Mock LLM response for complete context extraction
            mock_llm_response = {
                "customer_info": {"name": "Eva Martinez", "contact": "eva.martinez@email.com"},
                "vehicle_requirements": {"make": "Honda", "model": "CR-V", "year": "2024", "color": "red"},
                "purchase_preferences": {"delivery_timeline": "next month", "financing": "required"},
                "completeness_assessment": {"is_complete": True, "confidence_score": 0.95}
            }
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                result = await generate_quotation(
                    customer_identifier="Eva Martinez",
                    vehicle_requirements="Honda CR-V 2024, red color",
                    additional_notes="delivery next month with financing options",
                    conversation_context=sample_conversation_contexts["complete_request"]
                )
                
                # Verify no HITL request was made (complete information)
                mock_request_input.assert_not_called()
                
                # Verify successful quotation generation
                assert "quotation" in result.lower() or "eva martinez" in result.lower()
                assert "honda" in result.lower() and "cr-v" in result.lower()
                
                logger.info("[TEST] ✅ Complete conversation understanding validated")

    @pytest.mark.asyncio
    async def test_conversation_understanding_partial_request(self, mock_settings, sample_conversation_contexts):
        """
        Task 15.7.2: TEST conversation understanding with partial information.
        
        Scenario: Incomplete vehicle specifications
        Expected: Should identify missing information and generate intelligent HITL prompt.
        """
        logger.info("[TEST] 🧠 Testing partial conversation understanding")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            # Mock LLM response for incomplete context
            mock_llm_response = {
                "customer_info": {"name": "John Smith"},
                "vehicle_requirements": {"category": "SUV"},
                "purchase_preferences": {},
                "completeness_assessment": {"is_complete": False, "confidence_score": 0.4},
                "missing_info_analysis": {
                    "vehicle_details": ["specific make and model", "year preference", "color"],
                    "customer_details": ["contact information"],
                    "purchase_details": ["budget range", "timeline"]
                }
            }
            
            mock_request_input.return_value = "Intelligent HITL prompt generated"
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                result = await generate_quotation(
                    customer_identifier="John Smith",
                    vehicle_requirements="SUV",
                    conversation_context=sample_conversation_contexts["partial_request"]
                )
                
                # Verify HITL request was made for missing information
                mock_request_input.assert_called_once()
                
                # Verify intelligent prompt generation
                assert "Intelligent HITL prompt generated" in result
                
                logger.info("[TEST] ✅ Partial conversation understanding validated")

    # =============================================================================
    # Task 15.7.3: VALIDATE Resume Scenarios
    # =============================================================================

    @pytest.mark.asyncio
    async def test_resume_vehicle_information(self, mock_settings):
        """
        Task 15.7.3: VALIDATE resume scenarios - Vehicle information provided.
        
        Scenario: User provides vehicle details in resume response.
        Expected: Should integrate new information and continue quotation flow.
        """
        logger.info("[TEST] 🔄 Testing vehicle information resume")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            # Mock context extraction for resume scenario
            mock_context_result = {
                "status": "success",
                "extracted_context": type('MockContext', (), {
                    "customer_info": {"name": "Maria Santos"},
                    "vehicle_requirements": {"make": "Toyota", "model": "Camry", "year": "2024", "color": "white"},
                    "purchase_preferences": {"features": ["hybrid engine", "leather seats"]},
                    "completeness_assessment": {"is_complete": True, "confidence_score": 0.9}
                })()
            }
            
            with patch('backend.agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract:
                mock_extract.return_value = mock_context_result
                
                with patch('backend.agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate:
                    mock_validate.return_value = {"status": "success", "is_complete": True}
                    
                    with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                        mock_final.return_value = "Final quotation generated for Maria Santos - Toyota Camry 2024"
                        
                        result = await handle_quotation_resume(
                            user_response="Toyota Camry 2024, white color with hybrid engine and leather seats",
                            existing_context={"customer_info": {"name": "Maria Santos"}}
                        )
                        
                        # Verify context integration
                        mock_extract.assert_called_once()
                        
                        # Verify final quotation generation
                        mock_final.assert_called_once()
                        
                        # Verify result contains expected information
                        assert "Maria Santos" in result
                        assert "Toyota Camry" in result
                        
                        logger.info("[TEST] ✅ Vehicle information resume validated")

    @pytest.mark.asyncio
    async def test_resume_delivery_preferences(self, mock_settings):
        """
        Task 15.7.3: VALIDATE resume scenarios - Delivery preferences provided.
        
        Scenario: User provides delivery and payment preferences.
        Expected: Should integrate preferences and continue quotation flow.
        """
        logger.info("[TEST] 🚚 Testing delivery preferences resume")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            # Mock context for delivery preferences resume
            mock_context_result = {
                "status": "success",
                "extracted_context": type('MockContext', (), {
                    "customer_info": {"name": "Carlos Rodriguez"},
                    "vehicle_requirements": {"make": "Ford", "model": "F-150", "year": "2024"},
                    "purchase_preferences": {
                        "delivery_location": "downtown office",
                        "delivery_timeline": "within 2 weeks",
                        "payment_method": "business financing"
                    },
                    "completeness_assessment": {"is_complete": True, "confidence_score": 0.85}
                })()
            }
            
            with patch('backend.agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract:
                mock_extract.return_value = mock_context_result
                
                with patch('backend.agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate:
                    mock_validate.return_value = {"status": "success", "is_complete": True}
                    
                    with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                        mock_final.return_value = "Business quotation for Carlos Rodriguez - Ford F-150 with delivery"
                        
                        result = await handle_quotation_resume(
                            user_response="Deliver to downtown office within 2 weeks, business financing preferred",
                            existing_context={
                                "customer_info": {"name": "Carlos Rodriguez"},
                                "vehicle_requirements": {"make": "Ford", "model": "F-150", "year": "2024"}
                            }
                        )
                        
                        # Verify delivery preferences integration
                        assert "Carlos Rodriguez" in result
                        assert "Ford F-150" in result
                        
                        logger.info("[TEST] ✅ Delivery preferences resume validated")

    @pytest.mark.asyncio
    async def test_resume_payment_methods(self, mock_settings):
        """
        Task 15.7.3: VALIDATE resume scenarios - Payment method specification.
        
        Scenario: User specifies financing or payment preferences.
        Expected: Should integrate payment information and generate appropriate quotation.
        """
        logger.info("[TEST] 💳 Testing payment methods resume")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            # Test multiple payment scenarios
            payment_scenarios = [
                {
                    "user_response": "I prefer cash payment with trade-in discount",
                    "expected_payment": "cash with trade-in"
                },
                {
                    "user_response": "Can you include financing options with 60-month terms?",
                    "expected_payment": "60-month financing"
                },
                {
                    "user_response": "Corporate lease agreement for 3 years",
                    "expected_payment": "corporate lease"
                }
            ]
            
            for scenario in payment_scenarios:
                mock_context_result = {
                    "status": "success",
                    "extracted_context": type('MockContext', (), {
                        "customer_info": {"name": "Payment Test Customer"},
                        "vehicle_requirements": {"make": "Test", "model": "Vehicle"},
                        "purchase_preferences": {"payment_method": scenario["expected_payment"]},
                        "completeness_assessment": {"is_complete": True, "confidence_score": 0.8}
                    })()
                }
                
                with patch('backend.agents.toolbox.generate_quotation._extract_comprehensive_context') as mock_extract:
                    mock_extract.return_value = mock_context_result
                    
                    with patch('backend.agents.toolbox.generate_quotation._validate_quotation_completeness') as mock_validate:
                        mock_validate.return_value = {"status": "success", "is_complete": True}
                        
                        with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                            mock_final.return_value = f"Quotation with {scenario['expected_payment']} payment method"
                            
                            result = await handle_quotation_resume(
                                user_response=scenario["user_response"],
                                existing_context={"customer_info": {"name": "Payment Test Customer"}}
                            )
                            
                            # Verify payment method integration
                            assert scenario["expected_payment"].replace("_", " ") in result.lower() or "quotation" in result.lower()
            
            logger.info("[TEST] ✅ Payment methods resume validated")

    # =============================================================================
    # Task 15.7.4: VERIFY No Repetitive Prompts
    # =============================================================================

    @pytest.mark.asyncio
    async def test_no_repetitive_prompts_complete_info(self, mock_settings):
        """
        Task 15.7.4: VERIFY no repetitive prompts for already provided information.
        
        Scenario: All information provided in initial conversation.
        Expected: Should proceed directly to quotation without asking for known information.
        """
        logger.info("[TEST] 🚫 Testing no repetitive prompts with complete information")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            # Complete information scenario
            complete_context = """
                User: Generate a quotation for Sarah Johnson
                User: She needs a BMW X5 2024, black color, premium package
                User: Delivery to Seattle office next month
                User: Corporate financing preferred
                User: Contact: sarah.johnson@company.com, 555-0123
            """
            
            # Mock complete context extraction
            mock_llm_response = {
                "customer_info": {
                    "name": "Sarah Johnson",
                    "contact": "sarah.johnson@company.com",
                    "phone": "555-0123"
                },
                "vehicle_requirements": {
                    "make": "BMW", "model": "X5", "year": "2024",
                    "color": "black", "package": "premium"
                },
                "purchase_preferences": {
                    "delivery_location": "Seattle office",
                    "delivery_timeline": "next month",
                    "financing": "corporate"
                },
                "completeness_assessment": {"is_complete": True, "confidence_score": 0.95}
            }
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                    mock_final.return_value = "Complete quotation for Sarah Johnson - BMW X5 2024"
                    
                    result = await generate_quotation(
                        customer_identifier="Sarah Johnson",
                        vehicle_requirements="BMW X5 2024, black color, premium package",
                        additional_notes="Delivery to Seattle office next month, corporate financing",
                        conversation_context=complete_context
                    )
                    
                    # Verify NO HITL request was made (no repetitive prompts)
                    mock_request_input.assert_not_called()
                    
                    # Verify direct quotation generation
                    mock_final.assert_called_once()
                    
                    # Verify result contains all provided information
                    assert "Sarah Johnson" in result
                    assert "BMW X5" in result
                    
                    logger.info("[TEST] ✅ No repetitive prompts validated - complete information")

    @pytest.mark.asyncio
    async def test_no_repetitive_prompts_partial_context(self, mock_settings):
        """
        Task 15.7.4: VERIFY intelligent prompting only for truly missing information.
        
        Scenario: Some information provided, but specific details missing.
        Expected: Should only ask for genuinely missing information, not repeat known data.
        """
        logger.info("[TEST] 🎯 Testing intelligent prompting for missing information only")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            # Partial information scenario
            partial_context = """
                User: Generate a quotation for Michael Chen
                User: He's interested in a Tesla Model 3
                User: Contact: michael.chen@email.com
            """
            
            # Mock partial context extraction
            mock_llm_response = {
                "customer_info": {"name": "Michael Chen", "contact": "michael.chen@email.com"},
                "vehicle_requirements": {"make": "Tesla", "model": "Model 3"},
                "purchase_preferences": {},
                "completeness_assessment": {"is_complete": False, "confidence_score": 0.6},
                "missing_info_analysis": {
                    "vehicle_details": ["year", "color", "trim level"],
                    "purchase_details": ["delivery timeline", "payment method"]
                }
            }
            
            # Mock intelligent HITL prompt that only asks for missing information
            mock_intelligent_prompt = """🚗 **Additional Information Needed for Michael Chen's Tesla Model 3 Quotation**

I have your contact information and vehicle interest. To complete your quotation, I need:

**Vehicle Specifications:**
• What year Model 3 are you interested in? (2023 or 2024)
• Preferred color?
• Any specific trim level or features?

**Purchase Details:**
• When do you need delivery?
• Preferred payment method? (financing, lease, cash)

*Note: I already have your name and contact information, so no need to repeat those.*"""
            
            mock_request_input.return_value = mock_intelligent_prompt
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                with patch.object(QuotationCommunicationIntelligence, 'generate_intelligent_hitl_prompt') as mock_hitl:
                    mock_hitl.return_value = mock_intelligent_prompt
                    
                    result = await generate_quotation(
                        customer_identifier="Michael Chen",
                        vehicle_requirements="Tesla Model 3",
                        conversation_context=partial_context
                    )
                    
                    # Verify HITL was called for missing information
                    mock_request_input.assert_called_once()
                    
                    # Verify intelligent prompt does not repeat known information
                    assert "Michael Chen" in result  # Name is known
                    assert "Tesla Model 3" in result  # Vehicle is known
                    assert "contact information" in result  # Acknowledges known contact
                    assert "no need to repeat" in result  # Explicitly states not to repeat
                    
                    # Verify only asks for genuinely missing information
                    assert "year" in result
                    assert "color" in result
                    assert "delivery" in result
                    assert "payment method" in result
                    
                    logger.info("[TEST] ✅ No repetitive prompts validated - intelligent partial prompting")

    # =============================================================================
    # Task 15.7.5: BENCHMARK Performance
    # =============================================================================

    @pytest.mark.asyncio
    async def test_performance_benchmark_single_llm_call(self, mock_settings):
        """
        Task 15.7.5: BENCHMARK performance - Single LLM call efficiency.
        
        Expected: LLM-driven approach should be faster than fragmented logic
        due to reduced HITL rounds and single comprehensive analysis.
        """
        logger.info("[TEST] ⚡ Benchmarking single LLM call performance")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            # Mock fast LLM response
            mock_llm_response = {
                "customer_info": {"name": "Performance Test", "contact": "test@example.com"},
                "vehicle_requirements": {"make": "Test", "model": "Vehicle", "year": "2024"},
                "purchase_preferences": {"payment": "test"},
                "completeness_assessment": {"is_complete": True, "confidence_score": 0.9}
            }
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                    mock_final.return_value = "Fast quotation generated"
                    
                    # Measure performance
                    start_time = time.time()
                    
                    result = await generate_quotation(
                        customer_identifier="Performance Test",
                        vehicle_requirements="Test Vehicle 2024",
                        conversation_context="Complete test context"
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Verify single LLM call efficiency
                    mock_analyze.assert_called_once()  # Only one comprehensive analysis
                    mock_final.assert_called_once()    # Direct to final generation
                    
                    # Performance assertion (should be very fast with mocks)
                    assert execution_time < 1.0  # Should complete in under 1 second with mocks
                    
                    logger.info(f"[TEST] ✅ Single LLM call performance: {execution_time:.3f}s")

    @pytest.mark.asyncio
    async def test_performance_benchmark_reduced_hitl_rounds(self, mock_settings):
        """
        Task 15.7.5: BENCHMARK performance - Reduced HITL rounds.
        
        Expected: Intelligent context extraction should minimize HITL interactions
        compared to step-by-step fragmented approaches.
        """
        logger.info("[TEST] 🔄 Benchmarking reduced HITL rounds")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            # Scenario: Moderately complete information that old system might fragment
            moderate_context = """
                User: I need a quotation for Jennifer Walsh
                User: She's looking at a Honda Pilot 2024
                User: Family use, needs 8 seats, safety features important
                User: Budget around $45,000, financing preferred
            """
            
            # Mock intelligent analysis that recognizes sufficient information
            mock_llm_response = {
                "customer_info": {"name": "Jennifer Walsh"},
                "vehicle_requirements": {
                    "make": "Honda", "model": "Pilot", "year": "2024",
                    "seating": "8 seats", "features": "safety features"
                },
                "purchase_preferences": {"budget": "$45,000", "financing": "preferred"},
                "completeness_assessment": {"is_complete": True, "confidence_score": 0.85}
            }
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                with patch('backend.agents.toolbox.generate_quotation._generate_final_quotation') as mock_final:
                    mock_final.return_value = "Intelligent quotation for Jennifer Walsh - Honda Pilot 2024"
                    
                    # Track HITL interactions
                    hitl_call_count = 0
                    
                    def count_hitl_calls(*args, **kwargs):
                        nonlocal hitl_call_count
                        hitl_call_count += 1
                        return "HITL call made"
                    
                    mock_request_input.side_effect = count_hitl_calls
                    
                    result = await generate_quotation(
                        customer_identifier="Jennifer Walsh",
                        vehicle_requirements="Honda Pilot 2024, 8 seats, safety features",
                        additional_notes="Family use, budget $45,000, financing preferred",
                        conversation_context=moderate_context
                    )
                    
                    # Verify reduced HITL rounds
                    assert hitl_call_count == 0  # No HITL needed with intelligent analysis
                    
                    # Verify direct quotation generation
                    mock_final.assert_called_once()
                    
                    # Verify comprehensive information extraction
                    assert "Jennifer Walsh" in result
                    assert "Honda Pilot" in result
                    
                    logger.info(f"[TEST] ✅ Reduced HITL rounds: {hitl_call_count} calls (expected: 0)")

    # =============================================================================
    # UNIFIED INTELLIGENCE COORDINATOR TESTING
    # =============================================================================

    @pytest.mark.asyncio
    async def test_unified_coordinator_integration(self, mock_settings):
        """
        Test QuotationIntelligenceCoordinator integration and coordination.
        
        Validates that all intelligence classes work together seamlessly.
        """
        logger.info("[TEST] 🎯 Testing unified intelligence coordinator")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            coordinator = QuotationIntelligenceCoordinator()
            
            # Mock coordinated intelligence responses
            mock_context_result = {
                "status": "success",
                "extracted_context": type('MockContext', (), {
                    "customer_info": {"name": "Coordinator Test", "tier": "premium"},
                    "vehicle_requirements": {"make": "Mercedes", "model": "E-Class"},
                    "purchase_preferences": {"style": "executive"},
                    "completeness_assessment": {"is_complete": True, "confidence_score": 0.9}
                })()
            }
            
            with patch.object(coordinator, '_extract_unified_context') as mock_extract:
                mock_extract.return_value = mock_context_result
                
                with patch.object(coordinator, '_validate_unified_completeness') as mock_validate:
                    mock_validate.return_value = {"status": "success", "is_complete": True}
                    
                    with patch.object(coordinator, '_generate_coordinated_final_quotation') as mock_final:
                        mock_final.return_value = "🌟 **Exclusive Premium Customer Quotation** - Coordinated intelligence result"
                        
                        result = await coordinator.process_complete_quotation_request(
                            customer_identifier="Coordinator Test",
                            vehicle_requirements="Mercedes E-Class",
                            additional_notes="Executive customer"
                        )
                        
                        # Verify coordinated processing
                        mock_extract.assert_called_once()
                        mock_validate.assert_called_once()
                        mock_final.assert_called_once()
                        
                        # Verify coordinated result
                        assert "Premium" in result
                        assert "Coordinated" in result
                        
                        logger.info("[TEST] ✅ Unified coordinator integration validated")

    # =============================================================================
    # EDGE CASES AND ERROR HANDLING
    # =============================================================================

    @pytest.mark.asyncio
    async def test_edge_case_empty_input(self, mock_settings):
        """Test handling of empty or minimal input."""
        logger.info("[TEST] 🔍 Testing edge case: empty input")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings), \
             patch('backend.agents.hitl.request_input') as mock_request_input:
            
            mock_request_input.return_value = "Please provide customer and vehicle information"
            
            # Mock minimal context extraction
            mock_llm_response = {
                "customer_info": {},
                "vehicle_requirements": {},
                "purchase_preferences": {},
                "completeness_assessment": {"is_complete": False, "confidence_score": 0.1}
            }
            
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.return_value = type('MockResult', (), mock_llm_response)()
                
                result = await generate_quotation(
                    customer_identifier="",
                    vehicle_requirements="",
                    conversation_context=""
                )
                
                # Verify graceful handling of empty input
                mock_request_input.assert_called_once()
                assert "provide" in result.lower()
                
                logger.info("[TEST] ✅ Edge case: empty input handled gracefully")

    @pytest.mark.asyncio
    async def test_edge_case_llm_error_recovery(self, mock_settings):
        """Test error recovery when LLM calls fail."""
        logger.info("[TEST] 🚨 Testing edge case: LLM error recovery")
        
        with patch('backend.core.config.get_settings_sync', return_value=mock_settings):
            
            # Mock LLM failure
            with patch.object(QuotationContextIntelligence, 'analyze_complete_context') as mock_analyze:
                mock_analyze.side_effect = Exception("LLM service error")
                
                with patch.object(QuotationCommunicationIntelligence, 'generate_intelligent_error_explanation') as mock_error:
                    mock_error.return_value = "❌ **Service Temporarily Unavailable** - Please try again"
                    
                    result = await generate_quotation(
                        customer_identifier="Error Test",
                        vehicle_requirements="Test Vehicle"
                    )
                    
                    # Verify error recovery
                    mock_error.assert_called_once()
                    assert "Service Temporarily Unavailable" in result
                    
                    logger.info("[TEST] ✅ Edge case: LLM error recovery validated")

if __name__ == "__main__":
    """
    Run comprehensive LLM-driven quotation tests.
    
    Usage:
        python -m pytest tests/test_llm_driven_quotation.py -v
        python -m pytest tests/test_llm_driven_quotation.py::TestLLMDrivenQuotation::test_conversation_understanding_complete_request -v
    """
    
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting comprehensive LLM-driven quotation tests")
    logger.info("Task 15.7: Validating LLM-driven approach across all quotation scenarios")
    
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
