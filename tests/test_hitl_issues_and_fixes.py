#!/usr/bin/env python3

"""
HITL Issues Analysis and Fixes Test Suite

This test suite identifies specific issues found in the HITL implementation
during the comprehensive PRD assessment and validates fixes.

ISSUES IDENTIFIED:
1. HITL node treats "approved"/"denied" phases as "unexpected" and clears them
2. LLM natural language interpretation not working correctly in test environment
3. Human response detection logic is complex and fragile
4. State transitions for completion states are not handled properly

EXPECTED FIXES:
1. Handle completion states properly without clearing them
2. Simplify human response detection
3. Improve LLM interpretation robustness
4. Ensure proper state transitions for all phases
"""

import asyncio
import pytest
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4

# Test framework setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import HITL components
from backend.agents.hitl import (
    hitl_node, 
    parse_tool_response, 
    request_approval, 
    request_input, 
    request_selection,
    _process_hitl_response_llm_driven,
    _interpret_user_intent_with_llm
)


class TestHITLIssuesAndFixes:
    """
    Test suite specifically focused on identifying and testing fixes for HITL implementation issues.
    """

    @pytest.fixture
    def base_agent_state(self):
        """Create a basic agent state for testing."""
        return {
            "messages": [],
            "conversation_id": str(uuid4()),
            "user_id": "test_user_123",
            "customer_id": None,
            "employee_id": "emp_456",
            "retrieved_docs": [],
            "sources": [],
            "long_term_context": None,
            "conversation_summary": None,
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None
        }

    # =========================================================================
    # ISSUE 1: HITL Node Incorrectly Handles Completion States
    # =========================================================================

    async def test_issue_completion_states_cleared_incorrectly(self, base_agent_state):
        """
        ISSUE TEST: HITL node treats "approved"/"denied" phases as unexpected and clears them.
        
        Found in logs: "Unexpected phase 'approved', clearing HITL state"
        This violates PRD requirement that completion states should be handled properly.
        """
        logger.info("🐛 TESTING ISSUE: Completion States Cleared Incorrectly")
        
        # Test approved state handling
        approved_state = {
            **base_agent_state,
            "hitl_phase": "approved",
            "hitl_context": {"source_tool": "test_tool", "action": "test_action"},
            "messages": [
                {"role": "human", "content": "Please proceed", "type": "human"},
                {"role": "assistant", "content": "Should I proceed?", "type": "ai"},
                {"role": "human", "content": "yes", "type": "human"}
            ]
        }
        
        try:
            result = await hitl_node(approved_state)
            
            # ISSUE: The current implementation clears approved states
            # This test documents the current broken behavior
            current_phase = result.get("hitl_phase")
            logger.info(f"  Current behavior: approved → {current_phase}")
            
            if current_phase is None:
                logger.error("  🐛 CONFIRMED ISSUE: Approved state was incorrectly cleared")
                logger.error("  📋 EXPECTED: Should maintain 'approved' phase or transition appropriately")
            else:
                logger.info("  ✅ Approved state preserved correctly")
                
        except Exception as e:
            logger.error(f"  🐛 ISSUE: Exception during approved state handling: {e}")

        # Test denied state handling  
        denied_state = {
            **base_agent_state,
            "hitl_phase": "denied",
            "hitl_context": {"source_tool": "test_tool", "action": "test_action"},
            "messages": [
                {"role": "human", "content": "Please proceed", "type": "human"},
                {"role": "assistant", "content": "Should I proceed?", "type": "ai"},
                {"role": "human", "content": "no", "type": "human"}
            ]
        }
        
        try:
            result = await hitl_node(denied_state)
            
            current_phase = result.get("hitl_phase")
            logger.info(f"  Current behavior: denied → {current_phase}")
            
            if current_phase is None:
                logger.error("  🐛 CONFIRMED ISSUE: Denied state was incorrectly cleared")
                logger.error("  📋 EXPECTED: Should maintain 'denied' phase or transition appropriately")
            else:
                logger.info("  ✅ Denied state preserved correctly")
                
        except Exception as e:
            logger.error(f"  🐛 ISSUE: Exception during denied state handling: {e}")

        logger.info("🐛 ISSUE DOCUMENTED: Completion States Cleared Incorrectly")

    # =========================================================================
    # ISSUE 2: LLM Natural Language Interpretation Problems
    # =========================================================================

    async def test_issue_llm_interpretation_not_working(self):
        """
        ISSUE TEST: LLM natural language interpretation not working correctly.
        
        Found in test results: "send it" not interpreted as approval
        This suggests the LLM interpretation logic has issues.
        """
        logger.info("🐛 TESTING ISSUE: LLM Natural Language Interpretation Problems")
        
        # Test direct LLM interpretation function
        test_responses = [
            ("send it", "approval"),
            ("go ahead", "approval"), 
            ("not now", "denial"),
            ("cancel", "denial"),
            ("john@example.com", "input")
        ]
        
        context = {"source_tool": "test_tool"}
        
        for response, expected_intent in test_responses:
            try:
                actual_intent = await _interpret_user_intent_with_llm(response, context)
                
                if actual_intent == expected_intent:
                    logger.info(f"  ✅ '{response}' → {actual_intent} (correct)")
                else:
                    logger.error(f"  🐛 ISSUE: '{response}' → {actual_intent}, expected {expected_intent}")
                    
            except Exception as e:
                logger.error(f"  🐛 ISSUE: LLM interpretation failed for '{response}': {e}")

        logger.info("🐛 ISSUE DOCUMENTED: LLM Natural Language Interpretation Problems")

    async def test_issue_response_processing_pipeline(self, base_agent_state):
        """
        ISSUE TEST: End-to-end response processing pipeline problems.
        
        Tests the complete pipeline from human response to phase transition.
        """
        logger.info("🐛 TESTING ISSUE: Response Processing Pipeline Problems")
        
        # Set up a realistic HITL response scenario
        test_state = {
            **base_agent_state,
            "hitl_phase": "awaiting_response",
            "hitl_prompt": "Send this message to customer John?",
            "hitl_context": {"source_tool": "trigger_customer_message", "customer_id": "123"},
            "messages": [
                {"role": "assistant", "content": "Send this message to customer John?", "type": "ai"},
                {"role": "human", "content": "send it", "type": "human"}
            ]
        }
        
        try:
            # Test the complete processing pipeline
            result = await hitl_node(test_state)
            
            final_phase = result.get("hitl_phase")
            logger.info(f"  Pipeline result: 'send it' → phase: {final_phase}")
            
            if final_phase == "approved":
                logger.info("  ✅ Pipeline working correctly")
            else:
                logger.error(f"  🐛 ISSUE: Expected 'approved', got '{final_phase}'")
                logger.error(f"  🔍 Messages in result: {len(result.get('messages', []))}")
                logger.error(f"  🔍 Context preserved: {bool(result.get('hitl_context'))}")
                
        except Exception as e:
            logger.error(f"  🐛 ISSUE: Pipeline processing failed: {e}")

        logger.info("🐛 ISSUE DOCUMENTED: Response Processing Pipeline Problems")

    # =========================================================================
    # ISSUE 3: Human Response Detection Logic Problems  
    # =========================================================================

    async def test_issue_human_response_detection(self, base_agent_state):
        """
        ISSUE TEST: Complex human response detection logic is fragile.
        
        The current logic for finding human responses after HITL prompts is complex
        and may not work reliably in different message formats.
        """
        logger.info("🐛 TESTING ISSUE: Human Response Detection Logic Problems")
        
        # Test different message format scenarios
        message_scenarios = [
            {
                "name": "simple_format",
                "messages": [
                    {"role": "ai", "content": "Test prompt?", "type": "ai"},
                    {"role": "human", "content": "yes", "type": "human"}
                ],
                "expected_response": "yes"
            },
            {
                "name": "object_format", 
                "messages": [
                    {"content": "Test prompt?", "type": "ai"},
                    {"content": "approve", "type": "human"}
                ],
                "expected_response": "approve"  
            },
            {
                "name": "mixed_format",
                "messages": [
                    {"role": "assistant", "content": "Test prompt?"},
                    {"role": "human", "content": "send it"}
                ],
                "expected_response": "send it"
            }
        ]
        
        for scenario in message_scenarios:
            logger.info(f"  Testing scenario: {scenario['name']}")
            
            test_state = {
                **base_agent_state,
                "hitl_phase": "awaiting_response", 
                "hitl_prompt": "Test prompt?",
                "hitl_context": {"source_tool": "test_tool"},
                "messages": scenario["messages"]
            }
            
            try:
                # Test if the HITL node can detect the human response
                result = await hitl_node(test_state)
                
                final_phase = result.get("hitl_phase")
                if final_phase in ["approved", "denied"] or final_phase != "awaiting_response":
                    logger.info(f"    ✅ Response detected and processed: {scenario['expected_response']} → {final_phase}")
                else:
                    logger.error(f"    🐛 ISSUE: Response not detected: still in awaiting_response")
                    
            except Exception as e:
                logger.error(f"    🐛 ISSUE: Exception in {scenario['name']}: {e}")

        logger.info("🐛 ISSUE DOCUMENTED: Human Response Detection Logic Problems")

    # =========================================================================
    # ISSUE 4: State Consistency Problems
    # =========================================================================

    async def test_issue_state_consistency_problems(self, base_agent_state):
        """
        ISSUE TEST: State consistency problems across HITL interactions.
        
        Tests that state fields are properly maintained and not incorrectly cleared.
        """
        logger.info("🐛 TESTING ISSUE: State Consistency Problems")
        
        # Test that context is preserved through interactions
        initial_context = {
            "source_tool": "test_tool",
            "customer_id": "123",
            "action_data": {"message": "Hello John"}
        }
        
        test_state = {
            **base_agent_state,
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Send message to John?",
            "hitl_context": initial_context
        }
        
        try:
            # Step 1: Show prompt
            prompted_state = await hitl_node(test_state)
            
            prompt_context = prompted_state.get("hitl_context")
            if prompt_context == initial_context:
                logger.info("  ✅ Context preserved during prompt phase")
            else:
                logger.error(f"  🐛 ISSUE: Context changed during prompt phase")
                logger.error(f"    Original: {initial_context}")
                logger.error(f"    After prompt: {prompt_context}")
            
            # Step 2: Process response (simulate)
            response_state = {
                **prompted_state,
                "hitl_phase": "awaiting_response",
                "messages": prompted_state.get("messages", []) + [
                    {"role": "human", "content": "yes", "type": "human"}
                ]
            }
            
            final_state = await hitl_node(response_state)
            
            final_context = final_state.get("hitl_context")
            final_phase = final_state.get("hitl_phase")
            
            logger.info(f"  Final phase: {final_phase}")
            logger.info(f"  Context preserved: {bool(final_context)}")
            
            if final_context and "source_tool" in final_context:
                logger.info("  ✅ Essential context preserved")
            else:
                logger.error("  🐛 ISSUE: Essential context lost")
                
        except Exception as e:
            logger.error(f"  🐛 ISSUE: State consistency test failed: {e}")

        logger.info("🐛 ISSUE DOCUMENTED: State Consistency Problems")

    # =========================================================================
    # PROPOSED FIXES VALIDATION
    # =========================================================================

    def test_proposed_fix_completion_state_handling(self):
        """
        PROPOSED FIX TEST: Proper completion state handling.
        
        Tests how completion states should be handled according to PRD requirements.
        """
        logger.info("🔧 TESTING PROPOSED FIX: Completion State Handling")
        
        # According to PRD, completion states should:
        # 1. Not be treated as "unexpected" 
        # 2. Should allow agent to process them appropriately
        # 3. Should maintain context for tool re-calling if needed
        
        completion_states = ["approved", "denied"]
        
        for state_name in completion_states:
            logger.info(f"  Proposed behavior for '{state_name}' state:")
            logger.info(f"    • Should NOT clear the state automatically")
            logger.info(f"    • Should preserve context for agent processing")
            logger.info(f"    • Should allow agent to decide next steps")
            logger.info(f"    • Should route back to agent node, not HITL recursion")

        logger.info("🔧 PROPOSED FIX: Update hitl_node logic to handle completion states properly")

    def test_proposed_fix_simplified_response_detection(self):
        """
        PROPOSED FIX TEST: Simplified human response detection.
        
        Tests a simpler approach to detecting human responses.
        """
        logger.info("🔧 TESTING PROPOSED FIX: Simplified Response Detection")
        
        # Proposed fix: Simply look at the last message if it's human
        # This is more reliable than complex prompt-matching logic
        
        simple_scenarios = [
            {
                "messages": [{"role": "human", "content": "yes", "type": "human"}],
                "expected": "yes"
            },
            {
                "messages": [
                    {"role": "ai", "content": "Prompt?", "type": "ai"},
                    {"role": "human", "content": "approve", "type": "human"}
                ],
                "expected": "approve"
            }
        ]
        
        for i, scenario in enumerate(simple_scenarios):
            messages = scenario["messages"]
            expected = scenario["expected"]
            
            # Simple detection logic (proposed fix)
            if messages:
                last_msg = messages[-1]
                if (hasattr(last_msg, 'type') and last_msg.type == 'human') or \
                   (isinstance(last_msg, dict) and last_msg.get('type') == 'human'):
                    content = getattr(last_msg, 'content', last_msg.get('content', ''))
                    logger.info(f"    Scenario {i+1}: Detected response '{content}' ✅")
                else:
                    logger.error(f"    Scenario {i+1}: No human response detected ❌")
            
        logger.info("🔧 PROPOSED FIX: Use simpler last-message detection logic")

    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================

    def test_hitl_issues_summary_and_recommendations(self):
        """
        SUMMARY: All identified HITL issues and recommended fixes.
        """
        logger.info("\n" + "="*80)
        logger.info("📋 HITL IMPLEMENTATION ISSUES SUMMARY AND RECOMMENDATIONS")
        logger.info("="*80)
        
        issues_and_fixes = [
            {
                "issue": "Completion States Cleared Incorrectly",
                "description": "HITL node treats 'approved'/'denied' as unexpected and clears them",
                "location": "hitl.py lines 684-697",
                "impact": "Breaks PRD requirement for proper completion state handling",
                "fix": "Remove 'unexpected phase' logic, handle completion states properly"
            },
            {
                "issue": "LLM Natural Language Interpretation Problems", 
                "description": "LLM not correctly interpreting natural responses like 'send it'",
                "location": "_interpret_user_intent_with_llm function",
                "impact": "Users can't use natural language as specified in PRD",
                "fix": "Debug and improve LLM prompt, add fallback logic, test with actual API"
            },
            {
                "issue": "Complex Human Response Detection",
                "description": "Fragile logic for finding human responses after HITL prompts",
                "location": "hitl.py lines 958-982",
                "impact": "Responses may not be detected reliably in different scenarios",
                "fix": "Simplify to check last message, make more robust to different formats"
            },
            {
                "issue": "State Consistency Problems",
                "description": "Context and state may be lost during transitions",
                "location": "Various state transition points",
                "impact": "Tool re-calling and recursive collection may not work",
                "fix": "Ensure context preservation, add comprehensive state validation"
            }
        ]
        
        logger.info("\n🐛 IDENTIFIED ISSUES:")
        for i, item in enumerate(issues_and_fixes, 1):
            logger.info(f"\n{i}. {item['issue']}")
            logger.info(f"   Description: {item['description']}")
            logger.info(f"   Location: {item['location']}")
            logger.info(f"   Impact: {item['impact']}")
            logger.info(f"   Recommended Fix: {item['fix']}")
        
        logger.info("\n" + "="*80)
        logger.info("🎯 PRIORITY FIXES NEEDED:")
        logger.info("  1. Fix completion state handling (HIGH - breaks basic functionality)")
        logger.info("  2. Improve LLM interpretation reliability (HIGH - core PRD requirement)")
        logger.info("  3. Simplify response detection (MEDIUM - reliability improvement)")
        logger.info("  4. Add comprehensive state validation (MEDIUM - robustness)")
        
        logger.info("\n✅ CURRENT STATUS:")
        logger.info("  • Ultra-minimal 3-field architecture: ✅ IMPLEMENTED")
        logger.info("  • Dedicated HITL request tools: ✅ IMPLEMENTED")
        logger.info("  • Tool-managed collection pattern: ✅ IMPLEMENTED")
        logger.info("  • No HITL recursion routing: ❌ NEEDS FIXING")
        logger.info("  • LLM-native interpretation: ❌ NEEDS FIXING")
        logger.info("  • State consistency: ❌ NEEDS FIXING")
        
        logger.info("\n📊 OVERALL ASSESSMENT:")
        logger.info("  Implementation Status: PARTIALLY COMPLIANT")
        logger.info("  Critical Issues: 4 identified")
        logger.info("  Estimated Fix Effort: MEDIUM (architectural issues but well-defined)")
        logger.info("="*80)


# Test execution
if __name__ == "__main__":
    """
    Run HITL issues analysis and fixes test suite.
    """
    logger.info("🔍 Starting HITL Issues Analysis and Fixes Test Suite")
    
    # Run all tests
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "--disable-warnings",
        "--asyncio-mode=auto"
    ])
    
    logger.info("🏁 HITL Issues Analysis Complete")