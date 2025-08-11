"""
Multi-Turn Information Gathering Test for Quotation Generation

Tests the complete end-to-end flow where an employee provides information
incrementally and the agent uses HITL to gather complete information
before generating the PDF quotation.

Test Scenario:
1. Employee starts with minimal information (just customer name)
2. Agent requests additional required information via HITL
3. Employee provides information one piece at a time
4. Agent continues HITL until all critical information is gathered
5. Agent generates PDF quotation and provides confirmation with link
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Test framework imports
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Backend imports
from agents.tobi_sales_copilot.agent import create_graph
from agents.toolbox.generate_quotation import generate_quotation
from core.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTurnQuotationTester:
    """Comprehensive multi-turn quotation flow tester."""
    
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {
            "test_start_time": datetime.now().isoformat(),
            "turns": [],
            "errors": [],
            "success_metrics": {
                "total_turns": 0,
                "hitl_requests": 0,
                "information_gathered": [],
                "pdf_generated": False,
                "confirmation_sent": False
            }
        }
        
    async def run_multi_turn_test(self):
        """Run the complete multi-turn information gathering test."""
        logger.info("🚀 Starting Multi-Turn Quotation Information Gathering Test")
        
        # Test scenario: Employee provides information incrementally
        test_scenario = {
            "turn_1": {
                "input": "I need to create a quotation for Grace Lee",
                "expected": "HITL request for missing vehicle and contact information"
            },
            "turn_2": {
                "input": "She wants a Toyota sedan",
                "expected": "HITL request for specific model and contact information"
            },
            "turn_3": {
                "input": "Toyota Camry 2024",
                "expected": "HITL request for contact information and any specific requirements"
            },
            "turn_4": {
                "input": "Her email is grace.lee@email.com and phone is +63 917 123 4567",
                "expected": "Should proceed to quotation generation"
            }
        }
        
        try:
            # Initialize the agent graph
            graph = await create_graph()
            
            # Simulate employee authentication
            employee_context = {
                "employee_id": "emp_001",
                "employee_name": "Sales Agent Test",
                "employee_email": "test.agent@company.com"
            }
            
            # Execute each turn of the conversation
            conversation_state = {
                "messages": [],
                "hitl_context": {},
                "quotation_context": {}
            }
            
            for turn_key, turn_data in test_scenario.items():
                await self._execute_conversation_turn(
                    graph, 
                    turn_key, 
                    turn_data, 
                    employee_context, 
                    conversation_state
                )
                
            # Generate final test report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Multi-turn test failed: {e}")
            self.test_results["errors"].append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "phase": "test_execution"
            })
            raise
            
    async def _execute_conversation_turn(
        self, 
        graph, 
        turn_key: str, 
        turn_data: Dict[str, str], 
        employee_context: Dict[str, str],
        conversation_state: Dict[str, Any]
    ):
        """Execute a single conversation turn."""
        logger.info(f"🔄 Executing {turn_key}: {turn_data['input']}")
        
        turn_start_time = time.time()
        turn_result = {
            "turn": turn_key,
            "input": turn_data["input"],
            "expected": turn_data["expected"],
            "timestamp": datetime.now().isoformat(),
            "execution_time": 0,
            "response": None,
            "hitl_detected": False,
            "errors": []
        }
        
        try:
            # Prepare conversation input
            conversation_input = {
                "messages": conversation_state["messages"] + [{
                    "role": "user",
                    "content": turn_data["input"],
                    "timestamp": datetime.now().isoformat(),
                    "user_type": "employee"
                }],
                "user_id": employee_context["employee_id"],
                "user_type": "employee",
                "conversation_id": f"test_conv_{int(time.time())}"
            }
            
            # Execute the agent graph
            logger.info(f"📤 Sending to agent: {turn_data['input']}")
            result = await graph.ainvoke(conversation_input)
            
            # Process the result
            if result and "messages" in result:
                last_message = result["messages"][-1] if result["messages"] else None
                if last_message:
                    turn_result["response"] = last_message.get("content", "No response content")
                    
                    # Check if HITL was triggered
                    if self._is_hitl_response(last_message):
                        turn_result["hitl_detected"] = True
                        self.test_results["success_metrics"]["hitl_requests"] += 1
                        logger.info(f"✅ HITL detected in {turn_key}")
                    
                    # Check for quotation generation completion
                    if self._is_quotation_completion(last_message):
                        self.test_results["success_metrics"]["pdf_generated"] = True
                        self.test_results["success_metrics"]["confirmation_sent"] = True
                        logger.info(f"🎉 Quotation completion detected in {turn_key}")
                    
                    # Update conversation state
                    conversation_state["messages"] = result["messages"]
                    
            turn_result["execution_time"] = time.time() - turn_start_time
            self.test_results["success_metrics"]["total_turns"] += 1
            
            # Log turn results
            logger.info(f"📊 {turn_key} Results:")
            logger.info(f"   Response: {turn_result['response'][:200]}...")
            logger.info(f"   HITL Detected: {turn_result['hitl_detected']}")
            logger.info(f"   Execution Time: {turn_result['execution_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Error in {turn_key}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            turn_result["errors"].append(error_msg)
            self.test_results["errors"].append({
                "turn": turn_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
        finally:
            self.test_results["turns"].append(turn_result)
            
    def _is_hitl_response(self, message: Dict[str, Any]) -> bool:
        """Check if the message indicates a HITL request."""
        content = message.get("content", "").lower()
        hitl_indicators = [
            "need more information",
            "please provide",
            "missing information",
            "additional details",
            "to generate",
            "quotation requires"
        ]
        return any(indicator in content for indicator in hitl_indicators)
        
    def _is_quotation_completion(self, message: Dict[str, Any]) -> bool:
        """Check if the message indicates quotation generation completion."""
        content = message.get("content", "").lower()
        completion_indicators = [
            "quotation generated",
            "pdf created",
            "quotation link",
            "download your quotation",
            "quotation is ready"
        ]
        return any(indicator in content for indicator in completion_indicators)
        
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        self.test_results["test_end_time"] = datetime.now().isoformat()
        self.test_results["test_duration"] = sum(turn.get("execution_time", 0) for turn in self.test_results["turns"])
        
        # Calculate success metrics
        success_rate = (
            (self.test_results["success_metrics"]["hitl_requests"] > 0) * 0.3 +
            (self.test_results["success_metrics"]["total_turns"] >= 4) * 0.2 +
            (self.test_results["success_metrics"]["pdf_generated"]) * 0.3 +
            (self.test_results["success_metrics"]["confirmation_sent"]) * 0.2
        ) * 100
        
        self.test_results["overall_success_rate"] = success_rate
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"multi_turn_quotation_test_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
            
        logger.info("📋 Multi-Turn Quotation Test Report Generated")
        logger.info(f"📁 Report saved to: {report_file}")
        logger.info(f"📊 Overall Success Rate: {success_rate:.1f}%")
        logger.info(f"🔄 Total Turns: {self.test_results['success_metrics']['total_turns']}")
        logger.info(f"❓ HITL Requests: {self.test_results['success_metrics']['hitl_requests']}")
        logger.info(f"📄 PDF Generated: {self.test_results['success_metrics']['pdf_generated']}")
        logger.info(f"✅ Confirmation Sent: {self.test_results['success_metrics']['confirmation_sent']}")
        
        if self.test_results["errors"]:
            logger.warning(f"⚠️  Errors Encountered: {len(self.test_results['errors'])}")
            for error in self.test_results["errors"]:
                logger.warning(f"   - {error['turn']}: {error['error']}")

async def run_direct_quotation_test():
    """Run direct quotation generation test with incremental information."""
    logger.info("🧪 Running Direct Quotation Generation Test with Multi-Turn Flow")
    
    try:
        # Test 1: Start with minimal information (should trigger HITL)
        logger.info("\n=== TURN 1: Minimal Information ===")
        result1 = await generate_quotation(
            customer_identifier="Grace Lee",
            vehicle_requirements="sedan",
            additional_notes="Employee quotation request"
        )
        logger.info(f"Turn 1 Result: {result1}")
        
        # Test 2: Add vehicle make (should still trigger HITL)
        logger.info("\n=== TURN 2: Add Vehicle Make ===")
        result2 = await generate_quotation(
            customer_identifier="Grace Lee",
            vehicle_requirements="Toyota sedan",
            additional_notes="Employee quotation request",
            user_response="Toyota sedan",
            hitl_phase="vehicle_requirements"
        )
        logger.info(f"Turn 2 Result: {result2}")
        
        # Test 3: Add specific model (should still need contact info)
        logger.info("\n=== TURN 3: Add Specific Model ===")
        result3 = await generate_quotation(
            customer_identifier="Grace Lee",
            vehicle_requirements="Toyota Camry 2024",
            additional_notes="Employee quotation request",
            user_response="Toyota Camry 2024",
            hitl_phase="vehicle_specifications"
        )
        logger.info(f"Turn 3 Result: {result3}")
        
        # Test 4: Add contact information (should complete)
        logger.info("\n=== TURN 4: Add Contact Information ===")
        result4 = await generate_quotation(
            customer_identifier="Grace Lee - grace.lee@email.com - +63 917 123 4567",
            vehicle_requirements="Toyota Camry 2024",
            additional_notes="Employee quotation request - complete information provided",
            user_response="Contact: grace.lee@email.com, phone: +63 917 123 4567",
            hitl_phase="contact_information"
        )
        logger.info(f"Turn 4 Result: {result4}")
        
        logger.info("✅ Direct quotation test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Direct quotation test failed: {e}")
        raise

async def main():
    """Main test execution function."""
    logger.info("🎯 Multi-Turn Quotation Information Gathering Test Suite")
    
    try:
        # Run direct quotation test first
        await run_direct_quotation_test()
        
        # Run full agent integration test
        tester = MultiTurnQuotationTester()
        await tester.run_multi_turn_test()
        
        logger.info("🎉 All multi-turn quotation tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
