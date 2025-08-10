"""
Simple Multi-Turn Quotation Test

Tests the quotation generation tool directly with incremental information
to assess the HITL data completeness flow.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.toolbox.generate_quotation import generate_quotation
from agents.toolbox.toolbox import UserContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_incremental_information_gathering():
    """Test quotation generation with incremental information gathering."""
    logger.info("🚀 Starting Simple Multi-Turn Quotation Test")
    
    test_results = []
    
    try:
        # Set up employee context for all tests with proper UUID format
        employee_context = UserContext(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            user_type="employee",
            employee_id="550e8400-e29b-41d4-a716-446655440000",  # Valid UUID format
            conversation_id="test_conv_001"
        )
        
        with employee_context:
            # Turn 1: Minimal information - should trigger HITL
            logger.info("\n=== TURN 1: Minimal Information (Customer Name Only) ===")
            turn1_start = time.time()
            
            result1 = await generate_quotation.ainvoke({
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "car",
                "additional_notes": "Employee needs quotation"
            })
            
            turn1_time = time.time() - turn1_start
            test_results.append({
                "turn": 1,
                "input": "Grace Lee, car",
                "result": result1,
                "execution_time": turn1_time,
                "hitl_expected": True
            })
            
            logger.info(f"Turn 1 Result ({turn1_time:.2f}s): {result1[:200]}...")
            
            # Turn 2: Add vehicle make - should still trigger HITL
            logger.info("\n=== TURN 2: Add Vehicle Make ===")
            turn2_start = time.time()
            
            result2 = await generate_quotation.ainvoke({
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "Toyota sedan",
                "additional_notes": "Employee needs quotation",
                "user_response": "Toyota sedan",
                "hitl_phase": "vehicle_requirements"
            })
            
            turn2_time = time.time() - turn2_start
            test_results.append({
                "turn": 2,
                "input": "Grace Lee, Toyota sedan",
                "result": result2,
                "execution_time": turn2_time,
                "hitl_expected": True
            })
            
            logger.info(f"Turn 2 Result ({turn2_time:.2f}s): {result2[:200]}...")
            
            # Turn 3: Add specific model - should still need contact
            logger.info("\n=== TURN 3: Add Specific Model ===")
            turn3_start = time.time()
            
            result3 = await generate_quotation.ainvoke({
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "Toyota Camry 2024",
                "additional_notes": "Employee needs quotation",
                "user_response": "Toyota Camry 2024 model",
                "hitl_phase": "vehicle_specifications"
            })
            
            turn3_time = time.time() - turn3_start
            test_results.append({
                "turn": 3,
                "input": "Grace Lee, Toyota Camry 2024",
                "result": result3,
                "execution_time": turn3_time,
                "hitl_expected": True
            })
            
            logger.info(f"Turn 3 Result ({turn3_time:.2f}s): {result3[:200]}...")
            
            # Turn 4: Add contact information - should complete
            logger.info("\n=== TURN 4: Add Complete Contact Information ===")
            turn4_start = time.time()
            
            result4 = await generate_quotation.ainvoke({
                "customer_identifier": "Grace Lee",
                "vehicle_requirements": "Toyota Camry 2024",
                "additional_notes": "Customer: Grace Lee, Email: grace.lee@email.com, Phone: +63 917 123 4567",
                "user_response": "Contact information: grace.lee@email.com, phone: +63 917 123 4567",
                "hitl_phase": "contact_information"
            })
            
            turn4_time = time.time() - turn4_start
            test_results.append({
                "turn": 4,
                "input": "Grace Lee with complete contact info, Toyota Camry 2024",
                "result": result4,
                "execution_time": turn4_time,
                "hitl_expected": False,  # Should complete and generate PDF
                "pdf_expected": True
            })
            
            logger.info(f"Turn 4 Result ({turn4_time:.2f}s): {result4[:200]}...")
        
        # Analyze results
        analyze_test_results(test_results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_multi_turn_quotation_test_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_name": "Simple Multi-Turn Quotation Test",
                "timestamp": timestamp,
                "results": test_results,
                "summary": analyze_test_results(test_results, return_summary=True)
            }, f, indent=2)
            
        logger.info(f"✅ Test results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_test_results(results, return_summary=False):
    """Analyze the test results for patterns and success metrics."""
    
    hitl_detections = 0
    pdf_generations = 0
    total_time = 0
    
    for result in results:
        total_time += result["execution_time"]
        
        # Check for HITL indicators - look for the HITL_REQUIRED prefix
        response = result["result"].lower()
        if "hitl_required" in response or any(indicator in response for indicator in [
            "critical information needed", "need more information", "please provide", 
            "missing information", "additional details", "essential information"
        ]):
            hitl_detections += 1
            
        # Check for PDF generation indicators
        if any(indicator in response for indicator in [
            "professional vehicle quotation", "quotation generated", "pdf created", 
            "quotation link", "download your quotation", "quotation is ready",
            "pricing strategy", "validity"
        ]):
            pdf_generations += 1
    
    summary = {
        "total_turns": len(results),
        "hitl_detections": hitl_detections,
        "pdf_generations": pdf_generations,
        "total_execution_time": total_time,
        "average_turn_time": total_time / len(results) if results else 0,
        "hitl_effectiveness": hitl_detections >= 3,  # Expected HITL in first 3 turns
        "completion_success": pdf_generations >= 1,  # Expected PDF in final turn
        "overall_success": hitl_detections >= 3 and pdf_generations >= 1
    }
    
    if not return_summary:
        logger.info("\n📊 TEST ANALYSIS SUMMARY")
        logger.info(f"   Total Turns: {summary['total_turns']}")
        logger.info(f"   HITL Detections: {summary['hitl_detections']}")
        logger.info(f"   PDF Generations: {summary['pdf_generations']}")
        logger.info(f"   Total Time: {summary['total_execution_time']:.2f}s")
        logger.info(f"   Avg Turn Time: {summary['average_turn_time']:.2f}s")
        logger.info(f"   HITL Effectiveness: {'✅' if summary['hitl_effectiveness'] else '❌'}")
        logger.info(f"   Completion Success: {'✅' if summary['completion_success'] else '❌'}")
        logger.info(f"   Overall Success: {'✅' if summary['overall_success'] else '❌'}")
    
    return summary

async def main():
    """Main test execution."""
    logger.info("🎯 Simple Multi-Turn Quotation Information Gathering Test")
    
    try:
        await test_incremental_information_gathering()
        logger.info("🎉 Test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
