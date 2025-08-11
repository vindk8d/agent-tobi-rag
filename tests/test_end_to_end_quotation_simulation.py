"""
End-to-End Quotation Generation Flow Simulation

This test simulates the complete flow of an employee requesting the bot to generate
a quotation for a customer, capturing each step and agent response for debugging.
"""

import os
import pytest
import asyncio
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip if no Supabase credentials
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available"
)

# Test parameters
EMPLOYEE_USER_ID = "550e8400-e29b-41d4-a716-446655440000"  # Test employee
TEST_CUSTOMER_NAME = "Maria Santos"
TEST_VEHICLE_REQUIREMENTS = "Toyota Camry sedan, white color preferred"

class QuotationFlowSimulator:
    """Simulates the complete quotation generation flow step by step."""
    
    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.step_responses = []
        self.agent = None
        
    async def initialize_agent(self):
        """Initialize the agent for testing."""
        from backend.agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        self.agent = UnifiedToolCallingRAGAgent()
        await self.agent._ensure_initialized()
        logger.info(f"✅ Agent initialized successfully")
        
    async def simulate_employee_message(self, message: str, step_name: str) -> Dict[str, Any]:
        """Simulate an employee sending a message and capture the response."""
        logger.info(f"\n🔄 STEP: {step_name}")
        logger.info(f"📤 Employee Message: '{message}'")
        
        try:
            # Process the message through the agent
            result = await self.agent.process_user_message(
                user_query=message,
                conversation_id=self.conversation_id,
                user_id=EMPLOYEE_USER_ID
            )
            
            # Log the response details
            logger.info(f"📥 Agent Response: {result.get('message', 'No message')}")
            
            if result.get('is_interrupted'):
                logger.info(f"⏸️  HITL Interruption Detected")
                if result.get('hitl_phase'):
                    logger.info(f"   Phase: {result.get('hitl_phase')}")
                if result.get('hitl_prompt'):
                    logger.info(f"   Prompt: {result.get('hitl_prompt')}")
                if result.get('hitl_context'):
                    logger.info(f"   Context: {json.dumps(result.get('hitl_context', {}), indent=2)}")
            
            if result.get('error'):
                logger.error(f"❌ Error in step '{step_name}': {result.get('error_details', 'Unknown error')}")
            
            # Store the step response
            step_data = {
                'step_name': step_name,
                'employee_message': message,
                'agent_response': result,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': self.conversation_id
            }
            self.step_responses.append(step_data)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Exception in step '{step_name}': {str(e)}")
            error_result = {
                'message': f"System error: {str(e)}",
                'error': True,
                'error_details': str(e),
                'is_interrupted': False
            }
            
            step_data = {
                'step_name': step_name,
                'employee_message': message,
                'agent_response': error_result,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': self.conversation_id,
                'exception': str(e)
            }
            self.step_responses.append(step_data)
            
            return error_result
    
    async def handle_hitl_response(self, user_response: str, step_name: str) -> Dict[str, Any]:
        """Handle HITL (Human-in-the-Loop) responses."""
        logger.info(f"\n🔄 HITL RESPONSE: {step_name}")
        logger.info(f"👤 User Response: '{user_response}'")
        
        # Process the HITL response
        result = await self.simulate_employee_message(user_response, f"{step_name} (HITL Response)")
        return result
    
    def print_step_summary(self):
        """Print a summary of all steps taken."""
        logger.info(f"\n📊 FLOW SUMMARY")
        logger.info(f"Conversation ID: {self.conversation_id}")
        logger.info(f"Total Steps: {len(self.step_responses)}")
        
        for i, step in enumerate(self.step_responses, 1):
            logger.info(f"\n{i}. {step['step_name']}")
            logger.info(f"   Message: {step['employee_message']}")
            
            response = step['agent_response']
            if response.get('is_interrupted'):
                logger.info(f"   Result: HITL Required ({response.get('hitl_phase', 'unknown')})")
            elif response.get('error'):
                logger.info(f"   Result: Error - {response.get('error_details', 'Unknown')}")
            else:
                logger.info(f"   Result: Success")
                
            # Check for quotation link in response
            message = response.get('message', '')
            if 'storage.supabase' in message or 'quotation' in message.lower():
                logger.info(f"   📄 Potential quotation generated!")
    
    def save_detailed_log(self, filename: str = None):
        """Save detailed step-by-step log to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quotation_flow_simulation_{timestamp}.json"
        
        log_data = {
            'simulation_metadata': {
                'conversation_id': self.conversation_id,
                'employee_user_id': EMPLOYEE_USER_ID,
                'customer_name': TEST_CUSTOMER_NAME,
                'vehicle_requirements': TEST_VEHICLE_REQUIREMENTS,
                'timestamp': datetime.now().isoformat(),
                'total_steps': len(self.step_responses)
            },
            'step_responses': self.step_responses
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"📁 Detailed log saved to: {filename}")
        return filename


@pytest.mark.asyncio
async def test_complete_quotation_flow_simulation():
    """Test the complete end-to-end quotation generation flow."""
    
    simulator = QuotationFlowSimulator()
    
    try:
        # Initialize the agent
        await simulator.initialize_agent()
        
        # Step 1: Employee initiates quotation request
        logger.info(f"\n🚀 STARTING QUOTATION FLOW SIMULATION")
        logger.info(f"Employee: {EMPLOYEE_USER_ID}")
        logger.info(f"Customer: {TEST_CUSTOMER_NAME}")
        logger.info(f"Vehicle: {TEST_VEHICLE_REQUIREMENTS}")
        
        result1 = await simulator.simulate_employee_message(
            f"Generate a quotation for {TEST_CUSTOMER_NAME}, they want {TEST_VEHICLE_REQUIREMENTS}",
            "Initial Quotation Request"
        )
        
        # Step 2: Handle any HITL requests iteratively
        current_result = result1
        max_hitl_iterations = 5
        hitl_iteration = 0
        
        while current_result.get('is_interrupted') and hitl_iteration < max_hitl_iterations:
            hitl_iteration += 1
            hitl_phase = current_result.get('hitl_phase', 'unknown')
            hitl_prompt = current_result.get('hitl_prompt', '')
            hitl_context = current_result.get('hitl_context', {})
            
            logger.info(f"\n🔄 HITL Iteration {hitl_iteration}")
            logger.info(f"Phase: {hitl_phase}")
            
            # Generate appropriate responses based on HITL phase
            if hitl_phase == 'needs_confirmation' or 'approval' in hitl_phase.lower():
                # Approve the quotation
                hitl_response = "Yes, approve and generate the quotation"
                
            elif 'customer' in hitl_prompt.lower() and 'not found' in hitl_prompt.lower():
                # Provide customer information
                hitl_response = """Customer Information:
                Name: Maria Santos
                Email: maria.santos@email.com
                Phone: +63 912 345 6789
                Company: Santos Trading Corp
                Address: 123 Business Ave, Makati City"""
                
            elif 'vehicle' in hitl_prompt.lower():
                # Provide vehicle selection or details
                hitl_response = "Toyota Camry 2024, Sedan, White color, Standard package"
                
            elif 'pricing' in hitl_prompt.lower():
                # Confirm pricing
                hitl_response = "Yes, proceed with the standard pricing"
                
            else:
                # Generic approval/confirmation
                hitl_response = "Yes, proceed"
            
            # Send the HITL response
            current_result = await simulator.handle_hitl_response(
                hitl_response,
                f"HITL Response {hitl_iteration} ({hitl_phase})"
            )
        
        # Step 3: Final verification
        if current_result.get('is_interrupted'):
            logger.warning(f"⚠️  Flow still interrupted after {hitl_iteration} HITL iterations")
        elif current_result.get('error'):
            logger.error(f"❌ Final error: {current_result.get('error_details')}")
        else:
            logger.info(f"✅ Flow completed successfully!")
            
            # Check for quotation link
            message = current_result.get('message', '')
            if 'http' in message and ('quotation' in message.lower() or 'storage' in message.lower()):
                logger.info(f"🎉 QUOTATION LINK FOUND!")
                # Extract the link
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
                for url in urls:
                    logger.info(f"📄 Quotation URL: {url}")
        
        # Print summary
        simulator.print_step_summary()
        
        # Save detailed log
        log_file = simulator.save_detailed_log()
        
        # Assert that we made progress
        assert len(simulator.step_responses) >= 1, "Should have at least one step response"
        
        # Return the simulator for further analysis
        return simulator
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {str(e)}")
        
        # Still save what we have
        if simulator.step_responses:
            simulator.save_detailed_log()
        
        raise


@pytest.mark.asyncio
async def test_quotation_flow_with_docker_logs():
    """Test quotation flow and capture Docker logs for debugging."""
    
    # Run the main simulation
    simulator = await test_complete_quotation_flow_simulation()
    
    # Try to capture Docker logs if available
    try:
        import subprocess
        
        logger.info(f"\n🐳 CAPTURING DOCKER LOGS")
        
        # Get backend container logs
        backend_logs = subprocess.run(
            ["docker", "logs", "--tail", "50", "agent-tobi-rag-backend-1"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if backend_logs.returncode == 0:
            logger.info(f"📋 Backend Docker Logs (last 50 lines):")
            for line in backend_logs.stdout.split('\n')[-20:]:  # Show last 20 lines
                if line.strip():
                    logger.info(f"   {line}")
        else:
            logger.warning(f"⚠️  Could not get backend logs: {backend_logs.stderr}")
            
    except Exception as e:
        logger.warning(f"⚠️  Could not capture Docker logs: {str(e)}")
    
    return simulator


if __name__ == "__main__":
    # Run the simulation directly
    asyncio.run(test_complete_quotation_flow_simulation())
