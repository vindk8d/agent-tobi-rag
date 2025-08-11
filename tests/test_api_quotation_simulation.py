"""
End-to-End Quotation Generation Flow Simulation via API

This test simulates the complete flow of an employee requesting the bot to generate
a quotation for a customer using the Docker API directly, capturing each step.
"""

import requests
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/message"

# Test Data from Database
EMPLOYEE_USER_ID = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # John Smith (user.id, not employee.id)
CUSTOMER_NAME = "Robert Brown"
VEHICLE_REQUIREMENTS = "Toyota Camry sedan, silver color"

class APIQuotationFlowSimulator:
    """Simulates the complete quotation generation flow via API calls."""
    
    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.step_responses = []
        self.session = requests.Session()
        
    def send_chat_message(self, message: str, step_name: str) -> Dict[str, Any]:
        """Send a message to the chat API and capture the response."""
        print(f"\n🔄 STEP: {step_name}")
        print(f"📤 Employee Message: '{message}'")
        
        payload = {
            "message": message,
            "conversation_id": self.conversation_id,
            "user_id": EMPLOYEE_USER_ID,
            "include_sources": True
        }
        
        try:
            response = self.session.post(
                CHAT_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"📥 Agent Response: {result.get('message', 'No message')[:200]}...")
                
                # Check for HITL interruption
                if result.get('is_interrupted'):
                    print(f"⏸️  HITL Interruption Detected")
                    if result.get('hitl_phase'):
                        print(f"   Phase: {result.get('hitl_phase')}")
                    if result.get('hitl_prompt'):
                        print(f"   Prompt Preview: {result.get('hitl_prompt')[:100]}...")
                    if result.get('hitl_context'):
                        print(f"   Context Keys: {list(result.get('hitl_context', {}).keys())}")
                
                if result.get('error'):
                    print(f"❌ Error: {result.get('error_type', 'Unknown')}")
                
                # Store the step response
                step_data = {
                    'step_name': step_name,
                    'employee_message': message,
                    'api_response': result,
                    'timestamp': datetime.now().isoformat(),
                    'conversation_id': self.conversation_id,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                self.step_responses.append(step_data)
                
                return result
                
            else:
                error_result = {
                    'message': f"API Error: {response.status_code}",
                    'error': True,
                    'error_details': response.text,
                    'is_interrupted': False
                }
                
                step_data = {
                    'step_name': step_name,
                    'employee_message': message,
                    'api_response': error_result,
                    'timestamp': datetime.now().isoformat(),
                    'conversation_id': self.conversation_id,
                    'http_status': response.status_code
                }
                self.step_responses.append(step_data)
                
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                return error_result
                
        except Exception as e:
            error_result = {
                'message': f"Request failed: {str(e)}",
                'error': True,
                'error_details': str(e),
                'is_interrupted': False
            }
            
            step_data = {
                'step_name': step_name,
                'employee_message': message,
                'api_response': error_result,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': self.conversation_id,
                'exception': str(e)
            }
            self.step_responses.append(step_data)
            
            print(f"❌ Exception: {str(e)}")
            return error_result
    
    def handle_hitl_response(self, user_response: str, step_name: str) -> Dict[str, Any]:
        """Handle HITL (Human-in-the-Loop) responses."""
        print(f"\n🔄 HITL RESPONSE: {step_name}")
        print(f"👤 User Response: '{user_response}'")
        
        # Process the HITL response
        result = self.send_chat_message(user_response, f"{step_name} (HITL Response)")
        return result
    
    def print_step_summary(self):
        """Print a summary of all steps taken."""
        print(f"\n📊 FLOW SUMMARY")
        print(f"Conversation ID: {self.conversation_id}")
        print(f"Total Steps: {len(self.step_responses)}")
        
        for i, step in enumerate(self.step_responses, 1):
            print(f"\n{i}. {step['step_name']}")
            print(f"   Message: {step['employee_message'][:50]}...")
            
            response = step['api_response']
            if response.get('is_interrupted'):
                print(f"   Result: HITL Required ({response.get('hitl_phase', 'unknown')})")
            elif response.get('error'):
                print(f"   Result: Error - {response.get('error_details', 'Unknown')[:50]}...")
            else:
                print(f"   Result: Success")
                
            # Check for quotation link in response
            message = response.get('message', '')
            if 'storage.supabase' in message or 'quotation' in message.lower():
                print(f"   📄 Potential quotation generated!")
                
            if 'response_time_ms' in step:
                print(f"   ⏱️  Response Time: {step['response_time_ms']:.0f}ms")
    
    def save_detailed_log(self, filename: str = None):
        """Save detailed step-by-step log to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_quotation_flow_simulation_{timestamp}.json"
        
        log_data = {
            'simulation_metadata': {
                'conversation_id': self.conversation_id,
                'employee_user_id': EMPLOYEE_USER_ID,
                'customer_name': CUSTOMER_NAME,
                'vehicle_requirements': VEHICLE_REQUIREMENTS,
                'api_base_url': API_BASE_URL,
                'timestamp': datetime.now().isoformat(),
                'total_steps': len(self.step_responses)
            },
            'step_responses': self.step_responses
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📁 Detailed log saved to: {filename}")
        return filename

    def check_api_health(self) -> bool:
        """Check if the API is healthy and responsive."""
        try:
            response = self.session.get(f"{API_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ API is healthy")
                return True
            else:
                print(f"⚠️  API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API health check failed: {str(e)}")
            return False

def run_quotation_flow_simulation():
    """Run the complete quotation flow simulation."""
    
    simulator = APIQuotationFlowSimulator()
    
    print(f"🚀 STARTING API QUOTATION FLOW SIMULATION")
    print(f"Employee: {EMPLOYEE_USER_ID} (John Smith)")
    print(f"Customer: {CUSTOMER_NAME}")
    print(f"Vehicle: {VEHICLE_REQUIREMENTS}")
    print(f"API: {API_BASE_URL}")
    
    # Check API health first
    if not simulator.check_api_health():
        print("❌ API is not healthy, aborting simulation")
        return None
    
    try:
        # Step 1: Employee initiates quotation request
        result1 = simulator.send_chat_message(
            f"Generate a quotation for {CUSTOMER_NAME}, they want {VEHICLE_REQUIREMENTS}",
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
            
            print(f"\n🔄 HITL Iteration {hitl_iteration}")
            print(f"Phase: {hitl_phase}")
            
            # Generate appropriate responses based on HITL phase
            if hitl_phase == 'needs_confirmation' or 'approval' in hitl_phase.lower():
                # Approve the quotation
                hitl_response = "Yes, approve and generate the quotation"
                
            elif 'customer' in hitl_prompt.lower() and 'not found' in hitl_prompt.lower():
                # Provide customer information
                hitl_response = """Customer Information:
                Name: Robert Brown
                Email: robert.brown@email.com
                Phone: +63 912 345 6789
                Company: Personal Purchase
                Address: 123 Main Street, Manila"""
                
            elif 'vehicle' in hitl_prompt.lower():
                # Provide vehicle selection or details
                hitl_response = "Toyota Camry 2024, Sedan, Silver color, Standard package"
                
            elif 'pricing' in hitl_prompt.lower():
                # Confirm pricing
                hitl_response = "Yes, proceed with the standard pricing"
                
            else:
                # Generic approval/confirmation
                hitl_response = "Yes, proceed"
            
            # Send the HITL response
            current_result = simulator.handle_hitl_response(
                hitl_response,
                f"HITL Response {hitl_iteration} ({hitl_phase})"
            )
            
            # Add a small delay between iterations
            time.sleep(1)
        
        # Step 3: Final verification
        if current_result.get('is_interrupted'):
            print(f"⚠️  Flow still interrupted after {hitl_iteration} HITL iterations")
        elif current_result.get('error'):
            print(f"❌ Final error: {current_result.get('error_details')}")
        else:
            print(f"✅ Flow completed successfully!")
            
            # Check for quotation link
            message = current_result.get('message', '')
            if 'http' in message and ('quotation' in message.lower() or 'storage' in message.lower()):
                print(f"🎉 QUOTATION LINK FOUND!")
                # Extract the link
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
                for url in urls:
                    print(f"📄 Quotation URL: {url}")
        
        # Print summary
        simulator.print_step_summary()
        
        # Save detailed log
        log_file = simulator.save_detailed_log()
        
        return simulator
        
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")
        
        # Still save what we have
        if simulator.step_responses:
            simulator.save_detailed_log()
        
        raise

def get_docker_logs():
    """Get recent Docker logs for debugging."""
    import subprocess
    
    try:
        print(f"\n🐳 CAPTURING DOCKER LOGS")
        
        # Get backend container logs
        backend_logs = subprocess.run(
            ["docker", "logs", "--tail", "50", "agent-tobi-rag-backend-1"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if backend_logs.returncode == 0:
            print(f"📋 Backend Docker Logs (last 50 lines):")
            for line in backend_logs.stdout.split('\n')[-20:]:  # Show last 20 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"⚠️  Could not get backend logs: {backend_logs.stderr}")
            
    except Exception as e:
        print(f"⚠️  Could not capture Docker logs: {str(e)}")

if __name__ == "__main__":
    # Run the simulation
    try:
        simulator = run_quotation_flow_simulation()
        
        # Get Docker logs for additional debugging
        get_docker_logs()
        
        if simulator:
            print(f"\n🎯 SIMULATION COMPLETED SUCCESSFULLY")
            print(f"Check the generated log file for detailed analysis.")
        
    except Exception as e:
        print(f"\n❌ SIMULATION FAILED: {str(e)}")
        
        # Get Docker logs for debugging
        get_docker_logs()
