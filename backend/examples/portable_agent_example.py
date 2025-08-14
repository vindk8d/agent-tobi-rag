"""
Example showing how any agent can use the portable background task system.

This demonstrates the portability principle - any agent can use the same
background task infrastructure without duplicating code.
"""

import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from agents.background_tasks import background_task_manager
from langchain_core.messages import HumanMessage, AIMessage


class ExamplePortableAgent:
    """
    Example agent demonstrating portable background task usage.
    
    This agent can use the same background task infrastructure as
    the Tobi Sales Copilot agent without any code duplication.
    """
    
    def __init__(self):
        # Any agent can use the global background task manager
        self.background_task_manager = background_task_manager
    
    async def process_message(self, user_message: str, state: dict) -> dict:
        """
        Example agent processing that uses portable background tasks.
        
        Args:
            user_message: User's input message
            state: Agent state (any format, as long as it has the expected fields)
            
        Returns:
            Updated state
        """
        # Simulate agent processing
        response = f"Hello! You said: {user_message}"
        
        # Add messages to state
        messages = state.get("messages", [])
        messages.extend([
            HumanMessage(content=user_message),
            AIMessage(content=response)
        ])
        
        updated_state = {
            **state,
            "messages": messages
        }
        
        # PORTABLE: Schedule message storage using the universal method
        # This works regardless of which agent is calling it
        if response:
            self.background_task_manager.schedule_message_from_agent_state(
                updated_state, response, "assistant"
            )
        
        # PORTABLE: Schedule summary generation when needed
        message_count = len(messages)
        if message_count >= 10:  # Example threshold
            self.background_task_manager.schedule_summary_from_agent_state(updated_state)
        
        return updated_state


class AnotherExampleAgent:
    """
    Another example agent showing the same portable usage pattern.
    
    This demonstrates that multiple agents can use the same infrastructure
    without any conflicts or code duplication.
    """
    
    def __init__(self):
        # Same portable background task manager
        self.background_task_manager = background_task_manager
    
    async def handle_request(self, request: str, context: dict) -> dict:
        """Different method signature, same portable background task usage."""
        
        # Different processing logic
        result = f"Processed: {request.upper()}"
        
        # Same portable background task usage
        if result:
            self.background_task_manager.schedule_message_from_agent_state(
                context, result, "assistant"
            )
        
        # Same portable summary logic
        if len(context.get("messages", [])) >= 15:
            self.background_task_manager.schedule_summary_from_agent_state(context)
        
        return context


# Example usage
async def demonstrate_portability():
    """Demonstrate how multiple agents can use the same portable system."""
    
    # Example state (any agent can use this format)
    example_state = {
        "conversation_id": "example-conv-123",
        "user_id": "user-456",
        "customer_id": "customer-789",
        "employee_id": None,
        "messages": []
    }
    
    # Agent 1 uses portable system
    agent1 = ExamplePortableAgent()
    state1 = await agent1.process_message("Hello from agent 1", example_state.copy())
    print("Agent 1 processed message using portable background tasks")
    
    # Agent 2 uses same portable system
    agent2 = AnotherExampleAgent()
    state2 = await agent2.handle_request("Hello from agent 2", example_state.copy())
    print("Agent 2 processed request using same portable background tasks")
    
    print("✅ Both agents used the same portable background task infrastructure!")
    print("✅ No code duplication between agents!")
    print("✅ Portability principle successfully demonstrated!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_portability())
