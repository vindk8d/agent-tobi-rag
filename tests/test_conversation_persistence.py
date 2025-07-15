import asyncio
import pytest
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.memory_manager import memory_manager
from backend.agents.rag_agent import UnifiedToolCallingRAGAgent


async def test_conversation_persistence():
    """Test that conversation context persists between turns."""
    
    # Initialize the agent
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create a test conversation
    conversation_id = str(uuid4())
    user_id = "test_user"
    
    # First turn: Ask about employees
    first_query = "how many employees in the company?"
    first_result = await agent.invoke(
        user_query=first_query,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    print(f"First turn result: {first_result}")
    
    # Second turn: Ask to list them (should have context from first turn)
    second_query = "can you list them all down?"
    second_result = await agent.invoke(
        user_query=second_query,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    print(f"Second turn result: {second_result}")
    
    # Check that the agent has conversation context
    final_messages = second_result.get("messages", [])
    
    # Should have messages from both turns
    assert len(final_messages) >= 4  # At least 2 human + 2 AI messages
    
    # Check that the conversation includes context from the first turn
    conversation_text = " ".join([msg.content for msg in final_messages if hasattr(msg, 'content')])
    assert "employees" in conversation_text.lower()
    
    print("✅ Conversation persistence test passed!")


if __name__ == "__main__":
    asyncio.run(test_conversation_persistence()) 