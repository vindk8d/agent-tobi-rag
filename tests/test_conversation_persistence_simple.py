import asyncio
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.memory import memory_manager
from backend.agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent


async def test_conversation_debug():
    """Debug conversation persistence step by step."""
    
    # Initialize the agent
    agent = UnifiedToolCallingRAGAgent()
    await agent._ensure_initialized()
    
    # Create a test conversation  
    conversation_id = str(uuid4())
    user_id = "test_user"
    
    print(f"🔍 Testing conversation persistence with ID: {conversation_id}")
    
    # First turn: Ask about employees
    print("\n--- FIRST TURN ---")
    first_query = "how many employees in the company?"
    print(f"Query: {first_query}")
    
    first_result = await agent.invoke(
        user_query=first_query,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    print(f"First turn messages count: {len(first_result.get('messages', []))}")
    print(f"First turn final message: {first_result.get('messages', [])[-1].content if first_result.get('messages') else 'No messages'}")
    
    # Second turn: Ask to list them (should have context from first turn)
    print("\n--- SECOND TURN ---") 
    second_query = "can you list them all down?"
    print(f"Query: {second_query}")
    
    second_result = await agent.invoke(
        user_query=second_query,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    print(f"Second turn messages count: {len(second_result.get('messages', []))}")
    
    # Check conversation context
    print("\n--- ANALYSIS ---")
    final_messages = second_result.get("messages", [])
    
    print(f"Total messages in second turn: {len(final_messages)}")
    for i, msg in enumerate(final_messages):
        print(f"  {i}: {msg.type} - {msg.content[:100]}...")
    
    # Check if first turn context is preserved
    conversation_text = " ".join([msg.content for msg in final_messages if hasattr(msg, 'content')])
    has_employees = "employees" in conversation_text.lower()
    has_nine = "9" in conversation_text
    
    print(f"\nContext preservation:")
    print(f"  - Contains 'employees': {has_employees}")
    print(f"  - Contains '9': {has_nine}")
    
    # Check conversation summary tool output
    tool_outputs = [msg.content for msg in final_messages if hasattr(msg, 'name') and msg.name == 'get_conversation_summary']
    if tool_outputs:
        print(f"  - Conversation summary tool output: {tool_outputs[0]}")
    else:
        print("  - No conversation summary tool output found")
    
    return has_employees and has_nine


if __name__ == "__main__":
    result = asyncio.run(test_conversation_debug())
    print(f"\n✅ Test result: {'PASSED' if result else 'FAILED'}") 