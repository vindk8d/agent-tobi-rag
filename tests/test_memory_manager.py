"""
Test script for ConversationMemoryManager sliding window context management and LLM-based summarization.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.memory_manager import ConversationMemoryManager
from langchain_core.messages import HumanMessage, AIMessage

async def test_sliding_window_context_management():
    """Test sliding window context management with configurable size."""
    print("🧪 Testing sliding window context management...")
    
    # Test with smaller window size for easier testing
    memory_manager = ConversationMemoryManager(window_size=5, summary_interval=4)
    
    # Create test messages
    messages = []
    for i in range(12):  # More than window size
        messages.append(HumanMessage(content=f"User message {i+1}"))
        messages.append(AIMessage(content=f"Assistant response {i+1}"))
    
    print(f"📝 Created {len(messages)} test messages")
    
    # Test sliding window application
    windowed_messages, summary = await memory_manager.apply_sliding_window(messages)
    
    print(f"📊 Window size: {memory_manager.window_size}")
    print(f"📊 Summary interval: {memory_manager.summary_interval}")
    print(f"📊 Original messages: {len(messages)}")
    print(f"📊 Windowed messages: {len(windowed_messages)}")
    print(f"📊 Summary generated: {summary is not None}")
    
    if summary:
        print(f"📝 Summary preview: {summary[:100]}...")
    
    # Test effective context
    context = await memory_manager.get_effective_context(messages)
    print(f"📊 Effective context: {context['messages_pruned']} messages pruned")
    print(f"📊 Has summary: {context['has_summary']}")
    
    return len(windowed_messages) == memory_manager.window_size

async def test_periodic_summary_generation():
    """Test periodic summary generation at specified intervals."""
    print("\n🧪 Testing periodic summary generation...")
    
    memory_manager = ConversationMemoryManager(window_size=10, summary_interval=6)
    
    # Test summary trigger logic
    messages = []
    for i in range(6):  # Exactly at summary interval
        messages.append(HumanMessage(content=f"Question {i+1}: What is the weather like?"))
        messages.append(AIMessage(content=f"Answer {i+1}: The weather is sunny and warm."))
    
    should_update = await memory_manager.should_update_summary(messages)
    print(f"📊 Should update summary at {len(messages)} messages: {should_update}")
    
    # Test summary generation
    if should_update:
        summary = await memory_manager.generate_conversation_summary(messages)
        print(f"📝 Generated summary: {len(summary)} characters")
        print(f"📝 Summary preview: {summary[:150]}...")
        return len(summary) > 0
    
    return False

async def test_llm_based_summarization():
    """Test LLM-based conversation summarization."""
    print("\n🧪 Testing LLM-based summarization...")
    
    memory_manager = ConversationMemoryManager()
    
    # Create a meaningful conversation
    messages = [
        HumanMessage(content="Hello, I need help with my CRM system."),
        AIMessage(content="I'd be happy to help you with your CRM system. What specific issue are you facing?"),
        HumanMessage(content="I'm looking for information about our top sales performers this quarter."),
        AIMessage(content="I can help you find information about sales performance. Let me query the CRM database for you."),
        HumanMessage(content="Also, I need to understand our current inventory levels for Tesla Model 3 vehicles."),
        AIMessage(content="I'll check both the sales performance data and Tesla Model 3 inventory levels for you."),
        HumanMessage(content="Thank you, this information will help me prepare for the quarterly sales meeting."),
        AIMessage(content="You're welcome! I've provided the sales performance and inventory data you requested."),
    ]
    
    # Generate summary
    summary = await memory_manager.generate_conversation_summary(messages)
    
    print(f"📝 Summary length: {len(summary)} characters")
    print(f"📝 Summary content: {summary}")
    
    # Verify summary contains relevant information
    contains_crm = "CRM" in summary or "crm" in summary
    contains_sales = "sales" in summary or "Sales" in summary
    contains_inventory = "inventory" in summary or "Inventory" in summary
    
    print(f"📊 Summary contains CRM: {contains_crm}")
    print(f"📊 Summary contains Sales: {contains_sales}")
    print(f"📊 Summary contains Inventory: {contains_inventory}")
    
    return len(summary) > 0 and (contains_crm or contains_sales)

async def main():
    """Run all ConversationMemoryManager tests."""
    print("🚀 Starting ConversationMemoryManager Tests")
    print("=" * 50)
    
    try:
        # Test sliding window context management
        test1_passed = await test_sliding_window_context_management()
        
        # Test periodic summary generation
        test2_passed = await test_periodic_summary_generation()
        
        # Test LLM-based summarization
        test3_passed = await test_llm_based_summarization()
        
        print("\n" + "=" * 50)
        print("📊 Test Results:")
        print(f"✅ Sliding window context management: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ Periodic summary generation: {'PASSED' if test2_passed else 'FAILED'}")
        print(f"✅ LLM-based summarization: {'PASSED' if test3_passed else 'FAILED'}")
        
        overall_success = test1_passed and test2_passed and test3_passed
        print(f"\n🎯 Overall Status: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 