#!/usr/bin/env python3
"""
Test script for moving context window memory management.
Demonstrates how previous messages are appended to current messages and how
conversations are summarized and reset when reaching the message limit.
"""

import asyncio
import sys
import os
from datetime import datetime
from uuid import uuid4

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_core.messages import HumanMessage, AIMessage
from agents.memory_manager import ConversationMemoryManager

async def test_moving_context_window():
    """Test the moving context window functionality."""
    print("🧠 Testing Moving Context Window Memory Management")
    print("=" * 60)
    
    # Initialize memory manager with small limit for testing
    memory_manager = ConversationMemoryManager(max_messages=5)
    
    print(f"📝 Configuration: max_messages = {memory_manager.max_messages}")
    print()
    
    # Test conversation messages
    test_messages = [
        HumanMessage(content="Hello, can you help me with Python?"),
        AIMessage(content="Of course! I'd be happy to help you with Python. What specific topic would you like to explore?"),
        HumanMessage(content="I want to learn about list comprehensions"),
        AIMessage(content="List comprehensions are a concise way to create lists. Here's the basic syntax: [expression for item in iterable if condition]"),
        HumanMessage(content="Can you give me an example?"),
        AIMessage(content="Sure! Here's an example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 9."),
        HumanMessage(content="How about filtering with conditions?"),
        AIMessage(content="You can add conditions like: even_squares = [x**2 for x in range(10) if x % 2 == 0]"),
        HumanMessage(content="That's helpful! Now I want to learn about decorators"),
        AIMessage(content="Decorators are functions that modify other functions. They use the @ symbol and are very powerful for adding functionality."),
        HumanMessage(content="Can you show me a simple decorator example?"),
        AIMessage(content="Here's a basic decorator: def my_decorator(func): def wrapper(): print('Before'); func(); print('After'); return wrapper"),
        HumanMessage(content="What about decorators with arguments?"),
    ]
    
    messages = []
    conversation_summary = None
    
    for i, message in enumerate(test_messages):
        print(f"🔄 Processing message {i+1}/{len(test_messages)}")
        print(f"💬 Message: {message.content[:50]}...")
        
        # Add message to conversation
        messages.append(message)
        
        # Get effective context
        context = await memory_manager.get_effective_context(messages, conversation_summary)
        effective_messages = context["messages"]
        conversation_summary = context["conversation_summary"]
        
        print(f"📊 Stats: {context['total_messages']} total → {context['processed_messages']} processed")
        
        if context.get("was_reset", False):
            print("🔄 CONVERSATION RESET! Context appended to current message")
            print(f"📋 New summary generated: {len(conversation_summary)} chars")
            
            # Show how the current message now contains context
            if effective_messages and isinstance(effective_messages[-1], HumanMessage):
                current_content = effective_messages[-1].content
                print(f"💭 Enhanced message preview: {current_content[:100]}...")
                
                # Show structure of the enhanced message
                if "Previous conversation summary:" in current_content:
                    print("   ✅ Contains previous conversation summary")
                if "Recent conversation context:" in current_content:
                    print("   ✅ Contains recent conversation context")
                if "Current request:" in current_content:
                    print("   ✅ Contains current request")
        else:
            print("📝 Context appended to current message")
            
            # Show how context is being appended
            if effective_messages and isinstance(effective_messages[-1], HumanMessage):
                current_content = effective_messages[-1].content
                if "Previous conversation:" in current_content:
                    print("   ✅ Previous conversation history appended")
                if "Current request:" in current_content:
                    print("   ✅ Current request clearly marked")
        
        print(f"🧠 Has summary: {'Yes' if context['has_summary'] else 'No'}")
        print("-" * 40)
        
        # Update messages list with processed messages for next iteration
        messages = effective_messages
    
    print("\n✅ Test completed successfully!")
    print(f"📊 Final stats:")
    print(f"   - Total messages processed: {len(test_messages)}")
    print(f"   - Final conversation length: {len(messages)}")
    print(f"   - Summary length: {len(conversation_summary)} chars" if conversation_summary else "   - No summary")


async def test_context_appending():
    """Test how context is appended to messages."""
    print("\n🧪 Testing Context Appending Behavior")
    print("=" * 60)
    
    memory_manager = ConversationMemoryManager(max_messages=3)
    
    # Create a simple conversation
    messages = [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
        HumanMessage(content="Tell me about its features"),
    ]
    
    context = await memory_manager.get_effective_context(messages, None)
    processed_messages = context["messages"]
    
    print(f"📊 Original messages: {len(messages)}")
    print(f"📊 Processed messages: {len(processed_messages)}")
    
    # Show the enhanced message content
    if processed_messages and isinstance(processed_messages[-1], HumanMessage):
        enhanced_content = processed_messages[-1].content
        print(f"\n💭 Enhanced message content:")
        print("-" * 30)
        print(enhanced_content)
        print("-" * 30)
    
    print("\n✅ Context appending test completed!")


async def main():
    """Run all tests."""
    try:
        await test_moving_context_window()
        await test_context_appending()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 