#!/usr/bin/env python3
"""
Test script to verify the moving context window implementation.
This script simulates adding messages to a conversation and demonstrates
how the context window maintains a consistent size.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.memory import MemoryManager
from core.database import db_client
from core.config import get_settings


class MovingContextWindowTester:
    """Test harness for moving context window functionality."""
    
    def __init__(self):
        self.memory = MemoryManager()
        self.test_conversation_id = str(uuid4())
        self.test_user_id = str(uuid4())
        
    async def setup_test_data(self):
        """Create a test conversation with multiple messages."""
        print("🔧 Setting up test data...")
        
        # Create test conversation
        conversation_data = {
            "id": self.test_conversation_id,
            "user_id": self.test_user_id,
            "title": "Moving Context Window Test",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "message_count": 0
        }
        
        await asyncio.to_thread(
            db_client.client.table("conversations").insert(conversation_data).execute
        )
        print(f"✅ Created test conversation: {self.test_conversation_id[:8]}...")
        
    async def add_test_messages(self, count: int, batch_name: str = ""):
        """Add a batch of test messages to the conversation."""
        print(f"📝 Adding {count} messages{' (' + batch_name + ')' if batch_name else ''}...")
        
        messages = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            # Alternate between human and AI messages
            role = "human" if i % 2 == 0 else "assistant"
            content = f"Test message {i + 1} from {batch_name or 'batch'}: This is {'a user question' if role == 'human' else 'an AI response'}."
            
            message = {
                "conversation_id": self.test_conversation_id,
                "user_id": self.test_user_id,
                "role": role,
                "content": content,
                "created_at": (base_time + timedelta(seconds=i)).isoformat()
            }
            messages.append(message)
        
        # Insert messages in batch
        await asyncio.to_thread(
            db_client.client.table("messages").insert(messages).execute
        )
        print(f"✅ Added {count} messages to conversation")
        
    async def test_context_window_retrieval(self, test_name: str):
        """Test the context window retrieval and show results."""
        print(f"\n🧪 {test_name}")
        print("=" * 60)
        
        # Get current total message count
        count_result = await asyncio.to_thread(
            lambda: db_client.client.table("messages")
            .select("id", count="exact")
            .eq("conversation_id", self.test_conversation_id)
            .execute()
        )
        total_messages = count_result.count
        print(f"📊 Total messages in conversation: {total_messages}")
        
        # Test the moving context window retrieval
        retrieved_messages = await self.memory._get_recent_conversation_messages(
            self.test_conversation_id, 
            previous_summary=None  # No previous summary to test the window behavior
        )
        
        print(f"🪟 Messages retrieved by context window: {len(retrieved_messages)}")
        
        # Show first and last few messages to verify window behavior
        if retrieved_messages:
            print(f"📋 First retrieved message: '{retrieved_messages[0]['content'][:50]}...'")
            print(f"📋 Last retrieved message: '{retrieved_messages[-1]['content'][:50]}...'")
            
            # Show which messages were selected
            if total_messages > len(retrieved_messages):
                skipped = total_messages - len(retrieved_messages)
                print(f"⏭️  Skipped first {skipped} messages (outside window)")
                print(f"✅ Retrieved last {len(retrieved_messages)} messages (within window)")
            else:
                print(f"✅ Retrieved all {len(retrieved_messages)} messages (conversation smaller than window)")
        
        return len(retrieved_messages)
    
    async def test_with_previous_summary(self):
        """Test behavior when a previous summary exists."""
        print(f"\n🧪 Testing with Previous Summary (should still use moving window)")
        print("=" * 60)
        
        # Create a fake previous summary
        fake_summary = {
            "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "content": "This is a fake previous summary for testing"
        }
        
        # Test with previous summary
        retrieved_messages = await self.memory._get_recent_conversation_messages(
            self.test_conversation_id, 
            previous_summary=fake_summary
        )
        
        print(f"🪟 Messages retrieved with previous summary: {len(retrieved_messages)}")
        print("✅ Should still use moving window (ignore summary timestamp)")
        
        return len(retrieved_messages)
    
    async def cleanup_test_data(self):
        """Clean up test data."""
        print(f"\n🧹 Cleaning up test data...")
        
        # Delete test messages
        await asyncio.to_thread(
            lambda: db_client.client.table("messages")
            .delete()
            .eq("conversation_id", self.test_conversation_id)
            .execute()
        )
        
        # Delete test conversation
        await asyncio.to_thread(
            lambda: db_client.client.table("conversations")
            .delete()
            .eq("id", self.test_conversation_id)
            .execute()
        )
        
        print("✅ Test data cleaned up")
    
    async def run_comprehensive_test(self):
        """Run the complete moving context window test."""
        try:
            print("🚀 Starting Moving Context Window Test")
            print("=" * 80)
            
            # Get configuration
            settings = await get_settings()
            window_size = settings.memory.context_window_size
            print(f"⚙️  Configured context window size: {window_size} messages")
            print()
            
            await self.setup_test_data()
            
            # Test 1: Few messages (less than window size)
            await self.add_test_messages(5, "initial batch")
            retrieved_count_1 = await self.test_context_window_retrieval("Test 1: Few Messages (< Window Size)")
            
            # Test 2: Exactly window size
            await self.add_test_messages(window_size - 5, "fill to window size")
            retrieved_count_2 = await self.test_context_window_retrieval("Test 2: Exactly Window Size")
            
            # Test 3: More than window size
            await self.add_test_messages(15, "exceed window size")
            retrieved_count_3 = await self.test_context_window_retrieval("Test 3: Exceed Window Size")
            
            # Test 4: Much more than window size (simulate long conversation)
            await self.add_test_messages(30, "long conversation")
            retrieved_count_4 = await self.test_context_window_retrieval("Test 4: Long Conversation")
            
            # Test 5: With previous summary
            retrieved_count_5 = await self.test_with_previous_summary()
            
            # Summary
            print(f"\n📊 TEST RESULTS SUMMARY")
            print("=" * 40)
            print(f"Window Size Configuration: {window_size}")
            print(f"Test 1 (5 messages): Retrieved {retrieved_count_1} ✅")
            print(f"Test 2 ({window_size} messages): Retrieved {retrieved_count_2} ✅")
            print(f"Test 3 ({window_size + 15} messages): Retrieved {retrieved_count_3} ✅")
            print(f"Test 4 ({window_size + 45} messages): Retrieved {retrieved_count_4} ✅")
            print(f"Test 5 (with prev summary): Retrieved {retrieved_count_5} ✅")
            
            # Validation
            print(f"\n🔍 VALIDATION:")
            expected_counts = [5, window_size, window_size, window_size, window_size]
            actual_counts = [retrieved_count_1, retrieved_count_2, retrieved_count_3, retrieved_count_4, retrieved_count_5]
            
            all_passed = True
            for i, (expected, actual) in enumerate(zip(expected_counts, actual_counts), 1):
                if expected == actual:
                    print(f"✅ Test {i}: Expected {expected}, Got {actual} - PASS")
                else:
                    print(f"❌ Test {i}: Expected {expected}, Got {actual} - FAIL")
                    all_passed = False
            
            if all_passed:
                print(f"\n🎉 ALL TESTS PASSED! Moving context window is working correctly!")
                print(f"✅ Window consistently maintains {window_size} message limit")
                print(f"✅ Older messages are properly excluded from context")
                print(f"✅ Previous summary logic doesn't interfere with window size")
            else:
                print(f"\n❌ SOME TESTS FAILED! Moving context window needs attention.")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            await self.cleanup_test_data()


async def main():
    """Main test function."""
    tester = MovingContextWindowTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())