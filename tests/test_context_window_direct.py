#!/usr/bin/env python3
"""
Direct test of the moving context window implementation.
This test properly initializes ConversationConsolidator with all dependencies.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.memory import ConversationConsolidator, SimpleDBManager, SupabaseLongTermMemoryStore
from core.database import db_client
from core.config import get_settings
from langchain_openai import ChatOpenAI
from rag.embeddings import OpenAIEmbeddings


class DirectContextWindowTester:
    """Direct test for moving context window functionality."""
    
    def __init__(self):
        self.consolidator = None
        
    async def setup_consolidator(self):
        """Setup ConversationConsolidator with all required dependencies."""
        print("🔧 Setting up ConversationConsolidator...")
        
        try:
            # Get settings
            settings = await get_settings()
            
            # Initialize components
            db_manager = SimpleDBManager()
            memory_store = SupabaseLongTermMemoryStore()
            llm = ChatOpenAI(
                api_key=settings.openai.api_key,
                model=settings.openai.chat_model,
                temperature=0.1
            )
            embeddings = OpenAIEmbeddings()
            
            # Create consolidator
            self.consolidator = ConversationConsolidator(
                db_manager=db_manager,
                memory_store=memory_store,
                llm=llm,
                embeddings=embeddings
            )
            
            print("✅ ConversationConsolidator initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup ConversationConsolidator: {e}")
            raise
    
    async def get_existing_conversation(self):
        """Get an existing conversation for testing."""
        print("🔍 Finding existing conversation...")
        
        result = await asyncio.to_thread(
            lambda: db_client.client.table("conversations")
            .select("id, user_id, title")
            .limit(1)
            .execute()
        )
        
        if not result.data:
            print("❌ No existing conversations found")
            return None
            
        conversation = result.data[0]
        print(f"✅ Found conversation: {conversation['id'][:8]}... ({conversation.get('title', 'No title')})")
        return conversation
    
    async def add_test_messages(self, conversation_id: str, user_id: str, count: int, prefix: str = ""):
        """Add test messages to an existing conversation."""
        print(f"📝 Adding {count} test messages{' (' + prefix + ')' if prefix else ''}...")
        
        messages = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            role = "human" if i % 2 == 0 else "assistant"
            content = f"Test message {i + 1} {prefix}: This is {'a user question' if role == 'human' else 'an AI response'}."
            
            message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "created_at": (base_time + timedelta(seconds=i)).isoformat()
            }
            messages.append(message)
        
        await asyncio.to_thread(
            lambda: db_client.client.table("messages").insert(messages).execute()
        )
        print(f"✅ Added {count} messages")
    
    async def count_messages(self, conversation_id: str):
        """Count total messages in conversation."""
        result = await asyncio.to_thread(
            lambda: db_client.client.table("messages")
            .select("id", count="exact")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        return result.count
    
    async def test_context_window(self, conversation_id: str, test_name: str, previous_summary=None):
        """Test the context window and show results."""
        print(f"\n🧪 {test_name}")
        print("=" * 50)
        
        # Get total message count
        total_messages = await self.count_messages(conversation_id)
        print(f"📊 Total messages in conversation: {total_messages}")
        
        # Test the moving context window
        retrieved_messages = await self.consolidator._get_recent_conversation_messages(
            conversation_id, 
            previous_summary=previous_summary
        )
        
        retrieved_count = len(retrieved_messages)
        print(f"🪟 Messages retrieved by context window: {retrieved_count}")
        
        # Show window behavior
        if retrieved_messages:
            print(f"📋 First message: '{retrieved_messages[0]['content'][:60]}...'")
            print(f"📋 Last message: '{retrieved_messages[-1]['content'][:60]}...'")
            
            if total_messages > retrieved_count:
                skipped = total_messages - retrieved_count
                print(f"⏭️  Skipped {skipped} older messages (outside window)")
                print(f"✅ Retrieved last {retrieved_count} messages (within window)")
            else:
                print(f"✅ Retrieved all {retrieved_count} messages (conversation smaller than window)")
        
        return retrieved_count, total_messages
    
    async def cleanup_test_messages(self, conversation_id: str, prefix: str):
        """Clean up test messages."""
        print(f"🧹 Cleaning up test messages with prefix '{prefix}'...")
        
        await asyncio.to_thread(
            lambda: db_client.client.table("messages")
            .delete()
            .eq("conversation_id", conversation_id)
            .like("content", f"Test message%{prefix}%")
            .execute()
        )
        print("✅ Test messages cleaned up")
    
    async def run_test(self):
        """Run the moving context window test."""
        try:
            print("🚀 Starting Direct Moving Context Window Test")
            print("=" * 60)
            
            # Setup consolidator
            await self.setup_consolidator()
            
            # Get configuration
            settings = await get_settings()
            window_size = settings.memory.context_window_size
            print(f"⚙️  Configured context window size: {window_size} messages")
            
            # Get existing conversation
            conversation = await self.get_existing_conversation()
            if not conversation:
                print("❌ Cannot run test without an existing conversation")
                return
                
            conversation_id = conversation['id']
            user_id = conversation['user_id']
            
            print(f"🎯 Testing with conversation: {conversation_id[:8]}...")
            
            # Test 1: Check current state
            count_1, total_1 = await self.test_context_window(conversation_id, "Test 1: Current State")
            
            # Test 2: Add a few messages
            await self.add_test_messages(conversation_id, user_id, 5, "batch1")
            count_2, total_2 = await self.test_context_window(conversation_id, "Test 2: After Adding 5 Messages")
            
            # Test 3: Add more messages to exceed window
            await self.add_test_messages(conversation_id, user_id, window_size, "batch2")
            count_3, total_3 = await self.test_context_window(conversation_id, f"Test 3: After Adding {window_size} More Messages")
            
            # Test 4: Add many more messages
            await self.add_test_messages(conversation_id, user_id, 15, "batch3")
            count_4, total_4 = await self.test_context_window(conversation_id, "Test 4: After Adding 15 More Messages")
            
            # Test 5: Test with previous summary (should still use window)
            fake_summary = {
                "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "content": "Fake summary for testing"
            }
            count_5, total_5 = await self.test_context_window(
                conversation_id, 
                "Test 5: With Previous Summary", 
                previous_summary=fake_summary
            )
            
            # Results
            print(f"\n📊 TEST RESULTS")
            print("=" * 30)
            print(f"Window Size: {window_size}")
            print(f"Test 1: {count_1}/{total_1} messages")
            print(f"Test 2: {count_2}/{total_2} messages")
            print(f"Test 3: {count_3}/{total_3} messages")
            print(f"Test 4: {count_4}/{total_4} messages")
            print(f"Test 5: {count_5}/{total_5} messages (with summary)")
            
            # Validation
            print(f"\n🔍 VALIDATION:")
            
            tests_passed = 0
            total_tests = 0
            
            # Check that later tests maintain window size
            for i, (count, total, test_name) in enumerate([
                (count_3, total_3, "Test 3"),
                (count_4, total_4, "Test 4"), 
                (count_5, total_5, "Test 5")
            ], 3):
                total_tests += 1
                expected_max = window_size
                if count <= expected_max and (total <= window_size or count == window_size):
                    print(f"✅ {test_name}: Retrieved {count} ≤ {expected_max} - PASS")
                    tests_passed += 1
                else:
                    print(f"❌ {test_name}: Retrieved {count} > {expected_max} - FAIL")
            
            # Check progression shows window behavior
            total_tests += 1
            if count_4 == count_5 == window_size:  # If we have enough messages, should hit window limit
                print(f"✅ Window consistency: Both final tests retrieved exactly {window_size} - PASS")
                tests_passed += 1
            elif count_4 <= window_size and count_5 <= window_size:
                print(f"✅ Window constraint: Both final tests ≤ {window_size} - PASS")
                tests_passed += 1
            else:
                print(f"❌ Window constraint violated - FAIL")
            
            if tests_passed == total_tests:
                print(f"\n🎉 ALL TESTS PASSED ({tests_passed}/{total_tests})!")
                print(f"✅ Moving context window is working correctly!")
                print(f"✅ Window size maintained at {window_size} messages")
                print(f"✅ Old behavior (52+ messages) is FIXED!")
            else:
                print(f"\n⚠️  {tests_passed}/{total_tests} TESTS PASSED")
                print(f"❌ Moving context window needs attention.")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if 'conversation_id' in locals():
                await self.cleanup_test_messages(conversation_id, "batch1")
                await self.cleanup_test_messages(conversation_id, "batch2")
                await self.cleanup_test_messages(conversation_id, "batch3")


async def main():
    """Main test function."""
    tester = DirectContextWindowTester()
    await tester.run_test()


if __name__ == "__main__":
    asyncio.run(main())