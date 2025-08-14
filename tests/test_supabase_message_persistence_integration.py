"""
Comprehensive Integration Tests for Supabase Messages Table Persistence

Task 5.1: Create comprehensive integration tests for existing Supabase messages table persistence

This module provides complete integration testing of the message persistence system,
validating that messages are correctly stored, retrieved, and managed in the Supabase
messages table through the BackgroundTaskManager.

Test Coverage:
1. Message storage with proper role mapping (human→user, ai→assistant)
2. Batch message insertion and retrieval
3. Conversation-based message querying
4. Message metadata preservation
5. Error handling and retry logic
6. Performance under concurrent operations
7. Data integrity validation
8. Background task processing reliability
"""

import asyncio
import pytest
import pytest_asyncio
import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test modules
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from agents.background_tasks import BackgroundTaskManager, BackgroundTask, TaskPriority, TaskStatus
from core.database import db_client
from core.config import get_settings

# Skip tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)


class MessagePersistenceTestHelper:
    """Helper class for message persistence integration testing."""
    
    def __init__(self):
        self.test_conversation_ids: List[str] = []
        self.test_user_ids: List[str] = []
        self.background_task_manager: Optional[BackgroundTaskManager] = None
    
    async def setup(self):
        """Set up test environment."""
        self.background_task_manager = BackgroundTaskManager()
        await self.background_task_manager.start()
    
    async def cleanup(self):
        """Clean up test data and resources."""
        if self.background_task_manager:
            await self.background_task_manager.stop()
        
        # Clean up only test messages (not existing conversations/users)
        for conversation_id in self.test_conversation_ids:
            try:
                # Only clean up messages we created during testing
                await asyncio.to_thread(
                    lambda cid=conversation_id: db_client.client.table("messages")
                    .delete()
                    .eq("conversation_id", cid)
                    .like("content", "Test message%")  # Only test messages
                    .execute()
                )
            except Exception as e:
                print(f"Warning: Failed to clean up test messages for conversation {conversation_id}: {e}")
    
    async def get_existing_conversation(self) -> tuple[str, str]:
        """Get an existing conversation ID and user ID from the database."""
        def _get_conversation():
            return (db_client.client.table("conversations")
                   .select("id,user_id")
                   .limit(1)
                   .execute())
        
        result = await asyncio.to_thread(_get_conversation)
        
        if result.data and len(result.data) > 0:
            conversation = result.data[0]
            conversation_id = conversation["id"]
            user_id = conversation["user_id"]
            
            # Track for cleanup (though we won't delete existing conversations)
            self.test_conversation_ids.append(conversation_id)
            self.test_user_ids.append(user_id)
            
            return conversation_id, user_id
        else:
            # Fallback: create minimal test identifiers (messages table may allow this)
            conversation_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            self.test_conversation_ids.append(conversation_id)
            self.test_user_ids.append(user_id)
            return conversation_id, user_id
    
    def create_test_messages(self, count: int = 5, role_mix: bool = True) -> List[Dict[str, Any]]:
        """Create test messages with various roles and content."""
        messages = []
        roles = ["human", "ai"] if role_mix else ["human"]
        
        for i in range(count):
            role = roles[i % len(roles)]
            messages.append({
                "role": role,
                "content": f"Test message {i+1} from {role}",
                "metadata": {
                    "test_message": True,
                    "message_number": i+1,
                    "created_by": "integration_test"
                }
            })
        
        return messages
    
    async def verify_messages_in_database(self, conversation_id: str, expected_count: int) -> List[Dict[str, Any]]:
        """Verify messages are correctly stored in the database."""
        def _get_test_messages():
            return (db_client.client.table("messages")
                   .select("*")
                   .eq("conversation_id", conversation_id)
                   .like("content", "Test message%")  # Only our test messages
                   .order("created_at")
                   .execute())
        
        result = await asyncio.to_thread(_get_test_messages)
        messages = result.data or []
        
        assert len(messages) == expected_count, f"Expected {expected_count} test messages, found {len(messages)}"
        return messages


@pytest_asyncio.fixture
async def test_helper():
    """Fixture providing test helper with setup and cleanup."""
    helper = MessagePersistenceTestHelper()
    await helper.setup()
    try:
        yield helper
    finally:
        await helper.cleanup()


class TestMessagePersistence:
    """Test suite for message persistence integration."""
    
    @pytest.mark.asyncio
    async def test_basic_message_storage(self, test_helper: MessagePersistenceTestHelper):
        """Test basic message storage functionality."""
        print("\n🧪 Testing Basic Message Storage")
        
        # Get existing conversation data
        conversation_id, user_id = await test_helper.get_existing_conversation()
        messages = test_helper.create_test_messages(3)
        
        # Create background task for message storage
        task = BackgroundTask(
            task_type="store_messages",
            priority=TaskPriority.NORMAL,
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": messages}
        )
        
        # Schedule and wait for task completion
        await test_helper.background_task_manager._handle_message_storage(task)
        
        # Verify messages were stored correctly
        stored_messages = await test_helper.verify_messages_in_database(conversation_id, 3)
        
        # Validate role mapping
        roles = [msg["role"] for msg in stored_messages]
        assert "user" in roles, "Human messages should be mapped to 'user' role"
        assert "assistant" in roles, "AI messages should be mapped to 'assistant' role"
        
        # Validate content preservation
        contents = [msg["content"] for msg in stored_messages]
        assert "Test message 1 from human" in contents
        assert "Test message 2 from ai" in contents
        
        # Validate metadata preservation
        for msg in stored_messages:
            assert msg["metadata"] is not None
            assert msg["metadata"].get("test_message") == True
            assert msg["conversation_id"] == conversation_id
            assert msg["user_id"] == user_id
        
        print("✅ Basic message storage test passed")
    
    @pytest.mark.asyncio
    async def test_role_mapping_comprehensive(self, test_helper: MessagePersistenceTestHelper):
        """Test comprehensive role mapping from LangChain to database format."""
        print("\n🧪 Testing Comprehensive Role Mapping")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Test all role mappings
        test_cases = [
            {"input_role": "human", "expected_role": "user"},
            {"input_role": "ai", "expected_role": "assistant"},
            {"input_role": "user", "expected_role": "user"},  # Should remain unchanged
            {"input_role": "assistant", "expected_role": "assistant"},  # Should remain unchanged
            {"input_role": "system", "expected_role": "system"},  # Should remain unchanged
        ]
        
        messages = []
        for i, case in enumerate(test_cases):
            messages.append({
                "role": case["input_role"],
                "content": f"Test message with {case['input_role']} role",
                "metadata": {"test_case": i}
            })
        
        # Store messages
        task = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": messages}
        )
        
        await test_helper.background_task_manager._handle_message_storage(task)
        
        # Verify role mappings
        stored_messages = await test_helper.verify_messages_in_database(conversation_id, len(test_cases))
        
        for i, (stored_msg, case) in enumerate(zip(stored_messages, test_cases)):
            assert stored_msg["role"] == case["expected_role"], \
                f"Role mapping failed: {case['input_role']} -> {stored_msg['role']} (expected {case['expected_role']})"
        
        print("✅ Role mapping test passed")
    
    @pytest.mark.asyncio
    async def test_batch_message_insertion(self, test_helper: MessagePersistenceTestHelper):
        """Test batch insertion of multiple messages."""
        print("\n🧪 Testing Batch Message Insertion")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create large batch of messages
        large_batch = test_helper.create_test_messages(50, role_mix=True)
        
        start_time = time.time()
        
        # Store large batch
        task = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": large_batch}
        )
        
        await test_helper.background_task_manager._handle_message_storage(task)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all messages were stored
        stored_messages = await test_helper.verify_messages_in_database(conversation_id, 50)
        
        # Validate performance (should be reasonably fast for 50 messages)
        assert processing_time < 5.0, f"Batch insertion took too long: {processing_time:.2f}s"
        
        # Validate ordering preservation
        for i, msg in enumerate(stored_messages):
            expected_content = f"Test message {i+1} from {'human' if i % 2 == 0 else 'ai'}"
            # Note: stored content will have mapped roles, so we check the pattern
            assert f"Test message {i+1}" in msg["content"]
        
        print(f"✅ Batch insertion test passed ({processing_time:.2f}s for 50 messages)")
    
    @pytest.mark.asyncio
    async def test_concurrent_message_storage(self, test_helper: MessagePersistenceTestHelper):
        """Test concurrent message storage operations."""
        print("\n🧪 Testing Concurrent Message Storage")
        
        # Create multiple conversations for concurrent testing
        conversation_data = []
        for i in range(5):
            conversation_id, user_id = await test_helper.get_existing_conversation()
            # Add unique identifier to avoid conflicts
            conversation_id = f"{conversation_id}-test-{i}"
            messages = test_helper.create_test_messages(10)
            conversation_data.append((conversation_id, user_id, messages))
        
        # Create concurrent tasks
        tasks = []
        for conversation_id, user_id, messages in conversation_data:
            task = BackgroundTask(
                task_type="store_messages",
                conversation_id=conversation_id,
                user_id=user_id,
                data={"messages": messages}
            )
            tasks.append(test_helper.background_task_manager._handle_message_storage(task))
        
        start_time = time.time()
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all conversations have their messages
        for conversation_id, user_id, messages in conversation_data:
            stored_messages = await test_helper.verify_messages_in_database(conversation_id, len(messages))
            
            # Verify data integrity
            for stored_msg in stored_messages:
                assert stored_msg["conversation_id"] == conversation_id
                assert stored_msg["user_id"] == user_id
        
        print(f"✅ Concurrent storage test passed ({processing_time:.2f}s for 5 conversations)")
    
    @pytest.mark.asyncio
    async def test_message_retrieval_and_querying(self, test_helper: MessagePersistenceTestHelper):
        """Test message retrieval and conversation-based querying."""
        print("\n🧪 Testing Message Retrieval and Querying")
        
        # Create test data with multiple conversations
        conversation1_id, user_id = await test_helper.get_existing_conversation()
        conversation2_id, _ = await test_helper.get_existing_conversation()
        # Add unique identifiers to avoid conflicts
        conversation1_id = f"{conversation1_id}-test-1"
        conversation2_id = f"{conversation2_id}-test-2"
        
        # Store messages in both conversations
        messages1 = test_helper.create_test_messages(5)
        messages2 = test_helper.create_test_messages(3)
        
        # Add distinguishing metadata
        for msg in messages1:
            msg["metadata"]["conversation"] = "first"
        for msg in messages2:
            msg["metadata"]["conversation"] = "second"
        
        # Store messages
        task1 = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation1_id,
            user_id=user_id,
            data={"messages": messages1}
        )
        
        task2 = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation2_id,
            user_id=user_id,
            data={"messages": messages2}
        )
        
        await test_helper.background_task_manager._handle_message_storage(task1)
        await test_helper.background_task_manager._handle_message_storage(task2)
        
        # Test conversation-specific retrieval
        conv1_messages = await test_helper.verify_messages_in_database(conversation1_id, 5)
        conv2_messages = await test_helper.verify_messages_in_database(conversation2_id, 3)
        
        # Verify conversation isolation
        for msg in conv1_messages:
            assert msg["conversation_id"] == conversation1_id
            assert msg["metadata"]["conversation"] == "first"
        
        for msg in conv2_messages:
            assert msg["conversation_id"] == conversation2_id
            assert msg["metadata"]["conversation"] == "second"
        
        # Test user-based querying
        def _get_user_messages():
            return (db_client.client.table("messages")
                   .select("*")
                   .eq("user_id", user_id)
                   .execute())
        
        user_messages_result = await asyncio.to_thread(_get_user_messages)
        user_messages = user_messages_result.data or []
        
        # Should have messages from both conversations
        assert len(user_messages) == 8  # 5 + 3
        
        conversation_ids = set(msg["conversation_id"] for msg in user_messages)
        assert conversation1_id in conversation_ids
        assert conversation2_id in conversation_ids
        
        print("✅ Message retrieval and querying test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_helper: MessagePersistenceTestHelper):
        """Test error handling and recovery mechanisms."""
        print("\n🧪 Testing Error Handling and Recovery")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Test with invalid data
        invalid_messages = [
            {"role": "human", "content": None},  # Invalid content
            {"role": "ai"},  # Missing content
            {"content": "Missing role"},  # Missing role
        ]
        
        task = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": invalid_messages}
        )
        
        # Should handle gracefully without crashing
        try:
            await test_helper.background_task_manager._handle_message_storage(task)
            # If it succeeds, verify what was actually stored
            stored_messages = await test_helper.verify_messages_in_database(conversation_id, 0)
        except Exception as e:
            # Error handling should be graceful
            assert "Failed to store messages" in str(e) or "content" in str(e).lower()
            print(f"✅ Error properly handled: {e}")
        
        # Test with valid messages to ensure system is still functional
        valid_messages = test_helper.create_test_messages(2)
        valid_task = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": valid_messages}
        )
        
        await test_helper.background_task_manager._handle_message_storage(valid_task)
        await test_helper.verify_messages_in_database(conversation_id, 2)
        
        print("✅ Error handling and recovery test passed")
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, test_helper: MessagePersistenceTestHelper):
        """Test that message metadata is properly preserved."""
        print("\n🧪 Testing Metadata Preservation")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create messages with rich metadata
        messages = [
            {
                "role": "human",
                "content": "Test message with complex metadata",
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "integration_test",
                    "nested_data": {
                        "level1": {
                            "level2": "deep_value"
                        }
                    },
                    "array_data": [1, 2, 3, "four"],
                    "boolean_flag": True,
                    "null_value": None
                }
            },
            {
                "role": "ai",
                "content": "Response with different metadata",
                "metadata": {
                    "model": "test-model",
                    "temperature": 0.7,
                    "tokens_used": 150,
                    "confidence": 0.95
                }
            }
        ]
        
        task = BackgroundTask(
            task_type="store_messages",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"messages": messages}
        )
        
        await test_helper.background_task_manager._handle_message_storage(task)
        
        # Verify metadata preservation
        stored_messages = await test_helper.verify_messages_in_database(conversation_id, 2)
        
        # Check first message metadata
        msg1 = stored_messages[0]
        assert msg1["metadata"]["source"] == "integration_test"
        assert msg1["metadata"]["nested_data"]["level1"]["level2"] == "deep_value"
        assert msg1["metadata"]["array_data"] == [1, 2, 3, "four"]
        assert msg1["metadata"]["boolean_flag"] is True
        assert msg1["metadata"]["null_value"] is None
        
        # Check second message metadata
        msg2 = stored_messages[1]
        assert msg2["metadata"]["model"] == "test-model"
        assert msg2["metadata"]["temperature"] == 0.7
        assert msg2["metadata"]["tokens_used"] == 150
        assert msg2["metadata"]["confidence"] == 0.95
        
        print("✅ Metadata preservation test passed")


async def run_comprehensive_message_persistence_tests():
    """Run all message persistence integration tests."""
    print("🚀 Starting Comprehensive Message Persistence Integration Tests")
    print("=" * 80)
    
    test_helper = MessagePersistenceTestHelper()
    await test_helper.setup()
    
    try:
        test_suite = TestMessagePersistence()
        
        # Run all tests
        await test_suite.test_basic_message_storage(test_helper)
        await test_suite.test_role_mapping_comprehensive(test_helper)
        await test_suite.test_batch_message_insertion(test_helper)
        await test_suite.test_concurrent_message_storage(test_helper)
        await test_suite.test_message_retrieval_and_querying(test_helper)
        await test_suite.test_error_handling_and_recovery(test_helper)
        await test_suite.test_metadata_preservation(test_helper)
        
        print("\n" + "=" * 80)
        print("✅ ALL MESSAGE PERSISTENCE INTEGRATION TESTS PASSED!")
        print("✅ Supabase messages table persistence is working correctly")
        print("✅ Role mapping, batch operations, and metadata preservation validated")
        print("✅ Error handling and concurrent operations tested successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        await test_helper.cleanup()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_message_persistence_tests())
