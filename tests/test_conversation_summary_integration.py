"""
Comprehensive Integration Tests for Conversation Summary Generation and Storage

Task 5.2: Test conversation summary generation and storage to existing conversation_summaries table

This module provides complete integration testing of the conversation summary system,
validating that summaries are correctly generated, stored, and managed in the Supabase
conversation_summaries table through the BackgroundTaskManager.

Test Coverage:
1. Summary generation with configurable thresholds
2. Storage to existing conversation_summaries table
3. Metadata preservation and structure validation
4. Message count and date range tracking
5. Threshold-based summary generation logic
6. Performance under different conversation sizes
7. Error handling for edge cases
8. Integration with existing database schema
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


class ConversationSummaryTestHelper:
    """Helper class for conversation summary integration testing."""
    
    def __init__(self):
        self.test_conversation_ids: List[str] = []
        self.test_user_ids: List[str] = []
        self.test_summary_ids: List[str] = []
        self.background_task_manager: Optional[BackgroundTaskManager] = None
    
    async def setup(self):
        """Set up test environment."""
        self.background_task_manager = BackgroundTaskManager()
        await self.background_task_manager.start()
    
    async def cleanup(self):
        """Clean up test data and resources."""
        if self.background_task_manager:
            await self.background_task_manager.stop()
        
        # Clean up test summaries
        for summary_id in self.test_summary_ids:
            try:
                await asyncio.to_thread(
                    lambda sid=summary_id: db_client.client.table("conversation_summaries")
                    .delete()
                    .eq("id", sid)
                    .execute()
                )
            except Exception as e:
                print(f"Warning: Failed to clean up summary {summary_id}: {e}")
        
        # Clean up test messages
        for conversation_id in self.test_conversation_ids:
            try:
                await asyncio.to_thread(
                    lambda cid=conversation_id: db_client.client.table("messages")
                    .delete()
                    .eq("conversation_id", cid)
                    .like("content", "Test message%")
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
            
            self.test_conversation_ids.append(conversation_id)
            self.test_user_ids.append(user_id)
            
            return conversation_id, user_id
        else:
            # Fallback: create test identifiers
            conversation_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            self.test_conversation_ids.append(conversation_id)
            self.test_user_ids.append(user_id)
            return conversation_id, user_id
    
    def create_test_messages(self, count: int = 10, role_mix: bool = True) -> List[Dict[str, Any]]:
        """Create test messages for summary generation."""
        messages = []
        roles = ["user", "assistant"] if role_mix else ["user"]
        
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(count):
            role = roles[i % len(roles)]
            # Create more realistic message content for better summaries
            if role == "user":
                content = f"Test user message {i+1}: This is a customer inquiry about our services."
            else:
                content = f"Test assistant message {i+1}: Thank you for your inquiry. Let me help you with that."
            
            messages.append({
                "role": role,
                "content": content,
                "created_at": (base_time + timedelta(minutes=i*2)).isoformat(),
                "metadata": {
                    "test_message": True,
                    "message_number": i+1,
                    "created_by": "summary_integration_test"
                }
            })
        
        return messages
    
    async def create_messages_in_database(self, conversation_id: str, user_id: str, messages: List[Dict[str, Any]]):
        """Store test messages in the database for summary generation."""
        # Prepare messages for database
        db_messages = []
        for msg in messages:
            db_messages.append({
                "conversation_id": conversation_id,
                "role": msg["role"],
                "content": msg["content"],
                "user_id": user_id,
                "created_at": msg["created_at"],
                "metadata": msg.get("metadata", {})
            })
        
        def _store_messages():
            return db_client.client.table("messages").insert(db_messages).execute()
        
        result = await asyncio.to_thread(_store_messages)
        return result.data
    
    async def verify_summary_in_database(self, conversation_id: str, created_after: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Verify that a summary was created and stored correctly."""
        def _get_summary():
            query = (db_client.client.table("conversation_summaries")
                    .select("*")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=True))
            
            # Filter by creation time if specified
            if created_after:
                query = query.gte("created_at", created_after.isoformat())
            
            return query.limit(1).execute()
        
        result = await asyncio.to_thread(_get_summary)
        
        if result.data and len(result.data) > 0:
            summary = result.data[0]
            # Track for cleanup
            if "id" in summary:
                self.test_summary_ids.append(summary["id"])
            return summary
        return None
    
    async def count_messages_in_conversation(self, conversation_id: str) -> int:
        """Count messages in a conversation."""
        def _count_messages():
            return (db_client.client.table("messages")
                   .select("id", count="exact")
                   .eq("conversation_id", conversation_id)
                   .execute())
        
        result = await asyncio.to_thread(_count_messages)
        return result.count or 0


@pytest_asyncio.fixture
async def test_helper():
    """Fixture providing test helper with setup and cleanup."""
    helper = ConversationSummaryTestHelper()
    await helper.setup()
    try:
        yield helper
    finally:
        await helper.cleanup()


class TestConversationSummary:
    """Test suite for conversation summary integration."""
    
    @pytest.mark.asyncio
    async def test_basic_summary_generation(self, test_helper: ConversationSummaryTestHelper):
        """Test basic conversation summary generation and storage."""
        print("\n🧪 Testing Basic Summary Generation")
        
        # Get existing conversation data
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create test messages above the default threshold (10)
        messages = test_helper.create_test_messages(12)
        await test_helper.create_messages_in_database(conversation_id, user_id, messages)
        
        # Create summary generation task
        task = BackgroundTask(
            task_type="generate_summary",
            priority=TaskPriority.NORMAL,
            conversation_id=conversation_id,
            user_id=user_id,
            data={
                "max_messages": 15,
                "summary_threshold": 10
            }
        )
        
        # Execute summary generation
        await test_helper.background_task_manager._handle_summary_generation(task)
        
        # Verify summary was created
        summary = await test_helper.verify_summary_in_database(conversation_id)
        assert summary is not None, "Summary should be created"
        
        # Validate summary structure
        assert summary["conversation_id"] == conversation_id
        assert summary["user_id"] == user_id
        assert summary["summary_text"] is not None
        assert len(summary["summary_text"]) > 0
        # Message count should be limited by max_messages parameter (15), not our test messages (12)
        assert summary["message_count"] <= 15  # Should not exceed max_messages limit
        assert summary["message_count"] >= 12  # Should include at least our test messages
        
        # Validate metadata
        metadata = summary.get("metadata", {})
        assert metadata["max_messages_used"] == 15
        assert metadata["summary_threshold"] == 10
        assert metadata["generated_by"] == "background_task_manager"
        
        print("✅ Basic summary generation test passed")
    
    @pytest.mark.asyncio
    async def test_threshold_based_summary_generation(self, test_helper: ConversationSummaryTestHelper):
        """Test that summaries are only generated when threshold is met."""
        print("\n🧪 Testing Threshold-Based Summary Generation")
        
        # Record test start time to filter new summaries
        test_start_time = datetime.utcnow()
        
        # Test case 1: Check actual message count first
        conversation_id1, user_id1 = await test_helper.get_existing_conversation()
        
        # Count existing messages in the conversation
        existing_count = await test_helper.count_messages_in_conversation(conversation_id1)
        print(f"Existing messages in conversation: {existing_count}")
        
        # Set a very high threshold to test "below threshold" behavior
        very_high_threshold = existing_count + 100  # Much higher than current count
        
        task1 = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id1,
            user_id=user_id1,
            data={"summary_threshold": very_high_threshold}
        )
        
        # Should not generate summary with very high threshold
        await test_helper.background_task_manager._handle_summary_generation(task1)
        summary1 = await test_helper.verify_summary_in_database(conversation_id1, created_after=test_start_time)
        assert summary1 is None, f"Summary should not be generated when threshold ({very_high_threshold}) > message count ({existing_count})"
        
        # Test case 2: Above threshold - should generate summary
        conversation_id2, user_id2 = await test_helper.get_existing_conversation()
        
        # Count existing messages and set a low threshold to ensure generation
        existing_count2 = await test_helper.count_messages_in_conversation(conversation_id2)
        low_threshold = min(10, max(1, existing_count2 - 5))  # Keep threshold reasonable
        max_messages_to_use = max(20, existing_count2)  # Ensure max_messages >= threshold
        
        task2 = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id2,
            user_id=user_id2,
            data={
                "summary_threshold": low_threshold,
                "max_messages": max_messages_to_use
            }
        )
        
        # Should generate summary
        await test_helper.background_task_manager._handle_summary_generation(task2)
        summary2 = await test_helper.verify_summary_in_database(conversation_id2, created_after=test_start_time)
        assert summary2 is not None, f"Summary should be generated when threshold ({low_threshold}) <= message count ({existing_count2})"
        # The summary uses min(existing_count, max_messages), not necessarily >= low_threshold
        assert summary2["message_count"] >= low_threshold or summary2["message_count"] == min(existing_count2, max_messages_to_use)
        
        print("✅ Threshold-based summary generation test passed")
    
    @pytest.mark.asyncio
    async def test_configurable_thresholds(self, test_helper: ConversationSummaryTestHelper):
        """Test summary generation with different configurable thresholds."""
        print("\n🧪 Testing Configurable Thresholds")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create messages
        messages = test_helper.create_test_messages(8)
        await test_helper.create_messages_in_database(conversation_id, user_id, messages)
        
        # Test with custom threshold of 5 (should generate summary)
        task = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id,
            user_id=user_id,
            data={
                "summary_threshold": 5,
                "max_messages": 20
            }
        )
        
        await test_helper.background_task_manager._handle_summary_generation(task)
        
        # Verify summary was created with custom settings
        summary = await test_helper.verify_summary_in_database(conversation_id)
        assert summary is not None
        assert summary["message_count"] == 8
        
        metadata = summary.get("metadata", {})
        assert metadata["summary_threshold"] == 5
        assert metadata["max_messages_used"] == 20
        
        print("✅ Configurable thresholds test passed")
    
    @pytest.mark.asyncio
    async def test_summary_content_quality(self, test_helper: ConversationSummaryTestHelper):
        """Test that summary content contains expected information."""
        print("\n🧪 Testing Summary Content Quality")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create diverse messages for better summary testing
        messages = [
            {
                "role": "user",
                "content": "Hello, I'm interested in purchasing a vehicle",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"test_message": True}
            },
            {
                "role": "assistant", 
                "content": "Great! I'd be happy to help you find the perfect vehicle. What type are you looking for?",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"test_message": True}
            },
            {
                "role": "user",
                "content": "I need a reliable SUV for my family",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"test_message": True}
            },
            {
                "role": "assistant",
                "content": "Perfect! We have several family-friendly SUVs available. Let me show you our options.",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"test_message": True}
            }
        ]
        
        # Add more messages to meet threshold
        for i in range(7):
            messages.extend([
                {
                    "role": "user",
                    "content": f"Test follow-up question {i+1}",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": {"test_message": True}
                },
                {
                    "role": "assistant",
                    "content": f"Test response {i+1}",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": {"test_message": True}
                }
            ])
        
        await test_helper.create_messages_in_database(conversation_id, user_id, messages)
        
        # Generate summary
        task = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id,
            user_id=user_id,
            data={"summary_threshold": 5}
        )
        
        await test_helper.background_task_manager._handle_summary_generation(task)
        
        # Verify summary content
        summary = await test_helper.verify_summary_in_database(conversation_id)
        assert summary is not None
        
        summary_text = summary["summary_text"]
        assert len(summary_text) > 20, "Summary should have meaningful content"
        
        # Check that summary contains basic conversation info
        assert "messages" in summary_text.lower()
        assert str(len(messages)) in summary_text  # Message count should be mentioned
        
        print("✅ Summary content quality test passed")
    
    @pytest.mark.asyncio
    async def test_large_conversation_summary(self, test_helper: ConversationSummaryTestHelper):
        """Test summary generation for large conversations."""
        print("\n🧪 Testing Large Conversation Summary")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        # Create large conversation (100 messages)
        messages = test_helper.create_test_messages(100, role_mix=True)
        
        start_time = time.time()
        await test_helper.create_messages_in_database(conversation_id, user_id, messages)
        
        # Generate summary with limited max_messages
        task = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id,
            user_id=user_id,
            data={
                "summary_threshold": 50,
                "max_messages": 75  # Should only use 75 most recent messages
            }
        )
        
        await test_helper.background_task_manager._handle_summary_generation(task)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify summary
        summary = await test_helper.verify_summary_in_database(conversation_id)
        assert summary is not None
        
        # Should use limited message count, not full 100
        assert summary["message_count"] == 75
        
        metadata = summary.get("metadata", {})
        assert metadata["max_messages_used"] == 75
        
        # Performance check - should handle large conversations efficiently
        assert processing_time < 10.0, f"Large conversation processing took too long: {processing_time:.2f}s"
        
        print(f"✅ Large conversation summary test passed ({processing_time:.2f}s for 100 messages)")
    
    @pytest.mark.asyncio
    async def test_error_handling_edge_cases(self, test_helper: ConversationSummaryTestHelper):
        """Test error handling for edge cases in summary generation."""
        print("\n🧪 Testing Error Handling Edge Cases")
        
        # Test case 1: Conversation with no messages
        empty_conversation_id = str(uuid.uuid4())
        user_id = await test_helper.get_existing_conversation()
        if isinstance(user_id, tuple):
            user_id = user_id[1]
        
        task1 = BackgroundTask(
            task_type="generate_summary",
            conversation_id=empty_conversation_id,
            user_id=user_id,
            data={"summary_threshold": 1}
        )
        
        # Should handle gracefully without crashing
        try:
            await test_helper.background_task_manager._handle_summary_generation(task1)
            # If no exception, verify no summary was created
            summary1 = await test_helper.verify_summary_in_database(empty_conversation_id)
            assert summary1 is None, "Should not create summary for empty conversation"
        except Exception as e:
            # Should handle gracefully
            assert "No messages found" in str(e) or "not found" in str(e).lower()
        
        # Test case 2: Very high threshold
        conversation_id2, user_id2 = await test_helper.get_existing_conversation()
        
        messages = test_helper.create_test_messages(5)
        await test_helper.create_messages_in_database(conversation_id2, user_id2, messages)
        
        task2 = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id2,
            user_id=user_id2,
            data={"summary_threshold": 1000}  # Very high threshold
        )
        
        await test_helper.background_task_manager._handle_summary_generation(task2)
        summary2 = await test_helper.verify_summary_in_database(conversation_id2)
        assert summary2 is None, "Should not create summary when threshold not met"
        
        print("✅ Error handling edge cases test passed")
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, test_helper: ConversationSummaryTestHelper):
        """Test that summary metadata is properly preserved and structured."""
        print("\n🧪 Testing Metadata Preservation")
        
        conversation_id, user_id = await test_helper.get_existing_conversation()
        
        messages = test_helper.create_test_messages(15)
        await test_helper.create_messages_in_database(conversation_id, user_id, messages)
        
        # Generate summary with specific configuration
        custom_config = {
            "summary_threshold": 10,
            "max_messages": 20,
            "test_parameter": "integration_test_value"
        }
        
        task = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id,
            user_id=user_id,
            data=custom_config
        )
        
        await test_helper.background_task_manager._handle_summary_generation(task)
        
        # Verify metadata structure
        summary = await test_helper.verify_summary_in_database(conversation_id)
        assert summary is not None
        
        # Check required fields
        assert "id" in summary
        assert "conversation_id" in summary
        assert "user_id" in summary
        assert "summary_text" in summary
        assert "message_count" in summary
        assert "created_at" in summary
        assert "metadata" in summary
        
        # Check metadata content
        metadata = summary["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["max_messages_used"] == 20
        assert metadata["summary_threshold"] == 10
        assert metadata["generated_by"] == "background_task_manager"
        
        # Verify timestamp format
        created_at = summary["created_at"]
        assert isinstance(created_at, str)
        # Should be valid ISO format timestamp
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        print("✅ Metadata preservation test passed")


async def run_comprehensive_summary_tests():
    """Run all conversation summary integration tests."""
    print("🚀 Starting Comprehensive Conversation Summary Integration Tests")
    print("=" * 80)
    
    test_helper = ConversationSummaryTestHelper()
    await test_helper.setup()
    
    try:
        test_suite = TestConversationSummary()
        
        # Run all tests
        await test_suite.test_basic_summary_generation(test_helper)
        await test_suite.test_threshold_based_summary_generation(test_helper)
        await test_suite.test_configurable_thresholds(test_helper)
        await test_suite.test_summary_content_quality(test_helper)
        await test_suite.test_large_conversation_summary(test_helper)
        await test_suite.test_error_handling_edge_cases(test_helper)
        await test_suite.test_metadata_preservation(test_helper)
        
        print("\n" + "=" * 80)
        print("✅ ALL CONVERSATION SUMMARY INTEGRATION TESTS PASSED!")
        print("✅ Conversation summary generation and storage working correctly")
        print("✅ Threshold-based generation, metadata preservation, and performance validated")
        print("✅ Error handling and edge cases tested successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        await test_helper.cleanup()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_summary_tests())
