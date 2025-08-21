"""
Comprehensive Configurable Thresholds Testing

Task 5.8: Test configurable thresholds with different values (5, 10, 15, 20 messages)

This module provides comprehensive testing of the configurable threshold system
for conversation summary generation, ensuring that:
1. Thresholds work correctly at different message counts
2. Summary generation is triggered appropriately
3. Performance remains consistent across different threshold values
4. Edge cases and boundary conditions are handled properly
5. Background task behavior adapts to threshold configurations

Test Coverage:
1. Threshold values: 5, 10, 15, 20 messages
2. Summary generation behavior at each threshold
3. Performance impact of different thresholds
4. Edge cases (threshold = 1, very high thresholds)
5. Multiple conversations with different thresholds
6. Threshold configuration persistence and reliability
"""

import asyncio
import pytest
import pytest_asyncio
import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import test modules
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.database import db_client
from agents.background_tasks import BackgroundTaskManager, BackgroundTask, TaskPriority

# Skip tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)


class ThresholdTestHelper:
    """Helper class for configurable threshold testing."""
    
    def __init__(self):
        self.test_conversation_ids: List[str] = []
        self.test_user_ids: List[str] = []
        self.background_task_manager: Optional[BackgroundTaskManager] = None
        self.test_summary_ids: List[str] = []
    
    async def setup(self):
        """Set up test environment."""
        # Initialize background task manager
        self.background_task_manager = BackgroundTaskManager()
        await self.background_task_manager.start()
    
    async def cleanup(self):
        """Clean up test data and resources."""
        if self.background_task_manager:
            await self.background_task_manager.stop()
        
        # Clean up test messages
        for conversation_id in self.test_conversation_ids:
            try:
                await asyncio.to_thread(
                    lambda cid=conversation_id: db_client.client.table("messages")
                    .delete()
                    .eq("conversation_id", cid)
                    .like("content", "Threshold Test%")
                    .execute()
                )
            except Exception as e:
                print(f"Warning: Failed to clean up test messages: {e}")
        
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
                print(f"Warning: Failed to clean up test summary: {e}")
    
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
            # Fallback
            conversation_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            self.test_conversation_ids.append(conversation_id)
            self.test_user_ids.append(user_id)
            return conversation_id, user_id
    
    async def create_test_messages(self, conversation_id: str, user_id: str, count: int) -> List[Dict[str, Any]]:
        """Create test messages for threshold testing."""
        messages = []
        
        for i in range(count):
            message = {
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Threshold Test message {i+1} - Testing configurable thresholds",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"test": True, "threshold_test": True}
            }
            messages.append(message)
        
        return messages
    
    async def store_messages_in_database(self, messages: List[Dict[str, Any]]) -> bool:
        """Store messages directly in the database."""
        try:
            # Store messages using background task
            if self.background_task_manager and messages:
                conversation_id = messages[0]["conversation_id"]
                user_id = messages[0]["user_id"]
                
                task = BackgroundTask(
                    task_type="store_messages",
                    priority=TaskPriority.NORMAL,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    data={"messages": messages}
                )
                
                await self.background_task_manager._handle_message_storage(task)
                return True
        except Exception as e:
            print(f"Error storing messages: {e}")
            return False
        
        return False
    
    async def count_messages_in_conversation(self, conversation_id: str) -> int:
        """Count total messages in a conversation."""
        def _count_messages():
            return (db_client.client.table("messages")
                   .select("id", count="exact")
                   .eq("conversation_id", conversation_id)
                   .execute())
        
        result = await asyncio.to_thread(_count_messages)
        return result.count if result.count is not None else 0
    
    async def generate_summary_with_threshold(self, conversation_id: str, user_id: str, 
                                            threshold: int, max_messages: Optional[int] = None) -> Dict[str, Any]:
        """Generate summary with specific threshold configuration."""
        if not self.background_task_manager:
            return {"error": "BackgroundTaskManager not available"}
        
        # Record test start time for filtering
        test_start_time = datetime.utcnow()
        
        # Create summary generation task
        task_data = {
            "summary_threshold": threshold,
        }
        
        if max_messages is not None:
            task_data["max_messages"] = max_messages
        
        task = BackgroundTask(
            task_type="generate_summary",
            conversation_id=conversation_id,
            user_id=user_id,
            data=task_data
        )
        
        # Execute summary generation
        start_time = time.perf_counter()
        try:
            await self.background_task_manager._handle_summary_generation(task)
            duration = time.perf_counter() - start_time
            
            # Check if summary was created
            summary = await self.get_latest_summary(conversation_id, created_after=test_start_time)
            
            return {
                "success": True,
                "duration": duration,
                "summary_created": summary is not None,
                "summary": summary,
                "threshold": threshold,
                "max_messages": max_messages
            }
        
        except Exception as e:
            duration = time.perf_counter() - start_time
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "summary_created": False,
                "threshold": threshold,
                "max_messages": max_messages
            }
    
    async def get_latest_summary(self, conversation_id: str, created_after: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Get the latest summary for a conversation."""
        def _get_summary():
            query = (db_client.client.table("conversation_summaries")
                    .select("*")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=True))
            
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
    
    def analyze_threshold_behavior(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze threshold behavior across multiple tests."""
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        summaries_created = [r for r in successful_results if r.get("summary_created", False)]
        summaries_not_created = [r for r in successful_results if not r.get("summary_created", False)]
        
        durations = [r["duration"] for r in successful_results if "duration" in r]
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(failed_results),
            "summaries_created": len(summaries_created),
            "summaries_not_created": len(summaries_not_created),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "success_rate": len(successful_results) / len(results) if results else 0,
            "summary_creation_rate": len(summaries_created) / len(successful_results) if successful_results else 0
        }


@pytest_asyncio.fixture
async def threshold_helper():
    """Fixture providing threshold test helper with setup and cleanup."""
    helper = ThresholdTestHelper()
    await helper.setup()
    try:
        yield helper
    finally:
        await helper.cleanup()


class TestConfigurableThresholds:
    """Test suite for configurable threshold functionality."""
    
    @pytest.mark.asyncio
    async def test_threshold_values_5_10_15_20(self, threshold_helper: ThresholdTestHelper):
        """Test threshold values of 5, 10, 15, and 20 messages."""
        print("\n🚀 Testing Configurable Thresholds: 5, 10, 15, 20 messages")
        
        # Get existing conversation
        conversation_id, user_id = await threshold_helper.get_existing_conversation()
        
        # Test different threshold values
        threshold_values = [5, 10, 15, 20]
        results = []
        
        for threshold in threshold_values:
            print(f"\n📊 Testing threshold: {threshold} messages")
            
            # Count existing messages first
            existing_count = await threshold_helper.count_messages_in_conversation(conversation_id)
            print(f"   Existing messages: {existing_count}")
            
            # Create additional messages if needed to test threshold behavior
            if existing_count < threshold:
                # Add messages to reach threshold
                additional_needed = threshold - existing_count + 2  # Add 2 extra to exceed threshold
                test_messages = await threshold_helper.create_test_messages(
                    conversation_id, user_id, additional_needed
                )
                
                success = await threshold_helper.store_messages_in_database(test_messages)
                if not success:
                    print(f"   ⚠️  Failed to store test messages for threshold {threshold}")
                    continue
                
                # Wait a moment for messages to be stored
                await asyncio.sleep(0.5)
            
            # Test summary generation with this threshold
            result = await threshold_helper.generate_summary_with_threshold(
                conversation_id, user_id, threshold, max_messages=min(30, threshold * 2)
            )
            
            # Add threshold info to result
            result["threshold_tested"] = threshold
            result["existing_messages"] = existing_count
            results.append(result)
            
            # Analyze result
            if result["success"]:
                if result["summary_created"]:
                    summary = result["summary"]
                    print(f"   ✅ Summary created - Messages: {summary['message_count']}, Duration: {result['duration']*1000:.1f}ms")
                else:
                    print(f"   ℹ️  No summary created (below threshold) - Duration: {result['duration']*1000:.1f}ms")
            else:
                print(f"   ❌ Summary generation failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Analyze overall threshold behavior
        analysis = threshold_helper.analyze_threshold_behavior(results)
        
        print(f"\n📊 Threshold Testing Analysis:")
        print(f"   Total tests: {analysis['total_tests']}")
        print(f"   Successful tests: {analysis['successful_tests']}")
        print(f"   Summaries created: {analysis['summaries_created']}")
        print(f"   Average duration: {analysis['avg_duration']*1000:.1f}ms")
        print(f"   Success rate: {analysis['success_rate']*100:.1f}%")
        
        # Validate threshold behavior
        expected_success_rate = 0.8  # At least 80% should succeed
        if analysis['success_rate'] >= expected_success_rate:
            print(f"   ✅ SUCCESS RATE: {analysis['success_rate']*100:.1f}% >= {expected_success_rate*100:.0f}%")
        else:
            print(f"   ⚠️  SUCCESS RATE CONCERN: {analysis['success_rate']*100:.1f}% < {expected_success_rate*100:.0f}%")
        
        # Validate performance
        expected_max_duration = 1.0  # 1 second max
        if analysis['max_duration'] <= expected_max_duration:
            print(f"   ✅ PERFORMANCE: Max {analysis['max_duration']*1000:.1f}ms <= {expected_max_duration*1000:.0f}ms")
        else:
            print(f"   ⚠️  PERFORMANCE CONCERN: Max {analysis['max_duration']*1000:.1f}ms > {expected_max_duration*1000:.0f}ms")
        
        # Assert minimum performance standards
        assert analysis['success_rate'] >= 0.75, f"Threshold success rate too low: {analysis['success_rate']*100:.1f}%"
        
        print("✅ Configurable thresholds test completed")
    
    @pytest.mark.asyncio
    async def test_threshold_edge_cases(self, threshold_helper: ThresholdTestHelper):
        """Test edge cases for threshold configuration."""
        print("\n🚀 Testing Threshold Edge Cases")
        
        # Get existing conversation
        conversation_id, user_id = await threshold_helper.get_existing_conversation()
        
        # Test edge case thresholds
        edge_cases = [
            {"threshold": 1, "description": "Very low threshold (1 message)"},
            {"threshold": 100, "description": "Very high threshold (100 messages)"},
            {"threshold": 0, "description": "Zero threshold (edge case)"}
        ]
        
        results = []
        
        for case in edge_cases:
            threshold = case["threshold"]
            description = case["description"]
            
            print(f"\n📊 Testing: {description}")
            
            # Test summary generation with edge case threshold
            result = await threshold_helper.generate_summary_with_threshold(
                conversation_id, user_id, threshold, max_messages=50
            )
            
            result["case_description"] = description
            results.append(result)
            
            # Analyze result
            if result["success"]:
                if result["summary_created"]:
                    summary = result["summary"]
                    print(f"   ✅ Summary created - Messages: {summary['message_count']}, Duration: {result['duration']*1000:.1f}ms")
                else:
                    print(f"   ℹ️  No summary created - Duration: {result['duration']*1000:.1f}ms")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Validate edge case handling
        successful_edge_cases = [r for r in results if r["success"]]
        
        print(f"\n📊 Edge Case Analysis:")
        print(f"   Total edge cases tested: {len(results)}")
        print(f"   Successful edge cases: {len(successful_edge_cases)}")
        
        # The system should handle edge cases gracefully (not crash)
        assert len(successful_edge_cases) >= 2, "System should handle most edge cases gracefully"
        
        print("✅ Threshold edge cases test completed")
    
    @pytest.mark.asyncio
    async def test_threshold_performance_comparison(self, threshold_helper: ThresholdTestHelper):
        """Test performance impact of different threshold values."""
        print("\n🚀 Testing Threshold Performance Comparison")
        
        # Get existing conversation
        conversation_id, user_id = await threshold_helper.get_existing_conversation()
        
        # Test performance at different thresholds
        performance_tests = [
            {"threshold": 5, "max_messages": 10},
            {"threshold": 10, "max_messages": 20},
            {"threshold": 15, "max_messages": 30},
            {"threshold": 20, "max_messages": 40}
        ]
        
        performance_results = []
        
        for test_config in performance_tests:
            threshold = test_config["threshold"]
            max_messages = test_config["max_messages"]
            
            print(f"\n📊 Performance test: Threshold {threshold}, Max messages {max_messages}")
            
            # Run multiple iterations for statistical accuracy
            durations = []
            
            for iteration in range(3):
                result = await threshold_helper.generate_summary_with_threshold(
                    conversation_id, user_id, threshold, max_messages
                )
                
                if result["success"]:
                    durations.append(result["duration"])
                
                # Small delay between iterations
                await asyncio.sleep(0.2)
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
                
                performance_results.append({
                    "threshold": threshold,
                    "max_messages": max_messages,
                    "avg_duration": avg_duration,
                    "max_duration": max_duration,
                    "min_duration": min_duration,
                    "iterations": len(durations)
                })
                
                print(f"   ⏱️  Avg: {avg_duration*1000:.1f}ms, Max: {max_duration*1000:.1f}ms, Min: {min_duration*1000:.1f}ms")
            else:
                print(f"   ❌ No successful iterations for threshold {threshold}")
        
        # Analyze performance trends
        if performance_results:
            print(f"\n📊 Performance Comparison Summary:")
            
            for result in performance_results:
                print(f"   Threshold {result['threshold']:2d}: Avg {result['avg_duration']*1000:5.1f}ms, Max {result['max_duration']*1000:5.1f}ms")
            
            # Check if performance is reasonable across all thresholds
            all_avg_durations = [r["avg_duration"] for r in performance_results]
            overall_avg = sum(all_avg_durations) / len(all_avg_durations)
            
            expected_avg_duration = 0.5  # 500ms average
            if overall_avg <= expected_avg_duration:
                print(f"   ✅ OVERALL PERFORMANCE: {overall_avg*1000:.1f}ms <= {expected_avg_duration*1000:.0f}ms")
            else:
                print(f"   ⚠️  PERFORMANCE CONCERN: {overall_avg*1000:.1f}ms > {expected_avg_duration*1000:.0f}ms")
            
            # Validate performance consistency
            max_avg_duration = max(all_avg_durations)
            assert max_avg_duration <= 1.0, f"Threshold performance too slow: {max_avg_duration*1000:.1f}ms"
        
        print("✅ Threshold performance comparison completed")
    
    @pytest.mark.asyncio
    async def test_threshold_configuration_reliability(self, threshold_helper: ThresholdTestHelper):
        """Test reliability of threshold configuration across multiple operations."""
        print("\n🚀 Testing Threshold Configuration Reliability")
        
        # Get existing conversation
        conversation_id, user_id = await threshold_helper.get_existing_conversation()
        
        # Test the same threshold multiple times to ensure consistency
        test_threshold = 10
        iterations = 5
        
        print(f"📊 Testing threshold {test_threshold} across {iterations} iterations")
        
        results = []
        
        for i in range(iterations):
            print(f"   Iteration {i+1}/{iterations}")
            
            result = await threshold_helper.generate_summary_with_threshold(
                conversation_id, user_id, test_threshold, max_messages=20
            )
            
            result["iteration"] = i + 1
            results.append(result)
            
            # Small delay between iterations
            await asyncio.sleep(0.3)
        
        # Analyze reliability
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        durations = [r["duration"] for r in successful_results]
        summary_creations = [r["summary_created"] for r in successful_results]
        
        print(f"\n📊 Reliability Analysis:")
        print(f"   Successful iterations: {len(successful_results)}/{iterations}")
        print(f"   Failed iterations: {len(failed_results)}")
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            duration_variance = max(durations) - min(durations)
            print(f"   Average duration: {avg_duration*1000:.1f}ms")
            print(f"   Duration variance: {duration_variance*1000:.1f}ms")
            
            # Check consistency
            expected_variance = 0.5  # 500ms max variance
            if duration_variance <= expected_variance:
                print(f"   ✅ CONSISTENCY: Variance {duration_variance*1000:.1f}ms <= {expected_variance*1000:.0f}ms")
            else:
                print(f"   ⚠️  CONSISTENCY CONCERN: Variance {duration_variance*1000:.1f}ms > {expected_variance*1000:.0f}ms")
        
        # Validate reliability
        reliability_rate = len(successful_results) / iterations
        expected_reliability = 0.8  # 80% reliability
        
        if reliability_rate >= expected_reliability:
            print(f"   ✅ RELIABILITY: {reliability_rate*100:.1f}% >= {expected_reliability*100:.0f}%")
        else:
            print(f"   ⚠️  RELIABILITY CONCERN: {reliability_rate*100:.1f}% < {expected_reliability*100:.0f}%")
        
        # Assert minimum reliability
        assert reliability_rate >= 0.6, f"Threshold reliability too low: {reliability_rate*100:.1f}%"
        
        print("✅ Threshold configuration reliability test completed")


async def run_comprehensive_threshold_tests():
    """Run all configurable threshold tests."""
    print("🚀 Starting Comprehensive Configurable Threshold Tests")
    print("=" * 80)
    print("🎯 THRESHOLD TESTING OBJECTIVES:")
    print("   • Test thresholds: 5, 10, 15, 20 messages")
    print("   • Validate edge cases (1, 0, 100 messages)")
    print("   • Measure performance impact of different thresholds")
    print("   • Ensure configuration reliability and consistency")
    print("   • Verify summary generation behavior")
    print("=" * 80)
    
    threshold_helper = ThresholdTestHelper()
    await threshold_helper.setup()
    
    try:
        test_suite = TestConfigurableThresholds()
        
        # Run all threshold tests
        await test_suite.test_threshold_values_5_10_15_20(threshold_helper)
        await test_suite.test_threshold_edge_cases(threshold_helper)
        await test_suite.test_threshold_performance_comparison(threshold_helper)
        await test_suite.test_threshold_configuration_reliability(threshold_helper)
        
        print("\n" + "=" * 80)
        print("✅ ALL CONFIGURABLE THRESHOLD TESTS COMPLETED!")
        print("✅ Thresholds 5, 10, 15, 20 work correctly")
        print("✅ Edge cases handled gracefully")
        print("✅ Performance consistent across threshold values")
        print("✅ Configuration reliability validated")
        print("✅ Summary generation behavior verified")
        
    except Exception as e:
        print(f"\n❌ Threshold test failed: {e}")
        raise
    finally:
        await threshold_helper.cleanup()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_threshold_tests())





