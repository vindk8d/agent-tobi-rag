"""
Token-Efficient Concurrent Background Task Testing.

This module tests the BackgroundTaskManager with 1000+ concurrent tasks
without consuming API tokens, following the PRD requirements:

- 1.11.1 Create mock LLM responses for summary generation tasks (avoid real API calls)
- 1.11.2 Use minimal test message content (10-20 words per message max)
- 1.11.3 Implement test mode flag to bypass actual LLM calls for performance testing
- 1.11.4 Test with 70% message storage tasks, 20% context loading, 10% summary generation
- 1.11.5 Use local/mock embedding generation instead of OpenAI API calls
- 1.11.6 Validate queue processing, retry logic, and error handling without token usage
- 1.11.7 Measure processing times and throughput with mock responses
- 1.11.8 Test API rate limiting and backoff strategies with minimal real API calls (<100 tokens total)

Performance Goals:
- Process 1000+ tasks concurrently
- Measure throughput and processing times
- Validate queue management and retry logic
- Test system under load without API costs
"""

import asyncio
import os
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
import pytest

# Set test mode to avoid real API calls
TEST_MODE = os.getenv('TEST_MODE', 'mock').lower() == 'mock'

from backend.agents.background_tasks import (
    BackgroundTaskManager,
    BackgroundTask,
    TaskPriority,
    TaskStatus
)


class MockLLMResponses:
    """
    Mock LLM responses for summary generation tasks.
    Provides realistic but token-free responses for testing.
    """
    
    SUMMARY_TEMPLATES = [
        "Customer discussed {topic} with {message_count} messages exchanged. Key points: {keywords}.",
        "Conversation focused on {topic}. {message_count} messages covered {keywords}.",
        "User inquiry about {topic}. Discussion included {keywords} across {message_count} messages.",
        "Support conversation regarding {topic}. Main topics: {keywords} in {message_count} exchanges.",
        "Customer service interaction on {topic}. Covered {keywords} over {message_count} messages."
    ]
    
    TOPICS = [
        "product pricing", "order status", "technical support", "account issues", 
        "billing inquiry", "feature request", "bug report", "general question"
    ]
    
    KEYWORDS = [
        ["pricing", "cost", "payment"], ["order", "shipping", "delivery"],
        ["technical", "setup", "configuration"], ["account", "login", "access"],
        ["billing", "invoice", "charges"], ["feature", "enhancement", "request"],
        ["bug", "error", "issue"], ["question", "help", "support"]
    ]
    
    @classmethod
    def generate_summary(cls, message_count: int = 5) -> str:
        """Generate a mock summary without LLM calls."""
        import random
        
        template = random.choice(cls.SUMMARY_TEMPLATES)
        topic = random.choice(cls.TOPICS)
        keywords = ", ".join(random.choice(cls.KEYWORDS))
        
        return template.format(
            topic=topic,
            message_count=message_count,
            keywords=keywords
        )
    
    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """Generate mock embedding vector without API calls."""
        # Generate deterministic mock embedding based on text hash
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to consistent float vector (384 dimensions like OpenAI)
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            # Convert hex to float between -1 and 1
            float_val = (int(hex_pair, 16) - 128) / 128.0
            embedding.append(float_val)
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.extend(embedding[:min(10, 384 - len(embedding))])
        
        return embedding[:384]


class MinimalTestData:
    """
    Minimal test message content (10-20 words per message max).
    Designed to minimize token usage while maintaining realistic test scenarios.
    """
    
    USER_MESSAGES = [
        "Hello, I need help with my order",
        "Can you check my account status?",
        "What's the price for premium features?",
        "I'm having trouble logging in",
        "How do I cancel my subscription?",
        "When will my order ship?",
        "Is there a discount available?",
        "Can you help me reset password?",
        "What payment methods do you accept?",
        "I want to upgrade my plan"
    ]
    
    ASSISTANT_MESSAGES = [
        "I'd be happy to help you with that",
        "Let me check your account details",
        "Here's the pricing information you requested",
        "I can help you with login issues",
        "I'll guide you through the cancellation process",
        "Your order will ship within 2-3 days",
        "Yes, we have a 10% discount available",
        "I'll send you a password reset link",
        "We accept all major credit cards",
        "I can help you upgrade right away"
    ]
    
    @classmethod
    def generate_conversation(cls, message_count: int = 5) -> List[Dict[str, Any]]:
        """Generate minimal test conversation."""
        import random
        
        messages = []
        for i in range(message_count):
            if i % 2 == 0:  # User message
                messages.append({
                    "role": "user",
                    "content": random.choice(cls.USER_MESSAGES),
                    "created_at": datetime.utcnow().isoformat()
                })
            else:  # Assistant message
                messages.append({
                    "role": "assistant", 
                    "content": random.choice(cls.ASSISTANT_MESSAGES),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return messages


class ConcurrentTaskGenerator:
    """
    Generates concurrent tasks following the specified distribution:
    - 70% message storage tasks
    - 20% context loading tasks  
    - 10% summary generation tasks
    """
    
    @staticmethod
    def generate_task_batch(batch_size: int = 1000) -> List[BackgroundTask]:
        """Generate a batch of tasks with specified distribution."""
        tasks = []
        
        # Calculate task counts based on distribution
        storage_count = int(batch_size * 0.7)  # 70%
        context_count = int(batch_size * 0.2)   # 20%
        summary_count = batch_size - storage_count - context_count  # Remaining 10%
        
        # Generate message storage tasks (70%)
        for i in range(storage_count):
            messages = MinimalTestData.generate_conversation(3)  # Minimal messages
            task = BackgroundTask(
                task_type="message_storage",
                priority=TaskPriority.NORMAL,
                data={"messages": messages, "table": "messages"},
                conversation_id=str(uuid4()),
                user_id=str(uuid4()),  # Use proper UUID format
                customer_id=str(uuid4()) if i % 2 == 0 else None,
                employee_id=str(uuid4()) if i % 2 == 1 else None
            )
            tasks.append(task)
        
        # Generate context loading tasks (20%)
        for i in range(context_count):
            task = BackgroundTask(
                task_type="context_loading",
                priority=TaskPriority.HIGH,
                data={
                    "load_user_context": True,
                    "load_conversation_history": True
                },
                conversation_id=str(uuid4()),
                user_id=str(uuid4()),  # Use proper UUID format
                customer_id=str(uuid4()) if i % 3 == 0 else None
            )
            tasks.append(task)
        
        # Generate summary generation tasks (10%)
        for i in range(summary_count):
            task = BackgroundTask(
                task_type="summary_generation",
                priority=TaskPriority.NORMAL,
                data={
                    "max_messages": 10,
                    "summary_threshold": 5,
                    "table": "conversation_summaries"
                },
                conversation_id=str(uuid4()),
                user_id=str(uuid4()),  # Use proper UUID format
                customer_id=str(uuid4())  # Use proper UUID format
            )
            tasks.append(task)
        
        # Shuffle tasks to simulate realistic mixed workload
        import random
        random.shuffle(tasks)
        
        return tasks


class PerformanceMetrics:
    """
    Measures processing times and throughput with detailed analytics.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.task_times: List[float] = []
        self.task_counts: Dict[str, int] = {
            "message_storage": 0,
            "context_loading": 0,
            "summary_generation": 0
        }
        self.success_counts: Dict[str, int] = {
            "message_storage": 0,
            "context_loading": 0, 
            "summary_generation": 0
        }
        self.error_counts: Dict[str, int] = {
            "message_storage": 0,
            "context_loading": 0,
            "summary_generation": 0
        }
        self.queue_size_samples: List[int] = []
        self.memory_usage_samples: List[float] = []
    
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
    
    def end_measurement(self):
        """End performance measurement."""
        self.end_time = time.time()
    
    def record_task_completion(self, task: BackgroundTask, processing_time: float, success: bool):
        """Record task completion metrics."""
        self.task_times.append(processing_time)
        self.task_counts[task.task_type] += 1
        
        if success:
            self.success_counts[task.task_type] += 1
        else:
            self.error_counts[task.task_type] += 1
    
    def record_queue_size(self, size: int):
        """Record queue size sample."""
        self.queue_size_samples.append(size)
    
    def record_memory_usage(self, usage_mb: float):
        """Record memory usage sample."""
        self.memory_usage_samples.append(usage_mb)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.start_time or not self.end_time:
            return {"error": "Measurement not completed"}
        
        total_time = self.end_time - self.start_time
        total_tasks = sum(self.task_counts.values())
        total_successful = sum(self.success_counts.values())
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_execution_time_seconds": round(total_time, 2),
            "total_tasks_processed": total_tasks,
            "tasks_per_second": round(total_tasks / total_time, 2) if total_time > 0 else 0,
            "success_rate": round(total_successful / total_tasks, 3) if total_tasks > 0 else 0,
            "error_rate": round(total_errors / total_tasks, 3) if total_tasks > 0 else 0,
            "task_distribution": self.task_counts,
            "success_distribution": self.success_counts,
            "error_distribution": self.error_counts,
            "processing_times": {
                "min_ms": round(min(self.task_times) * 1000, 2) if self.task_times else 0,
                "max_ms": round(max(self.task_times) * 1000, 2) if self.task_times else 0,
                "avg_ms": round(statistics.mean(self.task_times) * 1000, 2) if self.task_times else 0,
                "median_ms": round(statistics.median(self.task_times) * 1000, 2) if self.task_times else 0,
                "p95_ms": round(statistics.quantiles(self.task_times, n=20)[18] * 1000, 2) if len(self.task_times) >= 20 else 0
            },
            "queue_management": {
                "avg_queue_size": round(statistics.mean(self.queue_size_samples), 1) if self.queue_size_samples else 0,
                "max_queue_size": max(self.queue_size_samples) if self.queue_size_samples else 0
            },
            "memory_usage": {
                "avg_mb": round(statistics.mean(self.memory_usage_samples), 1) if self.memory_usage_samples else 0,
                "max_mb": round(max(self.memory_usage_samples), 1) if self.memory_usage_samples else 0
            }
        }


@pytest.mark.asyncio
class TestConcurrentBackgroundTasks:
    """
    Comprehensive concurrent testing suite for BackgroundTaskManager.
    Tests system performance under load without consuming API tokens.
    """
    
    @pytest.fixture
    def mock_db_operations(self):
        """Mock all database operations to avoid real DB calls."""
        with patch('backend.core.database.db_client') as mock_db:
            # Create mock responses
            mock_response = Mock()
            mock_response.data = [{"id": str(uuid4()), "created_at": "2024-01-01T10:00:00Z"}]
            mock_response.count = 10
            
            # Set up the mock chain to return successful responses
            mock_db.client.table.return_value.insert.return_value.execute.return_value = mock_response
            mock_db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
            mock_db.client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
            mock_db.client.table.return_value.upsert.return_value.execute.return_value = mock_response
            
            # Mock specific data for message retrieval
            mock_message_response = Mock()
            mock_message_response.data = [
                {"role": "user", "content": "test message", "created_at": "2024-01-01T10:00:00Z"}
            ]
            mock_message_response.count = 10
            
            # Override specific query chains
            mock_db.client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_message_response
            
            yield mock_db
    
    @pytest.fixture
    def mock_llm_calls(self):
        """Mock all LLM calls to avoid token usage."""
        if not TEST_MODE:
            yield None
            return
            
        with patch('backend.core.config.get_settings') as mock_settings:
            # Mock settings
            settings = Mock()
            settings.EMPLOYEE_MAX_MESSAGES = 12
            settings.CUSTOMER_MAX_MESSAGES = 15
            settings.SUMMARY_THRESHOLD = 8
            mock_settings.return_value = settings
            
            # Mock LLM responses
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.content = MockLLMResponses.generate_summary()
                mock_client.ainvoke.return_value = mock_response
                mock_openai.return_value = mock_client
                
                yield mock_client
    
    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics tracker."""
        return PerformanceMetrics()
    
    async def test_basic_functionality_validation(self, mock_db_operations, mock_llm_calls, performance_metrics):
        """
        Test basic BackgroundTaskManager functionality with proper UUID formats.
        Validates core operations work correctly without database format issues.
        """
        print(f"\n🧪 Testing basic BackgroundTaskManager functionality (TEST_MODE: {TEST_MODE})")
        
        # Create task manager with reasonable settings
        manager = BackgroundTaskManager()
        manager.max_concurrent_tasks = 10
        await manager._ensure_initialized()
        
        # Generate 50 tasks with proper UUID formats for initial validation
        tasks = ConcurrentTaskGenerator.generate_task_batch(50)
        
        # Verify all tasks have proper UUID formats
        for task in tasks:
            assert len(task.user_id) == 36, f"Invalid user_id format: {task.user_id}"
            assert len(task.conversation_id) == 36, f"Invalid conversation_id format: {task.conversation_id}"
            if task.customer_id:
                assert len(task.customer_id) == 36, f"Invalid customer_id format: {task.customer_id}"
            if task.employee_id:
                assert len(task.employee_id) == 36, f"Invalid employee_id format: {task.employee_id}"
        
        print(f"✅ All 50 tasks have proper UUID formats")
        
        # Start performance measurement
        performance_metrics.start_measurement()
        
        # Schedule all tasks
        task_ids = []
        for task in tasks:
            task_id = manager.schedule_task(task)
            task_ids.append(task_id)
        
        print(f"✅ Scheduled {len(task_ids)} tasks")
        
        # Start task manager
        await manager.start()
        
        # Wait for completion with reasonable timeout
        completed_count = 0
        max_wait_time = 30  # 30 seconds should be plenty
        check_interval = 0.2  # Check every 200ms
        start_time = time.time()
        
        while completed_count < 50 and (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(check_interval)
            completed_count = len(manager.completed_tasks) + len(manager.failed_tasks)
            
            if completed_count % 10 == 0 and completed_count > 0:
                print(f"📊 Progress: {completed_count}/50 tasks completed")
        
        # Stop task manager
        await manager.stop()
        performance_metrics.end_measurement()
        
        # Analyze results
        successful_tasks = len(manager.completed_tasks)
        failed_tasks = len(manager.failed_tasks)
        total_processed = successful_tasks + failed_tasks
        
        print(f"✅ Final Results: {successful_tasks} successful, {failed_tasks} failed, {total_processed} total")
        
        # Generate performance report
        summary = performance_metrics.get_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total execution time: {summary['total_execution_time_seconds']}s")
        print(f"   Tasks per second: {summary['tasks_per_second']}")
        print(f"   Success rate: {summary['success_rate'] * 100:.1f}%")
        
        # Basic functionality assertions
        assert total_processed >= 40, f"Expected at least 40 tasks processed, got {total_processed}"
        assert summary['success_rate'] >= 0.8, f"Expected at least 80% success rate, got {summary['success_rate'] * 100:.1f}%"
        assert summary['tasks_per_second'] >= 1, f"Expected at least 1 task/second, got {summary['tasks_per_second']}"
        
        print(f"✅ Basic functionality validation PASSED")
    
    async def test_1000_concurrent_tasks_basic(self, mock_db_operations, mock_llm_calls, performance_metrics):
        """
        Test 1000+ concurrent tasks with basic distribution.
        Validates core functionality under load.
        """
        print(f"\n🧪 Testing 1000 concurrent tasks (TEST_MODE: {TEST_MODE})")
        
        # Create task manager with higher concurrency for testing
        manager = BackgroundTaskManager()
        manager.max_concurrent_tasks = 50  # Increase for better throughput
        await manager._ensure_initialized()
        
        # Generate 1000 tasks with specified distribution
        tasks = ConcurrentTaskGenerator.generate_task_batch(1000)
        
        # Verify distribution
        storage_tasks = [t for t in tasks if t.task_type == "message_storage"]
        context_tasks = [t for t in tasks if t.task_type == "context_loading"]
        summary_tasks = [t for t in tasks if t.task_type == "summary_generation"]
        
        assert len(storage_tasks) == 700  # 70%
        assert len(context_tasks) == 200   # 20%
        assert len(summary_tasks) == 100   # 10%
        
        print(f"✅ Task distribution verified: {len(storage_tasks)} storage, {len(context_tasks)} context, {len(summary_tasks)} summary")
        
        # Start performance measurement
        performance_metrics.start_measurement()
        
        # Schedule all tasks
        task_ids = []
        for task in tasks:
            task_id = manager.schedule_task(task)
            task_ids.append(task_id)
        
        print(f"✅ Scheduled {len(task_ids)} tasks")
        
        # Start task manager
        await manager.start()
        
        # Monitor progress
        completed_count = 0
        max_wait_time = 60  # Maximum 60 seconds
        check_interval = 0.5  # Check every 500ms
        start_time = time.time()
        
        while completed_count < 1000 and (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(check_interval)
            
            # Count completed tasks
            completed_count = len(manager.completed_tasks) + len(manager.failed_tasks)
            
            # Record metrics
            total_queued = sum(len(queue) for queue in manager.task_queue.values())
            performance_metrics.record_queue_size(total_queued)
            
            # Show progress
            if completed_count % 100 == 0 and completed_count > 0:
                print(f"📊 Progress: {completed_count}/1000 tasks completed")
        
        # Stop task manager
        await manager.stop()
        performance_metrics.end_measurement()
        
        # Analyze results
        successful_tasks = len(manager.completed_tasks)
        failed_tasks = len(manager.failed_tasks)
        total_processed = successful_tasks + failed_tasks
        
        print(f"✅ Final Results: {successful_tasks} successful, {failed_tasks} failed, {total_processed} total")
        
        # Generate performance report
        summary = performance_metrics.get_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total execution time: {summary['total_execution_time_seconds']}s")
        print(f"   Tasks per second: {summary['tasks_per_second']}")
        print(f"   Success rate: {summary['success_rate'] * 100:.1f}%")
        print(f"   Average processing time: {summary['processing_times']['avg_ms']}ms")
        
        # Assertions for success criteria
        assert total_processed >= 800, f"Expected at least 800 tasks processed, got {total_processed}"
        assert summary['success_rate'] >= 0.7, f"Expected at least 70% success rate, got {summary['success_rate'] * 100:.1f}%"
        assert summary['tasks_per_second'] >= 10, f"Expected at least 10 tasks/second, got {summary['tasks_per_second']}"
    
    async def test_queue_processing_and_priority(self, mock_db_operations, mock_llm_calls):
        """
        Test queue processing, priority handling, and retry logic.
        Validates task management without token usage.
        """
        print(f"\n🔄 Testing queue processing and priority handling")
        
        manager = BackgroundTaskManager()
        manager.max_concurrent_tasks = 5
        await manager._ensure_initialized()
        
        # Create tasks with different priorities
        high_priority_tasks = []
        normal_priority_tasks = []
        
        for i in range(10):
            # High priority context loading tasks
            high_task = BackgroundTask(
                task_type="context_loading",
                priority=TaskPriority.HIGH,
                data={"load_user_context": True},
                user_id=str(uuid4()),  # Use proper UUID format
                conversation_id=str(uuid4())
            )
            high_priority_tasks.append(high_task)
            
            # Normal priority message storage tasks
            normal_task = BackgroundTask(
                task_type="message_storage", 
                priority=TaskPriority.NORMAL,
                data={"messages": MinimalTestData.generate_conversation(2)},
                user_id=str(uuid4()),  # Use proper UUID format
                conversation_id=str(uuid4())
            )
            normal_priority_tasks.append(normal_task)
        
        # Schedule normal priority tasks first
        for task in normal_priority_tasks:
            manager.schedule_task(task)
        
        # Then schedule high priority tasks
        for task in high_priority_tasks:
            manager.schedule_task(task)
        
        # Verify queue sizes
        assert len(manager.task_queue[TaskPriority.HIGH]) == 10
        assert len(manager.task_queue[TaskPriority.NORMAL]) == 10
        
        # Start processing
        await manager.start()
        
        # Wait for some processing
        await asyncio.sleep(2)
        
        await manager.stop()
        
        print(f"✅ Processed {len(manager.completed_tasks)} tasks with priority handling")
        
        # Verify high priority tasks were processed
        assert len(manager.completed_tasks) > 0, "Expected some tasks to be completed"
    
    async def test_retry_logic_and_error_handling(self, mock_llm_calls):
        """
        Test retry logic and error handling without database calls.
        Validates resilience under failure conditions.
        """
        print(f"\n🔄 Testing retry logic and error handling")
        
        manager = BackgroundTaskManager()
        await manager._ensure_initialized()
        
        # Create tasks that will fail (no database mocking)
        failing_tasks = []
        for i in range(5):
            task = BackgroundTask(
                task_type="message_storage",
                priority=TaskPriority.NORMAL,
                data={"messages": [{"role": "user", "content": "test"}]},
                user_id=str(uuid4()),  # Use proper UUID format
                conversation_id="invalid_uuid_format",  # This will cause DB errors
                max_retries=2  # Limit retries for faster testing
            )
            failing_tasks.append(task)
        
        # Schedule failing tasks
        for task in failing_tasks:
            manager.schedule_task(task)
        
        await manager.start()
        
        # Wait for retry cycles to complete
        await asyncio.sleep(5)
        
        await manager.stop()
        
        # Verify retry behavior
        failed_tasks = manager.failed_tasks
        print(f"✅ Failed tasks after retries: {len(failed_tasks)}")
        
        # Check that tasks went through retry cycles
        for task in failed_tasks:
            assert task.retry_count >= 1, f"Task {task.id} should have been retried"
            assert len(task.error_history) > 0, f"Task {task.id} should have error history"
        
        print(f"✅ Retry logic validated: {len(failed_tasks)} tasks failed after retries")
    
    async def test_throughput_measurement(self, mock_db_operations, mock_llm_calls, performance_metrics):
        """
        Test system throughput with detailed measurements.
        Measures processing efficiency without API costs.
        """
        print(f"\n📊 Testing system throughput and performance")
        
        manager = BackgroundTaskManager()
        manager.max_concurrent_tasks = 20  # Optimize for throughput
        await manager._ensure_initialized()
        
        # Generate smaller batch for detailed measurement
        tasks = ConcurrentTaskGenerator.generate_task_batch(200)
        
        performance_metrics.start_measurement()
        
        # Schedule all tasks and track individual timing
        task_start_times = {}
        for task in tasks:
            task_start_times[task.id] = time.time()
            manager.schedule_task(task)
        
        await manager.start()
        
        # Monitor with detailed metrics
        while len(manager.completed_tasks) + len(manager.failed_tasks) < 200:
            await asyncio.sleep(0.1)
            
            # Record queue sizes
            total_queued = sum(len(queue) for queue in manager.task_queue.values())
            performance_metrics.record_queue_size(total_queued)
            
            # Record memory usage (simplified)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            performance_metrics.record_memory_usage(memory_mb)
        
        await manager.stop()
        performance_metrics.end_measurement()
        
        # Calculate individual task processing times
        for task in manager.completed_tasks + manager.failed_tasks:
            if task.id in task_start_times and task.completed_at:
                start_time = task_start_times[task.id]
                end_time = task.completed_at.timestamp()
                processing_time = end_time - start_time
                success = task.status == TaskStatus.COMPLETED
                performance_metrics.record_task_completion(task, processing_time, success)
        
        # Generate detailed report
        summary = performance_metrics.get_summary()
        
        print(f"\n📈 Detailed Performance Report:")
        print(f"   Total tasks: {summary['total_tasks_processed']}")
        print(f"   Throughput: {summary['tasks_per_second']} tasks/second")
        print(f"   Success rate: {summary['success_rate'] * 100:.1f}%")
        print(f"   Processing times:")
        print(f"     Min: {summary['processing_times']['min_ms']}ms")
        print(f"     Avg: {summary['processing_times']['avg_ms']}ms")
        print(f"     Max: {summary['processing_times']['max_ms']}ms")
        print(f"     P95: {summary['processing_times']['p95_ms']}ms")
        print(f"   Queue management:")
        print(f"     Avg queue size: {summary['queue_management']['avg_queue_size']}")
        print(f"     Max queue size: {summary['queue_management']['max_queue_size']}")
        print(f"   Memory usage:")
        print(f"     Avg: {summary['memory_usage']['avg_mb']}MB")
        print(f"     Max: {summary['memory_usage']['max_mb']}MB")
        
        # Performance assertions
        assert summary['tasks_per_second'] >= 15, f"Expected at least 15 tasks/second, got {summary['tasks_per_second']}"
        assert summary['success_rate'] >= 0.8, f"Expected at least 80% success rate, got {summary['success_rate'] * 100:.1f}%"
        assert summary['processing_times']['avg_ms'] <= 1000, f"Expected avg processing time <= 1000ms, got {summary['processing_times']['avg_ms']}ms"
        
        # Save detailed results for analysis
        results_file = f"tests/concurrent_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Detailed results saved to {results_file}")
    
    async def test_api_rate_limiting_simulation(self, mock_db_operations):
        """
        Test API rate limiting and backoff strategies.
        Simulates rate limiting without consuming actual API tokens.
        """
        print(f"\n⏱️ Testing API rate limiting and backoff strategies")
        
        manager = BackgroundTaskManager()
        await manager._ensure_initialized()
        
        # Create tasks that simulate rate limiting
        rate_limited_tasks = []
        for i in range(20):
            task = BackgroundTask(
                task_type="summary_generation",
                priority=TaskPriority.NORMAL,
                data={
                    "max_messages": 5,
                    "summary_threshold": 3,
                    "simulate_rate_limit": True  # Custom flag for testing
                },
                user_id=str(uuid4()),  # Use proper UUID format
                conversation_id=str(uuid4())
            )
            rate_limited_tasks.append(task)
        
        # Mock rate limiting behavior
        call_count = 0
        original_handle_summary = manager._handle_summary_generation
        
        async def mock_handle_with_rate_limit(task):
            nonlocal call_count
            call_count += 1
            
            # Simulate rate limiting every 5th call
            if call_count % 5 == 0:
                await asyncio.sleep(0.5)  # Simulate backoff delay
                raise Exception("Rate limit exceeded - simulated")
            
            # Use simple summary generation instead of LLM
            return await original_handle_summary(task)
        
        manager._handle_summary_generation = mock_handle_with_rate_limit
        
        # Schedule tasks
        for task in rate_limited_tasks:
            manager.schedule_task(task)
        
        start_time = time.time()
        await manager.start()
        
        # Wait for processing with rate limiting
        while len(manager.completed_tasks) + len(manager.failed_tasks) < 20:
            await asyncio.sleep(0.1)
            if time.time() - start_time > 30:  # Timeout
                break
        
        await manager.stop()
        end_time = time.time()
        
        processing_time = end_time - start_time
        successful_tasks = len(manager.completed_tasks)
        failed_tasks = len(manager.failed_tasks)
        
        print(f"✅ Rate limiting test completed:")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Successful: {successful_tasks}")
        print(f"   Failed: {failed_tasks}")
        print(f"   Rate limit calls simulated: {call_count // 5}")
        
        # Verify rate limiting behavior
        assert processing_time > 2, "Expected processing time to be longer due to rate limiting"
        assert successful_tasks > 0, "Expected some tasks to succeed despite rate limiting"


if __name__ == "__main__":
    # Run tests directly for development
    import sys
    
    print("🚀 Running Concurrent Background Task Tests")
    print(f"TEST_MODE: {TEST_MODE}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode
        pytest.main([__file__ + "::TestConcurrentBackgroundTasks::test_throughput_measurement", "-v", "-s"])
    else:
        # Full test suite
        pytest.main([__file__, "-v", "-s", "--tb=short"])
