"""
Comprehensive unit tests for Background Task Infrastructure.

Tests the BackgroundTaskManager and related components with focus on:
- Task scheduling, prioritization, and retry logic
- Error handling and logging
- Message storage functionality
- Summary generation
- Health monitoring
- Concurrent task processing

Follows pytest best practices with fixtures, mocking, and comprehensive coverage.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from backend.agents.background_tasks import (
    BackgroundTaskManager,
    BackgroundTask,
    TaskPriority,
    TaskStatus
)


class TestBackgroundTask:
    """Test the BackgroundTask dataclass and its functionality."""
    
    def test_background_task_creation(self):
        """Test creating a BackgroundTask with default values."""
        task = BackgroundTask(task_type="test_task")
        
        assert task.task_type == "test_task"
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.retry_delay == 1.0
        assert isinstance(task.created_at, datetime)
        assert task.data == {}
        assert task.error_history == []
    
    def test_background_task_custom_values(self):
        """Test creating a BackgroundTask with custom values."""
        custom_data = {"key": "value"}
        task = BackgroundTask(
            task_type="custom_task",
            priority=TaskPriority.HIGH,
            data=custom_data,
            max_retries=5,
            retry_delay=2.0,
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        assert task.task_type == "custom_task"
        assert task.priority == TaskPriority.HIGH
        assert task.data == custom_data
        assert task.max_retries == 5
        assert task.retry_delay == 2.0
        assert task.conversation_id == "conv_123"
        assert task.user_id == "user_456"


class TestBackgroundTaskManager:
    """Test the BackgroundTaskManager class."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a BackgroundTaskManager instance for testing."""
        return BackgroundTaskManager()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.EMPLOYEE_MAX_MESSAGES = 15
        settings.CUSTOMER_MAX_MESSAGES = 20
        settings.SUMMARY_THRESHOLD = 8
        return settings
    
    def test_task_manager_initialization(self, task_manager):
        """Test BackgroundTaskManager initialization."""
        assert task_manager.settings is None
        assert not task_manager.is_running
        assert len(task_manager.task_queue) == 4  # One for each priority
        assert task_manager.active_tasks == {}
        assert task_manager.completed_tasks == []
        assert task_manager.failed_tasks == []
        assert task_manager.max_concurrent_tasks == 10
        
        # Check health stats initialization
        assert task_manager.health_stats['total_tasks_processed'] == 0
        assert task_manager.health_stats['successful_tasks'] == 0
        assert task_manager.health_stats['failed_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_ensure_initialized(self, task_manager, mock_settings):
        """Test async initialization of task manager."""
        with patch('backend.core.config.get_settings', return_value=mock_settings):
            await task_manager._ensure_initialized()
            
            assert task_manager.settings == mock_settings
            assert task_manager.employee_max_messages == 15
            assert task_manager.customer_max_messages == 20
            assert task_manager.summary_threshold == 8
    
    def test_schedule_task(self, task_manager):
        """Test basic task scheduling."""
        task = BackgroundTask(task_type="test_task", priority=TaskPriority.HIGH)
        
        task_id = task_manager.schedule_task(task)
        
        assert task_id == task.id
        assert len(task_manager.task_queue[TaskPriority.HIGH]) == 1
        assert task_manager.task_queue[TaskPriority.HIGH][0] == task
        assert task.scheduled_at is not None
        assert task_manager.health_stats['queue_sizes']['HIGH'] == 1
    
    def test_schedule_task_with_delay(self, task_manager):
        """Test scheduling task with delay."""
        task = BackgroundTask(task_type="delayed_task")
        delay_seconds = 5.0
        
        before_time = datetime.utcnow()
        task_manager.schedule_task(task, delay_seconds)
        after_time = datetime.utcnow()
        
        assert task.scheduled_at > before_time + timedelta(seconds=delay_seconds - 1)
        assert task.scheduled_at < after_time + timedelta(seconds=delay_seconds + 1)
    
    def test_schedule_message_storage(self, task_manager):
        """Test scheduling message storage task."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        task_id = task_manager.schedule_message_storage(
            conversation_id="conv_123",
            user_id="user_456",
            messages=messages,
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert len(task_manager.task_queue[TaskPriority.HIGH]) == 1
        
        scheduled_task = task_manager.task_queue[TaskPriority.HIGH][0]
        assert scheduled_task.task_type == "message_storage"
        assert scheduled_task.data["messages"] == messages
        assert scheduled_task.data["table"] == "messages"
        assert scheduled_task.conversation_id == "conv_123"
        assert scheduled_task.user_id == "user_456"
    
    def test_schedule_summary_generation_customer(self, task_manager):
        """Test scheduling summary generation for customer."""
        task_manager.customer_max_messages = 20
        task_manager.summary_threshold = 10
        
        task_id = task_manager.schedule_summary_generation(
            conversation_id="conv_789",
            user_id="user_123",
            customer_id="cust_456"
        )
        
        assert task_id is not None
        scheduled_task = task_manager.task_queue[TaskPriority.NORMAL][0]
        assert scheduled_task.task_type == "summary_generation"
        assert scheduled_task.data["max_messages"] == 20
        assert scheduled_task.data["summary_threshold"] == 10
        assert scheduled_task.customer_id == "cust_456"
    
    def test_schedule_summary_generation_employee(self, task_manager):
        """Test scheduling summary generation for employee."""
        task_manager.employee_max_messages = 15
        
        task_id = task_manager.schedule_summary_generation(
            conversation_id="conv_789",
            user_id="user_123",
            employee_id="emp_456"
        )
        
        scheduled_task = task_manager.task_queue[TaskPriority.NORMAL][0]
        assert scheduled_task.data["max_messages"] == 15
        assert scheduled_task.employee_id == "emp_456"
        assert scheduled_task.customer_id is None
    
    def test_schedule_context_loading(self, task_manager):
        """Test scheduling context loading task."""
        task_id = task_manager.schedule_context_loading(
            user_id="user_123",
            conversation_id="conv_456",
            customer_id="cust_789"
        )
        
        assert task_id is not None
        scheduled_task = task_manager.task_queue[TaskPriority.HIGH][0]
        assert scheduled_task.task_type == "context_loading"
        assert scheduled_task.priority == TaskPriority.HIGH
        assert scheduled_task.data["load_user_context"] is True
        assert scheduled_task.data["load_conversation_history"] is True
    
    @pytest.mark.asyncio
    async def test_start_stop(self, task_manager):
        """Test starting and stopping the task manager."""
        with patch.object(task_manager, '_ensure_initialized') as mock_init:
            mock_init.return_value = None
            
            # Test start
            await task_manager.start()
            assert task_manager.is_running is True
            assert task_manager.worker_pool is not None
            assert task_manager.processing_loop_task is not None
            
            # Test stop
            await task_manager.stop()
            assert task_manager.is_running is False
    
    def test_get_health_status(self, task_manager):
        """Test health status reporting."""
        # Add some tasks to queues
        task_manager.task_queue[TaskPriority.HIGH].append(BackgroundTask(task_type="test1"))
        task_manager.task_queue[TaskPriority.NORMAL].append(BackgroundTask(task_type="test2"))
        task_manager.health_stats['total_tasks_processed'] = 10
        task_manager.health_stats['successful_tasks'] = 8
        task_manager.health_stats['average_processing_time'] = 1.5
        
        health = task_manager.get_health_status()
        
        assert health['is_running'] is False
        assert health['active_tasks'] == 0
        assert health['queue_sizes']['HIGH'] == 1
        assert health['queue_sizes']['NORMAL'] == 1
        assert health['total_processed'] == 10
        assert health['success_rate'] == 0.8
        assert health['average_processing_time'] == 1.5
        assert 'configuration' in health
        assert health['configuration']['max_concurrent_tasks'] == 10
    
    def test_get_task_status_queued(self, task_manager):
        """Test getting status of queued task."""
        task = BackgroundTask(task_type="test_task")
        task_manager.schedule_task(task)
        
        status = task_manager.get_task_status(task.id)
        
        assert status is not None
        assert status['id'] == task.id
        assert status['task_type'] == "test_task"
        assert status['status'] == TaskStatus.PENDING.value
        assert status['priority'] == TaskPriority.NORMAL.name
    
    def test_get_task_status_not_found(self, task_manager):
        """Test getting status of non-existent task."""
        status = task_manager.get_task_status("non_existent_id")
        assert status is None
    
    def test_task_to_dict(self, task_manager):
        """Test converting task to dictionary."""
        task = BackgroundTask(
            task_type="test_task",
            priority=TaskPriority.HIGH,
            conversation_id="conv_123",
            user_id="user_456"
        )
        task.started_at = datetime.utcnow()
        task.last_error = "Test error"
        
        task_dict = task_manager._task_to_dict(task)
        
        assert task_dict['id'] == task.id
        assert task_dict['task_type'] == "test_task"
        assert task_dict['priority'] == "HIGH"
        assert task_dict['status'] == TaskStatus.PENDING.value
        assert task_dict['conversation_id'] == "conv_123"
        assert task_dict['user_id'] == "user_456"
        assert task_dict['last_error'] == "Test error"
        assert 'created_at' in task_dict
        assert 'started_at' in task_dict


class TestTaskExecution:
    """Test task execution functionality."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a BackgroundTaskManager instance for testing."""
        return BackgroundTaskManager()
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock database client."""
        with patch('backend.core.database.db_client') as mock:
            yield mock
    
    @pytest.mark.asyncio
    async def test_handle_message_storage_success(self, task_manager, mock_db_client):
        """Test successful message storage handling."""
        # Setup mock
        mock_result = Mock()
        mock_result.data = [{"id": "msg_123"}]
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value = mock_result
        
        # Create task
        messages = [
            {"role": "human", "content": "Hello"},  # Test role mapping
            {"role": "ai", "content": "Hi!"},       # Test role mapping
            {"role": "user", "content": "Thanks"}   # Already correct role
        ]
        task = BackgroundTask(
            task_type="message_storage",
            data={"messages": messages},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Execute
        await task_manager._handle_message_storage(task)
        
        # Verify database call
        mock_db_client.client.table.assert_called_with("messages")
        insert_call = mock_db_client.client.table.return_value.insert.call_args[0][0]
        
        # Check role mapping
        assert insert_call[0]["role"] == "user"     # human -> user
        assert insert_call[1]["role"] == "assistant" # ai -> assistant
        assert insert_call[2]["role"] == "user"     # user -> user (unchanged)
        
        # Check other fields
        assert insert_call[0]["conversation_id"] == "conv_123"
        assert insert_call[0]["user_id"] == "user_456"
        assert insert_call[0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_handle_message_storage_failure(self, task_manager, mock_db_client):
        """Test message storage handling failure."""
        # Setup mock to return empty data (failure)
        mock_result = Mock()
        mock_result.data = None
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value = mock_result
        
        task = BackgroundTask(
            task_type="message_storage",
            data={"messages": [{"role": "user", "content": "Hello"}]},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Should raise exception
        with pytest.raises(Exception, match="Failed to store messages"):
            await task_manager._handle_message_storage(task)
    
    @pytest.mark.asyncio
    async def test_handle_summary_generation_success(self, task_manager, mock_db_client):
        """Test successful summary generation."""
        # Mock message count query
        count_result = Mock()
        count_result.count = 15  # Above threshold
        
        # Mock messages query
        messages_result = Mock()
        messages_result.data = [
            {"role": "user", "content": "Hello", "created_at": "2024-01-01T10:00:00Z"},
            {"role": "assistant", "content": "Hi there!", "created_at": "2024-01-01T10:01:00Z"}
        ]
        
        # Mock summary insert
        summary_result = Mock()
        summary_result.data = [{"id": "summary_123"}]
        
        # Setup mock chain
        table_mock = mock_db_client.client.table.return_value
        table_mock.select.return_value.eq.return_value.execute.return_value = count_result
        table_mock.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = messages_result
        table_mock.insert.return_value.execute.return_value = summary_result
        
        task = BackgroundTask(
            task_type="summary_generation",
            data={"max_messages": 10, "summary_threshold": 5},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Execute
        await task_manager._handle_summary_generation(task)
        
        # Verify database calls
        assert mock_db_client.client.table.call_count >= 2  # Count and messages queries
    
    @pytest.mark.asyncio
    async def test_handle_summary_generation_below_threshold(self, task_manager, mock_db_client):
        """Test summary generation when below threshold."""
        # Mock message count query - below threshold
        count_result = Mock()
        count_result.count = 3  # Below threshold of 5
        
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value = count_result
        
        task = BackgroundTask(
            task_type="summary_generation",
            data={"max_messages": 10, "summary_threshold": 5},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Execute - should return early without generating summary
        await task_manager._handle_summary_generation(task)
        
        # Should only call count query, not insert
        mock_db_client.client.table.return_value.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_context_loading(self, task_manager):
        """Test context loading handling."""
        task = BackgroundTask(
            task_type="context_loading",
            data={"load_user_context": True},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Should complete without error (placeholder implementation)
        await task_manager._handle_context_loading(task)
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, task_manager, mock_db_client):
        """Test successful task execution flow."""
        # Setup mock for message storage
        mock_result = Mock()
        mock_result.data = [{"id": "msg_123"}]
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value = mock_result
        
        task = BackgroundTask(
            task_type="message_storage",
            data={"messages": [{"role": "user", "content": "Hello"}]},
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Execute
        await task_manager._execute_task(task)
        
        # Verify task completion
        assert task.status == TaskStatus.COMPLETED
        assert task.started_at is not None
        assert task.completed_at is not None
        assert task in task_manager.completed_tasks
        assert task.id not in task_manager.active_tasks
        
        # Verify health stats update
        assert task_manager.health_stats['total_tasks_processed'] == 1
        assert task_manager.health_stats['successful_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_failure_with_retry(self, task_manager):
        """Test task execution failure with retry logic."""
        task = BackgroundTask(
            task_type="unknown_task",  # Will cause failure
            max_retries=2
        )
        
        # Execute
        await task_manager._execute_task(task)
        
        # Verify retry scheduling
        assert task.status == TaskStatus.RETRYING
        assert task.retry_count == 1
        assert len(task.error_history) == 1
        assert task.last_error is not None
        assert len(task_manager.task_queue[TaskPriority.NORMAL]) == 1  # Rescheduled
    
    @pytest.mark.asyncio
    async def test_execute_task_max_retries_exceeded(self, task_manager):
        """Test task execution with max retries exceeded."""
        task = BackgroundTask(
            task_type="unknown_task",
            max_retries=1,
            retry_count=1  # Already at max retries
        )
        
        # Execute
        await task_manager._execute_task(task)
        
        # Verify permanent failure
        assert task.status == TaskStatus.FAILED
        assert task in task_manager.failed_tasks
        assert task_manager.health_stats['failed_tasks'] == 1
    
    def test_generate_simple_summary(self, task_manager):
        """Test simple summary generation."""
        messages = [
            {"role": "user", "content": "Hello", "created_at": "2024-01-01T10:00:00Z"},
            {"role": "assistant", "content": "Hi!", "created_at": "2024-01-01T10:01:00Z"},
            {"role": "user", "content": "How are you?", "created_at": "2024-01-01T10:02:00Z"}
        ]
        
        summary = task_manager._generate_simple_summary(messages)
        
        assert "3 messages" in summary
        assert "2 user" in summary
        assert "1 assistant" in summary
        assert "2024-01-01" in summary
    
    def test_generate_simple_summary_empty(self, task_manager):
        """Test simple summary generation with empty messages."""
        summary = task_manager._generate_simple_summary([])
        assert summary == "Empty conversation"


class TestTaskPriorityHandling:
    """Test task priority and queue management."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a BackgroundTaskManager instance for testing."""
        return BackgroundTaskManager()
    
    def test_priority_queue_ordering(self, task_manager):
        """Test that tasks are queued by priority correctly."""
        # Schedule tasks with different priorities
        low_task = BackgroundTask(task_type="low", priority=TaskPriority.LOW)
        high_task = BackgroundTask(task_type="high", priority=TaskPriority.HIGH)
        critical_task = BackgroundTask(task_type="critical", priority=TaskPriority.CRITICAL)
        normal_task = BackgroundTask(task_type="normal", priority=TaskPriority.NORMAL)
        
        task_manager.schedule_task(low_task)
        task_manager.schedule_task(high_task)
        task_manager.schedule_task(critical_task)
        task_manager.schedule_task(normal_task)
        
        # Verify tasks are in correct queues
        assert len(task_manager.task_queue[TaskPriority.LOW]) == 1
        assert len(task_manager.task_queue[TaskPriority.NORMAL]) == 1
        assert len(task_manager.task_queue[TaskPriority.HIGH]) == 1
        assert len(task_manager.task_queue[TaskPriority.CRITICAL]) == 1
        
        # Verify health stats
        assert task_manager.health_stats['queue_sizes']['LOW'] == 1
        assert task_manager.health_stats['queue_sizes']['NORMAL'] == 1
        assert task_manager.health_stats['queue_sizes']['HIGH'] == 1
        assert task_manager.health_stats['queue_sizes']['CRITICAL'] == 1


class TestConcurrentTaskProcessing:
    """Test concurrent task processing and performance."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a BackgroundTaskManager instance for testing."""
        manager = BackgroundTaskManager()
        manager.max_concurrent_tasks = 3  # Limit for testing
        return manager
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, task_manager):
        """Test that concurrent task limit is respected."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = BackgroundTask(
                task_type="context_loading",  # Quick task type
                data={"load_user_context": True},
                user_id=f"user_{i}",
                conversation_id=f"conv_{i}"
            )
            tasks.append(task)
            task_manager.schedule_task(task)
        
        # Simulate processing loop logic
        ready_tasks = []
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = task_manager.task_queue[priority]
            ready_tasks.extend([
                task for task in queue 
                if task.scheduled_at <= datetime.utcnow() and len(task_manager.active_tasks) < task_manager.max_concurrent_tasks
            ])
        
        # Should only process up to max_concurrent_tasks
        assert len(ready_tasks) <= task_manager.max_concurrent_tasks


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""
    
    @pytest.fixture
    async def initialized_task_manager(self):
        """Create and initialize a task manager for testing."""
        manager = BackgroundTaskManager()
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.EMPLOYEE_MAX_MESSAGES = 12
        mock_settings.CUSTOMER_MAX_MESSAGES = 15
        mock_settings.SUMMARY_THRESHOLD = 10
        
        with patch('backend.core.config.get_settings', return_value=mock_settings):
            await manager._ensure_initialized()
        
        return manager
    
    async def test_customer_conversation_workflow(self, initialized_task_manager):
        """Test complete customer conversation processing workflow."""
        manager = initialized_task_manager
        
        # Step 1: Store customer messages
        messages = [
            {"role": "user", "content": "I need help with my order"},
            {"role": "assistant", "content": "I'd be happy to help you with that!"}
        ]
        
        storage_task_id = manager.schedule_message_storage(
            conversation_id="conv_customer_123",
            user_id="user_789",
            messages=messages,
            priority=TaskPriority.HIGH
        )
        
        # Step 2: Schedule summary generation for customer
        summary_task_id = manager.schedule_summary_generation(
            conversation_id="conv_customer_123",
            user_id="user_789",
            customer_id="customer_456"
        )
        
        # Step 3: Schedule context loading
        context_task_id = manager.schedule_context_loading(
            user_id="user_789",
            conversation_id="conv_customer_123",
            customer_id="customer_456"
        )
        
        # Verify all tasks are scheduled
        assert storage_task_id is not None
        assert summary_task_id is not None
        assert context_task_id is not None
        
        # Verify task types and priorities
        storage_task = manager.get_task_status(storage_task_id)
        summary_task = manager.get_task_status(summary_task_id)
        context_task = manager.get_task_status(context_task_id)
        
        assert storage_task['task_type'] == "message_storage"
        assert storage_task['priority'] == "HIGH"
        assert summary_task['task_type'] == "summary_generation"
        assert context_task['task_type'] == "context_loading"
        assert context_task['priority'] == "HIGH"  # Default for context loading
    
    async def test_employee_conversation_workflow(self, initialized_task_manager):
        """Test complete employee conversation processing workflow."""
        manager = initialized_task_manager
        
        # Schedule tasks for employee user
        storage_task_id = manager.schedule_message_storage(
            conversation_id="conv_employee_456",
            user_id="employee_123",
            messages=[{"role": "user", "content": "Customer inquiry about pricing"}]
        )
        
        summary_task_id = manager.schedule_summary_generation(
            conversation_id="conv_employee_456",
            user_id="employee_123",
            employee_id="employee_123"
        )
        
        # Verify employee-specific configuration
        summary_task = manager.get_task_status(summary_task_id)
        assert summary_task is not None
        
        # Employee should use different max_messages threshold
        summary_queue_task = None
        for task in manager.task_queue[TaskPriority.NORMAL]:
            if task.id == summary_task_id:
                summary_queue_task = task
                break
        
        assert summary_queue_task is not None
        assert summary_queue_task.data["max_messages"] == manager.employee_max_messages
    
    async def test_health_monitoring_workflow(self, initialized_task_manager):
        """Test health monitoring throughout task processing."""
        manager = initialized_task_manager
        
        # Initial health check
        initial_health = manager.get_health_status()
        assert initial_health['total_processed'] == 0
        assert initial_health['success_rate'] == 0
        assert initial_health['active_tasks'] == 0
        
        # Schedule multiple tasks
        for i in range(3):
            manager.schedule_message_storage(
                conversation_id=f"conv_{i}",
                user_id=f"user_{i}",
                messages=[{"role": "user", "content": f"Message {i}"}]
            )
        
        # Check queue sizes
        health_after_scheduling = manager.get_health_status()
        assert health_after_scheduling['queue_sizes']['NORMAL'] == 3
        
        # Verify configuration is included
        config = health_after_scheduling['configuration']
        assert config['employee_max_messages'] == 12
        assert config['customer_max_messages'] == 15
        assert config['summary_threshold'] == 10


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
