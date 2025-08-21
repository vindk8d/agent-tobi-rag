"""
Test vehicle cleanup integration with existing background task patterns.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.background_tasks import BackgroundTaskManager, TaskPriority, TaskStatus


class TestVehicleCleanupIntegration:
    """Test vehicle cleanup integration with existing background task patterns."""

    @pytest.mark.asyncio
    async def test_schedule_vehicle_cleanup_task(self):
        """Test scheduling vehicle cleanup tasks through BackgroundTaskManager."""
        # Create a fresh instance for testing
        task_manager = BackgroundTaskManager()
        
        # Schedule a backup cleanup task
        task_id = task_manager.schedule_vehicle_cleanup_task(
            cleanup_type="backup",
            days_old=30,
            max_delete=200
        )
        
        # Verify task was created and queued
        assert task_id is not None
        assert len(task_manager.task_queue[TaskPriority.LOW]) == 1
        
        # Verify task data
        queued_task = task_manager.task_queue[TaskPriority.LOW][0]
        assert queued_task.task_type == "vehicle_cleanup"
        assert queued_task.priority == TaskPriority.LOW
        assert queued_task.data["cleanup_type"] == "backup"
        assert queued_task.data["days_old"] == 30
        assert queued_task.data["max_delete"] == 200

    @pytest.mark.asyncio
    async def test_schedule_vehicle_orphan_cleanup_task(self):
        """Test scheduling vehicle orphan cleanup tasks."""
        task_manager = BackgroundTaskManager()
        
        # Schedule an orphan cleanup task
        task_id = task_manager.schedule_vehicle_cleanup_task(
            cleanup_type="orphan",
            max_delete=100
        )
        
        # Verify task was created
        assert task_id is not None
        queued_task = task_manager.task_queue[TaskPriority.LOW][0]
        assert queued_task.data["cleanup_type"] == "orphan"
        assert queued_task.data["max_delete"] == 100

    @pytest.mark.asyncio
    async def test_handle_vehicle_backup_cleanup(self):
        """Test handling vehicle backup cleanup tasks."""
        task_manager = BackgroundTaskManager()
        
        # Create a mock task
        from agents.background_tasks import BackgroundTask
        task = BackgroundTask(
            task_type="vehicle_cleanup",
            priority=TaskPriority.LOW,
            data={
                "cleanup_type": "backup",
                "days_old": 30,
                "max_delete": 200
            }
        )
        
        # Mock the cleanup function
        mock_result = {
            "selected": 5,
            "deleted": 5,
            "failed": 0,
            "errors": [],
            "status": "success"
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_specifications') as mock_cleanup:
            mock_cleanup.return_value = mock_result
            
            result = await task_manager._handle_vehicle_cleanup(task)
            
            # Verify cleanup function was called with correct parameters
            mock_cleanup.assert_called_once_with(days_old=30, max_delete=200)
            
            # Verify result
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_handle_vehicle_orphan_cleanup(self):
        """Test handling vehicle orphan cleanup tasks."""
        task_manager = BackgroundTaskManager()
        
        from agents.background_tasks import BackgroundTask
        task = BackgroundTask(
            task_type="vehicle_cleanup",
            priority=TaskPriority.LOW,
            data={
                "cleanup_type": "orphan",
                "max_delete": 100
            }
        )
        
        mock_result = {
            "selected": 2,
            "deleted": 1,
            "failed": 0,
            "errors": [],
            "status": "success"
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_orphaned_vehicle_documents') as mock_cleanup:
            mock_cleanup.return_value = mock_result
            
            result = await task_manager._handle_vehicle_cleanup(task)
            
            mock_cleanup.assert_called_once_with(max_delete=100)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_handle_vehicle_cleanup_error(self):
        """Test error handling in vehicle cleanup tasks."""
        task_manager = BackgroundTaskManager()
        
        from agents.background_tasks import BackgroundTask
        task = BackgroundTask(
            task_type="vehicle_cleanup",
            priority=TaskPriority.LOW,
            data={
                "cleanup_type": "invalid_type"
            }
        )
        
        # Should raise ValueError for invalid cleanup type
        with pytest.raises(Exception) as exc_info:
            await task_manager._handle_vehicle_cleanup(task)
        
        assert "Unknown cleanup type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_task_execution_routing(self):
        """Test that vehicle cleanup tasks are properly routed in _execute_task."""
        task_manager = BackgroundTaskManager()
        
        from agents.background_tasks import BackgroundTask
        task = BackgroundTask(
            task_type="vehicle_cleanup",
            priority=TaskPriority.LOW,
            data={
                "cleanup_type": "backup",
                "days_old": 30,
                "max_delete": 200
            }
        )
        
        # Mock the handler method
        with patch.object(task_manager, '_handle_vehicle_cleanup') as mock_handler:
            mock_handler.return_value = {"status": "success"}
            
            # Execute the task
            await task_manager._execute_task(task)
            
            # Verify handler was called
            mock_handler.assert_called_once_with(task)
            
            # Verify task status
            assert task.status == TaskStatus.COMPLETED
            assert task.completed_at is not None

    def test_integration_with_existing_patterns(self):
        """Test that vehicle cleanup follows existing background task patterns."""
        task_manager = BackgroundTaskManager()
        
        # Schedule different types of tasks
        cleanup_task_id = task_manager.schedule_vehicle_cleanup_task("backup", 30, 200)
        
        # For comparison, create a summary task (which uses NORMAL priority)
        summary_task_id = task_manager.schedule_summary_generation("conv1", "user1", 10)
        
        # Verify both tasks are in appropriate queues
        assert len(task_manager.task_queue[TaskPriority.NORMAL]) == 1  # Summary generation
        assert len(task_manager.task_queue[TaskPriority.LOW]) == 1     # Vehicle cleanup
        
        # Verify task types
        summary_task = task_manager.task_queue[TaskPriority.NORMAL][0]
        cleanup_task = task_manager.task_queue[TaskPriority.LOW][0]
        
        assert summary_task.task_type == "summary_generation"
        assert cleanup_task.task_type == "vehicle_cleanup"
        
        # Verify priority ordering (cleanup should be lower priority, so smaller numeric value)
        assert cleanup_task.priority.value < summary_task.priority.value

    def test_scheduler_integration_patterns(self):
        """Test that scheduler integration follows existing patterns."""
        # Test that DataSourceScheduler includes vehicle cleanup jobs
        from monitoring.scheduler import DataSourceScheduler
        
        scheduler = DataSourceScheduler()
        
        # Verify scheduler has the expected methods
        assert hasattr(scheduler, 'run_vehicle_cleanup')
        assert hasattr(scheduler, 'run_vehicle_orphan_cleanup')
        assert hasattr(scheduler, 'start')
        assert hasattr(scheduler, 'stop')
        
        # Verify it follows the same pattern as quotation cleanup
        assert hasattr(scheduler, 'run_quotation_cleanup')


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestVehicleCleanupIntegration()
    
    # Test synchronous functions
    test_instance.test_integration_with_existing_patterns()
    test_instance.test_scheduler_integration_patterns()
    
    print("✅ All synchronous vehicle cleanup integration tests passed!")
    print("🔧 Integration follows existing background task patterns!")
    
    # Note: Async tests require pytest to run properly
    print("📝 Run 'pytest tests/test_vehicle_cleanup_integration.py -v' for full async test suite")
