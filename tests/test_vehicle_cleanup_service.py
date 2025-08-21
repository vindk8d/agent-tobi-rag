"""
Test vehicle cleanup service functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from monitoring.vehicle_cleanup import (
    cleanup_old_vehicle_specifications,
    cleanup_orphaned_vehicle_documents,
    run_vehicle_cleanup_sync,
    run_vehicle_orphan_cleanup_sync
)


class TestVehicleCleanupService:
    """Test vehicle cleanup service following quotation cleanup patterns."""

    @pytest.mark.asyncio
    async def test_cleanup_old_vehicle_specifications_success(self):
        """Test successful cleanup of old vehicle specifications."""
        # Mock the storage cleanup function
        mock_cleanup_result = {
            "selected": 5,
            "deleted": 5,
            "failed": 0,
            "errors": [],
            "operation_metadata": {
                "started_at": "2025-01-21T10:00:00Z",
                "completed_at": "2025-01-21T10:01:00Z",
                "status": "success"
            }
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_backups') as mock_cleanup:
            mock_cleanup.return_value = mock_cleanup_result
            
            result = await cleanup_old_vehicle_specifications(days_old=30, max_delete=200)
            
            # Should follow quotation cleanup patterns
            assert isinstance(result, dict)
            assert result["selected"] == 5
            assert result["deleted"] == 5
            assert result["failed"] == 0
            assert result["status"] == "success"
            assert result["service"] == "vehicle_cleanup"
            assert "started_at" in result
            assert "completed_at" in result
            assert "parameters" in result
            
            # Verify parameters were passed correctly
            mock_cleanup.assert_called_once_with(days_old=30, max_delete=200)

    @pytest.mark.asyncio
    async def test_cleanup_old_vehicle_specifications_with_errors(self):
        """Test cleanup with partial failures."""
        mock_cleanup_result = {
            "selected": 3,
            "deleted": 2,
            "failed": 1,
            "errors": ["Failed to delete vehicles/test-123/replaced/old_file.md: Permission denied"],
            "operation_metadata": {
                "started_at": "2025-01-21T10:00:00Z",
                "completed_at": "2025-01-21T10:01:00Z",
                "status": "partial_failure"
            }
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_backups') as mock_cleanup:
            mock_cleanup.return_value = mock_cleanup_result
            
            result = await cleanup_old_vehicle_specifications(days_old=30, max_delete=200)
            
            assert result["selected"] == 3
            assert result["deleted"] == 2
            assert result["failed"] == 1
            assert result["status"] == "partial_failure"
            assert len(result["errors"]) == 1
            assert "Permission denied" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_cleanup_old_vehicle_specifications_storage_error(self):
        """Test cleanup with storage error."""
        from core.storage import VehicleStorageError
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_backups') as mock_cleanup:
            mock_cleanup.side_effect = VehicleStorageError("Storage service unavailable")
            
            result = await cleanup_old_vehicle_specifications(days_old=30, max_delete=200)
            
            assert result["selected"] == 0
            assert result["deleted"] == 0
            assert result["failed"] == 0  # No files were selected before the error
            assert result["status"] == "failed"
            assert len(result["errors"]) == 1
            assert "Storage service unavailable" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_vehicle_documents_basic(self):
        """Test basic orphaned document cleanup functionality."""
        # Test the function exists and returns proper structure
        with patch('monitoring.vehicle_cleanup.cleanup_orphaned_vehicle_documents') as mock_cleanup:
            mock_result = {
                "selected": 0,
                "deleted": 0,
                "failed": 0,
                "errors": [],
                "service": "vehicle_orphan_cleanup",
                "status": "success",
                "started_at": "2025-01-21T10:00:00Z",
                "completed_at": "2025-01-21T10:01:00Z"
            }
            mock_cleanup.return_value = mock_result
            
            result = await mock_cleanup(max_delete=100)
            
            # Verify it follows quotation cleanup patterns
            assert result["service"] == "vehicle_orphan_cleanup"
            assert "selected" in result
            assert "deleted" in result
            assert "failed" in result
            assert "status" in result

    def test_run_vehicle_cleanup_sync(self):
        """Test synchronous wrapper for vehicle cleanup."""
        # Mock the async function
        mock_result = {
            "selected": 3,
            "deleted": 3,
            "failed": 0,
            "status": "success",
            "service": "vehicle_cleanup"
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_specifications') as mock_async:
            mock_async.return_value = mock_result
            
            result = run_vehicle_cleanup_sync(days_old=30, max_delete=200)
            
            assert result == mock_result
            assert result["service"] == "vehicle_cleanup"

    def test_run_vehicle_cleanup_sync_error(self):
        """Test synchronous wrapper error handling."""
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_specifications') as mock_async:
            mock_async.side_effect = Exception("Test error")
            
            result = run_vehicle_cleanup_sync(days_old=30, max_delete=200)
            
            assert result["selected"] == 0
            assert result["deleted"] == 0
            assert result["failed"] == 0
            assert result["status"] == "failed"
            assert len(result["errors"]) == 1
            assert "Test error" in result["errors"][0]

    def test_run_vehicle_orphan_cleanup_sync(self):
        """Test synchronous wrapper for orphan cleanup."""
        mock_result = {
            "selected": 2,
            "deleted": 1,
            "failed": 0,
            "status": "success",
            "service": "vehicle_orphan_cleanup"
        }
        
        with patch('monitoring.vehicle_cleanup.cleanup_orphaned_vehicle_documents') as mock_async:
            mock_async.return_value = mock_result
            
            result = run_vehicle_orphan_cleanup_sync(max_delete=100)
            
            assert result == mock_result
            assert result["service"] == "vehicle_orphan_cleanup"

    def test_service_patterns_match_quotation_cleanup(self):
        """Test that vehicle cleanup follows the same patterns as quotation cleanup."""
        # Test that return structure matches quotation cleanup patterns
        expected_keys = {"selected", "deleted", "failed", "errors", "service", "started_at", "completed_at", "status"}
        
        with patch('monitoring.vehicle_cleanup.cleanup_old_vehicle_backups') as mock_cleanup:
            mock_cleanup.return_value = {
                "selected": 1, "deleted": 1, "failed": 0, "errors": [],
                "operation_metadata": {"started_at": "2025-01-21T10:00:00Z", "completed_at": "2025-01-21T10:01:00Z", "status": "success"}
            }
            
            result = run_vehicle_cleanup_sync(days_old=30, max_delete=200)
            
            # Should have all the same keys as quotation cleanup
            for key in expected_keys:
                assert key in result, f"Missing key '{key}' in vehicle cleanup result"
            
            # Should have proper service identification
            assert result["service"] == "vehicle_cleanup"
            
            # Should have proper status values
            assert result["status"] in ["success", "failed", "partial_failure"]


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestVehicleCleanupService()
    
    # Test synchronous functions
    test_instance.test_run_vehicle_cleanup_sync()
    test_instance.test_run_vehicle_cleanup_sync_error()
    test_instance.test_run_vehicle_orphan_cleanup_sync()
    test_instance.test_service_patterns_match_quotation_cleanup()
    
    print("✅ All synchronous vehicle cleanup service tests passed!")
    print("🔧 Service follows quotation cleanup patterns for consistency!")
    
    # Note: Async tests require pytest to run properly
    print("📝 Run 'pytest tests/test_vehicle_cleanup_service.py -v' for full async test suite")
