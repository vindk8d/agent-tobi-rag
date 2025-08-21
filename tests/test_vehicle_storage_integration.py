"""
Test vehicle storage integration with organized folder structure.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.storage import (
    generate_vehicle_storage_path,
    upload_vehicle_specification,
    move_vehicle_specification_to_backup,
    extract_vehicle_storage_path_from_document,
    cleanup_old_vehicle_backups,
    VehicleStorageError
)


class TestVehicleStorageIntegration:
    """Test vehicle storage functions and folder organization."""

    def test_generate_vehicle_storage_path_current(self):
        """Test generating storage path for current vehicle documents."""
        vehicle_id = "123e4567-e89b-12d3-a456-426614174000"
        filename = "ford_bronco_specs.md"
        
        result = generate_vehicle_storage_path(vehicle_id, filename, "current")
        expected = f"vehicles/{vehicle_id}/current/{filename}"
        
        assert result == expected

    def test_generate_vehicle_storage_path_replaced(self):
        """Test generating storage path for replaced vehicle documents."""
        vehicle_id = "123e4567-e89b-12d3-a456-426614174000"
        filename = "ford_bronco_specs.md"
        
        result = generate_vehicle_storage_path(vehicle_id, filename, "replaced")
        expected = f"vehicles/{vehicle_id}/replaced/{filename}"
        
        assert result == expected

    def test_generate_vehicle_storage_path_sanitization(self):
        """Test path sanitization for security."""
        vehicle_id = "../../malicious/path"
        filename = "../../../etc/passwd"
        
        result = generate_vehicle_storage_path(vehicle_id, filename, "current")
        
        # Should sanitize dangerous characters
        assert "../" not in result
        assert "../../" not in result
        assert "/etc/passwd" not in result
        assert result.startswith("vehicles/")

    @pytest.mark.asyncio
    async def test_upload_vehicle_specification_success(self):
        """Test successful vehicle specification upload."""
        vehicle_id = "123e4567-e89b-12d3-a456-426614174000"
        filename = "test_spec.md"
        file_content = b"# Vehicle Specification\nThis is a test document."
        content_type = "text/markdown"
        
        # Mock Supabase client
        mock_storage_result = {"error": None}
        
        with patch('core.storage.db_client') as mock_db:
            mock_db.client.storage.from_.return_value.upload.return_value = mock_storage_result
            
            result = await upload_vehicle_specification(
                file_bytes=file_content,
                vehicle_id=vehicle_id,
                filename=filename,
                content_type=content_type
            )
            
            assert result["bucket"] == "documents"
            assert result["vehicle_id"] == vehicle_id
            assert result["filename"] == filename
            assert result["size"] == len(file_content)
            assert result["content_type"] == content_type
            assert "uploaded_at" in result
            assert result["path"].startswith(f"vehicles/{vehicle_id}/current/")

    @pytest.mark.asyncio
    async def test_upload_vehicle_specification_error(self):
        """Test vehicle specification upload with storage error."""
        vehicle_id = "123e4567-e89b-12d3-a456-426614174000"
        filename = "test_spec.md"
        file_content = b"# Vehicle Specification\nThis is a test document."
        
        # Mock Supabase client with error
        mock_storage_result = {"error": "Storage quota exceeded"}
        
        with patch('core.storage.db_client') as mock_db:
            mock_db.client.storage.from_.return_value.upload.return_value = mock_storage_result
            
            with pytest.raises(VehicleStorageError) as exc_info:
                await upload_vehicle_specification(
                    file_bytes=file_content,
                    vehicle_id=vehicle_id,
                    filename=filename
                )
            
            assert "Storage quota exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_move_vehicle_specification_to_backup_success(self):
        """Test successful backup of vehicle specification following quotation patterns."""
        current_path = "vehicles/123e4567-e89b-12d3-a456-426614174000/current/test_spec.md"
        document_id = "doc-123"
        
        # Mock Supabase client operations
        mock_copy_result = {"error": None}
        mock_delete_result = {"error": None}
        mock_update_result = {"data": [{"id": document_id}]}
        
        with patch('core.storage.db_client') as mock_db:
            mock_storage = mock_db.client.storage.from_.return_value
            mock_storage.copy.return_value = mock_copy_result
            mock_storage.remove.return_value = mock_delete_result
            mock_db.client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_result
            
            result = await move_vehicle_specification_to_backup(current_path, document_id)
            
            # Should return detailed backup information (following quotation patterns)
            assert isinstance(result, dict)
            assert "backup_path" in result
            assert "original_path" in result
            assert "moved_at" in result
            assert "document_id" in result
            assert "operation_metadata" in result
            
            assert result["original_path"] == current_path
            assert result["document_id"] == document_id
            assert result["backup_path"].startswith("vehicles/123e4567-e89b-12d3-a456-426614174000/replaced/")
            assert "test_spec.md" in result["backup_path"]
            
            # Verify copy and delete were called
            mock_storage.copy.assert_called_once()
            mock_storage.remove.assert_called_once_with([current_path])
            
            # Verify database update was called
            mock_db.client.table.assert_called_with('documents')

    @pytest.mark.asyncio
    async def test_move_vehicle_specification_to_backup_copy_error(self):
        """Test backup failure during copy operation."""
        current_path = "vehicles/123e4567-e89b-12d3-a456-426614174000/current/test_spec.md"
        
        # Mock Supabase client with copy error
        mock_copy_result = {"error": "File not found"}
        
        with patch('core.storage.db_client') as mock_db:
            mock_storage = mock_db.client.storage.from_.return_value
            mock_storage.copy.return_value = mock_copy_result
            
            with pytest.raises(VehicleStorageError) as exc_info:
                await move_vehicle_specification_to_backup(current_path)
            
            assert "Backup copy failed" in str(exc_info.value)
            assert "File not found" in str(exc_info.value)

    def test_folder_organization_structure(self):
        """Test that the folder organization follows the expected structure."""
        vehicle_id = "test-vehicle-123"
        filename = "specification.md"
        
        # Test current folder
        current_path = generate_vehicle_storage_path(vehicle_id, filename, "current")
        assert current_path == "vehicles/test-vehicle-123/current/specification.md"
        
        # Test replaced folder
        replaced_path = generate_vehicle_storage_path(vehicle_id, filename, "replaced")
        assert replaced_path == "vehicles/test-vehicle-123/replaced/specification.md"
        
        # Test invalid folder type defaults to current
        default_path = generate_vehicle_storage_path(vehicle_id, filename, "invalid")
        assert default_path == "vehicles/test-vehicle-123/current/specification.md"

    @pytest.mark.asyncio
    async def test_upload_with_invalid_content(self):
        """Test upload with invalid file content."""
        vehicle_id = "123e4567-e89b-12d3-a456-426614174000"
        filename = "test_spec.md"
        
        # Test with empty bytes
        with pytest.raises(VehicleStorageError) as exc_info:
            await upload_vehicle_specification(
                file_bytes=b"",
                vehicle_id=vehicle_id,
                filename=filename
            )
        assert "Invalid file bytes" in str(exc_info.value)
        
        # Test with None
        with pytest.raises(VehicleStorageError) as exc_info:
            await upload_vehicle_specification(
                file_bytes=None,
                vehicle_id=vehicle_id,
                filename=filename
            )
        assert "Invalid file bytes" in str(exc_info.value)

    def test_extract_vehicle_storage_path_from_document(self):
        """Test extracting storage path from document record following quotation patterns."""
        # Test with direct storage_path
        document_with_path = {
            "id": "doc-123",
            "storage_path": "vehicles/test-vehicle/current/spec.md",
            "vehicle_id": "test-vehicle",
            "original_filename": "spec.md"
        }
        
        result = extract_vehicle_storage_path_from_document(document_with_path)
        assert result == "vehicles/test-vehicle/current/spec.md"
        
        # Test with metadata storage_path
        document_with_metadata = {
            "id": "doc-124",
            "vehicle_id": "test-vehicle",
            "original_filename": "spec.md",
            "metadata": {
                "storage_path": "vehicles/test-vehicle/current/spec.md"
            }
        }
        
        result = extract_vehicle_storage_path_from_document(document_with_metadata)
        assert result == "vehicles/test-vehicle/current/spec.md"
        
        # Test fallback generation
        document_fallback = {
            "id": "doc-125",
            "vehicle_id": "test-vehicle",
            "original_filename": "spec.md"
        }
        
        result = extract_vehicle_storage_path_from_document(document_fallback)
        assert result == "vehicles/test-vehicle/current/spec.md"
        
        # Test with None/empty document
        assert extract_vehicle_storage_path_from_document(None) is None
        assert extract_vehicle_storage_path_from_document({}) is None

    @pytest.mark.asyncio
    async def test_cleanup_old_vehicle_backups_success(self):
        """Test successful cleanup of old vehicle backups following quotation patterns."""
        # Mock old files to be deleted
        mock_vehicle_list = [
            {"name": "vehicle-123"},
            {"name": "vehicle-456"}
        ]
        
        mock_old_files = [
            {
                "name": "20240101_120000_old_spec.md",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        ]
        
        with patch('core.storage.db_client') as mock_db:
            mock_storage = mock_db.client.storage.from_.return_value
            mock_storage.list.side_effect = [mock_vehicle_list, mock_old_files, []]  # vehicles, replaced files for vehicle-123, empty for vehicle-456
            mock_storage.remove.return_value = {"error": None}
            
            result = await cleanup_old_vehicle_backups(days_old=30, max_delete=100)
            
            # Should follow quotation cleanup patterns
            assert isinstance(result, dict)
            assert "selected" in result
            assert "deleted" in result
            assert "failed" in result
            assert "operation_metadata" in result
            
            assert result["selected"] == 1  # One old file found
            assert result["deleted"] == 1   # One file deleted
            assert result["failed"] == 0    # No failures
            
            # Verify batch deletion was called (following quotation patterns)
            mock_storage.remove.assert_called_once()
            
            # Verify operation metadata
            metadata = result["operation_metadata"]
            assert "started_at" in metadata
            assert "completed_at" in metadata
            assert metadata["status"] == "success"
            assert metadata["days_old"] == 30
            assert metadata["max_delete"] == 100

    @pytest.mark.asyncio
    async def test_cleanup_old_vehicle_backups_no_files(self):
        """Test cleanup when no old files are found."""
        from datetime import datetime, timedelta
        
        # Create a recent timestamp (5 days ago) that should NOT be deleted
        recent_date = (datetime.now() - timedelta(days=5)).isoformat() + "Z"
        
        mock_vehicle_list = [{"name": "vehicle-123"}]
        mock_recent_files = [
            {
                "name": "20250120_120000_recent_spec.md",
                "updated_at": recent_date  # Recent file (5 days old, within 30 day limit)
            }
        ]
        
        with patch('core.storage.db_client') as mock_db:
            mock_storage = mock_db.client.storage.from_.return_value
            mock_storage.list.side_effect = [mock_vehicle_list, mock_recent_files]
            # Set up remove mock to return success (even though it shouldn't be called)
            mock_storage.remove.return_value = {"error": None}
            
            result = await cleanup_old_vehicle_backups(days_old=30)
            
            assert result["selected"] == 0
            assert result["deleted"] == 0
            assert result["failed"] == 0
            
            # Should not call remove when no files to delete
            mock_storage.remove.assert_not_called()


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestVehicleStorageIntegration()
    
    # Test path generation
    test_instance.test_generate_vehicle_storage_path_current()
    test_instance.test_generate_vehicle_storage_path_replaced()
    test_instance.test_generate_vehicle_storage_path_sanitization()
    test_instance.test_folder_organization_structure()
    
    # Test utility functions
    test_instance.test_extract_vehicle_storage_path_from_document()
    
    print("✅ All synchronous vehicle storage tests passed!")
    print("🔧 Enhanced backup patterns following quotation system reliability!")
    
    # Note: Async tests require pytest to run properly
    print("📝 Run 'pytest tests/test_vehicle_storage_integration.py -v' for full async test suite")
