"""
Tests for vehicle specification document API endpoints
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import json
from uuid import uuid4

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

class TestVehicleSpecificationAPI:
    """Test vehicle specification API endpoints using direct function calls"""

    @pytest.fixture
    def mock_db_client(self):
        """Mock database client"""
        with patch('backend.api.documents.db_client') as mock:
            mock.client = Mock()
            yield mock

    @pytest.fixture 
    def mock_pipeline(self):
        """Mock document processing pipeline"""
        with patch('backend.api.documents.pipeline') as mock:
            mock.process_file = AsyncMock(return_value={'success': True, 'chunks': 5})
            yield mock

    @pytest.mark.asyncio
    async def test_list_vehicles_success(self, mock_db_client):
        """Test successful vehicle listing"""
        from backend.api.documents import list_vehicles
        
        # Mock database response
        mock_vehicles = [
            {
                'id': str(uuid4()),
                'brand': 'Toyota',
                'model': 'Camry',
                'year': 2024,
                'type': 'Sedan',
                'variant': 'LE',
                'key_features': ['Hybrid', 'Safety Sense'],
                'is_available': True
            },
            {
                'id': str(uuid4()),
                'brand': 'Honda', 
                'model': 'Civic',
                'year': 2024,
                'type': 'Sedan',
                'variant': 'Sport',
                'key_features': ['Turbo', 'Manual'],
                'is_available': True
            }
        ]
        
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = mock_vehicles

        response = await list_vehicles()
        
        assert response.success is True
        assert len(response.data['vehicles']) == 2
        assert response.data['total_count'] == 2
        assert response.data['vehicles'][0]['brand'] == 'Toyota'

    @pytest.mark.asyncio
    async def test_list_vehicles_empty(self, mock_db_client):
        """Test vehicle listing when no vehicles available"""
        from backend.api.documents import list_vehicles
        
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = []

        response = await list_vehicles()
        
        assert response.success is True
        assert len(response.data['vehicles']) == 0
        assert response.data['total_count'] == 0

    @pytest.mark.asyncio
    async def test_get_vehicle_specification_exists(self, mock_db_client):
        """Test checking for existing vehicle specification"""
        from backend.api.documents import get_vehicle_specification
        
        vehicle_id = str(uuid4())
        mock_document = {
            'id': str(uuid4()),
            'original_filename': 'toyota_camry_specs.md',
            'file_size': 15000,
            'created_at': '2024-01-20T10:00:00Z',
            'updated_at': '2024-01-20T10:00:00Z'
        }
        
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [mock_document]

        response = await get_vehicle_specification(vehicle_id)
        
        assert response.success is True
        assert response.data['exists'] is True
        assert response.data['document']['original_filename'] == 'toyota_camry_specs.md'

    @pytest.mark.asyncio
    async def test_get_vehicle_specification_not_exists(self, mock_db_client):
        """Test checking for non-existent vehicle specification"""
        from backend.api.documents import get_vehicle_specification
        
        vehicle_id = str(uuid4())
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        response = await get_vehicle_specification(vehicle_id)
        
        assert response.success is True
        assert response.data['exists'] is False
        assert response.data['document'] is None

    @pytest.mark.asyncio
    async def test_upload_vehicle_specification_new(self, mock_db_client, mock_pipeline):
        """Test uploading new vehicle specification"""
        from backend.api.documents import upload_vehicle_specification, VehicleSpecificationRequest
        from fastapi import BackgroundTasks
        
        vehicle_id = str(uuid4())
        data_source_id = str(uuid4())
        
        # Mock vehicle exists
        mock_vehicle = {
            'id': vehicle_id,
            'brand': 'Ford',
            'model': 'Bronco', 
            'year': 2024
        }
        
        # Setup mock responses in sequence
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = [
            Mock(data=[mock_vehicle]),  # Vehicle exists check
            Mock(data=[])  # No existing specification
        ]
        
        # Mock data source creation
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value.data = [{'id': data_source_id}]

        request = VehicleSpecificationRequest(
            file_path='vehicles/ford_bronco_specs.md',
            file_name='ford_bronco_specs.md',
            file_size=25000,
            vehicle_id=vehicle_id
        )

        background_tasks = BackgroundTasks()
        response = await upload_vehicle_specification(vehicle_id, request, background_tasks)
        
        assert response.success is True
        assert response.data['vehicle_id'] == vehicle_id
        assert response.data['status'] == 'processing'

    @pytest.mark.asyncio
    async def test_upload_vehicle_specification_vehicle_not_found(self, mock_db_client):
        """Test uploading specification for non-existent vehicle"""
        from backend.api.documents import upload_vehicle_specification, VehicleSpecificationRequest
        from fastapi import BackgroundTasks, HTTPException
        
        vehicle_id = str(uuid4())
        
        # Mock vehicle not found
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        request = VehicleSpecificationRequest(
            file_path='vehicles/nonexistent_vehicle_specs.md',
            file_name='nonexistent_vehicle_specs.md',
            file_size=15000,
            vehicle_id=vehicle_id
        )

        background_tasks = BackgroundTasks()
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_vehicle_specification(vehicle_id, request, background_tasks)
        
        assert exc_info.value.status_code == 404
        assert "Vehicle not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_vehicle_specification_success(self, mock_db_client):
        """Test successful deletion of vehicle specification"""
        from backend.api.documents import delete_vehicle_specification
        
        vehicle_id = str(uuid4())
        
        # Mock existing document
        mock_document = {
            'id': str(uuid4()),
            'storage_path': 'vehicles/ford_bronco_specs.md'
        }
        
        # Mock existing chunks
        mock_chunks = [{'id': str(uuid4())}, {'id': str(uuid4())}]
        
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.side_effect = [
            Mock(data=[mock_document]),  # Document exists
            Mock(data=mock_chunks)  # Chunks exist
        ]

        response = await delete_vehicle_specification(vehicle_id)
        
        assert response.success is True
        assert "deleted successfully" in response.data['message']

    @pytest.mark.asyncio
    async def test_delete_vehicle_specification_not_found(self, mock_db_client):
        """Test deletion of non-existent vehicle specification"""
        from backend.api.documents import delete_vehicle_specification
        from fastapi import HTTPException
        
        vehicle_id = str(uuid4())
        
        # Mock no existing document
        mock_db_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        with pytest.raises(HTTPException) as exc_info:
            await delete_vehicle_specification(vehicle_id)
        
        assert exc_info.value.status_code == 404
        assert "Vehicle specification not found" in str(exc_info.value.detail)


class TestDocumentTypeMapping:
    """Test document type detection for vehicle specifications"""

    @pytest.mark.parametrize("filename,expected_type", [
        ("ford_bronco_specs.md", "markdown"),
        ("toyota_camry_manual.pdf", "pdf"),
        ("honda_civic_features.docx", "word"),
        ("nissan_altima_specs.txt", "text"),
        ("hyundai_elantra_info.html", "html"),
        ("vehicle_specs", "text")  # No extension defaults to text
    ])
    def test_document_type_detection(self, filename, expected_type):
        """Test that file extensions are correctly mapped to document types"""
        from backend.models.document import DocumentType
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        document_type_mapping = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.WORD,
            'doc': DocumentType.WORD,
            'txt': DocumentType.TEXT,
            'md': DocumentType.MARKDOWN,
            'html': DocumentType.HTML
        }
        
        document_type = document_type_mapping.get(file_ext, DocumentType.TEXT)
        assert document_type.value == expected_type


class TestVehicleSpecificationModels:
    """Test vehicle specification Pydantic models"""

    def test_vehicle_specification_request_valid(self):
        """Test valid vehicle specification request model"""
        from backend.models.document import VehicleSpecificationRequest
        
        vehicle_id = str(uuid4())
        request = VehicleSpecificationRequest(
            file_path='vehicles/test_spec.md',
            file_name='test_spec.md',
            file_size=15000,
            vehicle_id=vehicle_id
        )
        
        assert request.file_path == 'vehicles/test_spec.md'
        assert request.file_name == 'test_spec.md'
        assert request.file_size == 15000
        assert str(request.vehicle_id) == vehicle_id

    def test_vehicle_model_valid(self):
        """Test valid vehicle model"""
        from backend.models.document import VehicleModel
        
        vehicle_id = str(uuid4())
        vehicle = VehicleModel(
            id=vehicle_id,
            brand='Toyota',
            model='Camry',
            year=2024,
            type='Sedan',
            variant='LE',
            key_features=['Hybrid', 'Safety Sense'],
            is_available=True
        )
        
        assert str(vehicle.id) == vehicle_id
        assert vehicle.brand == 'Toyota'
        assert vehicle.model == 'Camry'
        assert vehicle.year == 2024
        assert vehicle.is_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
