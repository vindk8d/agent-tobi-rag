"""
Tests for vehicle-specific document processing pipeline enhancements
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
from langchain_core.documents import Document

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class TestVehicleDocumentProcessing:
    """Test vehicle-specific document processing pipeline"""

    @pytest.fixture
    def mock_db_client(self):
        """Mock database client"""
        with patch('backend.core.database.db_client') as mock:
            mock.client = Mock()
            yield mock

    @pytest.fixture
    def mock_embedder(self):
        """Mock embeddings"""
        with patch('backend.rag.pipeline.OpenAIEmbeddings') as mock:
            mock_instance = Mock()
            mock_instance.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store"""
        with patch('backend.rag.pipeline.SupabaseVectorStore') as mock:
            mock_instance = Mock()
            mock_instance.upsert_embedding = AsyncMock(return_value=str(uuid4()))
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_document_loader(self):
        """Mock document loader"""
        with patch('backend.rag.pipeline.DocumentLoader') as mock:
            # Sample Ford Bronco specification content
            mock_content = """# Ford Bronco Outer Banks with Sasquatch™ Package

## Technical Specifications
- Engine: 2.7L EcoBoost V6
- Power: 330 HP @ 5,500 RPM
- Torque: 415 lb-ft @ 2,300 RPM

## Exterior Features
- Sasquatch™ Package includes 35-inch tires
- Rock rails and bash plates
- LED headlights and fog lamps

## Interior Features
- SYNC® 4A with 12-inch touchscreen
- Leather-trimmed seating surfaces
- Dual-zone automatic climate control"""

            mock.load_from_file.return_value = [
                Document(
                    page_content=mock_content,
                    metadata={
                        'source': 'ford_bronco_specs.md',
                        'document_type': 'vehicle_specification'
                    }
                )
            ]
            yield mock

    @pytest.mark.asyncio
    async def test_process_file_with_vehicle_id(self, mock_db_client, mock_embedder, mock_vector_store, mock_document_loader):
        """Test processing file with vehicle_id parameter"""
        from backend.rag.pipeline import DocumentProcessingPipeline
        
        # Setup mocks
        vehicle_id = str(uuid4())
        data_source_id = str(uuid4())
        chunk_id = str(uuid4())
        
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value.data = [{'id': chunk_id}]
        
        vehicle_info = {
            'brand': 'Ford',
            'model': 'Bronco',
            'year': 2024
        }
        
        metadata = {
            'upload_method': 'vehicle_specification_upload',
            'vehicle_info': vehicle_info,
            'vehicle_id': vehicle_id
        }

        # Test the pipeline
        pipeline = DocumentProcessingPipeline()
        result = await pipeline.process_file(
            file_path='test_ford_bronco.md',
            file_type='markdown',
            data_source_id=data_source_id,
            metadata=metadata,
            vehicle_id=vehicle_id
        )

        # Assertions
        assert result['success'] is True
        assert result['data_source_id'] == data_source_id
        assert result['chunks'] > 0
        assert len(result['stored_chunk_ids']) > 0
        assert len(result['embedding_ids']) > 0

    @pytest.mark.asyncio
    async def test_section_based_chunking(self):
        """Test section-based chunking for vehicle specifications"""
        from backend.rag.document_loader import split_by_sections
        
        # Sample document with sections
        content = """# Ford Bronco Specifications

## Technical Specifications
Engine: 2.7L EcoBoost V6
Power: 330 HP @ 5,500 RPM
Torque: 415 lb-ft @ 2,300 RPM

## Exterior Features
Sasquatch™ Package includes 35-inch tires
Rock rails and bash plates
LED headlights and fog lamps

## Interior Features
SYNC® 4A with 12-inch touchscreen
Leather-trimmed seating surfaces
Dual-zone automatic climate control"""

        doc = Document(
            page_content=content,
            metadata={'source': 'ford_bronco_specs.md'}
        )

        # Test section-based splitting
        chunks = split_by_sections([doc])
        
        # Should have 4 chunks (intro + 3 sections)
        assert len(chunks) >= 3
        
        # Check that sections are properly identified
        section_titles = [chunk.metadata.get('section_title') for chunk in chunks if chunk.metadata.get('section_title')]
        assert 'Technical Specifications' in section_titles
        assert 'Exterior Features' in section_titles
        assert 'Interior Features' in section_titles

    @pytest.mark.asyncio
    async def test_vehicle_context_injection(self, mock_db_client, mock_embedder, mock_vector_store, mock_document_loader):
        """Test that vehicle context is properly injected into chunks"""
        from backend.rag.pipeline import DocumentProcessingPipeline
        
        # Setup
        vehicle_id = str(uuid4())
        data_source_id = str(uuid4())
        chunk_id = str(uuid4())
        
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value.data = [{'id': chunk_id}]
        
        vehicle_info = {
            'brand': 'Ford',
            'model': 'Bronco',
            'year': 2024
        }
        
        metadata = {
            'upload_method': 'vehicle_specification_upload',
            'vehicle_info': vehicle_info,
            'vehicle_id': vehicle_id
        }

        # Test the pipeline
        pipeline = DocumentProcessingPipeline()
        result = await pipeline.process_file(
            file_path='test_ford_bronco.md',
            file_type='markdown',
            data_source_id=data_source_id,
            metadata=metadata,
            vehicle_id=vehicle_id
        )

        # Check that embedder was called with vehicle context
        mock_embedder.embed_documents.assert_called_once()
        embedded_texts = mock_embedder.embed_documents.call_args[0][0]
        
        # Verify vehicle context was prepended
        for text in embedded_texts:
            assert 'Vehicle: Ford Bronco 2024' in text

    @pytest.mark.asyncio
    async def test_store_document_chunk_with_vehicle_id(self, mock_db_client):
        """Test storing document chunks with vehicle_id"""
        from backend.rag.pipeline import DocumentProcessingPipeline
        
        # Setup
        vehicle_id = str(uuid4())
        data_source_id = str(uuid4())
        chunk_id = str(uuid4())
        
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value.data = [{'id': chunk_id}]
        
        chunk = Document(
            page_content="Test vehicle specification content",
            metadata={
                'section_title': 'Technical Specifications',
                'vehicle_id': vehicle_id,
                'vehicle_brand': 'Ford',
                'vehicle_model': 'Bronco',
                'vehicle_year': 2024
            }
        )

        # Test storing chunk
        pipeline = DocumentProcessingPipeline()
        result_id = pipeline._store_document_chunk(
            data_source_id=data_source_id,
            chunk_index=0,
            chunk=chunk,
            metadata={'test': 'metadata'},
            vehicle_id=vehicle_id
        )

        # Verify the call
        assert result_id == chunk_id
        mock_db_client.client.table.assert_called_with("document_chunks")
        
        # Check that vehicle_id was included in the insert data
        insert_call = mock_db_client.client.table.return_value.insert.call_args[0][0]
        assert insert_call['vehicle_id'] == vehicle_id
        assert insert_call['data_source_id'] == data_source_id
        assert insert_call['content'] == "Test vehicle specification content"

    @pytest.mark.asyncio
    async def test_chunking_method_selection(self):
        """Test that correct chunking method is selected based on metadata"""
        from backend.rag.document_loader import split_documents
        
        # Test document
        doc = Document(
            page_content="## Section 1\nContent 1\n\n## Section 2\nContent 2",
            metadata={'source': 'test.md'}
        )

        # Test section-based chunking
        section_chunks = await split_documents([doc], chunking_method="section_based")
        assert len(section_chunks) >= 2
        assert any('Section 1' in chunk.page_content for chunk in section_chunks)
        assert any('Section 2' in chunk.page_content for chunk in section_chunks)

        # Test recursive chunking
        recursive_chunks = await split_documents([doc], chunking_method="recursive")
        # Should have different chunking behavior
        assert len(recursive_chunks) >= 1

    @pytest.mark.asyncio
    async def test_process_file_without_vehicle_id(self, mock_db_client, mock_embedder, mock_vector_store, mock_document_loader):
        """Test processing file without vehicle_id (regular document)"""
        from backend.rag.pipeline import DocumentProcessingPipeline
        
        # Setup
        data_source_id = str(uuid4())
        chunk_id = str(uuid4())
        
        mock_db_client.client.table.return_value.insert.return_value.execute.return_value.data = [{'id': chunk_id}]

        # Test the pipeline without vehicle_id
        pipeline = DocumentProcessingPipeline()
        result = await pipeline.process_file(
            file_path='test_regular_doc.md',
            file_type='markdown',
            data_source_id=data_source_id,
            metadata={'upload_method': 'regular_upload'}
        )

        # Should still work but without vehicle context
        assert result['success'] is True
        assert result['data_source_id'] == data_source_id
        
        # Check that embedder was called without vehicle context
        mock_embedder.embed_documents.assert_called_once()
        embedded_texts = mock_embedder.embed_documents.call_args[0][0]
        
        # Verify no vehicle context was prepended
        for text in embedded_texts:
            assert 'Vehicle:' not in text


class TestSectionBasedChunking:
    """Test section-based chunking functionality"""

    def test_split_by_sections_basic(self):
        """Test basic section splitting"""
        from backend.rag.document_loader import split_by_sections
        
        content = """# Main Title

## Section 1
Content for section 1

## Section 2  
Content for section 2

## Section 3
Content for section 3"""

        doc = Document(page_content=content, metadata={'source': 'test.md'})
        chunks = split_by_sections([doc])
        
        assert len(chunks) == 4  # Title + 3 sections
        
        # Check section titles are extracted
        section_chunks = [c for c in chunks if c.metadata.get('section_title')]
        assert len(section_chunks) == 3
        
        section_titles = [c.metadata['section_title'] for c in section_chunks]
        assert 'Section 1' in section_titles
        assert 'Section 2' in section_titles
        assert 'Section 3' in section_titles

    def test_split_by_sections_empty_sections(self):
        """Test handling of empty sections"""
        from backend.rag.document_loader import split_by_sections
        
        content = """## Section 1
Content here

## Empty Section

## Section 3
More content"""

        doc = Document(page_content=content, metadata={'source': 'test.md'})
        chunks = split_by_sections([doc])
        
        # Should filter out empty sections
        non_empty_chunks = [c for c in chunks if c.page_content.strip()]
        assert len(non_empty_chunks) >= 2

    def test_split_by_sections_custom_header(self):
        """Test section splitting with custom header"""
        from backend.rag.document_loader import split_by_sections
        
        content = """### Custom Section 1
Content 1

### Custom Section 2
Content 2"""

        doc = Document(page_content=content, metadata={'source': 'test.md'})
        chunks = split_by_sections([doc], section_header="###")
        
        assert len(chunks) >= 2
        section_titles = [c.metadata.get('section_title') for c in chunks if c.metadata.get('section_title')]
        assert 'Custom Section 1' in section_titles
        assert 'Custom Section 2' in section_titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
