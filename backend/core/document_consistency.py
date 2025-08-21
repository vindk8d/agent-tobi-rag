"""
Application-level document consistency management.
Alternative to database triggers for maintaining vehicle_id consistency.
"""

from typing import Optional, List
from core.database import db_client
import logging

logger = logging.getLogger(__name__)


class DocumentConsistencyManager:
    """Manages consistency between documents and document_chunks at application level."""
    
    @staticmethod
    async def sync_document_vehicle_id(document_id: str, new_vehicle_id: Optional[str]) -> int:
        """
        Update all chunks of a document to have the same vehicle_id.
        
        Args:
            document_id: The document ID
            new_vehicle_id: The new vehicle_id to set (can be None)
            
        Returns:
            Number of chunks updated
        """
        try:
            # Update all chunks for this document
            result = db_client.client.table('document_chunks').update({
                'vehicle_id': new_vehicle_id
            }).eq('document_id', document_id).execute()
            
            updated_count = len(result.data) if result.data else 0
            logger.info(f"Updated {updated_count} chunks for document {document_id} with vehicle_id {new_vehicle_id}")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to sync vehicle_id for document {document_id}: {e}")
            raise
    
    @staticmethod
    async def ensure_chunk_vehicle_id(chunk_data: dict) -> dict:
        """
        Ensure a chunk has the correct vehicle_id based on its parent document.
        
        Args:
            chunk_data: The chunk data dictionary
            
        Returns:
            Updated chunk data with correct vehicle_id
        """
        document_id = chunk_data.get('document_id')
        if not document_id:
            return chunk_data
        
        try:
            # Get vehicle_id from parent document
            doc_result = db_client.client.table('documents').select('vehicle_id').eq('id', document_id).execute()
            
            if doc_result.data:
                parent_vehicle_id = doc_result.data[0]['vehicle_id']
                chunk_data['vehicle_id'] = parent_vehicle_id
                logger.debug(f"Chunk inherited vehicle_id {parent_vehicle_id} from document {document_id}")
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to inherit vehicle_id for chunk from document {document_id}: {e}")
            return chunk_data
    
    @staticmethod
    async def validate_consistency(document_id: str) -> dict:
        """
        Validate that all chunks of a document have consistent vehicle_id.
        
        Args:
            document_id: The document ID to validate
            
        Returns:
            Validation report dictionary
        """
        try:
            # Get document vehicle_id
            doc_result = db_client.client.table('documents').select('vehicle_id').eq('id', document_id).execute()
            if not doc_result.data:
                return {'valid': False, 'error': 'Document not found'}
            
            document_vehicle_id = doc_result.data[0]['vehicle_id']
            
            # Get all chunks for this document
            chunks_result = db_client.client.table('document_chunks').select('id, vehicle_id').eq('document_id', document_id).execute()
            
            chunks = chunks_result.data or []
            total_chunks = len(chunks)
            consistent_chunks = sum(1 for chunk in chunks if chunk['vehicle_id'] == document_vehicle_id)
            
            is_valid = consistent_chunks == total_chunks
            
            return {
                'valid': is_valid,
                'document_id': document_id,
                'document_vehicle_id': document_vehicle_id,
                'total_chunks': total_chunks,
                'consistent_chunks': consistent_chunks,
                'inconsistent_chunks': total_chunks - consistent_chunks
            }
            
        except Exception as e:
            logger.error(f"Failed to validate consistency for document {document_id}: {e}")
            return {'valid': False, 'error': str(e)}


# Convenience functions for common operations
async def update_document_vehicle_id(document_id: str, new_vehicle_id: Optional[str]) -> dict:
    """
    Update a document's vehicle_id and sync all its chunks.
    
    Returns:
        Operation result dictionary
    """
    try:
        # Update the document
        doc_result = db_client.client.table('documents').update({
            'vehicle_id': new_vehicle_id
        }).eq('id', document_id).execute()
        
        if not doc_result.data:
            return {'success': False, 'error': 'Document not found'}
        
        # Sync all chunks
        chunks_updated = await DocumentConsistencyManager.sync_document_vehicle_id(document_id, new_vehicle_id)
        
        return {
            'success': True,
            'document_id': document_id,
            'new_vehicle_id': new_vehicle_id,
            'chunks_updated': chunks_updated
        }
        
    except Exception as e:
        logger.error(f"Failed to update document vehicle_id: {e}")
        return {'success': False, 'error': str(e)}
