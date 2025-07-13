"""
Document API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
import asyncio
import logging
import os
import tempfile
import requests
from uuid import uuid4
from pydantic import BaseModel

from models.base import APIResponse
from models.document import (
    DocumentModel, DocumentUploadRequest, DocumentUploadResponse, 
    DocumentStatus, DocumentType
)
from rag.pipeline import DocumentProcessingPipeline
from database import db_client
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize processing pipeline
pipeline = DocumentProcessingPipeline()
settings = get_settings()

class ProcessUploadedRequest(BaseModel):
    """Request model for processing uploaded documents"""
    file_path: str
    file_name: str
    file_size: Optional[int] = None

async def download_from_supabase_storage(storage_path: str) -> str:
    """
    Download a file from Supabase Storage and return the local file path.
    """
    try:
        # Get the public URL for the file
        supabase_client = db_client.client
        
        # Get signed URL for download (valid for 1 hour)
        response = supabase_client.storage.from_('documents').create_signed_url(storage_path, 3600)
        
        if 'signedURL' not in response:
            raise Exception(f"Failed to get signed URL: {response}")
        
        download_url = response['signedURL']
        
        # Download the file to a temporary location
        file_response = requests.get(download_url)
        file_response.raise_for_status()
        
        # Create temporary file with appropriate extension
        file_ext = os.path.splitext(storage_path)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(file_response.content)
        temp_file.close()
        
        logger.info(f"Downloaded {storage_path} to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to download {storage_path}: {e}")
        raise

@router.post("/process-uploaded", response_model=APIResponse)
async def process_uploaded_document(
    request: ProcessUploadedRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a document that was uploaded to Supabase Storage.
    This endpoint is called after a file is uploaded to trigger processing.
    """
    try:
        # Determine document type from file extension
        file_ext = request.file_name.lower().split('.')[-1] if '.' in request.file_name else ''
        
        document_type_mapping = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.WORD,
            'doc': DocumentType.WORD,
            'txt': DocumentType.TEXT,
            'md': DocumentType.MARKDOWN,
            'html': DocumentType.HTML
        }
        
        document_type = document_type_mapping.get(file_ext, DocumentType.TEXT)
        
        # Create document record in database
        document_id = str(uuid4())
        
        # Insert document record with PENDING status using proper columns
        document_data = {
            'id': document_id,
            'title': request.file_name,
            'content': '',  # Empty content since file is stored in Supabase Storage
            'document_type': document_type.value,
            'status': DocumentStatus.PENDING.value,
            'storage_path': request.file_path,
            'original_filename': request.file_name,
            'file_size': request.file_size,
            'metadata': {
                'storage_bucket': 'documents',
                'upload_path': request.file_path  # Keep for legacy compatibility
            }
        }
        
        # Insert into database
        result = db_client.client.table('documents').insert(document_data).execute()
        
        if result.data:
            # Start background processing
            background_tasks.add_task(process_document_background, document_id, request.file_path, document_type.value, request.file_name)
            
            return APIResponse.success_response(
                data={
                    "document_id": document_id,
                    "status": DocumentStatus.PROCESSING.value,
                    "message": "Document processing started"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create document record")
            
    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def get_or_create_uploads_data_source() -> str:
    """
    Get or create a default data source for uploaded files.
    """
    try:
        # Look for existing "Uploaded Files" data source
        result = db_client.client.table('data_sources').select('id').eq('name', 'Uploaded Files').execute()
        
        if result.data:
            return result.data[0]['id']
        
        # Create default data source for uploads
        data_source_data = {
            'name': 'Uploaded Files',
            'source_type': 'document',
            'url': '',  # No URL for uploaded files
            'status': 'active',
            'metadata': {
                'description': 'Default data source for uploaded documents',
                'created_by': 'system',
                'auto_created': True
            }
        }
        
        result = db_client.client.table('data_sources').insert(data_source_data).execute()
        if result.data:
            logger.info(f"Created default 'Uploaded Files' data source: {result.data[0]['id']}")
            return result.data[0]['id']
        else:
            logger.error("Failed to create default uploads data source")
            return ""
            
    except Exception as e:
        logger.error(f"Error getting/creating uploads data source: {e}")
        return ""

async def process_document_background(document_id: str, storage_path: str, file_type: str, file_name: str):
    """
    Background task to process document: download, load, chunk, embed, and store.
    """
    local_file_path = None
    try:
        # Update status to PROCESSING
        db_client.client.table('documents').update({
            'status': DocumentStatus.PROCESSING.value
        }).eq('id', document_id).execute()
        
        logger.info(f"Starting processing for document {document_id}")
        
        # Get or create the uploads data source
        uploads_data_source_id = await get_or_create_uploads_data_source()
        if not uploads_data_source_id:
            raise Exception("Failed to get or create uploads data source")
        
        # Download file from Supabase Storage
        local_file_path = await download_from_supabase_storage(storage_path)
        
        metadata = {
            'original_document_id': document_id,  # Keep reference to the original document record
            'original_filename': file_name,
            'storage_path': storage_path,
            'processed_at': str(asyncio.get_event_loop().time()),
            'upload_method': 'file_upload'
        }
        
        # Process the document using the fixed pipeline
        result = await pipeline.process_file(local_file_path, file_type, uploads_data_source_id, metadata)
        
        if result['success']:
            # Update status to COMPLETED using proper columns
            db_client.client.table('documents').update({
                'status': DocumentStatus.COMPLETED.value,
                'processed_at': 'now()',  # Use PostgreSQL's now() function
                'embedding_count': result.get('chunks', 0),
                'metadata': {
                    'processing_result': result,
                    'document_id': document_id
                }
            }).eq('id', document_id).execute()
            
            logger.info(f"Document {document_id} processed successfully: {result['chunks']} chunks")
        else:
            # Update status to FAILED using proper columns
            db_client.client.table('documents').update({
                'status': DocumentStatus.FAILED.value,
                'processed_at': 'now()',  # Use PostgreSQL's now() function
                'metadata': {
                    'error': result.get('error', 'Processing failed'),
                    'document_id': document_id
                }
            }).eq('id', document_id).execute()
            
            logger.error(f"Document {document_id} processing failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {e}")
        
        # Update status to FAILED
        try:
            db_client.client.table('documents').update({
                'status': DocumentStatus.FAILED.value,
                'processed_at': 'now()',  # Use PostgreSQL's now() function
                'metadata': {
                    'error': str(e),
                    'failed_at': str(asyncio.get_event_loop().time())
                }
            }).eq('id', document_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update document status: {db_error}")
    finally:
        # Clean up temporary file
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
                logger.info(f"Cleaned up temporary file: {local_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {local_file_path}: {cleanup_error}")

@router.get("/", response_model=APIResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 10,
    status: Optional[str] = None
):
    """
    List documents with pagination and optional status filtering.
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Build and execute query
        if status:
            result = db_client.client.table('documents').select('*').eq('status', status).range(offset, offset + page_size - 1).execute()
            count_result = db_client.client.table('documents').select('id', count='exact').eq('status', status).execute()
        else:
            result = db_client.client.table('documents').select('*').range(offset, offset + page_size - 1).execute()
            count_result = db_client.client.table('documents').select('id', count='exact').execute()
        
        total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
        
        return APIResponse.success_response(
            data={
                "documents": result.data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}", response_model=APIResponse)
async def get_document(document_id: str):
    """
    Get a specific document by ID.
    """
    try:
        result = db_client.client.table('documents').select('*').eq('id', document_id).execute()
        
        if result.data:
            return APIResponse.success_response(data=result.data[0])
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.delete("/{document_id}", response_model=APIResponse)
async def delete_document(document_id: str):
    """
    Delete a document and its embeddings.
    """
    try:
        # Delete embeddings first
        db_client.client.table('embeddings').delete().eq('document_id', document_id).execute()
        
        # Delete document
        result = db_client.client.table('documents').delete().eq('id', document_id).execute()
        
        if result.data:
            return APIResponse.success_response(
                data={"message": f"Document {document_id} deleted successfully"}
            )
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.post("/{document_id}/reprocess", response_model=APIResponse)
async def reprocess_document(document_id: str, background_tasks: BackgroundTasks):
    """
    Reprocess an existing document.
    """
    try:
        # Get document details
        result = db_client.client.table('documents').select('*').eq('id', document_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = result.data[0]
        
        # Delete existing embeddings
        db_client.client.table('embeddings').delete().eq('document_id', document_id).execute()
        
        # Extract file info from proper columns (with metadata fallback for backward compatibility)
        file_path = document.get('storage_path')
        if not file_path:
            # Fallback to metadata for legacy documents
            metadata = document.get('metadata', {})
            file_path = metadata.get('storage_path', metadata.get('file_path', metadata.get('upload_path', '')))
        
        document_type = document.get('document_type')
        if not document_type:
            # Fallback to metadata for legacy documents  
            document_type = metadata.get('document_type', 'text')
        
        # Start reprocessing
        background_tasks.add_task(
            process_document_background, 
            document_id, 
            file_path, 
            document_type,
            document['title']
        )
        
        return APIResponse.success_response(
            data={
                "document_id": document_id,
                "status": DocumentStatus.PROCESSING.value,
                "message": "Document reprocessing started"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}") 