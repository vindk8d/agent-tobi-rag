"""
Document API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile
from typing import Optional
import asyncio
import logging
import os
import tempfile
import requests
from pydantic import BaseModel

from models.base import APIResponse
from models.document import (
    DocumentStatus, DocumentType
)
from rag.pipeline import DocumentProcessingPipeline
from core.database import db_client
from core.storage import (
    generate_vehicle_storage_path, 
    upload_vehicle_specification as upload_to_storage,
    move_vehicle_specification_to_backup,
    extract_vehicle_storage_path_from_document,
    VehicleStorageError
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize processing pipeline
pipeline = DocumentProcessingPipeline()
# Remove sync settings initialization to avoid blocking calls
# Settings will be loaded asynchronously when needed

class ProcessUploadedRequest(BaseModel):
    """Request model for processing uploaded documents"""
    file_path: str
    file_name: str
    file_size: Optional[int] = None
    vehicle_id: Optional[str] = None  # Added for vehicle specification uploads


class VehicleSpecificationRequest(BaseModel):
    """Request model for vehicle specification uploads"""
    file_path: str
    file_name: str
    file_size: Optional[int] = None
    vehicle_id: str


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

        # Create data source record first
        data_source_data = {
            'name': request.file_name,
            'description': f"Uploaded file: {request.file_name}",
            'source_type': 'document',
            'file_path': request.file_path,
            'status': 'active',
            'document_type': document_type.value,
            'chunk_count': 0,
            'metadata': {
                'storage_bucket': 'documents',
                'upload_path': request.file_path,
                'original_filename': request.file_name,
                'file_size': request.file_size
            }
        }

        # Insert data source into database
        result = db_client.client.table('data_sources').insert(data_source_data).execute()

        if result.data:
            data_source_id = result.data[0]['id']

            # Start background processing
            background_tasks.add_task(process_document_background, data_source_id, request.file_path, document_type.value, request.file_name)

            return APIResponse.success_response(
                data={
                    "data_source_id": data_source_id,
                    "status": DocumentStatus.PROCESSING.value,
                    "message": "Document processing started"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create data source record")

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


async def process_document_background(data_source_id: str, storage_path: str, file_type: str, file_name: str):
    """
    Background task to process document: download, load, chunk, embed, and store.
    """
    local_file_path = None
    try:
        # Update status to PENDING (processing will start)
        db_client.client.table('data_sources').update({
            'status': 'pending'
        }).eq('id', data_source_id).execute()

        logger.info(f"Starting processing for data source {data_source_id}")

        # Download file from Supabase Storage
        local_file_path = await download_from_supabase_storage(storage_path)

        metadata = {
            'data_source_id': data_source_id,
            'original_filename': file_name,
            'storage_path': storage_path,
            'processed_at': str(asyncio.get_event_loop().time()),
            'upload_method': 'file_upload'
        }

        # Process the document using the fixed pipeline
        result = await pipeline.process_file(local_file_path, file_type, data_source_id, metadata)

        if result['success']:
            # Update status to COMPLETED using proper columns
            db_client.client.table('data_sources').update({
                'status': 'active',
                'last_scraped_at': 'now()',
                'chunk_count': result.get('chunks', 0),
                'metadata': {
                    'processing_result': result,
                    'data_source_id': data_source_id
                }
            }).eq('id', data_source_id).execute()

            logger.info(f"Data source {data_source_id} processed successfully: {result['chunks']} chunks")
        else:
            # Update status to FAILED using proper columns
            db_client.client.table('data_sources').update({
                'status': 'failed',
                'last_scraped_at': 'now()',
                'last_error': result.get('error', 'Processing failed'),
                'error_count': 1,
                'metadata': {
                    'error': result.get('error', 'Processing failed'),
                    'data_source_id': data_source_id
                }
            }).eq('id', data_source_id).execute()

            logger.error(f"Data source {data_source_id} processing failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"Background processing failed for data source {data_source_id}: {e}")

        # Update status to FAILED
        try:
            db_client.client.table('data_sources').update({
                'status': 'failed',
                'last_scraped_at': 'now()',
                'last_error': str(e),
                'error_count': 1,
                'metadata': {
                    'error': str(e),
                    'failed_at': str(asyncio.get_event_loop().time())
                }
            }).eq('id', data_source_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update data source status: {db_error}")
    finally:
        # Clean up temporary file
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
                logger.info(f"Cleaned up temporary file: {local_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {local_file_path}: {cleanup_error}")


async def process_vehicle_document_background(document_id: str, temp_file_path: str, file_type: str, file_name: str, vehicle_id: str):
    """
    Background task to process vehicle specification document with vehicle context.
    Uses documents-centric approach.
    """
    try:
        logger.info(f"Starting vehicle specification processing for document {document_id}, vehicle {vehicle_id}")

        # Get vehicle information for context
        vehicle_result = db_client.client.table('vehicles').select('*').eq('id', vehicle_id).execute()
        vehicle_info = vehicle_result.data[0] if vehicle_result.data else {}

        metadata = {
            'original_filename': file_name,
            'processed_at': str(asyncio.get_event_loop().time()),
            'upload_method': 'vehicle_specification_upload',
            'vehicle_id': vehicle_id,
            'vehicle_info': vehicle_info
        }

        # Process the document using the pipeline with document_id
        result = await pipeline.process_file(temp_file_path, file_type, document_id, metadata, vehicle_id=vehicle_id)

        if result['success']:
            # Update document with processing results
            db_client.client.table('documents').update({
                'processed_at': 'now()',
                'metadata': {
                    'processing_result': result,
                    'vehicle_id': vehicle_id,
                    'vehicle_info': vehicle_info,
                    'chunks_created': result.get('chunks', 0)
                }
            }).eq('id', document_id).execute()

            logger.info(f"Vehicle specification {document_id} processed successfully: {result['chunks']} chunks for vehicle {vehicle_id}")
        else:
            # Update document with error information
            db_client.client.table('documents').update({
                'processed_at': 'now()',
                'metadata': {
                    'error': result.get('error', 'Processing failed'),
                    'vehicle_id': vehicle_id,
                    'failed_at': str(asyncio.get_event_loop().time())
                }
            }).eq('id', document_id).execute()

            logger.error(f"Vehicle specification {document_id} processing failed for vehicle {vehicle_id}: {result.get('error')}")

    except Exception as e:
        logger.error(f"Background processing failed for vehicle specification {document_id}, vehicle {vehicle_id}: {e}")

        # Update document with error information
        try:
            db_client.client.table('documents').update({
                'processed_at': 'now()',
                'metadata': {
                    'error': str(e),
                    'failed_at': str(asyncio.get_event_loop().time()),
                    'vehicle_id': vehicle_id
                }
            }).eq('id', document_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update document status: {db_error}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")

@router.get("/", response_model=APIResponse)
async def list_data_sources(
    page: int = 1,
    page_size: int = 10,
    status: Optional[str] = None
):
    """
    List data sources with pagination and optional status filtering.
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size

        # Build and execute query
        if status:
            result = db_client.client.table('data_sources').select('*').eq('status', status).range(offset, offset + page_size - 1).execute()
            count_result = db_client.client.table('data_sources').select('id', count='exact').eq('status', status).execute()
        else:
            result = db_client.client.table('data_sources').select('*').range(offset, offset + page_size - 1).execute()
            count_result = db_client.client.table('data_sources').select('id', count='exact').execute()

        total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)

        return APIResponse.success_response(
            data={
                "data_sources": result.data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size
            }
        )

    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list data sources: {str(e)}")

# Vehicle-specific endpoints (must come before generic /{data_source_id} route)

@router.get("/vehicles", response_model=APIResponse)
async def list_vehicles():
    """
    List all available vehicles for the dropdown selection.
    """
    try:
        result = db_client.client.table('vehicles').select(
            'id, brand, model, year, type, variant, key_features, is_available'
        ).eq('is_available', True).order('brand, model, year').execute()

        return APIResponse.success_response(
            data={
                "vehicles": result.data,
                "total_count": len(result.data)
            }
        )

    except Exception as e:
        logger.error(f"Error listing vehicles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list vehicles: {str(e)}")


@router.get("/vehicles/{vehicle_id}/specification", response_model=APIResponse)
async def get_vehicle_specification(vehicle_id: str):
    """
    Check if a vehicle already has a specification document.
    Returns the existing document info with chunk and embedding counts if found.
    """
    try:
        # Check for existing vehicle specification
        result = db_client.client.table('documents').select(
            'id, original_filename, file_size, created_at, updated_at, vehicle_id'
        ).eq('vehicle_id', vehicle_id).execute()

        if result.data:
            document = result.data[0]
            document_id = document['id']
            document_vehicle_id = document.get('vehicle_id')
            
            # Get chunk count for this document (check both document_id and vehicle_id for compatibility)
            chunks_by_doc_result = db_client.client.table('document_chunks').select(
                'id', count='exact'
            ).eq('document_id', document_id).execute()
            chunks_by_doc_count = chunks_by_doc_result.count if hasattr(chunks_by_doc_result, 'count') else 0
            
            # Also check by vehicle_id for backward compatibility
            chunks_by_vehicle_count = 0
            if document_vehicle_id:
                chunks_by_vehicle_result = db_client.client.table('document_chunks').select(
                    'id', count='exact'
                ).eq('vehicle_id', document_vehicle_id).execute()
                chunks_by_vehicle_count = chunks_by_vehicle_result.count if hasattr(chunks_by_vehicle_result, 'count') else 0
            
            # Use the higher count (for compatibility with both old and new approaches)
            chunk_count = max(chunks_by_doc_count, chunks_by_vehicle_count)
            
            # Get embedding count for this document's chunks
            if chunk_count > 0:
                # Get chunk IDs (prefer document_id, fallback to vehicle_id)
                if chunks_by_doc_count > 0:
                    chunk_ids_result = db_client.client.table('document_chunks').select('id').eq('document_id', document_id).execute()
                elif document_vehicle_id:
                    chunk_ids_result = db_client.client.table('document_chunks').select('id').eq('vehicle_id', document_vehicle_id).execute()
                else:
                    chunk_ids_result = None
                
                if chunk_ids_result and chunk_ids_result.data:
                    chunk_ids = [chunk['id'] for chunk in chunk_ids_result.data]
                    embeddings_result = db_client.client.table('embeddings').select(
                        'id', count='exact'
                    ).in_('document_chunk_id', chunk_ids).execute()
                    embedding_count = embeddings_result.count if hasattr(embeddings_result, 'count') else 0
                else:
                    embedding_count = 0
            else:
                embedding_count = 0
            
            # Add counts to document info
            document_with_counts = {
                **document,
                'chunk_count': chunk_count,
                'embedding_count': embedding_count
            }
            
            return APIResponse.success_response(
                data={
                    "exists": True,
                    "document": document_with_counts
                }
            )
        else:
            return APIResponse.success_response(
                data={
                    "exists": False,
                    "document": None
                }
            )

    except Exception as e:
        logger.error(f"Error checking vehicle specification for {vehicle_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check vehicle specification: {str(e)}")


@router.post("/vehicles/{vehicle_id}/specification", response_model=APIResponse)
async def upload_vehicle_specification(
    vehicle_id: str,
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    """
    Upload a specification document for a specific vehicle.
    Replaces any existing specification for that vehicle.
    """
    try:
        # Verify vehicle exists
        vehicle_result = db_client.client.table('vehicles').select('*').eq('id', vehicle_id).execute()
        if not vehicle_result.data:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        vehicle_info = vehicle_result.data[0]

        # Check for existing specification
        existing_result = db_client.client.table('documents').select('*').eq('vehicle_id', vehicle_id).execute()
        
        if existing_result.data:
            existing_doc = existing_result.data[0]
            logger.info(f"Found existing vehicle specification for {vehicle_id}: {existing_doc['original_filename']}")
            
            # Move existing file to replaced/ folder using proper storage functions
            old_path = extract_vehicle_storage_path_from_document(existing_doc)
            if old_path:
                try:
                    backup_result = await move_vehicle_specification_to_backup(old_path, existing_doc['id'])
                    logger.info(f"Successfully moved existing file to backup: {backup_result['original_path']} -> {backup_result['backup_path']}")
                except VehicleStorageError as storage_error:
                    logger.warning(f"Failed to backup old file {old_path}: {storage_error}")
                except Exception as move_error:
                    logger.warning(f"Unexpected error backing up old file {old_path}: {move_error}")
            else:
                logger.warning(f"No storage path found for existing document {existing_doc['id']}, skipping backup")
            
            # Delete old document and related data
            try:
                # Get chunk IDs for this document (using both document_id and data_source_id for compatibility)
                # Build the OR condition properly handling NULL values
                data_source_id = existing_doc.get('data_source_id')
                if data_source_id:
                    or_condition = f"document_id.eq.{existing_doc['id']},data_source_id.eq.{data_source_id}"
                else:
                    or_condition = f"document_id.eq.{existing_doc['id']},data_source_id.is.null"
                
                chunks_result = db_client.client.table('document_chunks').select('id').or_(or_condition).execute()
                chunk_ids = [chunk['id'] for chunk in chunks_result.data or []]
                
                # Delete embeddings first (foreign key constraint)
                if chunk_ids:
                    db_client.client.table('embeddings').delete().in_('document_chunk_id', chunk_ids).execute()
                
                # Delete document chunks
                db_client.client.table('document_chunks').delete().or_(or_condition).execute()
                
                # Delete document
                db_client.client.table('documents').delete().eq('id', existing_doc['id']).execute()
                
                # Delete data source if it exists (for legacy compatibility)
                if existing_doc.get('data_source_id'):
                    db_client.client.table('data_sources').delete().eq('id', existing_doc['data_source_id']).execute()
                
                logger.info(f"Deleted existing vehicle specification data for {vehicle_id}")
            except Exception as delete_error:
                logger.error(f"Error deleting existing specification: {delete_error}")
                # Continue with upload even if deletion fails

        # Generate proper storage path using organized folder structure
        storage_path = generate_vehicle_storage_path(vehicle_id, file.filename, "current")
        
        # Create document record directly (documents-centric approach)
        document_data = {
            'content': f"Vehicle specification for {vehicle_info['brand']} {vehicle_info['model']} {vehicle_info['year']}",
            'vehicle_id': vehicle_id,
            'document_type': 'vehicle_specification',
            'original_filename': file.filename,
            'file_size': file.size if hasattr(file, 'size') else None,
            'storage_path': storage_path,
            'metadata': {
                'storage_bucket': 'documents',
                'storage_path': storage_path,
                'vehicle_id': vehicle_id,
                'vehicle_info': vehicle_info,
                'upload_method': 'vehicle_specification_upload',
                'original_filename': file.filename,
                'file_size': file.size if hasattr(file, 'size') else None
            }
        }

        # Insert document record
        doc_result = db_client.client.table('documents').insert(document_data).execute()
        if not doc_result.data:
            raise HTTPException(status_code=500, detail="Failed to create document record")
        
        document_id = doc_result.data[0]['id']

        # Upload file to organized storage structure
        try:
            # Read file content
            file_content = await file.read()
            
            # Determine content type with proper MIME type detection
            content_type = file.content_type
            
            # If no content type provided or it's generic, detect from filename
            if not content_type or content_type == "application/octet-stream":
                filename_lower = file.filename.lower() if file.filename else ""
                if filename_lower.endswith('.md'):
                    content_type = "text/markdown"
                elif filename_lower.endswith('.txt'):
                    content_type = "text/plain"
                elif filename_lower.endswith('.pdf'):
                    content_type = "application/pdf"
                elif filename_lower.endswith('.doc'):
                    content_type = "application/msword"
                elif filename_lower.endswith('.docx'):
                    content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                else:
                    # Default to text/plain for unknown text files
                    content_type = "text/plain"
            
            # Upload to proper storage location
            upload_result = await upload_to_storage(
                file_bytes=file_content,
                vehicle_id=vehicle_id,
                filename=file.filename,
                content_type=content_type
            )
            
            logger.info(f"Successfully uploaded vehicle specification to storage: {upload_result['path']}")
            
            # Update document record with actual storage path (simplified to avoid conflicts)
            db_client.client.table('documents').update({
                'storage_path': upload_result['path']
            }).eq('id', document_id).execute()
            
        except VehicleStorageError as storage_error:
            logger.error(f"Failed to upload vehicle specification to storage: {storage_error}")
            # Clean up document record
            db_client.client.table('documents').delete().eq('id', document_id).execute()
            raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(storage_error)}")
        except Exception as upload_error:
            logger.error(f"Unexpected error during upload: {upload_error}")
            # Clean up document record
            db_client.client.table('documents').delete().eq('id', document_id).execute()
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(upload_error)}")
        
        # Save file to temporary location for processing
        import tempfile
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Determine file type from filename extension
        file_extension = file.filename.lower().split('.')[-1] if file.filename and '.' in file.filename else 'txt'
        file_type_mapping = {
            'md': 'markdown',
            'txt': 'text',
            'pdf': 'pdf',
            'doc': 'word',
            'docx': 'word'
        }
        actual_file_type = file_type_mapping.get(file_extension, 'text')
        
        # Trigger background processing with temporary file
        background_tasks.add_task(
            process_vehicle_document_background,
            document_id,
            temp_file_path,
            actual_file_type,  # Use actual file type, not document type
            file.filename,
            vehicle_id
        )

        return APIResponse.success_response(
            data={
                "message": "Vehicle specification upload initiated",
                "document_id": document_id,
                "vehicle_id": vehicle_id,
                "processing": True,
                "status": "processing"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading vehicle specification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload vehicle specification: {str(e)}")


@router.delete("/vehicles/{vehicle_id}/specification", response_model=APIResponse)
async def delete_vehicle_specification(vehicle_id: str):
    """
    Delete the specification document for a specific vehicle.
    """
    try:
        # Find existing specification
        result = db_client.client.table('documents').select('*').eq('vehicle_id', vehicle_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="No specification found for this vehicle")
        
        document = result.data[0]
        document_id = document['id']
        data_source_id = document.get('data_source_id')

        # Get chunk IDs for this document (using both document_id and data_source_id for compatibility)
        chunks_result = db_client.client.table('document_chunks').select('id').or_(
            f"document_id.eq.{document_id},data_source_id.eq.{data_source_id or 'null'}"
        ).execute()
        chunk_ids = [chunk['id'] for chunk in chunks_result.data or []]
        
        # Delete embeddings first (foreign key constraint)
        if chunk_ids:
            db_client.client.table('embeddings').delete().in_('document_chunk_id', chunk_ids).execute()
        
        # Delete document chunks
        db_client.client.table('document_chunks').delete().or_(
            f"document_id.eq.{document_id},data_source_id.eq.{data_source_id or 'null'}"
        ).execute()
        
        # Delete document
        db_client.client.table('documents').delete().eq('id', document_id).execute()
        
        # Delete data source if it exists (for legacy compatibility)
        if data_source_id:
            db_client.client.table('data_sources').delete().eq('id', data_source_id).execute()

        return APIResponse.success_response(
            data={
                "message": "Vehicle specification deleted successfully",
                "vehicle_id": vehicle_id
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vehicle specification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vehicle specification: {str(e)}")

@router.get("/{data_source_id}", response_model=APIResponse)
async def get_data_source(data_source_id: str):
    """
    Get a specific data source by ID.
    """
    try:
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()

        if result.data:
            return APIResponse.success_response(data=result.data[0])
        else:
            raise HTTPException(status_code=404, detail="Data source not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data source: {str(e)}")

@router.delete("/{data_source_id}", response_model=APIResponse)
async def delete_data_source(data_source_id: str):
    """
    Delete a data source and all its chunks and embeddings.
    """
    try:
        # Get all chunk IDs for this data source
        chunks_result = db_client.client.table('document_chunks').select('id').eq('data_source_id', data_source_id).execute()
        chunk_ids = [chunk['id'] for chunk in chunks_result.data or []]

        # Delete embeddings first (for all chunks in this data source)
        if chunk_ids:
            db_client.client.table('embeddings').delete().in_('document_chunk_id', chunk_ids).execute()

        # Delete document chunks
        db_client.client.table('document_chunks').delete().eq('data_source_id', data_source_id).execute()

        # Delete data source
        result = db_client.client.table('data_sources').delete().eq('id', data_source_id).execute()

        if result.data:
            return APIResponse.success_response(
                data={"message": f"Data source {data_source_id} deleted successfully"}
            )
        else:
            raise HTTPException(status_code=404, detail="Data source not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data source: {str(e)}")

@router.post("/{data_source_id}/reprocess", response_model=APIResponse)
async def reprocess_data_source(data_source_id: str, background_tasks: BackgroundTasks):
    """
    Reprocess an existing data source.
    """
    try:
        # Get data source details
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")

        data_source = result.data[0]

        # Get all chunk IDs for this data source
        chunks_result = db_client.client.table('document_chunks').select('id').eq('data_source_id', data_source_id).execute()
        chunk_ids = [chunk['id'] for chunk in chunks_result.data or []]

        # Delete existing embeddings and chunks
        if chunk_ids:
            db_client.client.table('embeddings').delete().in_('document_chunk_id', chunk_ids).execute()
        db_client.client.table('document_chunks').delete().eq('data_source_id', data_source_id).execute()

        # Extract file info from proper columns (with metadata fallback for backward compatibility)
        file_path = data_source.get('file_path')
        if not file_path:
            # Fallback to metadata for legacy data sources
            metadata = data_source.get('metadata', {})
            file_path = metadata.get('storage_path', metadata.get('file_path', metadata.get('upload_path', '')))

        document_type = data_source.get('document_type')
        if not document_type:
            # Fallback to metadata for legacy data sources
            document_type = metadata.get('document_type', 'text')

        # Start reprocessing
        background_tasks.add_task(
            process_document_background,
            data_source_id,
            file_path,
            document_type,
            data_source['name']
        )

        return APIResponse.success_response(
            data={
                "data_source_id": data_source_id,
                "status": DocumentStatus.PROCESSING.value,
                "message": "Data source reprocessing started"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess data source: {str(e)}")


