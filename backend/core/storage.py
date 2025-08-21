"""
Supabase Storage utilities for Quotation PDFs and Vehicle Specifications

Provides functions to upload generated quotation PDFs to the
`quotations` bucket and vehicle specification documents to the
`documents` bucket with organized folder structures and cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from .database import db_client

logger = logging.getLogger(__name__)


class QuotationStorageError(Exception):
    """Raised when an operation related to quotation storage fails."""


class VehicleStorageError(Exception):
    """Raised when an operation related to vehicle storage fails."""


def _generate_quotation_filename(customer_id: str, timestamp: Optional[str] = None) -> str:
    """
    Generate a standardized file name for quotation PDFs.

    Naming convention: quotation_{customer_id}_{timestamp}.pdf

    Args:
        customer_id: UUID or identifier of the customer
        timestamp: Optional timestamp string (YYYYMMDDHHMMSS); when omitted, current time is used

    Returns:
        File name string
    """
    # Comprehensive sanitization to prevent directory traversal, injection attacks, and data leakage
    safe_customer_id = (customer_id or "unknown")
    
    # Remove/replace dangerous characters
    safe_customer_id = safe_customer_id.replace("/", "-")  # Unix path separator
    safe_customer_id = safe_customer_id.replace("\\", "-")  # Windows path separator
    safe_customer_id = safe_customer_id.replace("..", "-")  # Directory traversal
    safe_customer_id = safe_customer_id.replace("\x00", "")  # Null byte injection
    safe_customer_id = safe_customer_id.replace("\n", "-")  # Newline injection
    safe_customer_id = safe_customer_id.replace("\r", "-")  # Carriage return
    safe_customer_id = safe_customer_id.replace(";", "-")   # Command injection
    safe_customer_id = safe_customer_id.replace("|", "-")   # Pipe injection
    safe_customer_id = safe_customer_id.replace("&", "-")   # Command chaining
    safe_customer_id = safe_customer_id.replace("$", "-")   # Variable expansion
    safe_customer_id = safe_customer_id.replace("`", "-")   # Command substitution
    safe_customer_id = safe_customer_id.replace("'", "-")   # Single quote
    safe_customer_id = safe_customer_id.replace('"', "-")   # Double quote
    
    # Remove sensitive keywords that could leak information
    # Use word boundaries that account for underscores and other separators
    sensitive_patterns = [
        r'(?i)secret', r'(?i)password', r'(?i)key', r'(?i)token',
        r'(?i)confidential', r'(?i)private', r'(?i)admin', r'(?i)root',
        r'(?i)apikey', r'(?i)api_key', r'(?i)api-key', r'(?i)auth', r'(?i)credential'
    ]
    
    for pattern in sensitive_patterns:
        safe_customer_id = re.sub(pattern, "REDACTED", safe_customer_id)
    
    # Limit length to prevent buffer overflow attacks
    safe_customer_id = safe_customer_id[:50]
    
    # Ensure it's not empty after sanitization
    if not safe_customer_id or safe_customer_id.replace("-", "").replace("REDACTED", "").strip() == "":
        safe_customer_id = "sanitized"
    # Include microseconds + short random suffix to avoid collisions within the same second
    ts = timestamp or datetime.now().strftime("%Y%m%d%H%M%S%f")
    suffix = uuid.uuid4().hex[:6]
    return f"quotation_{safe_customer_id}_{ts}{suffix}.pdf"


async def upload_quotation_pdf(
    pdf_bytes: bytes,
    customer_id: str,
    employee_id: Optional[str] = None,
    quotation_number: Optional[str] = None,
    folder: str = "",
    retry_attempts: int = 3
) -> Dict[str, Any]:
    """
    Upload a quotation PDF to the `quotations` storage bucket with metadata.

    Args:
        pdf_bytes: The PDF content as bytes
        customer_id: The customer identifier used in the file name
        employee_id: Employee ID responsible for the quotation (stored as metadata)
        quotation_number: Optional human-friendly quotation number (stored as metadata)
        folder: Optional folder prefix inside the bucket (e.g., "2025/08/")
        retry_attempts: Number of times to retry on transient failures

    Returns:
        Dict with details: {bucket, path, size, content_type, uploaded_at}

    Raises:
        QuotationStorageError on failure
    """
    if not isinstance(pdf_bytes, (bytes, bytearray)) or len(pdf_bytes) == 0:
        raise QuotationStorageError("Invalid PDF bytes provided")

    filename = _generate_quotation_filename(customer_id)
    storage_path = f"{folder.rstrip('/')}/{filename}" if folder else filename

    supabase_client = db_client.client

    # Use raw bytes for upload for compatibility with Supabase Storage SDK

    # Options for upload (align with Supabase Storage expectations)
    upload_options = {
        "content-type": "application/pdf",
        "cache-control": "3600",
        # Some SDK versions expect string values in file_options
        "upsert": "true",
    }

    # Add lightweight metadata via path-based convention and separate DB if needed.
    # Supabase Storage does not persist arbitrary metadata fields per object in all SDKs,
    # so we primarily rely on path + downstream DB records for rich metadata.

    last_error: Optional[Exception] = None
    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(f"Uploading quotation PDF to storage (attempt {attempt}) -> {storage_path}")

            def _do_upload():
                # Upload raw bytes
                return supabase_client.storage.from_("quotations").upload(storage_path, pdf_bytes, upload_options)

            result = await asyncio.to_thread(_do_upload)

            # Handle different SDK return shapes
            error = None
            if isinstance(result, dict):
                error = result.get("error")
            elif hasattr(result, "error"):
                error = getattr(result, "error")

            if error:
                raise QuotationStorageError(str(error))

            logger.info(f"Uploaded quotation PDF successfully -> {storage_path}")
            return {
                "bucket": "quotations",
                "path": storage_path,
                "size": len(pdf_bytes),
                "content_type": "application/pdf",
                "uploaded_at": datetime.now().isoformat(),
                "customer_id": customer_id,
                "employee_id": employee_id,
                "quotation_number": quotation_number,
            }

        except Exception as e:
            last_error = e
            logger.warning(f"Upload failed (attempt {attempt}/{retry_attempts}) for {storage_path}: {e}")
            # No rewinding needed when using raw bytes
            if attempt < retry_attempts:
                # Basic exponential backoff: 0.5s, 1s, 2s ...
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

    # Exhausted retries
    raise QuotationStorageError(f"Failed to upload quotation PDF after {retry_attempts} attempts: {last_error}")


async def create_signed_quotation_url(storage_path: str, expires_in_seconds: int = 48 * 3600) -> str:
    """
    Create a signed URL for a stored quotation PDF in the `quotations` bucket.

    Args:
        storage_path: Path of the file inside the bucket
        expires_in_seconds: URL validity period (default: 48 hours)

    Returns:
        Signed URL string

    Raises:
        QuotationStorageError on failure
    """
    if not storage_path:
        raise QuotationStorageError("Invalid storage path")

    supabase_client = db_client.client

    def _do_sign():
        return supabase_client.storage.from_("quotations").create_signed_url(storage_path, expires_in_seconds)

    result = await asyncio.to_thread(_do_sign)

    # The project already expects 'signedURL' key in similar code paths
    signed_url = None
    if hasattr(result, "get"):
        signed_url = result.get("signedURL") or result.get("signed_url")
    elif hasattr(result, "signedURL"):
        signed_url = getattr(result, "signedURL")
    elif hasattr(result, "signed_url"):
        signed_url = getattr(result, "signed_url")

    if not signed_url:
        raise QuotationStorageError(f"Failed to create signed URL for {storage_path}: {result}")

    return signed_url


# ============================================================================
# VEHICLE SPECIFICATION STORAGE FUNCTIONS
# ============================================================================

def extract_vehicle_storage_path_from_document(document_record: Dict[str, Any]) -> Optional[str]:
    """
    Extract the storage path from a vehicle document record.
    Follows quotation system pattern for path extraction.
    
    Args:
        document_record: Document record from database
        
    Returns:
        Storage path string or None if not found
    """
    if not document_record:
        return None
    
    # Try direct storage_path field first
    storage_path = document_record.get('storage_path')
    if storage_path:
        return storage_path
    
    # Try metadata storage_path
    metadata = document_record.get('metadata', {})
    if isinstance(metadata, dict):
        storage_path = metadata.get('storage_path')
        if storage_path:
            return storage_path
    
    # Fallback: generate path from vehicle_id and filename
    vehicle_id = document_record.get('vehicle_id')
    filename = document_record.get('original_filename')
    if vehicle_id and filename:
        return generate_vehicle_storage_path(vehicle_id, filename, "current")
    
    return None


def generate_vehicle_storage_path(vehicle_id: str, filename: str, folder_type: str = "current") -> str:
    """
    Generate a standardized storage path for vehicle specification documents.
    
    Path convention: vehicles/{vehicle_id}/{folder_type}/{filename}
    
    Args:
        vehicle_id: UUID of the vehicle
        filename: Original filename of the document
        folder_type: Type of folder ("current" or "replaced")
        
    Returns:
        Storage path string
    """
    # Sanitize vehicle_id to prevent path traversal
    safe_vehicle_id = str(vehicle_id).replace("/", "-").replace("\\", "-").replace("..", "-")
    
    # Sanitize filename
    safe_filename = filename.replace("/", "-").replace("\\", "-").replace("..", "-")
    
    # Ensure folder_type is valid
    if folder_type not in ["current", "replaced"]:
        folder_type = "current"
    
    return f"vehicles/{safe_vehicle_id}/{folder_type}/{safe_filename}"


def generate_replaced_vehicle_path(current_path: str) -> str:
    """
    Generate a backup path for replaced vehicle specification documents.
    
    Converts: vehicles/{vehicle_id}/current/{filename}
    To: vehicles/{vehicle_id}/replaced/{timestamp}_{filename}
    
    Args:
        current_path: Current storage path
        
    Returns:
        Backup storage path with timestamp
    """
    if "/current/" not in current_path:
        # If it's not a current path, just move to replaced folder
        return current_path.replace("/current/", "/replaced/")
    
    # Extract filename and add timestamp
    parts = current_path.split("/")
    if len(parts) >= 3:
        vehicle_id = parts[1]
        filename = parts[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"vehicles/{vehicle_id}/replaced/{timestamp}_{filename}"
    
    # Fallback
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_path.replace("/current/", f"/replaced/{timestamp}_")


async def upload_vehicle_specification(
    file_bytes: bytes,
    vehicle_id: str,
    filename: str,
    content_type: str = "text/plain",
    retry_attempts: int = 3
) -> Dict[str, Any]:
    """
    Upload a vehicle specification document to the `documents` storage bucket.
    
    Args:
        file_bytes: The file content as bytes
        vehicle_id: The vehicle identifier
        filename: Original filename
        content_type: MIME type of the file
        retry_attempts: Number of times to retry on transient failures
        
    Returns:
        Dict with details: {bucket, path, size, content_type, uploaded_at}
        
    Raises:
        VehicleStorageError on failure
    """
    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        raise VehicleStorageError("Invalid file bytes provided")
    
    storage_path = generate_vehicle_storage_path(vehicle_id, filename, "current")
    supabase_client = db_client.client
    
    # Upload options
    upload_options = {
        "content-type": content_type,
        "cache-control": "3600",
        "upsert": "true",
    }
    
    last_error: Optional[Exception] = None
    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(f"Uploading vehicle specification to storage (attempt {attempt}) -> {storage_path}")
            
            def _do_upload():
                return supabase_client.storage.from_("documents").upload(storage_path, file_bytes, upload_options)
            
            result = await asyncio.to_thread(_do_upload)
            
            # Handle different SDK return shapes
            error = None
            if isinstance(result, dict):
                error = result.get("error")
            elif hasattr(result, "error"):
                error = getattr(result, "error")
            
            if error:
                raise VehicleStorageError(str(error))
            
            logger.info(f"Uploaded vehicle specification successfully -> {storage_path}")
            return {
                "bucket": "documents",
                "path": storage_path,
                "size": len(file_bytes),
                "content_type": content_type,
                "uploaded_at": datetime.now().isoformat(),
                "vehicle_id": vehicle_id,
                "filename": filename,
            }
            
        except Exception as e:
            last_error = e
            logger.warning(f"Upload failed (attempt {attempt}/{retry_attempts}) for {storage_path}: {e}")
            if attempt < retry_attempts:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
    
    raise VehicleStorageError(f"Failed to upload vehicle specification after {retry_attempts} attempts: {last_error}")


async def move_vehicle_specification_to_backup(current_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Move a vehicle specification document to the backup/replaced folder.
    Follows quotation system backup patterns for reliability.
    
    Args:
        current_path: Current storage path of the document
        document_id: Optional document ID for database synchronization
        
    Returns:
        Dict with backup details: {backup_path, original_path, moved_at, document_id}
        
    Raises:
        VehicleStorageError on failure
    """
    backup_path = generate_replaced_vehicle_path(current_path)
    supabase_client = db_client.client
    
    # Prepare operation metadata (following quotation patterns)
    operation_metadata = {
        "original_path": current_path,
        "backup_path": backup_path,
        "document_id": document_id,
        "operation": "vehicle_backup",
        "started_at": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"Starting vehicle specification backup operation: {current_path} -> {backup_path}")
        
        def _do_backup_operation():
            # Step 1: Copy file to backup location (following quotation error handling patterns)
            copy_result = supabase_client.storage.from_("documents").copy(current_path, backup_path)
            
            # Handle different SDK return shapes (same as quotation system)
            copy_error = None
            if hasattr(copy_result, "get"):
                copy_error = copy_result.get("error")
            elif hasattr(copy_result, "error"):
                copy_error = getattr(copy_result, "error")
            
            if copy_error:
                raise VehicleStorageError(f"Backup copy failed: {copy_error}")
            
            # Step 2: Delete original file (following quotation batch deletion patterns)
            delete_result = supabase_client.storage.from_("documents").remove([current_path])
            
            # Handle different SDK return shapes
            delete_error = None
            if hasattr(delete_result, "get"):
                delete_error = delete_result.get("error")
            elif hasattr(delete_result, "error"):
                delete_error = getattr(delete_result, "error")
            
            if delete_error:
                # Attempt to clean up the backup copy if original deletion failed
                try:
                    supabase_client.storage.from_("documents").remove([backup_path])
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup backup after deletion failure: {cleanup_error}")
                
                raise VehicleStorageError(f"Original file deletion failed: {delete_error}")
            
            return backup_path
        
        # Execute backup operation
        result_path = await asyncio.to_thread(_do_backup_operation)
        
        # Update operation metadata with success
        operation_metadata.update({
            "completed_at": datetime.now().isoformat(),
            "status": "success",
            "result_path": result_path
        })
        
        # Update document record if document_id provided (following quotation DB sync patterns)
        if document_id:
            try:
                db_client.client.table('documents').update({
                    'storage_path': backup_path,
                    'metadata': {
                        'backup_operation': operation_metadata,
                        'previous_path': current_path,
                        'backup_timestamp': operation_metadata["completed_at"]
                    }
                }).eq('id', document_id).execute()
                
                logger.info(f"Updated document record {document_id} with new backup path")
            except Exception as db_error:
                logger.warning(f"Failed to update document record after backup: {db_error}")
                # Don't fail the backup operation for DB update issues
        
        logger.info(f"Successfully completed vehicle specification backup: {current_path} -> {backup_path}")
        
        return {
            "backup_path": backup_path,
            "original_path": current_path,
            "moved_at": operation_metadata["completed_at"],
            "document_id": document_id,
            "operation_metadata": operation_metadata
        }
        
    except VehicleStorageError:
        # Re-raise storage errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error during vehicle specification backup: {e}")
        operation_metadata.update({
            "completed_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        })
        raise VehicleStorageError(f"Backup operation failed: {str(e)}")


async def cleanup_old_vehicle_backups(days_old: int = 30, max_delete: int = 200) -> Dict[str, Any]:
    """
    Clean up vehicle specification backup files older than specified days.
    Follows quotation system cleanup patterns for reliability and batch operations.
    
    Args:
        days_old: Number of days after which backup files should be deleted
        max_delete: Maximum number of files to delete in a single run (following quotation patterns)
        
    Returns:
        Dict with cleanup results: {selected, deleted, failed, errors, operation_metadata}
        
    Raises:
        VehicleStorageError on failure
    """
    supabase_client = db_client.client
    cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    
    # Initialize summary following quotation cleanup patterns
    summary = {
        "selected": 0,
        "deleted": 0, 
        "failed": 0,
        "errors": [],
        "operation_metadata": {
            "started_at": datetime.now().isoformat(),
            "cutoff_date": datetime.fromtimestamp(cutoff_date).isoformat(),
            "max_delete": max_delete,
            "days_old": days_old
        }
    }
    
    try:
        logger.info(f"Starting cleanup of vehicle backups older than {days_old} days (max: {max_delete} files)")
        
        def _do_cleanup_scan():
            # List all files in the vehicles folder
            list_result = supabase_client.storage.from_("documents").list("vehicles", {
                "limit": 1000,
                "offset": 0
            })
            
            # Handle different SDK return shapes (following quotation patterns)
            if isinstance(list_result, dict) and list_result.get("error"):
                raise VehicleStorageError(f"Failed to list vehicle folders: {list_result['error']}")
            
            files_to_delete = []
            
            # Process each vehicle folder (following quotation batch processing patterns)
            for item in list_result or []:
                item_name = item.get("name")
                if item_name and not item_name.startswith("."):
                    vehicle_id = item_name
                    
                    try:
                        # List replaced folder contents
                        replaced_result = supabase_client.storage.from_("documents").list(f"vehicles/{vehicle_id}/replaced", {
                            "limit": 1000,
                            "offset": 0
                        })
                        
                        if replaced_result and hasattr(replaced_result, "__iter__"):
                            for file_item in replaced_result:
                                if file_item.get("updated_at"):
                                    try:
                                        # Parse timestamp and check if old enough
                                        file_timestamp = datetime.fromisoformat(
                                            file_item["updated_at"].replace("Z", "+00:00")
                                        ).timestamp()
                                        
                                        if file_timestamp < cutoff_date:
                                            file_path = f"vehicles/{vehicle_id}/replaced/{file_item['name']}"
                                            files_to_delete.append({
                                                "path": file_path,
                                                "vehicle_id": vehicle_id,
                                                "filename": file_item['name'],
                                                "updated_at": file_item["updated_at"],
                                                "age_days": (datetime.now().timestamp() - file_timestamp) / (24 * 60 * 60)
                                            })
                                            
                                            # Respect max_delete limit (following quotation patterns)
                                            if len(files_to_delete) >= max_delete:
                                                logger.info(f"Reached max_delete limit of {max_delete}, stopping scan")
                                                return files_to_delete
                                    except Exception as parse_error:
                                        logger.warning(f"Failed to parse timestamp for {file_item}: {parse_error}")
                    except Exception as folder_error:
                        logger.warning(f"Failed to scan vehicle {vehicle_id} replaced folder: {folder_error}")
                        continue
            
            return files_to_delete
        
        # Scan for files to delete
        files_to_delete = await asyncio.to_thread(_do_cleanup_scan)
        summary["selected"] = len(files_to_delete)
        
        if not files_to_delete:
            summary["operation_metadata"]["completed_at"] = datetime.now().isoformat()
            logger.info("No old vehicle backup files found for cleanup")
            return summary
        
        # Extract paths for batch deletion (following quotation batch patterns)
        paths_to_delete = [file_info["path"] for file_info in files_to_delete]
        
        def _do_batch_delete():
            # Perform batch deletion (following quotation system patterns)
            delete_result = supabase_client.storage.from_("documents").remove(paths_to_delete)
            
            # Handle different SDK return shapes (same as quotation system)
            delete_error = None
            if hasattr(delete_result, "get"):
                delete_error = delete_result.get("error")
            elif hasattr(delete_result, "error"):
                delete_error = getattr(delete_result, "error")
            
            if delete_error:
                raise VehicleStorageError(f"Batch deletion failed: {delete_error}")
            
            return delete_result
        
        # Execute batch deletion only if there are files to delete
        if paths_to_delete:
            await asyncio.to_thread(_do_batch_delete)
        
        # Update summary with success (following quotation patterns)
        summary["deleted"] = len(paths_to_delete)
        summary["operation_metadata"].update({
            "completed_at": datetime.now().isoformat(),
            "status": "success",
            "deleted_files": [file_info["path"] for file_info in files_to_delete]
        })
        
        logger.info(f"Successfully deleted {summary['deleted']} old vehicle backup files")
        
        # Log details for audit trail (following quotation logging patterns)
        for file_info in files_to_delete:
            logger.info(f"Deleted old backup: {file_info['path']} (age: {file_info['age_days']:.1f} days)")
        
        return summary
        
    except VehicleStorageError:
        # Re-raise storage errors as-is
        summary["failed"] = summary["selected"]
        summary["operation_metadata"].update({
            "completed_at": datetime.now().isoformat(),
            "status": "failed"
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error during vehicle backup cleanup: {e}")
        summary["failed"] = summary["selected"]
        summary["errors"].append(str(e))
        summary["operation_metadata"].update({
            "completed_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        })
        raise VehicleStorageError(f"Cleanup operation failed: {str(e)}")


async def create_signed_vehicle_url(storage_path: str, expires_in_seconds: int = 24 * 3600) -> str:
    """
    Create a signed URL for a stored vehicle specification document.
    
    Args:
        storage_path: Path of the file inside the documents bucket
        expires_in_seconds: URL validity period (default: 24 hours)
        
    Returns:
        Signed URL string
        
    Raises:
        VehicleStorageError on failure
    """
    if not storage_path:
        raise VehicleStorageError("Invalid storage path")
    
    supabase_client = db_client.client
    
    def _do_sign():
        return supabase_client.storage.from_("documents").create_signed_url(storage_path, expires_in_seconds)
    
    result = await asyncio.to_thread(_do_sign)
    
    # Handle different SDK return shapes
    signed_url = None
    if hasattr(result, "get"):
        signed_url = result.get("signedURL") or result.get("signed_url")
    elif hasattr(result, "signedURL"):
        signed_url = getattr(result, "signedURL")
    elif hasattr(result, "signed_url"):
        signed_url = getattr(result, "signed_url")
    
    if not signed_url:
        raise VehicleStorageError(f"Failed to create signed URL for {storage_path}: {result}")
    
    return signed_url


