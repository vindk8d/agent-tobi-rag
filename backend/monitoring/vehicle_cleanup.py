"""
Vehicle Specification Backup Cleanup Service

Removes old vehicle specification backup files from the `documents` storage bucket
following the same patterns as the quotation cleanup service.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from core.database import db_client
from core.storage import cleanup_old_vehicle_backups, VehicleStorageError

logger = logging.getLogger(__name__)


async def cleanup_old_vehicle_specifications(
    days_old: int = 30, 
    max_delete: int = 200
) -> Dict[str, Any]:
    """
    Remove old vehicle specification backup files from storage.
    Follows quotation cleanup service patterns for consistency.
    
    Args:
        days_old: Number of days after which backup files should be deleted (default: 30)
        max_delete: Maximum number of files to delete in a single run (default: 200)
        
    Returns:
        Summary dict: {
            "selected": int,
            "deleted": int, 
            "failed": int,
            "errors": List[str],
            "operation_metadata": Dict[str, Any]
        }
    """
    logger.info(f"Starting vehicle specification cleanup: removing backups older than {days_old} days")
    
    # Initialize summary following quotation cleanup patterns
    summary = {
        "selected": 0,
        "deleted": 0,
        "failed": 0,
        "errors": [],
        "service": "vehicle_cleanup",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "days_old": days_old,
            "max_delete": max_delete
        }
    }
    
    try:
        # Use the enhanced cleanup function from storage module
        cleanup_result = await cleanup_old_vehicle_backups(days_old=days_old, max_delete=max_delete)
        
        # Merge results into summary (following quotation patterns)
        summary.update({
            "selected": cleanup_result.get("selected", 0),
            "deleted": cleanup_result.get("deleted", 0),
            "failed": cleanup_result.get("failed", 0),
            "errors": cleanup_result.get("errors", []),
            "operation_metadata": cleanup_result.get("operation_metadata", {}),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success" if cleanup_result.get("failed", 0) == 0 else "partial_failure"
        })
        
        # Log results (following quotation logging patterns)
        if summary["deleted"] > 0:
            logger.info(f"Vehicle cleanup completed successfully: deleted {summary['deleted']} old backup files")
        else:
            logger.info("Vehicle cleanup completed: no old backup files found")
            
        # Log any errors
        if summary["errors"]:
            for error in summary["errors"]:
                logger.warning(f"Vehicle cleanup error: {error}")
        
        return summary
        
    except VehicleStorageError as storage_error:
        # Handle storage-specific errors (following quotation error patterns)
        error_msg = f"Vehicle storage cleanup failed: {str(storage_error)}"
        logger.error(error_msg)
        
        summary.update({
            "failed": summary.get("selected", 0),
            "errors": [error_msg],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed"
        })
        
        return summary
        
    except Exception as e:
        # Handle unexpected errors (following quotation error patterns)
        error_msg = f"Unexpected error during vehicle cleanup: {str(e)}"
        logger.error(error_msg)
        
        summary.update({
            "failed": summary.get("selected", 0),
            "errors": [error_msg],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed"
        })
        
        return summary


async def cleanup_orphaned_vehicle_documents(max_delete: int = 100) -> Dict[str, Any]:
    """
    Clean up orphaned vehicle document records that have no corresponding storage files.
    Follows quotation cleanup service patterns for database operations.
    
    Args:
        max_delete: Maximum number of records to process in a single run
        
    Returns:
        Summary dict with cleanup results
    """
    logger.info("Starting cleanup of orphaned vehicle document records")
    
    summary = {
        "selected": 0,
        "deleted": 0,
        "failed": 0,
        "errors": [],
        "service": "vehicle_orphan_cleanup",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"max_delete": max_delete}
    }
    
    try:
        client = db_client.client
        
        # Find vehicle documents that might be orphaned (following quotation patterns)
        # Look for documents with storage_path but older than 7 days
        cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - 7)  # 7 days ago
        
        result = (
            client.table("documents")
            .select("id, storage_path, original_filename, created_at, vehicle_id")
            .eq("document_type", "vehicle_specification")
            .not_.is_("storage_path", None)
            .lt("created_at", cutoff_date.isoformat())
            .limit(max_delete)
            .execute()
        )
        
        records = result.data or []
        summary["selected"] = len(records)
        
        if not records:
            summary.update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "success"
            })
            logger.info("No orphaned vehicle documents found")
            return summary
        
        # Check which documents have missing storage files
        orphaned_ids = []
        
        for record in records:
            storage_path = record.get("storage_path")
            if storage_path:
                try:
                    # Check if file exists in storage
                    file_result = client.storage.from_("documents").list(
                        path=storage_path.rsplit('/', 1)[0],  # Get directory path
                        search=storage_path.rsplit('/', 1)[1]  # Get filename
                    )
                    
                    # If file doesn't exist, mark as orphaned
                    if not file_result or len(file_result) == 0:
                        orphaned_ids.append(record["id"])
                        logger.info(f"Found orphaned document: {record['id']} -> {storage_path}")
                        
                except Exception as check_error:
                    logger.warning(f"Error checking storage for document {record['id']}: {check_error}")
                    # Don't delete if we can't verify - err on the side of caution
        
        if not orphaned_ids:
            summary.update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "success"
            })
            logger.info("No orphaned vehicle documents found after storage verification")
            return summary
        
        # Delete orphaned document records (following quotation patterns)
        # First delete related chunks and embeddings
        for doc_id in orphaned_ids:
            try:
                # Delete embeddings first (foreign key constraint)
                chunks_result = client.table('document_chunks').select('id').eq('document_id', doc_id).execute()
                chunk_ids = [chunk['id'] for chunk in chunks_result.data or []]
                
                if chunk_ids:
                    client.table('embeddings').delete().in_('document_chunk_id', chunk_ids).execute()
                    client.table('document_chunks').delete().eq('document_id', doc_id).execute()
                
                # Delete document record
                client.table('documents').delete().eq('id', doc_id).execute()
                
                logger.info(f"Deleted orphaned document record: {doc_id}")
                
            except Exception as delete_error:
                error_msg = f"Failed to delete orphaned document {doc_id}: {delete_error}"
                logger.error(error_msg)
                summary["errors"].append(error_msg)
        
        # Calculate results
        deleted_count = len(orphaned_ids) - len(summary["errors"])
        summary.update({
            "deleted": deleted_count,
            "failed": len(summary["errors"]),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success" if len(summary["errors"]) == 0 else "partial_failure"
        })
        
        logger.info(f"Orphaned document cleanup completed: deleted {deleted_count} records")
        return summary
        
    except Exception as e:
        error_msg = f"Orphaned document cleanup failed: {str(e)}"
        logger.error(error_msg)
        
        summary.update({
            "failed": summary.get("selected", 0),
            "errors": [error_msg],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed"
        })
        
        return summary


def run_vehicle_cleanup_sync(days_old: int = 30, max_delete: int = 200) -> Dict[str, Any]:
    """
    Synchronous wrapper for vehicle cleanup to match quotation cleanup patterns.
    Used by the scheduler which expects synchronous functions.
    
    Args:
        days_old: Number of days after which backup files should be deleted
        max_delete: Maximum number of files to delete in a single run
        
    Returns:
        Summary dict with cleanup results
    """
    try:
        # Run the async cleanup function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(cleanup_old_vehicle_specifications(days_old, max_delete))
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Synchronous vehicle cleanup wrapper failed: {e}")
        return {
            "selected": 0,
            "deleted": 0,
            "failed": 0,
            "errors": [str(e)],
            "service": "vehicle_cleanup",
            "status": "failed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }


def run_vehicle_orphan_cleanup_sync(max_delete: int = 100) -> Dict[str, Any]:
    """
    Synchronous wrapper for orphaned document cleanup.
    Used by the scheduler which expects synchronous functions.
    
    Args:
        max_delete: Maximum number of records to process in a single run
        
    Returns:
        Summary dict with cleanup results
    """
    try:
        # Run the async cleanup function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(cleanup_orphaned_vehicle_documents(max_delete))
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Synchronous orphan cleanup wrapper failed: {e}")
        return {
            "selected": 0,
            "deleted": 0,
            "failed": 0,
            "errors": [str(e)],
            "service": "vehicle_orphan_cleanup",
            "status": "failed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
