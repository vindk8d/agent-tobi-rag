"""
Supabase Storage utilities for Quotation PDFs

Provides functions to upload generated quotation PDFs to the
`quotations` bucket and create signed URLs for time-limited sharing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from .database import db_client

logger = logging.getLogger(__name__)


class QuotationStorageError(Exception):
    """Raised when an operation related to quotation storage fails."""


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
    safe_customer_id = (customer_id or "unknown").replace("/", "-")
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
    if not storage_path or not isinstance(storage_path, str):
        raise QuotationStorageError("Invalid storage path")

    supabase_client = db_client.client

    def _do_sign():
        return supabase_client.storage.from_("quotations").create_signed_url(storage_path, expires_in_seconds)

    result = await asyncio.to_thread(_do_sign)

    # The project already expects 'signedURL' key in similar code paths
    signed_url = None
    if isinstance(result, dict):
        signed_url = result.get("signedURL") or result.get("signed_url")
    elif hasattr(result, "signedURL"):
        signed_url = getattr(result, "signedURL")
    elif hasattr(result, "signed_url"):
        signed_url = getattr(result, "signed_url")

    if not signed_url:
        raise QuotationStorageError(f"Failed to create signed URL for {storage_path}: {result}")

    return signed_url


