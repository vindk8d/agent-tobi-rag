"""
Quotation PDF cleanup task

Removes expired quotation PDFs from the `quotations` storage bucket and
updates database records accordingly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from core.database import db_client

logger = logging.getLogger(__name__)


def _extract_storage_path_from_pdf_url(pdf_url: str) -> Optional[str]:
    """
    Extract the storage path inside the `quotations` bucket from a Supabase URL.

    Supports both signed and public URLs that contain "/quotations/".
    Returns None if path cannot be determined confidently.
    """
    if not pdf_url or "/quotations/" not in pdf_url:
        return None

    try:
        # Path is everything after the first occurrence of "quotations/"
        path = pdf_url.split("/quotations/", 1)[1]
        # Strip query/hash if present
        for sep in ("?", "#"):
            if sep in path:
                path = path.split(sep, 1)[0]
        # Normalize any accidental leading slashes
        return path.lstrip("/")
    except Exception:
        return None


def cleanup_expired_quotations(max_delete: int = 200) -> dict:
    """
    Remove expired quotation PDFs from storage and mark quotations as expired.

    Args:
        max_delete: Maximum number of records to process in a single run

    Returns:
        Summary dict: {"selected": n, "deleted": n, "failed": n}
    """
    now_utc = datetime.now(timezone.utc)
    client = db_client.client

    summary = {"selected": 0, "deleted": 0, "failed": 0}

    try:
        # Find quotations whose validity has ended and still have a stored PDF URL
        # We target statuses that are not final acceptance; accepted quotations are retained.
        result = (
            client.table("quotations")
            .select("id, pdf_url, status, expires_at")
            .lt("expires_at", now_utc.isoformat())
            .in_("status", ["draft", "pending", "sent", "viewed", "expired", "rejected"])
            .not_.is_("pdf_url", None)
            .limit(max_delete)
            .execute()
        )

        records = result.data or []
        summary["selected"] = len(records)
        if not records:
            return summary

        # Collect storage paths to delete and corresponding quotation ids
        paths: List[str] = []
        ids: List[str] = []
        for row in records:
            storage_path = _extract_storage_path_from_pdf_url(row.get("pdf_url", ""))
            if storage_path:
                paths.append(storage_path)
                ids.append(row.get("id"))

        if not paths:
            return summary

        # Delete from storage in one batch
        remove_result = client.storage.from_("quotations").remove(paths)

        # Some SDKs return { data, error }, others may just throw;
        # treat the absence of an error as success.
        delete_error = None
        if isinstance(remove_result, dict):
            delete_error = remove_result.get("error")
        elif hasattr(remove_result, "error"):
            delete_error = getattr(remove_result, "error")

        if delete_error:
            logger.error(f"Storage remove error: {delete_error}")
            summary["failed"] = len(paths)
            return summary

        # Update DB: clear pdf_url and set status to expired (if not already)
        update_result = (
            client.table("quotations")
            .update({"pdf_url": None, "status": "expired"})
            .in_("id", ids)
            .execute()
        )

        # Consider all removed as deleted if DB update returned successfully
        summary["deleted"] = len(paths)
        return summary

    except Exception as e:
        logger.error(f"Quotation cleanup failed: {e}")
        summary["failed"] = summary.get("selected", 0)
        return summary


