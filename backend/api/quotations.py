"""
Quotation API endpoints for downloading and managing quotations
"""
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import RedirectResponse
from core.database import get_db_client
from core.storage import create_signed_quotation_url

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/quotations/{quotation_number}/download")
async def download_quotation(quotation_number: str):
    """
    Download a quotation PDF by quotation number.
    This creates a user-friendly short URL that redirects to the actual Supabase signed URL.
    
    Example: GET /api/v1/quotations/Q20250810-F1808036/download
    """
    try:
        # Get the PDF filename from database
        db_client = get_db_client()
        result = await asyncio.to_thread(
            lambda: db_client.client.table('quotations')
            .select('pdf_filename, title')
            .eq('quotation_number', quotation_number)
            .execute()
        )
        
        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail=f"Quotation {quotation_number} not found"
            )
        
        pdf_filename = result.data[0].get('pdf_filename')
        quotation_title = result.data[0].get('title', 'Quotation')
        
        if not pdf_filename:
            raise HTTPException(
                status_code=404,
                detail=f"PDF file not found for quotation {quotation_number}"
            )
        
        # Generate a fresh signed URL (24-hour validity for downloads)
        signed_url = await create_signed_quotation_url(pdf_filename, expires_in_seconds=24*3600)
        
        logger.info(f"[QUOTATION_DOWNLOAD] Generated download link for {quotation_number}")
        
        # Redirect to the signed URL
        return RedirectResponse(url=signed_url, status_code=302)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUOTATION_DOWNLOAD] Error generating download link for {quotation_number}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate download link. Please try again later."
        )

@router.get("/quotations/{quotation_number}/info")
async def get_quotation_info(quotation_number: str):
    """
    Get quotation information without downloading the PDF
    """
    try:
        db_client = get_db_client()
        result = await asyncio.to_thread(
            lambda: db_client.client.table('quotations')
            .select('quotation_number, title, status, created_at, expires_at, vehicle_specs, pricing_data')
            .eq('quotation_number', quotation_number)
            .execute()
        )
        
        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail=f"Quotation {quotation_number} not found"
            )
        
        return result.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUOTATION_INFO] Error getting info for {quotation_number}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quotation information"
        )
