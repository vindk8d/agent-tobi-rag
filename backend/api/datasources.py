"""
Data source management API endpoints for RAG-Tobi
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
from uuid import uuid4
import validators
from pydantic import BaseModel, Field, HttpUrl

from models.base import APIResponse
from models.datasource import (
    DataSourceModel, DataSourceRequest, DataSourceType, DataSourceStatus, 
    ScrapingFrequency
)
from database import db_client
from config import get_settings
from rag.pipeline import DocumentProcessingPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
settings = get_settings()
pipeline = DocumentProcessingPipeline()

class DataSourceCreateRequest(BaseModel):
    """Request model for creating a new data source"""
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable name for the data source")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    url: HttpUrl = Field(..., description="Website URL to scrape")
    scraping_frequency: ScrapingFrequency = Field(ScrapingFrequency.DAILY, description="How often to scrape")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Scraping configuration options")

class DataSourceUpdateRequest(BaseModel):
    """Request model for updating an existing data source"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    url: Optional[HttpUrl] = None
    scraping_frequency: Optional[ScrapingFrequency] = None
    configuration: Optional[Dict[str, Any]] = None

class DataSourceTestRequest(BaseModel):
    """Request model for testing a data source URL"""
    url: HttpUrl = Field(..., description="Website URL to test")

def validate_url(url: str) -> bool:
    """Validate URL format and accessibility"""
    try:
        # Basic URL validation
        if not validators.url(url):
            return False
        
        # Additional checks for supported protocols
        if not url.startswith(('http://', 'https://')):
            return False
        
        return True
    except Exception:
        return False

@router.post("/", response_model=APIResponse)
async def create_data_source(
    request: DataSourceCreateRequest,
    background_tasks: BackgroundTasks
):
    """Create a new data source for scraping"""
    try:
        # Validate URL
        if not validate_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Check if URL already exists
        existing = db_client.client.table('data_sources').select('id').eq('url', str(request.url)).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="Data source with this URL already exists")
        
        # Create data source
        data_source_id = str(uuid4())
        data_source_data = {
            'id': data_source_id,
            'name': request.name,
            'description': request.description,
            'source_type': DataSourceType.WEBSITE.value,
            'url': str(request.url),
            'status': DataSourceStatus.PENDING.value,
            'scraping_frequency': request.scraping_frequency.value,
            'configuration': request.configuration or {},
            'metadata': {}
        }
        
        # Insert into database
        result = db_client.client.table('data_sources').insert(data_source_data).execute()
        
        if result.data:
            # Start initial scrape test in background
            background_tasks.add_task(initial_scrape_test, data_source_id, str(request.url))
            
            return APIResponse.success_response(
                data={
                    "id": data_source_id,
                    "name": request.name,
                    "url": str(request.url),
                    "status": DataSourceStatus.PENDING.value,
                    "message": "Data source created successfully. Initial scraping test in progress."
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create data source")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating data source: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create data source: {str(e)}")

async def initial_scrape_test(data_source_id: str, url: str):
    """Test initial scraping capability for a data source"""
    try:
        # Test URL accessibility
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Update status to active
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.ACTIVE.value,
            'last_scraped_at': datetime.utcnow().isoformat(),
            'metadata': {'initial_test': 'passed'}
        }).eq('id', data_source_id).execute()
        
        # Start content processing
        await process_data_source_content(data_source_id, url)
        
    except Exception as e:
        logger.error(f"Initial scrape test failed for {data_source_id}: {e}")
        # Update status to failed
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.FAILED.value,
            'metadata': {'initial_test': 'failed', 'error': str(e)}
        }).eq('id', data_source_id).execute()

async def process_data_source_content(data_source_id: str, url: str):
    """Process content from a data source URL"""
    try:
        # This is a placeholder for actual content processing
        # In a real implementation, this would:
        # 1. Scrape the content
        # 2. Process and chunk the text
        # 3. Generate embeddings
        # 4. Store in vector database
        
        logger.info(f"Processing content for data source {data_source_id} from {url}")
        
        # Simulate processing
        await asyncio.sleep(2)
        
        # Update last processed time
        db_client.client.table('data_sources').update({
            'last_processed_at': datetime.utcnow().isoformat(),
            'metadata': {'last_process': 'completed'}
        }).eq('id', data_source_id).execute()
        
    except Exception as e:
        logger.error(f"Content processing failed for {data_source_id}: {e}")
        # Update status to failed
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.FAILED.value,
            'metadata': {'processing_error': str(e)}
        }).eq('id', data_source_id).execute()

@router.get("/", response_model=APIResponse)
async def list_data_sources(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: Optional[DataSourceStatus] = Query(None),
    source_type: Optional[DataSourceType] = Query(None)
):
    """List all data sources with pagination and filtering"""
    try:
        # Build query
        query = db_client.client.table('data_sources').select('*')
        
        # Apply filters
        if status:
            query = query.eq('status', status.value)
        if source_type:
            query = query.eq('source_type', source_type.value)
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.range(offset, offset + page_size - 1)
        
        # Execute query
        result = query.execute()
        
        # Get total count for pagination
        count_query = db_client.client.table('data_sources').select('id', count='exact')
        if status:
            count_query = count_query.eq('status', status.value)
        if source_type:
            count_query = count_query.eq('source_type', source_type.value)
        
        count_result = count_query.execute()
        total_count = count_result.count
        
        return APIResponse.success_response(
            data={
                "data_sources": result.data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list data sources: {str(e)}")

@router.get("/{data_source_id}", response_model=APIResponse)
async def get_data_source(data_source_id: str):
    """Get a specific data source by ID"""
    try:
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        return APIResponse.success_response(data=result.data[0])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data source: {str(e)}")

@router.put("/{data_source_id}", response_model=APIResponse)
async def update_data_source(
    data_source_id: str,
    request: DataSourceUpdateRequest
):
    """Update an existing data source"""
    try:
        # Check if data source exists
        existing = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Build update data
        update_data = {}
        if request.name is not None:
            update_data['name'] = request.name
        if request.description is not None:
            update_data['description'] = request.description
        if request.url is not None:
            if not validate_url(str(request.url)):
                raise HTTPException(status_code=400, detail="Invalid URL format")
            update_data['url'] = str(request.url)
        if request.scraping_frequency is not None:
            update_data['scraping_frequency'] = request.scraping_frequency.value
        if request.configuration is not None:
            update_data['configuration'] = request.configuration
        
        # Update in database
        result = db_client.client.table('data_sources').update(update_data).eq('id', data_source_id).execute()
        
        if result.data:
            return APIResponse.success_response(
                data=result.data[0], 
                message="Data source updated successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update data source")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update data source: {str(e)}")

@router.delete("/{data_source_id}", response_model=APIResponse)
async def delete_data_source(data_source_id: str):
    """Delete a data source"""
    try:
        # Check if data source exists
        existing = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Delete from database
        result = db_client.client.table('data_sources').delete().eq('id', data_source_id).execute()
        
        return APIResponse.success_response(
            data={"id": data_source_id},
            message="Data source deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data source: {str(e)}")

@router.post("/test", response_model=APIResponse)
async def test_data_source_url(request: DataSourceTestRequest):
    """Test a data source URL for accessibility and basic metrics"""
    try:
        # Validate URL
        if not validate_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Test URL accessibility
        import requests
        import time
        
        start_time = time.time()
        response = requests.get(str(request.url), timeout=10)
        response_time = time.time() - start_time
        
        response.raise_for_status()
        
        # Basic content analysis
        content_length = len(response.text)
        
        # Extract title
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title"
        
        return APIResponse.success_response(
            data={
                "url": str(request.url),
                "accessible": True,
                "response_time": response_time,
                "content_length": content_length,
                "title": title,
                "status_code": response.status_code
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing URL {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test URL: {str(e)}")

@router.post("/{data_source_id}/scrape", response_model=APIResponse)
async def trigger_scrape(
    data_source_id: str,
    background_tasks: BackgroundTasks
):
    """Manually trigger scraping for a data source"""
    try:
        # Check if data source exists
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        data_source = result.data[0]
        
        # Start scraping in background
        background_tasks.add_task(manual_scrape_task, data_source_id, data_source['url'])
        
        return APIResponse.success_response(
            data={"id": data_source_id, "status": "scraping_started"},
            message="Manual scraping started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering scrape for {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger scrape: {str(e)}")

async def manual_scrape_task(data_source_id: str, url: str):
    """Background task for manual scraping"""
    try:
        # Update status
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.ACTIVE.value,
            'last_scraped_at': datetime.utcnow().isoformat()
        }).eq('id', data_source_id).execute()
        
        # Process content
        await process_data_source_content(data_source_id, url)
        
    except Exception as e:
        logger.error(f"Manual scrape task failed for {data_source_id}: {e}")
        # Update status to failed
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.FAILED.value,
            'metadata': {'manual_scrape_error': str(e)}
        }).eq('id', data_source_id).execute()

@router.get("/stats/overview", response_model=APIResponse)
async def get_data_source_stats():
    """Get overview statistics for all data sources"""
    try:
        # Get counts by status
        stats = {}
        for status in DataSourceStatus:
            count_result = db_client.client.table('data_sources').select('id', count='exact').eq('status', status.value).execute()
            stats[status.value] = count_result.count
        
        # Get total count
        total_result = db_client.client.table('data_sources').select('id', count='exact').execute()
        stats['total'] = total_result.count
        
        return APIResponse.success_response(data=stats)
    
    except Exception as e:
        logger.error(f"Error getting data source stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# ============================================================================
# DEPRIORITIZED CODE - Website scraping functionality has been deprioritized
# ============================================================================

# DEPRIORITIZED: Website scraping functionality has been deprioritized
# from scrapers.web_scraper import WebScraper
# web_scraper = WebScraper()

# class WebsiteCrawlRequest(BaseModel):
#     """Request model for crawling an entire website"""
#     url: HttpUrl = Field(..., description="Base URL to start crawling from")
#     data_source_name: Optional[str] = Field(None, description="Name for the data source (defaults to domain name)")
#     max_pages: int = Field(50, ge=1, le=200, description="Maximum number of pages to crawl")
#     max_depth: int = Field(3, ge=1, le=10, description="Maximum crawl depth")
#     delay: float = Field(1.0, ge=0.1, le=10.0, description="Delay between requests in seconds")
#     include_patterns: Optional[List[str]] = Field(None, description="URL patterns to include (e.g., ['/docs/', '/api/'])")
#     exclude_patterns: Optional[List[str]] = Field(None, description="URL patterns to exclude (e.g., ['/admin/', '.pdf'])")

# async def test_scraping_capability(url: str) -> Dict[str, Any]:
#     """Test if a URL can be scraped and return basic metrics"""
#     try:
#         # Test scraping with the web scraper
#         result = await web_scraper.scrape_url(url)
        
#         if result['success']:
#             return {
#                 'success': True,
#                 'content_length': len(result.get('content', '')),
#                 'title': result.get('title', ''),
#                 'scraping_method': result.get('method', 'unknown'),
#                 'response_time': result.get('response_time', 0)
#             }
#         else:
#             return {
#                 'success': False,
#                 'error': result.get('error', 'Unknown error'),
#                 'status_code': result.get('status_code')
#             }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }

# @router.post("/crawl", response_model=APIResponse)
# async def crawl_website(
#     request: WebsiteCrawlRequest,
#     background_tasks: BackgroundTasks
# ):
#     """
#     Crawl an entire website starting from the base URL.
#     Discovers and processes all child links and related pages.
#     """
#     try:
#         # Validate URL
#         if not validate_url(str(request.url)):
#             raise HTTPException(status_code=400, detail="Invalid URL format")
        
#         # Generate data source name if not provided
#         data_source_name = request.data_source_name
#         if not data_source_name:
#             from urllib.parse import urlparse
#             parsed_url = urlparse(str(request.url))
#             data_source_name = f"{parsed_url.netloc} Website Crawl"
        
#         # Check if URL already exists as a data source
#         existing = db_client.client.table('data_sources').select('id').eq('url', str(request.url)).execute()
#         if existing.data:
#             raise HTTPException(status_code=409, detail="Data source with this URL already exists")
        
#         # Create data source record
#         data_source_id = str(uuid4())
#         data_source_data = {
#             'id': data_source_id,
#             'name': data_source_name,
#             'source_type': DataSourceType.WEBSITE.value,
#             'url': str(request.url),
#             'status': DataSourceStatus.ACTIVE.value,
#             'scraping_frequency': ScrapingFrequency.MANUAL.value,
#             'configuration': {
#                 'crawling_enabled': True,
#                 'max_pages': request.max_pages,
#                 'max_depth': request.max_depth,
#                 'delay': request.delay,
#                 'include_patterns': request.include_patterns,
#                 'exclude_patterns': request.exclude_patterns
#             },
#             'metadata': {
#                 'created_via': 'api_crawl',
#                 'created_at': datetime.utcnow().isoformat(),
#                 'crawl_request': {
#                     'max_pages': request.max_pages,
#                     'max_depth': request.max_depth,
#                     'delay': request.delay,
#                     'include_patterns': request.include_patterns,
#                     'exclude_patterns': request.exclude_patterns
#                 }
#             }
#         }
        
#         # Insert into database
#         result = db_client.client.table('data_sources').insert(data_source_data).execute()
        
#         if result.data:
#             # Start crawling in background
#             background_tasks.add_task(
#                 crawl_website_task,
#                 data_source_id,
#                 str(request.url),
#                 data_source_name,
#                 request.max_pages,
#                 request.max_depth,
#                 request.delay,
#                 request.include_patterns,
#                 request.exclude_patterns
#             )
            
#             return APIResponse.success_response(
#                 data={
#                     "data_source_id": data_source_id,
#                     "name": data_source_name,
#                     "url": str(request.url),
#                     "status": DataSourceStatus.ACTIVE.value,
#                     "crawl_config": {
#                         "max_pages": request.max_pages,
#                         "max_depth": request.max_depth,
#                         "delay": request.delay,
#                         "include_patterns": request.include_patterns,
#                         "exclude_patterns": request.exclude_patterns
#                     },
#                     "message": "Website crawl started successfully. This may take several minutes..."
#                 }
#             )
#         else:
#             raise HTTPException(status_code=500, detail="Failed to create data source for crawling")
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error starting website crawl: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to start crawl: {str(e)}")

# async def crawl_website_task(
#     data_source_id: str,
#     base_url: str,
#     data_source_name: str,
#     max_pages: int,
#     max_depth: int,
#     delay: float,
#     include_patterns: Optional[List[str]],
#     exclude_patterns: Optional[List[str]]
# ):
#     """
#     Background task for crawling an entire website.
#     """
#     try:
#         logger.info(f"Starting website crawl for data source {data_source_id}: {base_url}")
        
#         # Update status to processing
#         db_client.client.table('data_sources').update({
#             'status': DataSourceStatus.ACTIVE.value,
#             'last_scraped_at': datetime.utcnow().isoformat()
#         }).eq('id', data_source_id).execute()
        
#         # Process website using the enhanced pipeline
#         result = await pipeline.process_website(
#             base_url=base_url,
#             data_source_name=data_source_name,
#             max_pages=max_pages,
#             max_depth=max_depth,
#             delay=delay,
#             include_patterns=include_patterns,
#             exclude_patterns=exclude_patterns
#         )
        
#         if result["success"]:
#             logger.info(f"Website crawl completed for {data_source_id}: {result['processed_pages']} pages processed")
            
#             # Update data source with results
#             db_client.client.table('data_sources').update({
#                 'status': DataSourceStatus.ACTIVE.value,
#                 'last_processed_at': datetime.utcnow().isoformat(),
#                 'metadata': {
#                     'last_crawl': {
#                         'processed_pages': result['processed_pages'],
#                         'total_pages_found': result['total_pages_found'],
#                         'total_chunks': result['total_chunks'],
#                         'crawl_time': result['crawl_time'],
#                         'discovered_urls': result['discovered_urls'],
#                         'completed_at': datetime.utcnow().isoformat()
#                     }
#                 }
#             }).eq('id', data_source_id).execute()
            
#         else:
#             logger.error(f"Website crawl failed for {data_source_id}: {result.get('error', 'Unknown error')}")
            
#             # Update status to failed
#             db_client.client.table('data_sources').update({
#                 'status': DataSourceStatus.FAILED.value,
#                 'metadata': {
#                     'crawl_error': result.get('error', 'Unknown error'),
#                     'failed_at': datetime.utcnow().isoformat()
#                 }
#             }).eq('id', data_source_id).execute()
            
#     except Exception as e:
#         logger.error(f"Website crawl task failed for {data_source_id}: {e}")
        
#         # Update status to failed
#         db_client.client.table('data_sources').update({
#             'status': DataSourceStatus.FAILED.value,
#             'metadata': {
#                 'crawl_task_error': str(e),
#                 'failed_at': datetime.utcnow().isoformat()
#             }
#         }).eq('id', data_source_id).execute() 