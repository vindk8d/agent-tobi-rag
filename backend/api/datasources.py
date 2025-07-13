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
from scrapers.web_scraper import WebScraper
from rag.pipeline import DocumentProcessingPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
settings = get_settings()
web_scraper = WebScraper()
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

class WebsiteCrawlRequest(BaseModel):
    """Request model for crawling an entire website"""
    url: HttpUrl = Field(..., description="Base URL to start crawling from")
    data_source_name: Optional[str] = Field(None, description="Name for the data source (defaults to domain name)")
    max_pages: int = Field(50, ge=1, le=200, description="Maximum number of pages to crawl")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum crawl depth")
    delay: float = Field(1.0, ge=0.1, le=10.0, description="Delay between requests in seconds")
    include_patterns: Optional[List[str]] = Field(None, description="URL patterns to include (e.g., ['/docs/', '/api/'])")
    exclude_patterns: Optional[List[str]] = Field(None, description="URL patterns to exclude (e.g., ['/admin/', '.pdf'])")

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

async def test_scraping_capability(url: str) -> Dict[str, Any]:
    """Test if a URL can be scraped and return basic metrics"""
    try:
        # Test scraping with the web scraper
        result = await web_scraper.scrape_url(url)
        
        if result['success']:
            return {
                'success': True,
                'content_length': len(result.get('content', '')),
                'title': result.get('title', ''),
                'scraping_method': result.get('method', 'unknown'),
                'response_time': result.get('response_time', 0)
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'status_code': result.get('status_code')
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@router.post("/", response_model=APIResponse)
async def create_data_source(
    request: DataSourceCreateRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new data source and optionally test it immediately.
    """
    try:
        # Validate URL
        if not validate_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Check if URL already exists
        existing = db_client.client.table('data_sources').select('id').eq('url', str(request.url)).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="Data source with this URL already exists")
        
        # Create data source record
        data_source_id = str(uuid4())
        data_source_data = {
            'id': data_source_id,
            'name': request.name,
            'source_type': DataSourceType.WEBSITE.value,
            'url': str(request.url),
            'status': DataSourceStatus.PENDING.value,
            'scraping_frequency': request.scraping_frequency.value,
            'configuration': request.configuration or {},
            'metadata': {
                'created_via': 'api',
                'created_at': datetime.utcnow().isoformat()
            }
        }
        
        if request.description:
            data_source_data['description'] = request.description
        
        # Insert into database
        result = db_client.client.table('data_sources').insert(data_source_data).execute()
        
        if result.data:
            # Test scraping capability in background
            background_tasks.add_task(initial_scrape_test, data_source_id, str(request.url))
            
            return APIResponse.success_response(
                data={
                    "data_source_id": data_source_id,
                    "name": request.name,
                    "url": str(request.url),
                    "status": DataSourceStatus.PENDING.value,
                    "message": "Data source created successfully. Testing scraping capability..."
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
    """
    Background task to test scraping capability and update status.
    """
    try:
        logger.info(f"Testing scraping capability for data source {data_source_id}: {url}")
        
        # Update status to processing
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.ACTIVE.value,
            'last_scraped_at': datetime.utcnow().isoformat()
        }).eq('id', data_source_id).execute()
        
        # Test scraping
        scrape_result = await test_scraping_capability(url)
        
        if scrape_result['success']:
            # Update with success
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ACTIVE.value,
                'last_success': datetime.utcnow().isoformat(),
                'metadata': {
                    'scraping_test': scrape_result,
                    'last_test': datetime.utcnow().isoformat()
                }
            }).eq('id', data_source_id).execute()
            
            logger.info(f"Data source {data_source_id} scraping test successful")
            
            # Optionally process the scraped content immediately
            await process_data_source_content(data_source_id, url)
            
        else:
            # Update with error
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ERROR.value,
                'last_error': scrape_result.get('error', 'Scraping test failed'),
                'metadata': {
                    'scraping_test': scrape_result,
                    'last_test': datetime.utcnow().isoformat()
                }
            }).eq('id', data_source_id).execute()
            
            logger.error(f"Data source {data_source_id} scraping test failed: {scrape_result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error in initial scrape test for {data_source_id}: {e}")
        
        # Update status to error
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.ERROR.value,
            'last_error': str(e),
            'metadata': {
                'test_error': str(e),
                'last_test': datetime.utcnow().isoformat()
            }
        }).eq('id', data_source_id).execute()

async def process_data_source_content(data_source_id: str, url: str):
    """
    Process scraped content through the RAG pipeline.
    The pipeline will automatically create document chunks and embeddings.
    """
    try:
        # Get data source info for metadata
        data_source_result = db_client.client.table('data_sources').select('name').eq('id', data_source_id).execute()
        data_source_name = data_source_result.data[0]['name'] if data_source_result.data else 'Unknown Source'
        
        logger.info(f"Processing content for data source {data_source_id}: {url}")
        
        # Process the URL content using the fixed pipeline
        # The pipeline will find the existing data source by URL
        result = await pipeline.process_url(url, data_source_name=data_source_name)
        
        if result.get('success'):
            # Update data source with processing results
            db_client.client.table('data_sources').update({
                'document_count': result.get('chunks', 0),
                'last_success': datetime.utcnow().isoformat(),
                'status': DataSourceStatus.ACTIVE.value,
                'metadata': {
                    'last_processing_result': {
                        'chunks_created': result.get('chunks', 0),
                        'stored_chunk_ids': result.get('stored_chunk_ids', []),
                        'embedding_ids': result.get('embedding_ids', []),
                        'processed_at': datetime.utcnow().isoformat()
                    }
                }
            }).eq('id', data_source_id).execute()
            
            logger.info(f"Successfully processed {result.get('chunks', 0)} chunks for data source {data_source_id}")
        else:
            # Update data source with error
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ERROR.value,
                'last_error': result.get('error', 'Processing failed'),
                'metadata': {
                    'last_processing_error': {
                        'error': result.get('error', 'Processing failed'),
                        'failed_at': datetime.utcnow().isoformat()
                    }
                }
            }).eq('id', data_source_id).execute()
            
            logger.error(f"Processing failed for data source {data_source_id}: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error processing content for data source {data_source_id}: {e}")
        
        # Update data source with error
        try:
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ERROR.value,
                'last_error': str(e),
                'metadata': {
                    'processing_exception': {
                        'error': str(e),
                        'failed_at': datetime.utcnow().isoformat()
                    }
                }
            }).eq('id', data_source_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update data source status after error: {db_error}")

@router.get("/", response_model=APIResponse)
async def list_data_sources(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: Optional[DataSourceStatus] = Query(None),
    source_type: Optional[DataSourceType] = Query(None)
):
    """
    List data sources with pagination and filtering.
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Build query
        query = db_client.client.table('data_sources').select('*')
        
        # Apply filters
        if status:
            query = query.eq('status', status.value)
        if source_type:
            query = query.eq('source_type', source_type.value)
        
        # Execute query with pagination
        result = query.range(offset, offset + page_size - 1).order('created_at', desc=True).execute()
        
        # Get total count
        count_query = db_client.client.table('data_sources').select('id', count='exact')
        if status:
            count_query = count_query.eq('status', status.value)
        if source_type:
            count_query = count_query.eq('source_type', source_type.value)
        
        count_result = count_query.execute()
        total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
        
        return APIResponse.success_response(
            data={
                "data_sources": result.data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list data sources: {str(e)}")

@router.get("/{data_source_id}", response_model=APIResponse)
async def get_data_source(data_source_id: str):
    """
    Get a specific data source by ID.
    """
    try:
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        
        if result.data:
            data_source = result.data[0]
            
            # Get related documents count
            docs_result = db_client.client.table('documents').select('id', count='exact').eq('data_source_id', data_source_id).execute()
            document_count = docs_result.count if hasattr(docs_result, 'count') else 0
            
            data_source['document_count'] = document_count
            
            return APIResponse.success_response(data=data_source)
        else:
            raise HTTPException(status_code=404, detail="Data source not found")
            
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
    """
    Update a data source.
    """
    try:
        # Check if data source exists
        existing = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Prepare update data
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
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        # Update in database
        result = db_client.client.table('data_sources').update(update_data).eq('id', data_source_id).execute()
        
        if result.data:
            return APIResponse.success_response(
                data={
                    "data_source_id": data_source_id,
                    "message": "Data source updated successfully"
                }
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
    """
    Delete a data source and all related documents.
    """
    try:
        # Check if data source exists
        existing = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Delete related documents and their embeddings
        docs_result = db_client.client.table('documents').select('id').eq('data_source_id', data_source_id).execute()
        for doc in docs_result.data:
            # Delete embeddings
            db_client.client.table('embeddings').delete().eq('document_id', doc['id']).execute()
        
        # Delete documents
        db_client.client.table('documents').delete().eq('data_source_id', data_source_id).execute()
        
        # Delete data source
        result = db_client.client.table('data_sources').delete().eq('id', data_source_id).execute()
        
        if result.data:
            return APIResponse.success_response(
                data={
                    "data_source_id": data_source_id,
                    "message": "Data source and related documents deleted successfully"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete data source")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data source {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data source: {str(e)}")

@router.post("/test", response_model=APIResponse)
async def test_data_source_url(request: DataSourceTestRequest):
    """
    Test if a URL can be scraped without creating a data source.
    """
    try:
        # Validate URL
        if not validate_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Test scraping
        result = await test_scraping_capability(str(request.url))
        
        return APIResponse.success_response(
            data={
                "url": str(request.url),
                "test_result": result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing URL {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test URL: {str(e)}")

@router.post("/crawl", response_model=APIResponse)
async def crawl_website(
    request: WebsiteCrawlRequest,
    background_tasks: BackgroundTasks
):
    """
    Crawl an entire website starting from the base URL.
    Discovers and processes all child links and related pages.
    """
    try:
        # Validate URL
        if not validate_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Generate data source name if not provided
        data_source_name = request.data_source_name
        if not data_source_name:
            from urllib.parse import urlparse
            parsed_url = urlparse(str(request.url))
            data_source_name = f"{parsed_url.netloc} Website Crawl"
        
        # Check if URL already exists as a data source
        existing = db_client.client.table('data_sources').select('id').eq('url', str(request.url)).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="Data source with this URL already exists")
        
        # Create data source record
        data_source_id = str(uuid4())
        data_source_data = {
            'id': data_source_id,
            'name': data_source_name,
            'source_type': DataSourceType.WEBSITE.value,
            'url': str(request.url),
            'status': DataSourceStatus.ACTIVE.value,
            'scraping_frequency': ScrapingFrequency.MANUAL.value,
            'configuration': {
                'crawling_enabled': True,
                'max_pages': request.max_pages,
                'max_depth': request.max_depth,
                'delay': request.delay,
                'include_patterns': request.include_patterns,
                'exclude_patterns': request.exclude_patterns
            },
            'metadata': {
                'created_via': 'api_crawl',
                'created_at': datetime.utcnow().isoformat(),
                'crawl_request': {
                    'max_pages': request.max_pages,
                    'max_depth': request.max_depth,
                    'delay': request.delay,
                    'include_patterns': request.include_patterns,
                    'exclude_patterns': request.exclude_patterns
                }
            }
        }
        
        # Insert into database
        result = db_client.client.table('data_sources').insert(data_source_data).execute()
        
        if result.data:
            # Start crawling in background
            background_tasks.add_task(
                crawl_website_task,
                data_source_id,
                str(request.url),
                data_source_name,
                request.max_pages,
                request.max_depth,
                request.delay,
                request.include_patterns,
                request.exclude_patterns
            )
            
            return APIResponse.success_response(
                data={
                    "data_source_id": data_source_id,
                    "name": data_source_name,
                    "url": str(request.url),
                    "status": DataSourceStatus.ACTIVE.value,
                    "crawl_config": {
                        "max_pages": request.max_pages,
                        "max_depth": request.max_depth,
                        "delay": request.delay,
                        "include_patterns": request.include_patterns,
                        "exclude_patterns": request.exclude_patterns
                    },
                    "message": "Website crawl started successfully. This may take several minutes..."
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create data source for crawling")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting website crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start crawl: {str(e)}")

async def crawl_website_task(
    data_source_id: str,
    base_url: str,
    data_source_name: str,
    max_pages: int,
    max_depth: int,
    delay: float,
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]]
):
    """
    Background task for crawling an entire website.
    """
    try:
        logger.info(f"Starting website crawl for data source {data_source_id}: {base_url}")
        
        # Update status to processing
        db_client.client.table('data_sources').update({
            'status': DataSourceStatus.ACTIVE.value,
            'last_scraped_at': datetime.utcnow().isoformat()
        }).eq('id', data_source_id).execute()
        
        # Process website using the enhanced pipeline
        result = await pipeline.process_website(
            base_url=base_url,
            data_source_name=data_source_name,
            max_pages=max_pages,
            max_depth=max_depth,
            delay=delay,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns
        )
        
        if result['success']:
            # Update with success
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ACTIVE.value,
                'last_success': datetime.utcnow().isoformat(),
                'error_count': 0,
                'last_error': None,
                'metadata': {
                    'created_via': 'api_crawl',
                    'created_at': datetime.utcnow().isoformat(),
                    'crawl_results': {
                        'processed_pages': result.get('processed_pages', 0),
                        'total_pages_found': result.get('total_pages_found', 0),
                        'total_chunks': result.get('total_chunks', 0),
                        'crawl_time': result.get('crawl_time', 0),
                        'discovered_urls': result.get('discovered_urls', 0)
                    }
                }
            }).eq('id', data_source_id).execute()
            
            logger.info(f"Website crawl completed for data source {data_source_id}: "
                       f"{result.get('processed_pages', 0)} pages processed, "
                       f"{result.get('total_chunks', 0)} chunks created")
            
        else:
            # Update with error
            error_msg = result.get('error', 'Unknown crawling error')
            db_client.client.table('data_sources').update({
                'status': DataSourceStatus.ERROR.value,
                'last_error': error_msg,
                'error_count': 1
            }).eq('id', data_source_id).execute()
            
            logger.error(f"Website crawl failed for data source {data_source_id}: {error_msg}")
            
    except Exception as e:
        logger.error(f"Website crawl failed for data source {data_source_id}: {e}")
        
        # Update error information
        try:
            current_data = db_client.client.table('data_sources').select('error_count').eq('id', data_source_id).execute()
            current_error_count = current_data.data[0]['error_count'] if current_data.data else 0
            
            db_client.client.table('data_sources').update({
                'last_error': str(e),
                'error_count': current_error_count + 1,
                'status': DataSourceStatus.ERROR.value
            }).eq('id', data_source_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update error information: {db_error}")

@router.post("/{data_source_id}/scrape", response_model=APIResponse)
async def trigger_scrape(
    data_source_id: str,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger scraping for a data source.
    """
    try:
        # Check if data source exists
        result = db_client.client.table('data_sources').select('*').eq('id', data_source_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        data_source = result.data[0]
        
        # Trigger scraping in background
        background_tasks.add_task(
            manual_scrape_task,
            data_source_id,
            data_source['url']
        )
        
        return APIResponse.success_response(
            data={
                "data_source_id": data_source_id,
                "message": "Scraping triggered successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering scrape for {data_source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger scrape: {str(e)}")

async def manual_scrape_task(data_source_id: str, url: str):
    """
    Background task for manual scraping.
    """
    try:
        logger.info(f"Manual scraping triggered for data source {data_source_id}: {url}")
        
        # Update last scraped time
        db_client.client.table('data_sources').update({
            'last_scraped_at': datetime.utcnow().isoformat()
        }).eq('id', data_source_id).execute()
        
        # Process content (this will now create document records properly)
        await process_data_source_content(data_source_id, url)
        
        # Update success time
        db_client.client.table('data_sources').update({
            'last_success': datetime.utcnow().isoformat(),
            'error_count': 0,
            'last_error': None
        }).eq('id', data_source_id).execute()
        
        logger.info(f"Manual scraping completed for data source {data_source_id}")
        
    except Exception as e:
        logger.error(f"Manual scraping failed for data source {data_source_id}: {e}")
        
        # Update error information - get current error count safely
        try:
            current_data = db_client.client.table('data_sources').select('error_count').eq('id', data_source_id).execute()
            current_error_count = current_data.data[0]['error_count'] if current_data.data else 0
            
            db_client.client.table('data_sources').update({
                'last_error': str(e),
                'error_count': current_error_count + 1,
                'status': DataSourceStatus.ERROR.value
            }).eq('id', data_source_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update error information: {db_error}")

@router.get("/stats/overview", response_model=APIResponse)
async def get_data_source_stats():
    """
    Get overview statistics for all data sources.
    """
    try:
        # Get counts by status
        stats = {}
        for status in DataSourceStatus:
            count_result = db_client.client.table('data_sources').select('id', count='exact').eq('status', status.value).execute()
            stats[status.value] = count_result.count if hasattr(count_result, 'count') else 0
        
        # Get total documents processed
        total_docs = db_client.client.table('documents').select('id', count='exact').execute()
        total_documents = total_docs.count if hasattr(total_docs, 'count') else 0
        
        # Get recent scraping activity
        recent_scrapes = db_client.client.table('data_sources').select('id, name, last_scraped_at, last_success').order('last_scraped_at', desc=True).limit(5).execute()
        
        return APIResponse.success_response(
            data={
                "status_counts": stats,
                "total_documents": total_documents,
                "recent_activity": recent_scrapes.data
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting data source stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}") 