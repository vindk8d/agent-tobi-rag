"""
Data source models for website and document source management
"""

from typing import List, Optional, Dict, Any
from pydantic import Field, HttpUrl
from enum import Enum
from datetime import datetime
from uuid import UUID

from .base import BaseModel, IdentifiedModel


class DataSourceType(str, Enum):
    """Type of data source"""
    WEBSITE = "website"
    FILE_UPLOAD = "file_upload"
    API = "api"
    DATABASE = "database"


class DataSourceStatus(str, Enum):
    """Status of data source"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class ScrapingFrequency(str, Enum):
    """Frequency of web scraping"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MANUAL = "manual"


class DataSourceModel(IdentifiedModel):
    """Data source model for database storage"""
    
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    source_type: DataSourceType
    url: Optional[HttpUrl] = None
    status: DataSourceStatus = DataSourceStatus.PENDING
    scraping_frequency: ScrapingFrequency = ScrapingFrequency.DAILY
    last_scraped: Optional[datetime] = None
    last_success: Optional[datetime] = None
    document_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    

class DataSourceRequest(BaseModel):
    """Request model for creating/updating data sources"""
    
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    source_type: DataSourceType
    url: Optional[HttpUrl] = None
    scraping_frequency: ScrapingFrequency = ScrapingFrequency.DAILY
    configuration: Optional[Dict[str, Any]] = None
    

class DataSourceResponse(BaseModel):
    """Response model for data source operations"""
    
    data_source: DataSourceModel
    message: str = "Data source operation completed successfully"
    

class DataSourceListResponse(BaseModel):
    """Response model for data source listing"""
    
    data_sources: List[DataSourceModel]
    total_count: int
    page: int = 1
    page_size: int = 10
    

class ScrapingResult(BaseModel):
    """Result of a scraping operation"""
    
    data_source_id: UUID
    success: bool
    documents_found: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    error_message: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = None  # seconds 