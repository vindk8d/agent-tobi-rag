"""
Document models for file and content management
"""

from typing import List, Optional, Dict, Any
from pydantic import Field, HttpUrl
from enum import Enum
from uuid import UUID

from .base import BaseModel, IdentifiedModel


class DocumentType(str, Enum):
    """Type of document"""
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    WEB_PAGE = "web_page"
    VEHICLE_SPECIFICATION = "vehicle_specification"


class DocumentStatus(str, Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INDEXED = "indexed"


class DocumentModel(IdentifiedModel):
    """Document model for database storage"""

    title: str = Field(..., min_length=1, max_length=500)
    content: Optional[str] = None
    document_type: DocumentType
    file_path: Optional[str] = None
    url: Optional[HttpUrl] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    status: DocumentStatus = DocumentStatus.PENDING
    embedding_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    data_source_id: Optional[UUID] = None
    vehicle_id: Optional[UUID] = None  # Added for vehicle specification documents


class DocumentChunk(IdentifiedModel):
    """Document chunk model for vector storage"""

    document_id: UUID
    chunk_index: int
    content: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    page_number: Optional[int] = None
    vehicle_id: Optional[UUID] = None  # Added for vehicle specification chunks


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""

    title: str = Field(..., min_length=1, max_length=500)
    document_type: DocumentType
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""

    document_id: UUID
    upload_url: Optional[str] = None
    status: DocumentStatus
    message: str = "Document uploaded successfully"


class DocumentListResponse(BaseModel):
    """Response model for document listing"""

    documents: List[DocumentModel]
    total_count: int
    page: int = 1
    page_size: int = 10


# Vehicle-specific models

class VehicleModel(BaseModel):
    """Vehicle model for API responses"""
    
    id: UUID
    brand: str
    model: str
    year: int
    type: Optional[str] = None
    variant: Optional[str] = None
    key_features: Optional[List[str]] = None
    is_available: bool = True


class VehicleListResponse(BaseModel):
    """Response model for vehicle listing"""
    
    vehicles: List[VehicleModel]
    total_count: int


class VehicleSpecificationRequest(BaseModel):
    """Request model for vehicle specification upload"""
    
    file_path: str
    file_name: str
    file_size: Optional[int] = None
    vehicle_id: UUID


class VehicleSpecificationResponse(BaseModel):
    """Response model for vehicle specification operations"""
    
    data_source_id: Optional[UUID] = None
    vehicle_id: UUID
    status: DocumentStatus
    message: str
    document: Optional[Dict[str, Any]] = None
