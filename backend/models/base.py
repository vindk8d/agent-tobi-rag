"""
Base model classes for the RAG-Tobi application
"""

from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Optional, Dict, Any, Generic, TypeVar, List
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

T = TypeVar('T')


class BaseModel(PydanticBaseModel):
    """Base model with common functionality"""

    class Config:
        # Enable validation on assignment
        validate_assignment = True
        # Use enum values in JSON serialization
        use_enum_values = True
        # Allow population by field name (Pydantic v2 syntax)
        validate_by_name = True
        # JSON encoders for custom types
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TimestampedModel(BaseModel):
    """Base model with timestamp fields"""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()


class IdentifiedModel(TimestampedModel):
    """Base model with ID and timestamps"""

    id: UUID = Field(default_factory=uuid4)


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def success_response(cls, data: Optional[T] = None, message: str = "Success", metadata: Optional[Dict[str, Any]] = None) -> "APIResponse[T]":
        return cls(
            success=True,
            message=message,
            data=data,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_response(cls, message: str, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "APIResponse[T]":
        return cls(
            success=False,
            message=message,
            error=error,
            metadata=metadata or {}
        )

class ConfirmationStatus(str, Enum):
    """Status values for confirmation requests"""
    PENDING = "pending"
    APPROVED = "approved"
    CANCELLED = "cancelled"
    MODIFIED = "modified"
    TIMEOUT = "timeout"

class ConfirmationRequest(BaseModel):
    """Model for customer message confirmation requests"""
    confirmation_id: str = Field(..., description="Unique confirmation ID")
    customer_id: str = Field(..., description="Target customer ID")
    customer_name: str = Field(..., description="Customer name for display")
    customer_email: Optional[str] = Field(None, description="Customer email")
    message_content: str = Field(..., min_length=1, max_length=2000, description="Message to send")
    message_type: str = Field(default="follow_up", description="Type of message")
    requested_by: str = Field(..., description="Employee requesting the message")
    requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    status: ConfirmationStatus = Field(default=ConfirmationStatus.PENDING, description="Current status")
    conversation_id: str = Field(..., description="Associated conversation ID")
    
    @classmethod
    def create_new(cls, customer_id: str, customer_name: str, message_content: str, 
                   requested_by: str, conversation_id: str, customer_email: Optional[str] = None,
                   message_type: str = "follow_up", timeout_minutes: int = 5) -> "ConfirmationRequest":
        """Create a new confirmation request with auto-generated ID and expiration"""
        from uuid import uuid4
        
        confirmation_id = str(uuid4())
        expires_at = datetime.utcnow() + timedelta(minutes=timeout_minutes)
        
        return cls(
            confirmation_id=confirmation_id,
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=customer_email,
            message_content=message_content,
            message_type=message_type,
            requested_by=requested_by,
            expires_at=expires_at,
            conversation_id=conversation_id
        )

class ConfirmationResponse(BaseModel):
    """Model for confirmation responses from users"""
    confirmation_id: str = Field(..., description="Confirmation ID being responded to")
    action: ConfirmationStatus = Field(..., description="User's decision")
    modified_message: Optional[str] = Field(None, description="Modified message if action is MODIFIED")
    responded_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    notes: Optional[str] = Field(None, description="Optional notes from the user")

class DeliveryResult(BaseModel):
    """Model for message delivery results"""
    confirmation_id: str = Field(..., description="Associated confirmation ID")
    delivery_status: str = Field(..., description="Delivery status (success/failure)")
    delivery_message: str = Field(..., description="Delivery status message")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    delivery_details: Dict[str, Any] = Field(default_factory=dict, description="Additional delivery details")
