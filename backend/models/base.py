"""
Base model classes for the RAG-Tobi application
"""

from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Optional, Dict, Any, Generic, TypeVar
from datetime import datetime
from uuid import UUID, uuid4

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
    """Standard API response format"""

    success: bool = True
    message: str = "Success"
    data: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def success_response(cls, data: Optional[T] = None, message: str = "Success"):
        return cls(success=True, message=message, data=data)

    @classmethod
    def error_response(cls, error: str, message: str = "Error"):
        return cls(success=False, message=message, error=error)
