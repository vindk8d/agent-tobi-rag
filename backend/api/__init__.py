"""
API routes and endpoints for the RAG-Tobi application
"""

from fastapi import APIRouter

# Import route modules (will be implemented in future tasks)
from .chat import router as chat_router
from .documents import router as documents_router
from .datasources import router as datasources_router
from .memory_debug import router as memory_debug_router
from .test_pdf_generation import router as pdf_test_router
from .quotations import router as quotations_router

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Register route modules (will be added when implemented)
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(datasources_router, prefix="/datasources", tags=["datasources"])
api_router.include_router(memory_debug_router, prefix="/memory-debug", tags=["memory-debug"])
api_router.include_router(pdf_test_router, tags=["pdf-testing"])
api_router.include_router(quotations_router, tags=["quotations"])

__version__ = "1.0.0"
