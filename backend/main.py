"""
Main FastAPI application for RAG-Tobi
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from contextlib import asynccontextmanager

# Import models and responses
from models.base import APIResponse
from core.database import db_client
from api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly set agents.tools logger to INFO level for debugging customer lookup issues
logging.getLogger('agents.tools').setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG-Tobi API...")

    # Test database connection
    try:
        if db_client.health_check():
            logger.info("Database connection established")
        else:
            logger.warning("Database connection failed")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    yield

    # Shutdown
    logger.info("Shutting down RAG-Tobi API...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG-Tobi API",
    description="Salesperson Copilot API with RAG functionality",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    try:
        db_healthy = db_client.health_check()
        return APIResponse.success_response(
            data={
                "service": "RAG-Tobi API",
                "status": "healthy",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "database": "healthy" if db_healthy else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return APIResponse.error_response(f"Health check failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information"""
    return APIResponse.success_response(
        data={
            "service": "RAG-Tobi API",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "features": [
                "RAG (Retrieval-Augmented Generation)",
                "Document Processing",
                "Web Scraping",
                "Conversation Management",
                "LangGraph Agent Architecture"
            ]
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
