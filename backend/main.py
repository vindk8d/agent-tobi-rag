"""
Main FastAPI application for RAG-Tobi
"""

import sys
import os
from pathlib import Path

# Smart path setup for deployment compatibility
# Development: running from /backend directory, need to add current dir to path
# Production: running from /app directory, current dir is already in path
current_dir = Path(__file__).parent

# In development, we're in /project-root/backend/
# In production, we're in /app/
# Add current directory to path so imports like 'from core.config' work in both contexts
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import asyncio
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
    """Application lifespan manager with Railway optimizations"""
    # Startup
    logger.info("Starting RAG-Tobi API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    # Test database connection with retries for Railway
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if db_client.health_check():
                logger.info("Database connection established successfully")
                break
            else:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Database initialization error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Failed to establish database connection after all retries")

    # Log successful startup
    logger.info("RAG-Tobi API startup completed")

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

# Configure CORS for Railway deployment
allowed_origins = [
    "http://localhost:3000",  # Local development
    "http://localhost:3001",  # Alternative local port
    "https://tobi-frontend-production.up.railway.app",  # Production frontend
]

# Add environment-specific origins
if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

if os.getenv("ENVIRONMENT") == "production":
    # In production, keep specific Railway URLs and remove localhost
    production_origins = [
        "https://tobi-frontend-production.up.railway.app"
    ]
    if os.getenv("FRONTEND_URL"):
        production_origins.append(os.getenv("FRONTEND_URL"))
    allowed_origins = production_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for Railway and monitoring"""
    import time
    start_time = time.time()
    
    try:
        # Check database connection
        db_healthy = db_client.health_check()
        
        # Basic system checks
        checks = {
            "database": "healthy" if db_healthy else "unhealthy",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "port": os.getenv("PORT", "8000"),
            "python_path": os.getenv("PYTHONPATH", "not_set"),
        }
        
        # Calculate response time
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Overall health status
        overall_healthy = db_healthy
        status_code = 200 if overall_healthy else 503
        
        response_data = {
            "service": "RAG-Tobi API",
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": int(time.time()),
            "response_time_ms": response_time,
            "checks": checks,
            "version": "1.0.0"
        }
        
        if overall_healthy:
            return APIResponse.success_response(data=response_data)
        else:
            from fastapi import Response
            return Response(
                content=APIResponse.error_response("Service unhealthy", response_data).json(),
                status_code=status_code,
                media_type="application/json"
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        from fastapi import Response
        return Response(
            content=APIResponse.error_response(f"Health check failed: {str(e)}").json(),
            status_code=503,
            media_type="application/json"
        )


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
