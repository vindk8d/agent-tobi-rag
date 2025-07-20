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
from models.conversation import ConversationRequest, ConversationResponse
from database import db_client
from api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


# Initialize the agent (will be lazily loaded)
_agent_instance = None

async def get_agent():
    """Get or create the agent instance"""
    global _agent_instance
    if _agent_instance is None:
        from agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
        _agent_instance = UnifiedToolCallingRAGAgent()
    return _agent_instance

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Chat endpoint using the full RAG agent with memory system"""
    try:
        # Get the agent instance
        agent = await get_agent()
        
        # Use the provided conversation_id or generate a new one
        conversation_id = request.conversation_id
        if not conversation_id:
            from uuid import uuid4
            conversation_id = str(uuid4())
        
        logger.info(f"Processing chat message for conversation {conversation_id}")
        
        # Invoke the agent with memory system
        result = await agent.invoke(
            user_query=request.message,
            conversation_id=conversation_id,
            user_id=request.user_id
        )
        
        # Extract the final AI message from the result
        final_message = ""
        sources = []
        
        if result and 'messages' in result:
            # Get the last AI message
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and msg.content and not msg.content.startswith('['):
                    final_message = msg.content
                    break
        
        # Extract sources if available
        if result and 'retrieved_docs' in result:
            retrieved_docs = result.get('retrieved_docs', [])
            for doc in retrieved_docs[:3]:  # Limit to top 3
                if isinstance(doc, dict):
                    sources.append({
                        'content': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                        'metadata': doc.get('metadata', {})
                    })
        
        return ConversationResponse(
            message=final_message or "I apologize, but I couldn't generate a proper response. Please try again.",
            conversation_id=conversation_id,
            response_metadata={"status": "success", "agent_used": "UnifiedToolCallingRAGAgent"},
            sources=sources,
            suggestions=["Ask me about your documents", "Upload some files to get started"],
            confidence_score=0.9
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Return error response but don't crash
        return ConversationResponse(
            message=f"I'm experiencing some technical difficulties. Please try again. Error: {str(e)}",
            conversation_id=request.conversation_id or "error-conversation-id",
            response_metadata={"status": "error"},
            sources=[],
            suggestions=["Try rephrasing your question", "Check if the system is running properly"],
            confidence_score=0.0
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 