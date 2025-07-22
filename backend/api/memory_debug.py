"""
Memory Debug API endpoints for debugging and monitoring memory system
"""

import logging
import json
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from database import db_client
from models.base import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Response models for memory debug endpoints
class UserInfo(BaseModel):
    """User information for memory debug"""
    id: str
    email: str
    name: Optional[str] = None
    role: Optional[str] = None
    created_at: str


class LongTermMemoryItem(BaseModel):
    """Long-term memory item structure"""
    id: str
    namespace: List[str]
    key: str
    value: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    memory_type: str
    source_thread_id: Optional[str] = None
    expiry_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationSummaryItem(BaseModel):
    """Conversation summary structure"""
    id: str
    conversation_id: str
    user_id: str
    summary_text: str
    summary_type: str
    message_count: int
    start_message_id: Optional[str] = None
    end_message_id: Optional[str] = None
    created_at: str
    consolidation_status: str
    metadata: Optional[Dict[str, Any]] = None


class DatabaseMessage(BaseModel):
    """Database message structure"""
    id: str
    conversation_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class UserSummary(BaseModel):
    """User summary based on latest conversation summary"""
    user_id: str
    latest_summary: str
    conversation_count: int
    has_history: bool
    latest_conversation_id: Optional[str] = None


class CustomerData(BaseModel):
    """CRM customer data structure"""
    id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    status: str
    branch_id: str
    warmth_score: Optional[float] = None
    created_at: str


class MemoryAccessPattern(BaseModel):
    """Memory access pattern structure"""
    id: str
    user_id: str
    memory_namespace: List[str]
    memory_key: Optional[str] = None
    access_frequency: int
    last_accessed_at: str
    context_relevance: float
    access_context: Optional[str] = None
    retrieval_method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryConsolidationRequest(BaseModel):
    """Request to trigger memory consolidation"""
    user_id: str
    force: bool = False


class MemorySearchRequest(BaseModel):
    """Request to search memories"""
    query: str
    user_id: Optional[str] = None
    memory_type: Optional[str] = None
    similarity_threshold: float = 0.7
    limit: int = 10

@router.get("/users", response_model=APIResponse[List[UserInfo]])
async def get_users():
    """Get list of users for memory debugging"""
    try:
        # Try to get users from users table first
        try:
            result = db_client.client.table('users').select('*').order('created_at', desc=True).execute()
            users_data = result.data or []

            users = [
                UserInfo(
                    id=user['id'],
                    email=user.get('email', ''),
                    name=user.get('display_name'),
                    role=user.get('user_type', 'user'),
                    created_at=user['created_at']
                )
                for user in users_data
            ]
        except Exception as users_error:
            logger.warning(f"Users table not found or error: {users_error}, falling back to customers")

            # Fallback to customers table
            result = db_client.client.table('customers').select('*').order('created_at', desc=True).limit(20).execute()
            customers_data = result.data or []

            users = [
                UserInfo(
                    id=customer['id'],
                    email=customer.get('email', f"customer_{customer['id']}"),
                    name=customer.get('name', 'Unnamed Customer'),
                    role='customer',
                    created_at=customer['created_at']
                )
                for customer in customers_data
            ]

        return APIResponse(
            success=True,
            message="Users retrieved successfully",
            data=users
        )
    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve users: {str(e)}")

@router.get("/users/{user_id}/crm", response_model=APIResponse[Optional[CustomerData]])
async def get_user_crm_data(user_id: str):
    """Get CRM data for a specific user"""
    try:
        # First, get the user's customer_id from the users table
        user_result = db_client.client.table('users').select('customer_id').eq('id', user_id).execute()

        if not user_result.data or not user_result.data[0].get('customer_id'):
            return APIResponse(
                success=True,
                message="User is not a customer or no CRM data found",
                data=None
            )

        customer_id = user_result.data[0]['customer_id']

        # Now get the customer data using the correct customer_id
        result = db_client.client.table('customers').select('*').eq('id', customer_id).execute()

        if not result.data:
            return APIResponse(
                success=True,
                message="No CRM data found for user",
                data=None
            )

        customer = result.data[0]
        customer_data = CustomerData(
            id=customer['id'],
            first_name=customer.get('name', 'Unknown'),
            last_name='',
            email=customer.get('email'),
            phone=customer.get('phone'),
            status=customer.get('status', 'active'),
            branch_id=customer.get('branch_id', ''),
            warmth_score=customer.get('warmth_score'),
            created_at=customer['created_at']
        )

        return APIResponse(
            success=True,
            message="CRM data retrieved successfully",
            data=customer_data
        )
    except Exception as e:
        logger.error(f"Error retrieving CRM data for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve CRM data: {str(e)}")

@router.get("/users/{user_id}/memories", response_model=APIResponse[List[LongTermMemoryItem]])
async def get_user_long_term_memories(user_id: str):
    """Get long-term memories for a specific user"""
    try:
        # Query memories where user_id is in the namespace array
        result = db_client.client.table('long_term_memories').select('*').or_(
            f'namespace.cs.{{{user_id}}},namespace.cs.{{user,{user_id}}}'
        ).order('accessed_at', desc=True).execute()

        memories = []
        for memory in result.data or []:
            # Parse JSON fields that might be stored as strings
            try:
                value = memory['value']
                if isinstance(value, str):
                    value = json.loads(value)

                embedding = memory.get('embedding')
                if embedding and isinstance(embedding, str):
                    embedding = json.loads(embedding)

                metadata = memory.get('metadata')
                if metadata and isinstance(metadata, str):
                    metadata = json.loads(metadata)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON fields for memory {memory['id']}: {e}")
                value = memory['value'] if not isinstance(memory['value'], str) else {}
                embedding = None
                metadata = memory.get('metadata')

            memories.append(LongTermMemoryItem(
                id=memory['id'],
                namespace=memory['namespace'],
                key=memory['key'],
                value=value,
                embedding=embedding,
                created_at=memory['created_at'],
                updated_at=memory['updated_at'],
                accessed_at=memory['accessed_at'],
                access_count=memory['access_count'],
                memory_type=memory['memory_type'],
                source_thread_id=memory.get('source_thread_id'),
                expiry_at=memory.get('expiry_at'),
                metadata=metadata
            ))

        return APIResponse(
            success=True,
            message="Long-term memories retrieved successfully",
            data=memories
        )
    except Exception as e:
        logger.error(f"Error retrieving long-term memories for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@router.get("/users/{user_id}/conversation-summaries", response_model=APIResponse[List[ConversationSummaryItem]])
async def get_user_conversation_summaries(user_id: str):
    """Get conversation summaries for a specific user"""
    try:
        result = db_client.client.table('conversation_summaries').select('*').eq(
            'user_id', user_id
        ).order('created_at', desc=True).execute()

        summaries = []
        for summary in result.data or []:
            summaries.append(ConversationSummaryItem(
                id=summary['id'],
                conversation_id=summary['conversation_id'],
                user_id=summary['user_id'],
                summary_text=summary['summary_text'],
                summary_type=summary['summary_type'],
                message_count=summary['message_count'],
                start_message_id=summary.get('start_message_id'),
                end_message_id=summary.get('end_message_id'),
                created_at=summary['created_at'],
                consolidation_status=summary['consolidation_status'],
                metadata=summary.get('metadata')
            ))

        return APIResponse(
            success=True,
            message="Conversation summaries retrieved successfully",
            data=summaries
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation summaries for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve summaries: {str(e)}")

@router.get("/users/{user_id}/summary", response_model=APIResponse[UserSummary])
async def get_user_summary(user_id: str):
    """Get user summary based on latest conversation summary"""
    try:
        # Use the new database function to get user context
        result = db_client.client.rpc('get_user_context_from_conversations', {'target_user_id': user_id}).execute()

        if result.data and len(result.data) > 0:
            context = result.data[0]
            user_summary = UserSummary(
                user_id=user_id,
                latest_summary=context.get('latest_summary', ''),
                conversation_count=context.get('conversation_count', 0),
                has_history=context.get('has_history', False),
                latest_conversation_id=str(context.get('latest_conversation_id')) if context.get('latest_conversation_id') else None
            )
        else:
            user_summary = UserSummary(
                user_id=user_id,
                latest_summary="New user - no conversation history yet.",
                conversation_count=0,
                has_history=False,
                latest_conversation_id=None
            )

        return APIResponse(
            success=True,
            message="User summary retrieved successfully",
            data=user_summary
        )
    except Exception as e:
        logger.error(f"Error retrieving user summary for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user summary: {str(e)}")

@router.get("/users/{user_id}/messages", response_model=APIResponse[List[DatabaseMessage]])
async def get_user_messages(user_id: str, limit: int = Query(50, ge=1, le=200), conversation_id: Optional[str] = Query(None)):
    """Get recent messages for a specific user, optionally filtered by conversation_id"""
    try:
        if conversation_id:
            # If conversation_id is provided, get messages only from that conversation
            conversation_result = db_client.client.table('conversations').select('id').eq('id', conversation_id).eq('user_id', user_id).execute()
            if not conversation_result.data:
                return APIResponse(
                    success=True,
                    message="Conversation not found or doesn't belong to user",
                    data=[]
                )
            
            messages_result = db_client.client.table('messages').select('*').eq(
                'conversation_id', conversation_id
            ).order('created_at', asc=True).limit(limit).execute()
        else:
            # Get messages from all conversations for this user (original behavior)
            conversations_result = db_client.client.table('conversations').select('id').eq(
                'user_id', user_id
            ).order('created_at', desc=True).limit(5).execute()

            if not conversations_result.data:
                return APIResponse(
                    success=True,
                    message="No conversations found for user",
                    data=[]
                )

            conversation_ids = [conv['id'] for conv in conversations_result.data]

            messages_result = db_client.client.table('messages').select('*').in_(
                'conversation_id', conversation_ids
            ).order('created_at', desc=True).limit(limit).execute()

        messages = []
        for message in messages_result.data or []:
            messages.append(DatabaseMessage(
                id=message['id'],
                conversation_id=message['conversation_id'],
                role=message['role'],
                content=message['content'],
                created_at=message['created_at'],
                metadata=message.get('metadata')
            ))

        return APIResponse(
            success=True,
            message="Messages retrieved successfully",
            data=messages
        )
    except Exception as e:
        logger.error(f"Error retrieving messages for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")

@router.get("/conversations/{conversation_id}/messages", response_model=APIResponse[List[DatabaseMessage]])
async def get_conversation_messages(conversation_id: str, limit: int = Query(50, ge=1, le=200)):
    """Get messages for a specific conversation"""
    try:
        # Check if conversation exists
        conversation_result = db_client.client.table('conversations').select('id').eq('id', conversation_id).execute()
        if not conversation_result.data:
            return APIResponse(
                success=True,
                message="Conversation not found",
                data=[]
            )

        # Get messages from this specific conversation
        messages_result = db_client.client.table('messages').select('*').eq(
            'conversation_id', conversation_id
        ).order('created_at', asc=True).limit(limit).execute()

        messages = []
        for message in messages_result.data or []:
            messages.append(DatabaseMessage(
                id=message['id'],
                conversation_id=message['conversation_id'],
                role=message['role'],
                content=message['content'],
                created_at=message['created_at'],
                metadata=message.get('metadata')
            ))

        return APIResponse(
            success=True,
            message="Messages retrieved successfully",
            data=messages
        )
    except Exception as e:
        logger.error(f"Error retrieving messages for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")

@router.get("/users/{user_id}/memory-patterns", response_model=APIResponse[List[MemoryAccessPattern]])
async def get_user_memory_access_patterns(user_id: str):
    """Get memory access patterns for a specific user"""
    try:
        result = db_client.client.table('memory_access_patterns').select('*').eq(
            'user_id', user_id
        ).order('access_frequency', desc=True).execute()

        patterns = []
        for pattern in result.data or []:
            patterns.append(MemoryAccessPattern(
                id=pattern['id'],
                user_id=pattern['user_id'],
                memory_namespace=pattern['memory_namespace'],
                memory_key=pattern.get('memory_key'),
                access_frequency=pattern['access_frequency'],
                last_accessed_at=pattern['last_accessed_at'],
                context_relevance=pattern['context_relevance'],
                access_context=pattern.get('access_context'),
                retrieval_method=pattern.get('retrieval_method'),
                metadata=pattern.get('metadata')
            ))

        return APIResponse(
            success=True,
            message="Memory access patterns retrieved successfully",
            data=patterns
        )
    except Exception as e:
        logger.error(f"Error retrieving memory access patterns for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patterns: {str(e)}")

@router.post("/memory/consolidate", response_model=APIResponse[Dict[str, Any]])
async def trigger_memory_consolidation(request: MemoryConsolidationRequest):
    """Trigger memory consolidation for a user"""
    try:
        from agents.memory import memory_manager
        from datetime import datetime

        # Ensure memory manager is initialized
        await memory_manager._ensure_initialized()

        # Trigger actual consolidation
        logger.info(f"Starting manual memory consolidation for user {request.user_id}")

        if request.force:
            logger.info(f"Force flag enabled - bypassing all thresholds")

        # Call the actual consolidation method
        consolidation_result = await memory_manager.consolidator.consolidate_user_summary_with_llm(
            user_id=request.user_id
        )

        result = {
            "user_id": request.user_id,
            "consolidation_triggered": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "force": request.force,
            "consolidation_result": consolidation_result[:200] + "..." if consolidation_result and len(consolidation_result) > 200 else consolidation_result,
            "success": bool(consolidation_result and not consolidation_result.startswith("Error"))
        }

        logger.info(f"Memory consolidation completed for user {request.user_id}")

        return APIResponse(
            success=True,
            message="Memory consolidation triggered and executed successfully",
            data=result
        )
    except Exception as e:
        logger.error(f"Error triggering memory consolidation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger consolidation: {str(e)}")

@router.post("/memory/search", response_model=APIResponse[List[LongTermMemoryItem]])
async def search_memories(request: MemorySearchRequest):
    """Search memories using semantic similarity"""
    try:
        # This would integrate with the embedding search system
        # For now, return a basic text search as placeholder

        query_filter = db_client.client.table('long_term_memories').select('*')

        if request.user_id:
            query_filter = query_filter.or_(
                f'namespace.cs.{{{request.user_id}}},namespace.cs.{{user,{request.user_id}}}'
            )

        if request.memory_type:
            query_filter = query_filter.eq('memory_type', request.memory_type)

        # Simple text search on key and value for now
        # In a full implementation, this would use vector similarity
        result = query_filter.ilike('key', f'%{request.query}%').limit(request.limit).execute()

        memories = []
        for memory in result.data or []:
            # Parse JSON fields that might be stored as strings
            try:
                value = memory['value']
                if isinstance(value, str):
                    value = json.loads(value)

                embedding = memory.get('embedding')
                if embedding and isinstance(embedding, str):
                    embedding = json.loads(embedding)

                metadata = memory.get('metadata')
                if metadata and isinstance(metadata, str):
                    metadata = json.loads(metadata)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON fields for memory {memory['id']}: {e}")
                value = memory['value'] if not isinstance(memory['value'], str) else {}
                embedding = None
                metadata = memory.get('metadata')

            memories.append(LongTermMemoryItem(
                id=memory['id'],
                namespace=memory['namespace'],
                key=memory['key'],
                value=value,
                embedding=embedding,
                created_at=memory['created_at'],
                updated_at=memory['updated_at'],
                accessed_at=memory['accessed_at'],
                access_count=memory['access_count'],
                memory_type=memory['memory_type'],
                source_thread_id=memory.get('source_thread_id'),
                expiry_at=memory.get('expiry_at'),
                metadata=metadata
            ))

        return APIResponse(
            success=True,
            message="Memory search completed successfully",
            data=memories
        )
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.get("/memory/stats", response_model=APIResponse[Dict[str, Any]])
async def get_memory_stats():
    """Get overall memory system statistics"""
    try:
        # Get counts from various tables
        memory_count_result = db_client.client.table('long_term_memories').select('id', count='exact').execute()
        summary_count_result = db_client.client.table('conversation_summaries').select('id', count='exact').execute()
        message_count_result = db_client.client.table('messages').select('id', count='exact').execute()

        stats = {
            "total_long_term_memories": memory_count_result.count or 0,
            "total_conversation_summaries": summary_count_result.count or 0,
            "total_messages": message_count_result.count or 0,
            "timestamp": "2024-01-01T00:00:00Z"  # Current timestamp would be used here
        }

        return APIResponse(
            success=True,
            message="Memory statistics retrieved successfully",
            data=stats
        )
    except Exception as e:
        logger.error(f"Error retrieving memory statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.get("/connection-pool-status")
async def get_db_connection_pool_status():
    """Get current database connection pool status for monitoring connection usage."""
    try:
        from agents.tools import get_connection_pool_status
        pool_status = await get_connection_pool_status()
        
        return APIResponse(
            success=True,
            message="Connection pool status retrieved successfully",
            data=pool_status
        )
    except Exception as e:
        logger.error(f"Error retrieving connection pool status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve connection pool status: {str(e)}")


@router.post("/close-connections")
async def close_db_connections():
    """Close all database connections to free up the connection pool (use in emergencies)."""
    try:
        from agents.tools import close_database_connections
        await close_database_connections()
        
        return APIResponse(
            success=True,
            message="Database connections closed successfully",
            data={"connections_closed": True}
        )
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close connections: {str(e)}")
