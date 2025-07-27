"""
Redis-based confirmation storage to persist confirmations across container restarts.
"""
import json
import redis
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from models.base import ConfirmationRequest, ConfirmationStatus
import logging

logger = logging.getLogger(__name__)

class RedisConfirmationStorage:
    """Redis-based storage for confirmation requests that persists across restarts."""
    
    def __init__(self, redis_url: str = "redis://redis:6379", default_ttl: int = 1800):
        """Initialize Redis connection with 30-minute default TTL."""
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl  # 30 minutes
        self.key_prefix = "confirmation:"
        
    def _get_key(self, confirmation_id: str) -> str:
        """Generate Redis key for confirmation."""
        return f"{self.key_prefix}{confirmation_id}"
    
    def _serialize_confirmation(self, confirmation: ConfirmationRequest) -> str:
        """Serialize confirmation to JSON string."""
        return json.dumps({
            "confirmation_id": confirmation.confirmation_id,
            "conversation_id": confirmation.conversation_id,
            "customer_id": confirmation.customer_id,
            "customer_name": confirmation.customer_name,
            "customer_email": confirmation.customer_email,
            "message_content": confirmation.message_content,
            "message_type": confirmation.message_type,
            "requested_by": confirmation.requested_by,
            "requested_at": confirmation.requested_at.isoformat(),
            "expires_at": confirmation.expires_at.isoformat(),
            "status": confirmation.status.value if hasattr(confirmation.status, 'value') else confirmation.status
        })
    
    def _deserialize_confirmation(self, data: str) -> ConfirmationRequest:
        """Deserialize JSON string to ConfirmationRequest."""
        from datetime import datetime
        obj = json.loads(data)
        return ConfirmationRequest(
            confirmation_id=obj["confirmation_id"],
            conversation_id=obj["conversation_id"],
            customer_id=obj["customer_id"],
            customer_name=obj["customer_name"],
            customer_email=obj.get("customer_email"),
            message_content=obj["message_content"],
            message_type=obj.get("message_type", "follow_up"),
            requested_by=obj["requested_by"],
            requested_at=datetime.fromisoformat(obj["requested_at"]),
            expires_at=datetime.fromisoformat(obj["expires_at"]),
            status=ConfirmationStatus(obj["status"]) if isinstance(obj["status"], str) else obj["status"]
        )
    
    def store_confirmation(self, confirmation: ConfirmationRequest, ttl: Optional[int] = None) -> bool:
        """Store confirmation in Redis with TTL."""
        try:
            key = self._get_key(confirmation.confirmation_id)
            data = self._serialize_confirmation(confirmation)
            ttl = ttl or self.default_ttl
            
            result = self.redis_client.setex(key, ttl, data)
            logger.info(f"🔍 [REDIS_CONFIRMATION] Stored confirmation {confirmation.confirmation_id} with TTL {ttl}s")
            return result
        except Exception as e:
            logger.error(f"🔍 [REDIS_CONFIRMATION] Failed to store confirmation {confirmation.confirmation_id}: {e}")
            return False
    
    def get_confirmation(self, confirmation_id: str) -> Optional[ConfirmationRequest]:
        """Get confirmation from Redis."""
        try:
            key = self._get_key(confirmation_id)
            data = self.redis_client.get(key)
            if data:
                confirmation = self._deserialize_confirmation(data)
                logger.debug(f"🔍 [REDIS_CONFIRMATION] Retrieved confirmation {confirmation_id}")
                return confirmation
            return None
        except Exception as e:
            logger.error(f"🔍 [REDIS_CONFIRMATION] Failed to get confirmation {confirmation_id}: {e}")
            return None
    
    def delete_confirmation(self, confirmation_id: str) -> bool:
        """Delete confirmation from Redis."""
        try:
            key = self._get_key(confirmation_id)
            result = self.redis_client.delete(key)
            logger.info(f"🔍 [REDIS_CONFIRMATION] Deleted confirmation {confirmation_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"🔍 [REDIS_CONFIRMATION] Failed to delete confirmation {confirmation_id}: {e}")
            return False
    
    def get_all_confirmations(self) -> Dict[str, ConfirmationRequest]:
        """Get all confirmations from Redis."""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            confirmations = {}
            
            if keys:
                # Use pipeline for efficiency
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.get(key)
                results = pipe.execute()
                
                for key, data in zip(keys, results):
                    if data:
                        confirmation_id = key.replace(self.key_prefix, "")
                        confirmation = self._deserialize_confirmation(data)
                        confirmations[confirmation_id] = confirmation
            
            logger.debug(f"🔍 [REDIS_CONFIRMATION] Retrieved {len(confirmations)} total confirmations")
            return confirmations
        except Exception as e:
            logger.error(f"🔍 [REDIS_CONFIRMATION] Failed to get all confirmations: {e}")
            return {}
    
    def get_confirmations_by_conversation(self, conversation_id: str) -> List[ConfirmationRequest]:
        """Get all confirmations for a specific conversation."""
        all_confirmations = self.get_all_confirmations()
        return [conf for conf in all_confirmations.values() if conf.conversation_id == conversation_id]
    
    def get_pending_confirmations(self, conversation_id: Optional[str] = None) -> List[ConfirmationRequest]:
        """Get all pending confirmations, optionally filtered by conversation."""
        all_confirmations = self.get_all_confirmations()
        pending = [
            conf for conf in all_confirmations.values() 
            if conf.status == ConfirmationStatus.PENDING
        ]
        
        if conversation_id:
            pending = [conf for conf in pending if conf.conversation_id == conversation_id]
            
        logger.info(f"🔍 [REDIS_CONFIRMATION] Found {len(pending)} pending confirmations" + 
                   (f" for conversation {conversation_id}" if conversation_id else ""))
        return pending
    
    def update_confirmation_status(self, confirmation_id: str, status: ConfirmationStatus) -> bool:
        """Update confirmation status in Redis."""
        confirmation = self.get_confirmation(confirmation_id)
        if confirmation:
            confirmation.status = status
            return self.store_confirmation(confirmation)
        return False
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"🔍 [REDIS_CONFIRMATION] Health check failed: {e}")
            return False

# Global instance
_redis_storage = None

def get_confirmation_storage() -> RedisConfirmationStorage:
    """Get or create the global Redis confirmation storage instance."""
    global _redis_storage
    if _redis_storage is None:
        from config import get_settings_sync
        settings = get_settings_sync()
        # Use default Redis URL since we know Redis is running on redis:6379
        _redis_storage = RedisConfirmationStorage(redis_url="redis://redis:6379")
    return _redis_storage 