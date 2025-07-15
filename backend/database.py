"""
Database client setup and utilities for Supabase integration
"""

import os
from supabase import create_client, Client
from typing import Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Supabase database client wrapper"""
    
    def __init__(self):
        self._client: Optional[Client] = None
        # Don't initialize synchronously to avoid blocking calls
        
    async def _async_initialize_client(self):
        """Initialize the Supabase client asynchronously to avoid blocking calls"""
        if self._client is not None:
            return
            
        try:
            # Load environment variables directly without blocking calls
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url or not supabase_service_key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            
            # Create client with service key for backend operations (bypasses RLS)
            self._client = create_client(supabase_url, supabase_service_key)
            logger.info("Supabase client initialized successfully with service key")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def _initialize_client(self):
        """Legacy sync initialization - kept for backward compatibility"""
        if self._client is not None:
            return
            
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url or not supabase_service_key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            
            self._client = create_client(supabase_url, supabase_service_key)
            logger.info("Supabase client initialized successfully with service key")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    async def async_client(self) -> Client:
        """Get the Supabase client instance asynchronously"""
        await self._async_initialize_client()
        return self._client
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            # Simple query to test connection
            result = self.client.from_("data_sources").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def async_health_check(self) -> bool:
        """Check database connection health asynchronously"""
        try:
            client = await self.async_client()
            # Simple query to test connection - use async thread wrapper
            result = await asyncio.to_thread(
                lambda: client.from_("data_sources").select("id").limit(1).execute()
            )
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database client instance
db_client = DatabaseClient()


def get_db_client():
    """Get the global database client instance"""
    return db_client


class DBClientProxy:
    """Proxy class for database client access"""
    @property
    def client(self):
        return db_client.client
    
    def health_check(self):
        return db_client.health_check()
    
    async def async_client(self):
        return await db_client.async_client()
    
    async def async_health_check(self):
        return await db_client.async_health_check() 