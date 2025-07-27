"""
Comprehensive Connection Reset Utility

This utility provides functions to completely reset all database connections
to resolve "Max client connections reached" issues and start with a clean slate.
"""

import asyncio
import logging
import gc
from typing import Optional

logger = logging.getLogger(__name__)

async def reset_all_connections(force: bool = True) -> dict:
    """
    Comprehensive reset of all database connections.
    
    Args:
        force: If True, aggressively reset all connections regardless of state
        
    Returns:
        Dictionary with reset results and statistics
    """
    results = {
        "supabase_client_reset": False,
        "sql_connections_reset": False,
        "execution_manager_reset": False,
        "memory_manager_reset": False,
        "garbage_collection": False,
        "errors": []
    }
    
    logger.info("🔄 Starting comprehensive connection reset...")
    
    try:
        # 1. Reset Supabase client (user_verification connections)
        logger.info("1️⃣ Resetting Supabase client...")
        supabase_result = await reset_supabase_client(force=force)
        results["supabase_client_reset"] = supabase_result
        
        # 2. Reset SQL connection pools (tools connections)
        logger.info("2️⃣ Resetting SQL connection pools...")
        sql_result = await reset_sql_connections(force=force)
        results["sql_connections_reset"] = sql_result
        
        # 3. Reset execution connection manager state
        logger.info("3️⃣ Resetting execution connection manager...")
        manager_result = await reset_execution_manager(force=force)
        results["execution_manager_reset"] = manager_result
        
        # 4. Reset memory manager connections
        logger.info("4️⃣ Resetting memory manager...")
        memory_result = await reset_memory_manager(force=force)
        results["memory_manager_reset"] = memory_result
        
        # 5. Force garbage collection to clean up any remaining references
        logger.info("5️⃣ Running garbage collection...")
        gc.collect()
        results["garbage_collection"] = True
        
        # 6. Wait a moment for connections to fully close
        logger.info("6️⃣ Waiting for connections to settle...")
        await asyncio.sleep(2)
        
        logger.info("✅ Comprehensive connection reset completed successfully!")
        return results
        
    except Exception as e:
        error_msg = f"Error during connection reset: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
        return results

async def reset_supabase_client(force: bool = True) -> bool:
    """Reset the global Supabase client to clear REST API connections."""
    try:
        from database import db_client
        
        # Force reset the client
        if hasattr(db_client, '_client') and db_client._client:
            # Close any internal connections the Supabase client might have
            if hasattr(db_client._client, 'auth') and hasattr(db_client._client.auth, 'sign_out'):
                try:
                    # Don't actually sign out, just clear any cached sessions
                    pass
                except:
                    pass
            
            # Reset the client to None to force re-initialization
            db_client._client = None
            logger.info("   ✅ Supabase client reset")
        
        # Force re-initialization on next use
        if hasattr(db_client, '_ensure_initialized'):
            # The client will be re-initialized on next access
            pass
            
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error resetting Supabase client: {e}")
        return False

async def reset_sql_connections(force: bool = True) -> bool:
    """Reset all SQL connection pools and engines."""
    try:
        # Reset legacy global connections (if any)
        try:
            from agents.tools import close_database_connections
            await close_database_connections()
            logger.info("   ✅ Legacy SQL connections closed")
        except Exception as e:
            logger.warning(f"   ⚠️ Legacy connection cleanup warning: {e}")
        
        # Reset execution-scoped connection manager
        from agents.memory import _connection_manager
        
        # Force cleanup of all tracked connections
        if hasattr(_connection_manager, '_execution_connections'):
            for exec_id, connections in list(_connection_manager._execution_connections.items()):
                try:
                    _connection_manager._cleanup_execution_connections(exec_id, connections)
                except Exception as e:
                    logger.warning(f"   ⚠️ Error cleaning execution {exec_id}: {e}")
            
            # Clear all tracking state
            _connection_manager._execution_connections.clear()
            _connection_manager._active_executions.clear()
            logger.info("   ✅ Execution connection manager state cleared")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error resetting SQL connections: {e}")
        return False

async def reset_execution_manager(force: bool = True) -> bool:
    """Reset the execution connection manager to a clean state."""
    try:
        from agents.memory import _connection_manager, get_connection_manager
        
        # Get current stats before reset
        stats_before = _connection_manager.get_execution_stats()
        logger.info(f"   📊 Connections before reset: {stats_before}")
        
        # Force cleanup of all active executions
        active_executions = list(_connection_manager._active_executions)
        for exec_id in active_executions:
            try:
                _connection_manager.unregister_execution(exec_id)
                logger.info(f"   🧹 Cleaned up execution: {exec_id}")
            except Exception as e:
                logger.warning(f"   ⚠️ Error cleaning execution {exec_id}: {e}")
        
        # Verify clean state
        stats_after = _connection_manager.get_execution_stats()
        logger.info(f"   📊 Connections after reset: {stats_after}")
        
        return stats_after['total_tracked_connections'] == 0
        
    except Exception as e:
        logger.error(f"   ❌ Error resetting execution manager: {e}")
        return False

async def reset_memory_manager(force: bool = True) -> bool:
    """Reset memory manager connections."""
    try:
        from agents.memory import memory_manager
        
        # Reset memory manager connections if initialized
        if hasattr(memory_manager, 'connection_pool') and memory_manager.connection_pool:
            try:
                await memory_manager.connection_pool.close()
                memory_manager.connection_pool = None
                logger.info("   ✅ Memory manager connection pool closed")
            except Exception as e:
                logger.warning(f"   ⚠️ Error closing memory manager pool: {e}")
        
        # Reset checkpointer
        if hasattr(memory_manager, 'checkpointer'):
            memory_manager.checkpointer = None
            logger.info("   ✅ Memory manager checkpointer reset")
        
        # Reset other components
        if hasattr(memory_manager, 'db_manager'):
            memory_manager.db_manager = None
        if hasattr(memory_manager, 'long_term_store'):
            memory_manager.long_term_store = None
        if hasattr(memory_manager, 'consolidator'):
            memory_manager.consolidator = None
            
        logger.info("   ✅ Memory manager components reset")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error resetting memory manager: {e}")
        return False

async def get_connection_status() -> dict:
    """Get comprehensive status of all connections."""
    status = {
        "supabase_client": {"initialized": False, "status": "unknown"},
        "execution_manager": {"active_executions": 0, "total_connections": 0, "executions": {}},  
        "memory_manager": {"initialized": False, "components": {}},
        "timestamp": asyncio.get_event_loop().time()
    }
    
    try:
        # Check Supabase client
        from database import db_client
        status["supabase_client"]["initialized"] = db_client._client is not None
        status["supabase_client"]["status"] = "initialized" if db_client._client else "not_initialized"
        
        # Check execution manager
        from agents.memory import get_connection_statistics
        exec_stats = await get_connection_statistics()
        status["execution_manager"] = exec_stats
        
        # Check memory manager
        from agents.memory import memory_manager
        status["memory_manager"]["initialized"] = memory_manager.checkpointer is not None
        status["memory_manager"]["components"] = {
            "checkpointer": memory_manager.checkpointer is not None,
            "connection_pool": getattr(memory_manager, 'connection_pool', None) is not None,
            "db_manager": getattr(memory_manager, 'db_manager', None) is not None,
            "long_term_store": getattr(memory_manager, 'long_term_store', None) is not None,
            "consolidator": getattr(memory_manager, 'consolidator', None) is not None
        }
        
    except Exception as e:
        status["error"] = str(e)
        logger.error(f"Error getting connection status: {e}")
    
    return status

async def emergency_connection_reset() -> dict:
    """
    Emergency connection reset for when everything is stuck.
    This is the nuclear option - resets everything aggressively.
    """
    logger.warning("🚨 EMERGENCY CONNECTION RESET - This will aggressively close all connections!")
    
    results = {
        "emergency_reset": True,
        "actions_taken": [],
        "errors": []
    }
    
    try:
        # 1. Kill all Python connection objects
        logger.info("1️⃣ Killing all connection objects...")
        
        # Reset database client
        try:
            from database import db_client
            db_client._client = None
            results["actions_taken"].append("Supabase client nullified")
        except Exception as e:
            results["errors"].append(f"Supabase reset error: {e}")
        
        # 2. Clear execution manager state aggressively
        try:
            from agents.memory import _connection_manager
            
            # Force dispose all engines
            for exec_id, connections in _connection_manager._execution_connections.items():
                for engine in connections.get('sql_engines', []):
                    try:
                        engine.dispose()
                    except:
                        pass
                for db in connections.get('sql_databases', []):
                    try:
                        if hasattr(db, '_engine'):
                            db._engine.dispose()
                    except:
                        pass
            
            # Clear all state
            _connection_manager._execution_connections.clear()
            _connection_manager._active_executions.clear()
            results["actions_taken"].append("Execution manager state cleared")
            
        except Exception as e:
            results["errors"].append(f"Execution manager reset error: {e}")
        
        # 3. Reset memory manager
        try:
            from agents.memory import memory_manager
            if hasattr(memory_manager, 'connection_pool') and memory_manager.connection_pool:
                await memory_manager.connection_pool.close()
            memory_manager.connection_pool = None
            memory_manager.checkpointer = None
            memory_manager.db_manager = None
            memory_manager.long_term_store = None
            memory_manager.consolidator = None
            results["actions_taken"].append("Memory manager reset")
        except Exception as e:
            results["errors"].append(f"Memory manager reset error: {e}")
        
        # 4. Force garbage collection multiple times
        for i in range(3):
            gc.collect()
        results["actions_taken"].append("Multiple garbage collections")
        
        # 5. Wait for connections to close
        await asyncio.sleep(5)
        results["actions_taken"].append("5-second wait for connection cleanup")
        
        logger.warning("🚨 Emergency connection reset completed!")
        return results
        
    except Exception as e:
        results["errors"].append(f"Emergency reset error: {e}")
        return results

# Convenience functions for quick access
async def quick_reset():
    """Quick connection reset for common issues."""
    return await reset_all_connections(force=True)

async def status_check():
    """Quick status check of all connections."""
    return await get_connection_status() 