# Execution-Scoped Connection Cleanup Guide

## Overview

This guide explains the **execution-scoped connection cleanup system** implemented to solve the "Max client connections reached" error in LangGraph Studio and agent executions.

## Problem Statement

The original issue occurred because multiple connection sources were simultaneously hitting Supabase's PostgreSQL connection limits:

1. **LangGraph Studio**: Creates connections for graph persistence and checkpointing
2. **SQL Tools**: Used SQLAlchemy with global connection pooling
3. **Concurrent Executions**: Multiple agent instances running simultaneously
4. **User Verification**: Uses Supabase REST API (not affected by PostgreSQL limits)

## Solution: Execution-Scoped Connection Management

### Key Components

#### 1. ExecutionConnectionManager (`backend/agents/memory.py`)
- Tracks connections per agent execution ID  
- Provides automatic cleanup when executions complete
- Thread-safe with concurrent execution support
- Handles exceptions gracefully

#### 2. ExecutionScope Context Manager
```python
with ExecutionScope(execution_id):
    # All connections created here are tracked
    # Automatic cleanup happens on exit
    result = await agent.graph.ainvoke(state, config)
```

#### 3. Execution-Scoped Connection Creation
```python
# Old way (global connections)
sql_db = await _get_sql_database()

# New way (execution-scoped)
sql_db = await create_execution_scoped_sql_database()
```

## How It Works

### Connection Lifecycle

1. **Execution Start**: `ExecutionScope` registers a unique execution ID
2. **Connection Creation**: Each SQL connection is tracked per execution
3. **Normal Execution**: Connections are used normally
4. **Execution End**: All connections for that execution are automatically disposed
5. **Exception Handling**: Cleanup happens even if the execution fails

### Connection Tracking

```python
# Each execution tracks its own connections
execution_connections = {
    'execution_id_1': {
        'sql_engines': [engine1, engine2],
        'sql_databases': [db1, db2],
        'custom_connections': []
    },
    'execution_id_2': {
        'sql_engines': [engine3],
        'sql_databases': [db3],
        'custom_connections': []
    }
}
```

### Optimized Connection Settings

Each execution uses smaller, more aggressive connection pools:

```python
engine = create_engine(
    database_url,
    pool_size=2,          # Smaller pool per execution
    max_overflow=3,       # Limited overflow
    pool_timeout=20,      # Shorter timeout
    pool_recycle=1800,    # Recycle every 30 minutes
    pool_pre_ping=True,   # Verify connections
    pool_reset_on_return='commit'  # Close connections aggressively
)
```

## Implementation Details

### Agent Integration

The agent execution is wrapped with `ExecutionScope`:

```python
# In backend/agents/tobi_sales_copilot/rag_agent.py
execution_id = f"{user_id}_{conversation_id}_{int(asyncio.get_event_loop().time())}"

with ExecutionScope(execution_id):
    result = await self.graph.ainvoke(initial_state, config=config)
```

### Tools Integration

SQL tools now use execution-scoped connections:

```python
# In backend/agents/tools.py
async def _get_sql_database() -> Optional[SQLDatabase]:
    """Get SQL database connection using execution-scoped management."""
    try:
        sql_db = await create_execution_scoped_sql_database()
        return sql_db
    except Exception as e:
        logger.error(f"Failed to create execution-scoped SQL database: {e}")
        return None
```

### What Remains Global

The following connections remain global and are **not** cleaned up:

1. **Supabase Client (`db_client`)**: Used by user verification
   - REST API based, not PostgreSQL connections
   - Lightweight and shared across executions
   - Does not contribute to connection limit issues

2. **LangGraph Checkpointer**: Managed by LangGraph itself
   - Uses its own connection management
   - Outside our control but optimized by LangGraph

## Monitoring and Debugging

### API Endpoints

Two new debug endpoints are available:

```bash
# Get current connection statistics
GET /api/memory/debug/connections/status

# Trigger manual cleanup logging
POST /api/memory/debug/connections/cleanup
```

### Response Format

```json
{
  "success": true,
  "data": {
    "connection_stats": {
      "active_executions": 2,
      "total_tracked_connections": 6,
      "executions": {
        "user1_conv1_1640995200": {
          "sql_engines": 2,
          "sql_databases": 2,
          "custom_connections": 0
        },
        "user2_conv2_1640995201": {
          "sql_engines": 1,
          "sql_databases": 1,
          "custom_connections": 0
        }
      }
    },
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

### Testing

Run the connection cleanup test:

```bash
cd /path/to/project
python tests/test_connection_cleanup.py
```

This will verify:
- Single execution cleanup
- Concurrent execution isolation
- Exception handling and cleanup

## Benefits

### ✅ Solved Problems

1. **Connection Limit Errors**: No more "Max client connections reached"
2. **Resource Leaks**: Automatic cleanup prevents connection buildup
3. **Concurrent Safety**: Each execution manages its own connections
4. **Exception Safety**: Cleanup happens even when executions fail

### ✅ Preserved Functionality

1. **User Verification**: Still uses shared Supabase client efficiently
2. **LangGraph Persistence**: Checkpointing continues to work normally
3. **Performance**: No significant performance impact
4. **Monitoring**: Enhanced visibility into connection usage

## Migration Notes

### Deprecated Functions

```python
# DEPRECATED: No longer needed
await close_database_connections()

# NEW: Automatic cleanup via ExecutionScope
with ExecutionScope(execution_id):
    # Connections auto-cleaned on exit
```

### Backward Compatibility

- Old `close_database_connections()` function is preserved but does nothing
- Existing code will continue to work without modification
- New executions automatically use the scoped system

## Best Practices

### For New Development

1. **Always use ExecutionScope** for agent executions
2. **Use execution-scoped connections** for SQL operations
3. **Let the system handle cleanup** - don't manually close connections
4. **Monitor connection stats** during development

### For Troubleshooting

1. **Check connection stats** via `/api/memory/debug/connections/status`
2. **Look for cleanup logs** in agent execution logs
3. **Verify ExecutionScope usage** around agent invocations
4. **Monitor execution IDs** for proper tracking

## Example Usage

### Complete Agent Execution

```python
import uuid
from agents.memory import ExecutionScope

async def handle_user_query(user_id: str, query: str):
    """Handle a user query with proper connection cleanup."""
    
    # Create unique execution ID
    execution_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    
    # Wrap execution with connection tracking
    with ExecutionScope(execution_id):
        # All connections created here will be tracked and cleaned up
        result = await agent.handle_query(
            user_id=user_id,
            query_text=query,
            conversation_id=conversation_id
        )
        
        return result
    
    # Cleanup happens automatically here
```

### SQL Tool Usage

```python
from agents.memory import create_execution_scoped_sql_database

@tool
async def query_database(query: str):
    """Query database with execution-scoped connection."""
    
    # This connection will be automatically tracked and cleaned up
    db = await create_execution_scoped_sql_database()
    if not db:
        return "Database connection failed"
    
    # Use the connection normally
    result = db.run(query)
    return result
    
    # No manual cleanup needed!
```

## Conclusion

The execution-scoped connection cleanup system provides a robust solution to connection limit issues while maintaining all existing functionality. It's designed to be:

- **Automatic**: No manual intervention required
- **Safe**: Works with exceptions and concurrent executions  
- **Transparent**: Existing code continues to work
- **Monitorable**: Full visibility into connection usage
- **Organized**: Integrated into the memory system for better code organization

The system ensures that each agent execution gets its own connection resources and cleans them up properly, eliminating the "Max client connections reached" error while keeping the codebase simple and maintainable. Connection management is now part of the unified memory system in `backend/agents/memory.py`. [[memory:3452634]] 