"""
Test script to demonstrate execution-scoped connection cleanup.

This test verifies that:
1. Connections are properly tracked per execution
2. Cleanup happens automatically after execution
3. Multiple concurrent executions don't interfere with each other
"""

import asyncio
import uuid
from backend.agents.memory import (
    ExecutionScope,
    get_connection_statistics,
    create_execution_scoped_sql_database,
    log_connection_status
)

async def test_single_execution_cleanup():
    """Test that a single execution properly cleans up its connections."""
    print("=== Testing Single Execution Cleanup ===")
    
    # Check initial state
    initial_stats = await get_connection_statistics()
    print(f"Initial connections: {initial_stats}")
    
    execution_id = f"test_single_{uuid.uuid4().hex[:8]}"
    
    # Use ExecutionScope to track connections
    with ExecutionScope(execution_id):
        print(f"Inside execution scope: {execution_id}")
        
        # Create some connections
        db1 = await create_execution_scoped_sql_database()
        db2 = await create_execution_scoped_sql_database()
        
        # Check stats during execution
        during_stats = await get_connection_statistics()
        print(f"During execution connections: {during_stats}")
        
        # Simulate some work
        if db1 and db2:
            print("Created 2 database connections successfully")
        
    # After exiting scope, connections should be cleaned up
    final_stats = await get_connection_statistics()
    print(f"Final connections: {final_stats}")
    
    # Verify cleanup worked
    if final_stats['total_tracked_connections'] == initial_stats['total_tracked_connections']:
        print("✅ Single execution cleanup: PASSED")
    else:
        print("❌ Single execution cleanup: FAILED")
    
    print()

async def test_concurrent_executions():
    """Test that concurrent executions don't interfere with each other."""
    print("=== Testing Concurrent Executions ===")
    
    async def worker_execution(worker_id: int):
        """Simulate an agent execution with connections."""
        execution_id = f"test_worker_{worker_id}_{uuid.uuid4().hex[:8]}"
        
        with ExecutionScope(execution_id):
            print(f"Worker {worker_id}: Started execution {execution_id}")
            
            # Create connections
            db = await create_execution_scoped_sql_database()
            
            # Simulate work
            await asyncio.sleep(0.1)
            
            # Check stats during execution
            stats = await get_connection_statistics()
            print(f"Worker {worker_id}: Active executions: {stats['active_executions']}")
            
            if db:
                print(f"Worker {worker_id}: Database connection created")
            
            # More work
            await asyncio.sleep(0.1)
            
        print(f"Worker {worker_id}: Completed execution {execution_id}")
    
    # Run multiple concurrent executions
    initial_stats = await get_connection_statistics()
    print(f"Initial connections: {initial_stats}")
    
    # Start 3 concurrent workers
    tasks = [worker_execution(i) for i in range(3)]
    await asyncio.gather(*tasks)
    
    # Check final state
    final_stats = await get_connection_statistics()
    print(f"Final connections: {final_stats}")
    
    # All executions should be cleaned up
    if final_stats['active_executions'] == 0:
        print("✅ Concurrent executions cleanup: PASSED")
    else:
        print("❌ Concurrent executions cleanup: FAILED")
    
    print()

async def test_exception_handling():
    """Test that cleanup happens even when exceptions occur."""
    print("=== Testing Exception Handling ===")
    
    initial_stats = await get_connection_statistics()
    print(f"Initial connections: {initial_stats}")
    
    execution_id = f"test_exception_{uuid.uuid4().hex[:8]}"
    
    try:
        with ExecutionScope(execution_id):
            print(f"Inside execution scope: {execution_id}")
            
            # Create a connection
            db = await create_execution_scoped_sql_database()
            
            # Check stats during execution
            during_stats = await get_connection_statistics()
            print(f"During execution connections: {during_stats}")
            
            if db:
                print("Database connection created")
            
            # Simulate an exception
            raise ValueError("Simulated error during execution")
            
    except ValueError as e:
        print(f"Caught expected exception: {e}")
    
    # Check that cleanup still happened
    final_stats = await get_connection_statistics()
    print(f"Final connections: {final_stats}")
    
    # Verify cleanup worked despite exception
    if final_stats['total_tracked_connections'] == initial_stats['total_tracked_connections']:
        print("✅ Exception handling cleanup: PASSED")
    else:
        print("❌ Exception handling cleanup: FAILED")
    
    print()

async def main():
    """Run all connection cleanup tests."""
    print("🧪 Testing Execution-Scoped Connection Cleanup")
    print("=" * 50)
    
    try:
        # Run all tests
        await test_single_execution_cleanup()
        await test_concurrent_executions() 
        await test_exception_handling()
        
        # Final status
        print("=== Final System Status ===")
        await log_connection_status()
        
        print("\n🎉 All connection cleanup tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 