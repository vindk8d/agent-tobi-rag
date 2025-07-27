#!/usr/bin/env python3
"""
Connection Reset CLI Tool

Quick command-line utility to reset all database connections
when experiencing "Max client connections reached" errors.

Usage:
    python scripts/reset_connections.py [--emergency] [--status-only]
    
Options:
    --emergency    : Use emergency reset (nuclear option)
    --status-only  : Just check connection status without reset
    --quick        : Use quick reset (default)
"""

import asyncio
import sys
import argparse
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

async def main():
    parser = argparse.ArgumentParser(description="Reset database connections")
    parser.add_argument("--emergency", action="store_true", 
                       help="Use emergency reset (nuclear option)")
    parser.add_argument("--status-only", action="store_true",
                       help="Just check connection status without reset")
    parser.add_argument("--quick", action="store_true", default=True,
                       help="Use quick reset (default)")
    
    args = parser.parse_args()
    
    try:
        # Import after setting path
        from agents.connection_reset import (
            quick_reset, 
            emergency_connection_reset, 
            get_connection_status,
            reset_all_connections
        )
        
        print("🔗 Database Connection Reset Tool")
        print("=" * 40)
        
        if args.status_only:
            print("📊 Checking connection status...")
            status = await get_connection_status()
            print_status(status)
            return
        
        # Check status before reset
        print("📊 Connection status BEFORE reset:")
        status_before = await get_connection_status()
        print_status(status_before)
        print()
        
        # Perform reset based on options
        if args.emergency:
            print("🚨 EMERGENCY CONNECTION RESET")
            print("⚠️  This will aggressively close ALL connections!")
            confirm = input("Are you sure? Type 'yes' to continue: ")
            if confirm.lower() != 'yes':
                print("❌ Emergency reset cancelled.")
                return
            
            results = await emergency_connection_reset()
            print("🚨 Emergency reset completed!")
            
        else:
            print("⚡ Quick connection reset...")
            results = await quick_reset()
            print("✅ Quick reset completed!")
        
        # Show results
        print("\n📋 Reset Results:")
        for key, value in results.items():
            if key == "errors" and value:
                print(f"   ❌ {key}: {value}")
            elif key == "actions_taken" and value:
                print(f"   ✅ {key}:")
                for action in value:
                    print(f"      - {action}")
            else:
                status_icon = "✅" if value else "❌"
                print(f"   {status_icon} {key}: {value}")
        
        # Check status after reset
        print("\n📊 Connection status AFTER reset:")
        status_after = await get_connection_status()
        print_status(status_after)
        
        print("\n🎉 Connection reset completed successfully!")
        print("\n💡 You can now try running your LangGraph Studio test again.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error during connection reset: {e}")
        sys.exit(1)

def print_status(status):
    """Print connection status in a readable format."""
    if "error" in status:
        print(f"   ❌ Error: {status['error']}")
        return
    
    # Supabase client status
    supabase = status.get("supabase_client", {})
    supabase_icon = "✅" if supabase.get("initialized") else "❌"
    print(f"   {supabase_icon} Supabase client: {supabase.get('status', 'unknown')}")
    
    # Execution manager status
    exec_mgr = status.get("execution_manager", {})
    active_execs = exec_mgr.get("active_executions", 0)
    total_conns = exec_mgr.get("total_connections", 0)
    exec_icon = "✅" if active_execs == 0 and total_conns == 0 else "⚠️"
    print(f"   {exec_icon} Execution manager: {active_execs} active, {total_conns} total connections")
    
    # Memory manager status
    memory = status.get("memory_manager", {})
    memory_init = memory.get("initialized", False)
    memory_icon = "✅" if not memory_init else "⚠️"  # Not initialized is good after reset
    print(f"   {memory_icon} Memory manager: {'initialized' if memory_init else 'clean'}")
    
    # Show individual executions if any
    executions = exec_mgr.get("executions", {})
    if executions:
        print(f"   📝 Active executions:")
        for exec_id, exec_data in executions.items():
            engines = exec_data.get("sql_engines", 0)
            databases = exec_data.get("sql_databases", 0)
            custom = exec_data.get("custom_connections", 0)
            print(f"      - {exec_id[:16]}...: {engines} engines, {databases} databases, {custom} custom")

if __name__ == "__main__":
    asyncio.run(main()) 