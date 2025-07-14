"""
Test script for SQL database connection and table access restrictions.
Run this to verify that the LangChain SQLDatabase setup is working correctly.
"""

import asyncio
import logging
import sys
import os

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import _get_sql_database, CRM_TABLES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sql_connection():
    """Test the SQL database connection and verify table restrictions."""
    print("🧪 Testing SQL Database Connection...")
    print("=" * 50)
    
    try:
        # Test connection
        print("1. Testing database connection...")
        db = await _get_sql_database()
        
        if db is None:
            print("❌ FAILED: Database connection returned None")
            print("💡 Make sure SUPABASE_DB_PASSWORD is set in your .env file")
            return False
            
        print("✅ SUCCESS: Database connected")
        
        # Test table access restrictions
        print("\n2. Testing table access restrictions...")
        table_info = db.get_table_info()
        print(f"📋 Available tables: {db.get_usable_table_names()}")
        
        # Verify only CRM tables are accessible
        accessible_tables = set(db.get_usable_table_names())
        expected_tables = set(CRM_TABLES)
        
        if accessible_tables.issubset(expected_tables):
            print("✅ SUCCESS: Table access properly restricted to CRM tables")
        else:
            print(f"⚠️  WARNING: Unexpected tables accessible: {accessible_tables - expected_tables}")
        
        # Test sample query
        print("\n3. Testing sample query...")
        try:
            result = db.run("SELECT COUNT(*) as branch_count FROM branches;")
            print(f"✅ SUCCESS: Sample query executed - {result}")
        except Exception as e:
            print(f"❌ FAILED: Sample query failed - {e}")
            
        print("\n🎉 SQL Database setup test completed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure SUPABASE_DB_PASSWORD is set in your .env file")
        print("   - Verify your Supabase project has the CRM tables created")
        print("   - Check that your database password is correct")
        return False

if __name__ == "__main__":
    asyncio.run(test_sql_connection()) 