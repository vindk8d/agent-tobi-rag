#!/usr/bin/env python3
"""
Working test suite for fixed long-term memory functionality (Task 5.5.4)

This test suite verifies that the critical fixes have been properly implemented
using the correct database and embeddings interfaces.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.memory import SupabaseLongTermMemoryStore, MemoryManager
from backend.rag.embeddings import OpenAIEmbeddings
from backend.database import db_client
from langgraph.store.base import Item

class LongTermMemoryWorkingTester:
    """Working test runner for fixed long-term memory functionality."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def log_result(self, test_name: str, passed: bool, details: str = "", error: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {error}")
        
        print(f"{status} {test_name}")
        if details:
            print(f"    Details: {details}")
        if error:
            print(f"    Error: {error}")
    
    async def test_database_functions_direct(self):
        """Test database functions directly through Supabase client."""
        print("\n🔧 Testing Database Functions (Direct)...")
        
        # Test 1: search_conversation_summaries function
        try:
            # Create test embedding
            test_embedding = [0.1] * 1536
            
            # Use a valid UUID format for target_user_id
            test_user_uuid = "123e4567-e89b-12d3-a456-426614174000"
            
            # Test the function call using the Supabase client
            result = db_client.client.rpc('search_conversation_summaries', {
                'query_embedding': test_embedding,
                'target_user_id': test_user_uuid,
                'similarity_threshold': 0.5,
                'match_count': 5,
                'summary_type_filter': None
            }).execute()
            
            self.log_result(
                "search_conversation_summaries_function",
                True,
                f"Function executed successfully, returned {len(result.data)} results"
            )
            
        except Exception as e:
            self.log_result(
                "search_conversation_summaries_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 2: cleanup_expired_memories function
        try:
            result = db_client.client.rpc('cleanup_expired_memories').execute()
            
            self.log_result(
                "cleanup_expired_memories_function",
                True,
                f"Function executed successfully, cleaned up {result.data} expired memories"
            )
            
        except Exception as e:
            self.log_result(
                "cleanup_expired_memories_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 3: New helper functions
        try:
            test_embedding = [0.1] * 1536
            test_uuid = "123e4567-e89b-12d3-a456-426614174000"
            
            # Test search_long_term_memories_by_prefix
            result = db_client.client.rpc('search_long_term_memories_by_prefix', {
                'query_embedding': test_embedding,
                'namespace_prefix': [test_uuid],
                'similarity_threshold': 0.5,
                'match_count': 5
            }).execute()
            
            # Test list_long_term_memory_keys
            result = db_client.client.rpc('list_long_term_memory_keys', {
                'p_namespace': [test_uuid]
            }).execute()
            
            # Test put_long_term_memory
            result = db_client.client.rpc('put_long_term_memory', {
                'p_namespace': [test_uuid],
                'p_key': 'test_key',
                'p_value': {'test': 'value'},
                'p_embedding': None,
                'p_memory_type': 'semantic',
                'p_expiry_at': None
            }).execute()
            
            # Test get_long_term_memory
            result = db_client.client.rpc('get_long_term_memory', {
                'p_namespace': [test_uuid],
                'p_key': 'test_key'
            }).execute()
            
            # Test delete_long_term_memory
            result = db_client.client.rpc('delete_long_term_memory', {
                'p_namespace': [test_uuid],
                'p_key': 'test_key'
            }).execute()
            
            self.log_result(
                "new_helper_functions",
                True,
                "All new helper functions execute successfully"
            )
            
        except Exception as e:
            self.log_result(
                "new_helper_functions",
                False,
                error=f"Helper functions failed: {str(e)}"
            )
    
    async def test_store_interface_basic(self):
        """Test SupabaseLongTermMemoryStore basic functionality."""
        print("\n🏪 Testing Store Interface (Basic)...")
        
        try:
            # Initialize store with correct embeddings
            embeddings = OpenAIEmbeddings()
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: put operation
            test_uuid = "123e4567-e89b-12d3-a456-426614174001"
            await store.aput(
                namespace=(test_uuid, "preferences"),
                key="test_preference",
                value={"preference": "test_value", "priority": "high"}
            )
            
            self.log_result(
                "store_put_operation",
                True,
                "Successfully stored item"
            )
            
            # Test 2: get operation
            item = await store.aget(
                namespace=(test_uuid, "preferences"),
                key="test_preference"
            )
            
            if item is not None:
                self.log_result(
                    "store_get_operation",
                    True,
                    f"Successfully retrieved item: {item.value}"
                )
            else:
                self.log_result(
                    "store_get_operation",
                    False,
                    error="Item not found after storage"
                )
            
            # Test 3: delete operation
            await store.adelete(
                namespace=(test_uuid, "preferences"),
                key="test_preference"
            )
            
            # Verify deletion
            deleted_item = await store.aget(
                namespace=(test_uuid, "preferences"),
                key="test_preference"
            )
            
            if deleted_item is None:
                self.log_result(
                    "store_delete_operation",
                    True,
                    "Item successfully deleted"
                )
            else:
                self.log_result(
                    "store_delete_operation",
                    False,
                    error="Item still exists after deletion"
                )
            
        except Exception as e:
            self.log_result(
                "store_interface_basic",
                False,
                error=f"Store interface test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_memory_manager_basic(self):
        """Test MemoryManager basic functionality."""
        print("\n🧠 Testing Memory Manager (Basic)...")
        
        try:
            # Initialize memory manager
            memory_manager = MemoryManager()
            
            # Test 1: Store long-term memory
            test_uuid = "123e4567-e89b-12d3-a456-426614174002"
            success = await memory_manager.store_long_term_memory(
                user_id=test_uuid,
                namespace=["preferences"],
                key="test_integration",
                value={"integration": "test", "timestamp": datetime.now().isoformat()}
            )
            
            if success:
                self.log_result(
                    "memory_manager_store",
                    True,
                    "Successfully stored memory via MemoryManager"
                )
            else:
                self.log_result(
                    "memory_manager_store",
                    False,
                    error="Failed to store memory via MemoryManager"
                )
            
            # Test 2: Get checkpointer
            checkpointer = await memory_manager.get_checkpointer()
            
            if checkpointer is not None:
                self.log_result(
                    "memory_manager_checkpointer",
                    True,
                    "Successfully got checkpointer"
                )
            else:
                self.log_result(
                    "memory_manager_checkpointer",
                    False,
                    error="Failed to get checkpointer"
                )
            
        except Exception as e:
            self.log_result(
                "memory_manager_basic",
                False,
                error=f"Memory manager test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_error_handling_basic(self):
        """Test basic error handling."""
        print("\n🛡️ Testing Error Handling (Basic)...")
        
        try:
            embeddings = OpenAIEmbeddings()
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: Non-existent key handling
            test_uuid = "123e4567-e89b-12d3-a456-426614174003"
            result = await store.aget(
                namespace=(test_uuid, "nonexistent"),
                key="nonexistent_key"
            )
            
            if result is None:
                self.log_result(
                    "error_handling_nonexistent_key",
                    True,
                    "Correctly returned None for non-existent key"
                )
            else:
                self.log_result(
                    "error_handling_nonexistent_key",
                    False,
                    error="Should have returned None for non-existent key"
                )
            
            # Test 2: Valid value handling
            test_value = {"test": "data", "number": 42}
            
            await store.aput(
                namespace=(test_uuid, "valid_data"),
                key="valid_test",
                value=test_value
            )
            
            retrieved = await store.aget(
                namespace=(test_uuid, "valid_data"),
                key="valid_test"
            )
            
            if retrieved is not None and retrieved.value == test_value:
                self.log_result(
                    "error_handling_valid_value",
                    True,
                    "Successfully handled valid value storage/retrieval"
                )
                
                # Clean up
                await store.adelete(
                    namespace=(test_uuid, "valid_data"),
                    key="valid_test"
                )
            else:
                self.log_result(
                    "error_handling_valid_value",
                    False,
                    error="Failed to handle valid value"
                )
            
        except Exception as e:
            self.log_result(
                "error_handling_basic",
                False,
                error=f"Error handling test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_schema_validation(self):
        """Test that the database schema is properly set up."""
        print("\n📊 Testing Schema Validation...")
        
        try:
            # Test 1: Check long_term_memories table exists
            result = db_client.client.table('long_term_memories').select('*').limit(1).execute()
            
            self.log_result(
                "schema_long_term_memories_table",
                True,
                "long_term_memories table exists and accessible"
            )
            
            # Test 2: Check conversation_summaries table exists
            result = db_client.client.table('conversation_summaries').select('*').limit(1).execute()
            
            self.log_result(
                "schema_conversation_summaries_table",
                True,
                "conversation_summaries table exists and accessible"
            )
            
            # Test 3: Check memory_access_patterns table exists
            result = db_client.client.table('memory_access_patterns').select('*').limit(1).execute()
            
            self.log_result(
                "schema_memory_access_patterns_table",
                True,
                "memory_access_patterns table exists and accessible"
            )
            
        except Exception as e:
            self.log_result(
                "schema_validation",
                False,
                error=f"Schema validation failed: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Working Long-Term Memory Tests...")
        print("=" * 80)
        
        try:
            await self.test_schema_validation()
            await self.test_database_functions_direct()
            await self.test_store_interface_basic()
            await self.test_memory_manager_basic()
            await self.test_error_handling_basic()
            
        except Exception as e:
            print(f"❌ Test suite failed: {str(e)}\n{traceback.format_exc()}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        
        if self.errors:
            print("\n🚨 ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
        
        # Save results
        with open("long_term_memory_test_results_working.json", "w") as f:
            json.dump({
                "summary": {
                    "passed": self.passed,
                    "failed": self.failed,
                    "success_rate": (self.passed / (self.passed + self.failed) * 100) if (self.passed + self.failed) > 0 else 0,
                    "timestamp": datetime.now().isoformat()
                },
                "results": self.results,
                "errors": self.errors
            }, f, indent=2)
        
        print(f"\n📝 Full results saved to: long_term_memory_test_results_working.json")
        
        return self.passed > 0 and self.failed == 0


async def main():
    """Main test execution function."""
    tester = LongTermMemoryWorkingTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n🎉 ALL TESTS PASSED! Long-term memory fixes are working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the results above.")
        
        return success
        
    except Exception as e:
        print(f"💥 Test execution failed: {str(e)}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    asyncio.run(main()) 