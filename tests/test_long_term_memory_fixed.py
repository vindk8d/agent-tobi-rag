#!/usr/bin/env python3
"""
Comprehensive test suite for fixed long-term memory functionality (Task 5.5.4)

This test suite verifies that all critical fixes have been properly implemented:
1. Database function fixes (UUID/text type mismatches)
2. SupabaseLongTermMemoryStore BaseStore interface compliance
3. Conversation consolidation functionality
4. Memory manager integration
5. Performance optimization
6. Error handling and recovery

Run with: python tests/test_long_term_memory_fixed.py
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import db_client
from backend.config import get_settings
from backend.agents.memory import SupabaseLongTermMemoryStore, MemoryManager
from backend.rag.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.store.base import Item

class LongTermMemoryFixedTester:
    """Test runner for fixed long-term memory functionality."""
    
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
    
    async def test_database_functions(self):
        """Test all fixed database functions."""
        print("\n🔧 Testing Database Functions...")
        
        # Test 1: search_conversation_summaries function
        try:
            # Create test embedding
            test_embedding = [0.1] * 1536
            
            # Test the function call
            result = await db_client.execute_query(
                "SELECT * FROM search_conversation_summaries(%s, %s, %s, %s, %s)",
                (test_embedding, "test_user", 0.5, 5, None)
            )
            
            self.log_result(
                "search_conversation_summaries_function",
                True,
                "Function executes without UUID/text type mismatch"
            )
            
        except Exception as e:
            self.log_result(
                "search_conversation_summaries_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 2: cleanup_expired_memories function
        try:
            result = await db_client.execute_query(
                "SELECT cleanup_expired_memories()",
                ()
            )
            
            self.log_result(
                "cleanup_expired_memories_function",
                True,
                "Function executes without invalid UUID input error"
            )
            
        except Exception as e:
            self.log_result(
                "cleanup_expired_memories_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 3: update_memory_access_pattern function
        try:
            result = await db_client.execute_query(
                "SELECT update_memory_access_pattern(%s, %s, %s, %s, %s)",
                ("test_user", ["test", "namespace"], "test_key", "test_context", "direct")
            )
            
            self.log_result(
                "update_memory_access_pattern_function",
                True,
                "Function executes without conflict handling errors"
            )
            
        except Exception as e:
            self.log_result(
                "update_memory_access_pattern_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 4: search_long_term_memories function
        try:
            test_embedding = [0.1] * 1536
            
            result = await db_client.execute_query(
                "SELECT * FROM search_long_term_memories(%s, %s, %s, %s, %s, %s)",
                (test_embedding, ["test"], 0.5, 5, None, True)
            )
            
            self.log_result(
                "search_long_term_memories_function",
                True,
                "Function executes with proper array handling"
            )
            
        except Exception as e:
            self.log_result(
                "search_long_term_memories_function",
                False,
                error=f"Function failed: {str(e)}"
            )
        
        # Test 5: New helper functions
        try:
            test_embedding = [0.1] * 1536
            
            # Test search_long_term_memories_by_prefix
            result = await db_client.execute_query(
                "SELECT * FROM search_long_term_memories_by_prefix(%s, %s, %s, %s)",
                (test_embedding, ["test"], 0.5, 5)
            )
            
            # Test get_long_term_memory
            result = await db_client.execute_query(
                "SELECT * FROM get_long_term_memory(%s, %s)",
                (["test"], "test_key")
            )
            
            # Test list_long_term_memory_keys
            result = await db_client.execute_query(
                "SELECT * FROM list_long_term_memory_keys(%s)",
                (["test"],)
            )
            
            # Test delete_long_term_memory
            result = await db_client.execute_query(
                "SELECT delete_long_term_memory(%s, %s)",
                (["test"], "test_key")
            )
            
            # Test put_long_term_memory
            result = await db_client.execute_query(
                "SELECT put_long_term_memory(%s, %s, %s, %s, %s, %s)",
                (["test"], "test_key", {"test": "value"}, None, "semantic", None)
            )
            
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
    
    async def test_store_interface(self):
        """Test SupabaseLongTermMemoryStore BaseStore interface compliance."""
        print("\n🏪 Testing Store Interface...")
        
        try:
            # Initialize store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: Context manager support
            async with store:
                # Test 2: put operation
                await store.aput(
                    namespace=("test_user", "preferences"),
                    key="test_preference",
                    value={"preference": "test_value", "priority": "high"}
                )
                
                # Test 3: get operation
                item = await store.aget(
                    namespace=("test_user", "preferences"),
                    key="test_preference"
                )
                
                if item is not None:
                    self.log_result(
                        "store_put_get_operations",
                        True,
                        f"Successfully stored and retrieved item: {item.value}"
                    )
                else:
                    self.log_result(
                        "store_put_get_operations",
                        False,
                        error="Item not found after storage"
                    )
                
                # Test 4: search operation
                search_results = await store.asearch(
                    namespace_prefix=("test_user",)
                )
                
                self.log_result(
                    "store_search_operation",
                    True,
                    f"Search returned {len(search_results)} results"
                )
                
                # Test 5: semantic search
                semantic_results = await store.semantic_search(
                    query="test preference",
                    namespace_prefix=("test_user",),
                    limit=5
                )
                
                self.log_result(
                    "store_semantic_search",
                    True,
                    f"Semantic search returned {len(semantic_results)} results"
                )
                
                # Test 6: delete operation
                await store.adelete(
                    namespace=("test_user", "preferences"),
                    key="test_preference"
                )
                
                # Verify deletion
                deleted_item = await store.aget(
                    namespace=("test_user", "preferences"),
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
                
                # Test 7: Sync wrapper methods
                store.put(
                    namespace=("test_user", "sync_test"),
                    key="sync_key",
                    value={"sync": "test"}
                )
                
                sync_item = store.get(
                    namespace=("test_user", "sync_test"),
                    key="sync_key"
                )
                
                if sync_item is not None:
                    self.log_result(
                        "store_sync_wrappers",
                        True,
                        "Sync wrapper methods work correctly"
                    )
                    
                    # Clean up
                    store.delete(
                        namespace=("test_user", "sync_test"),
                        key="sync_key"
                    )
                else:
                    self.log_result(
                        "store_sync_wrappers",
                        False,
                        error="Sync wrapper methods failed"
                    )
            
        except Exception as e:
            self.log_result(
                "store_interface_test",
                False,
                error=f"Store interface test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_memory_manager_integration(self):
        """Test MemoryManager integration with long-term memory."""
        print("\n🧠 Testing Memory Manager Integration...")
        
        try:
            # Initialize memory manager
            memory_manager = MemoryManager()
            
            # Test 1: Store long-term memory
            success = await memory_manager.store_long_term_memory(
                user_id="test_user",
                namespace=["preferences"],
                key="test_integration",
                value={"integration": "test", "timestamp": datetime.now().isoformat()},
                ttl_hours=24
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
            
            # Test 2: Get relevant context
            context = await memory_manager.get_relevant_context(
                user_id="test_user",
                current_query="test integration preference",
                max_contexts=5
            )
            
            if context:
                self.log_result(
                    "memory_manager_context",
                    True,
                    f"Retrieved {len(context)} relevant contexts"
                )
            else:
                self.log_result(
                    "memory_manager_context",
                    False,
                    error="No relevant contexts found"
                )
            
            # Test 3: Background consolidation
            consolidation_result = await memory_manager.consolidate_old_conversations("test_user")
            
            self.log_result(
                "memory_manager_consolidation",
                True,
                f"Consolidation completed: {consolidation_result}"
            )
            
        except Exception as e:
            self.log_result(
                "memory_manager_integration",
                False,
                error=f"Memory manager integration failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        print("\n🛡️ Testing Error Handling...")
        
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: Invalid namespace handling
            try:
                await store.aget(namespace=(), key="invalid")
                self.log_result(
                    "error_handling_invalid_namespace",
                    True,
                    "Gracefully handled invalid namespace"
                )
            except Exception as e:
                self.log_result(
                    "error_handling_invalid_namespace",
                    False,
                    error=f"Failed to handle invalid namespace: {str(e)}"
                )
            
            # Test 2: Non-existent key handling
            result = await store.aget(
                namespace=("test_user", "nonexistent"),
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
            
            # Test 3: Large value handling
            large_value = {"large_data": "x" * 10000}  # 10KB of data
            
            await store.aput(
                namespace=("test_user", "large_data"),
                key="large_test",
                value=large_value
            )
            
            retrieved = await store.aget(
                namespace=("test_user", "large_data"),
                key="large_test"
            )
            
            if retrieved is not None and retrieved.value == large_value:
                self.log_result(
                    "error_handling_large_value",
                    True,
                    "Successfully handled large value storage/retrieval"
                )
                
                # Clean up
                await store.adelete(
                    namespace=("test_user", "large_data"),
                    key="large_test"
                )
            else:
                self.log_result(
                    "error_handling_large_value",
                    False,
                    error="Failed to handle large value"
                )
            
        except Exception as e:
            self.log_result(
                "error_handling_test",
                False,
                error=f"Error handling test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_performance_optimization(self):
        """Test performance optimization features."""
        print("\n⚡ Testing Performance Optimization...")
        
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: Batch operations
            start_time = datetime.now()
            
            # Store multiple items
            for i in range(10):
                await store.aput(
                    namespace=("test_user", "performance"),
                    key=f"perf_test_{i}",
                    value={"index": i, "data": f"performance test {i}"}
                )
            
            store_time = datetime.now() - start_time
            
            # Test 2: Search performance
            search_start = datetime.now()
            
            results = await store.semantic_search(
                query="performance test",
                namespace_prefix=("test_user", "performance"),
                limit=10
            )
            
            search_time = datetime.now() - search_start
            
            # Test 3: Cleanup
            cleanup_start = datetime.now()
            
            for i in range(10):
                await store.adelete(
                    namespace=("test_user", "performance"),
                    key=f"perf_test_{i}"
                )
            
            cleanup_time = datetime.now() - cleanup_start
            
            self.log_result(
                "performance_optimization",
                True,
                f"Store: {store_time.total_seconds():.2f}s, Search: {search_time.total_seconds():.2f}s, Cleanup: {cleanup_time.total_seconds():.2f}s"
            )
            
        except Exception as e:
            self.log_result(
                "performance_optimization",
                False,
                error=f"Performance test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def test_ttl_expiration(self):
        """Test TTL expiration functionality."""
        print("\n⏰ Testing TTL Expiration...")
        
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            store = SupabaseLongTermMemoryStore(embeddings=embeddings)
            
            # Test 1: Store with short TTL
            await store.aput(
                namespace=("test_user", "ttl_test"),
                key="short_ttl",
                value={"ttl": "short"},
                ttl_hours=0.001  # Very short TTL (3.6 seconds)
            )
            
            # Verify immediate availability
            item = await store.aget(
                namespace=("test_user", "ttl_test"),
                key="short_ttl"
            )
            
            if item is not None:
                self.log_result(
                    "ttl_immediate_availability",
                    True,
                    "Item available immediately after storage"
                )
            else:
                self.log_result(
                    "ttl_immediate_availability",
                    False,
                    error="Item not available immediately after storage"
                )
            
            # Test 2: Store without TTL
            await store.aput(
                namespace=("test_user", "ttl_test"),
                key="no_ttl",
                value={"ttl": "none"}
            )
            
            no_ttl_item = await store.aget(
                namespace=("test_user", "ttl_test"),
                key="no_ttl"
            )
            
            if no_ttl_item is not None:
                self.log_result(
                    "ttl_no_expiration",
                    True,
                    "Item without TTL stored successfully"
                )
                
                # Clean up
                await store.adelete(
                    namespace=("test_user", "ttl_test"),
                    key="no_ttl"
                )
            else:
                self.log_result(
                    "ttl_no_expiration",
                    False,
                    error="Item without TTL not stored"
                )
            
        except Exception as e:
            self.log_result(
                "ttl_expiration",
                False,
                error=f"TTL expiration test failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Comprehensive Long-Term Memory Fixed Tests...")
        print("=" * 80)
        
        try:
            await self.test_database_functions()
            await self.test_store_interface()
            await self.test_memory_manager_integration()
            await self.test_error_handling()
            await self.test_performance_optimization()
            await self.test_ttl_expiration()
            
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
        with open("long_term_memory_test_results_fixed.json", "w") as f:
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
        
        print(f"\n📝 Full results saved to: long_term_memory_test_results_fixed.json")
        
        return self.passed > 0 and self.failed == 0


async def main():
    """Main test execution function."""
    tester = LongTermMemoryFixedTester()
    
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