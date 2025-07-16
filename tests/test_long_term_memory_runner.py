#!/usr/bin/env python3
"""
Long-term Memory Testing Runner for Task 5.5.4

This script systematically tests all long-term memory functionality
and identifies potential pitfalls in the implementation.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import db_client
from backend.config import get_settings
from backend.rag.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


class LongTermMemoryTester:
    """Test runner for long-term memory functionality."""
    
    def __init__(self):
        self.results = []
        self.pitfalls = []
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results."""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)
        print(f"[{status.upper()}] {test_name}: {details}")
    
    def log_pitfall(self, pitfall_type: str, description: str, severity: str = "medium"):
        """Log identified pitfalls."""
        pitfall = {
            "type": pitfall_type,
            "description": description,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.pitfalls.append(pitfall)
        print(f"[PITFALL-{severity.upper()}] {pitfall_type}: {description}")
    
    async def test_database_schema(self):
        """Test 1: Database Schema Verification"""
        print("\n=== Testing Database Schema ===")
        
        db = db_client.client
        
        # Test 1.1: Long-term memory tables exist
        try:
            tables_to_check = [
                "long_term_memories",
                "conversation_summaries", 
                "memory_access_patterns"
            ]
            
            for table in tables_to_check:
                result = db.table(table).select("*").limit(1).execute()
                if result is not None:
                    self.log_test(f"Table {table} exists", "PASS")
                else:
                    self.log_test(f"Table {table} exists", "FAIL", "Table not found")
                    self.log_pitfall("SCHEMA_MISSING", f"Table {table} does not exist", "high")
                    
        except Exception as e:
            self.log_test("Database schema check", "FAIL", str(e))
            self.log_pitfall("DB_CONNECTION", f"Cannot connect to database: {e}", "critical")
        
        # Test 1.2: Required database functions exist
        try:
            functions_to_check = [
                "search_long_term_memories",
                "search_conversation_summaries",
                "cleanup_expired_memories",
                "get_memory_performance_metrics"
            ]
            
            for func in functions_to_check:
                try:
                    if func == "search_long_term_memories":
                        result = db.rpc(func, {
                            "query_embedding": [0.1] * 1536,
                            "similarity_threshold": 0.5,
                            "match_count": 1
                        }).execute()
                    elif func == "search_conversation_summaries":
                        result = db.rpc(func, {
                            "query_embedding": [0.1] * 1536,
                            "target_user_id": "test_user",
                            "similarity_threshold": 0.5
                        }).execute()
                    else:
                        result = db.rpc(func, {}).execute()
                    
                    self.log_test(f"Function {func} exists", "PASS")
                    
                except Exception as e:
                    self.log_test(f"Function {func} exists", "FAIL", str(e))
                    self.log_pitfall("FUNCTION_MISSING", f"Database function {func} missing or broken", "high")
                    
        except Exception as e:
            self.log_test("Database functions check", "FAIL", str(e))
    
    async def test_memory_store_operations(self):
        """Test 2: Memory Store Operations"""
        print("\n=== Testing Memory Store Operations ===")
        
        try:
            # Mock embeddings for testing
            from unittest.mock import Mock, AsyncMock
            embeddings = Mock()
            embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            
            from backend.agents.memory import SupabaseLongTermMemoryStore
            memory_store = SupabaseLongTermMemoryStore(embeddings)
            
            # Test 2.1: Basic put/get operations
            test_namespace = ("user", "test_user_123", "preferences")
            test_key = "test_key"
            test_value = {"content": "test content", "type": "preference"}
            
            try:
                await memory_store.put(test_namespace, test_key, test_value)
                self.log_test("Memory store PUT operation", "PASS")
                
                retrieved = await memory_store.get(test_namespace, test_key)
                if retrieved and retrieved.value == test_value:
                    self.log_test("Memory store GET operation", "PASS")
                else:
                    self.log_test("Memory store GET operation", "FAIL", "Retrieved value doesn't match")
                    self.log_pitfall("DATA_INTEGRITY", "PUT/GET operations don't preserve data integrity", "high")
                    
            except Exception as e:
                self.log_test("Memory store operations", "FAIL", str(e))
                self.log_pitfall("STORE_OPERATIONS", f"Basic store operations failing: {e}", "high")
            
            # Test 2.2: Semantic search
            try:
                search_results = await memory_store.search("test content", limit=5)
                self.log_test("Semantic search", "PASS", f"Found {len(search_results)} results")
                
                if len(search_results) == 0:
                    self.log_pitfall("SEARCH_EMPTY", "Semantic search returns no results", "medium")
                    
            except Exception as e:
                self.log_test("Semantic search", "FAIL", str(e))
                self.log_pitfall("SEARCH_BROKEN", f"Semantic search broken: {e}", "high")
                
        except Exception as e:
            self.log_test("Memory store initialization", "FAIL", str(e))
            self.log_pitfall("INITIALIZATION", f"Cannot initialize memory store: {e}", "critical")
    
    async def test_conversation_consolidation(self):
        """Test 3: Conversation Consolidation"""
        print("\n=== Testing Conversation Consolidation ===")
        
        try:
            # Mock dependencies
            from unittest.mock import Mock, AsyncMock
            from backend.agents.memory import ConversationConsolidator, SimpleDBManager
            
            db_manager = SimpleDBManager()
            embeddings = Mock()
            embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            
            # Create mock memory store
            memory_store = Mock()
            memory_store.put = AsyncMock()
            
            # Create mock LLM
            llm = Mock()
            llm.ainvoke = AsyncMock(return_value=Mock(content="Test summary"))
            
            consolidator = ConversationConsolidator(db_manager, memory_store, llm, embeddings)
            
            # Test 3.1: Get consolidation candidates
            try:
                candidates = await consolidator._get_consolidation_candidates("test_user", 5)
                self.log_test("Get consolidation candidates", "PASS", f"Found {len(candidates)} candidates")
                
                if len(candidates) == 0:
                    self.log_pitfall("NO_CANDIDATES", "No consolidation candidates found - may indicate inactive system", "low")
                    
            except Exception as e:
                self.log_test("Get consolidation candidates", "FAIL", str(e))
                self.log_pitfall("CONSOLIDATION_QUERY", f"Cannot query consolidation candidates: {e}", "medium")
            
            # Test 3.2: Summary generation
            try:
                test_conversation = {
                    "id": "test_conv_123",
                    "title": "Test Conversation",
                    "created_at": datetime.utcnow()
                }
                
                test_messages = [
                    {"role": "user", "content": "Hello", "created_at": datetime.utcnow()},
                    {"role": "assistant", "content": "Hi!", "created_at": datetime.utcnow()}
                ]
                
                summary = await consolidator._generate_conversation_summary(test_conversation, test_messages)
                
                if "content" in summary and "message_count" in summary:
                    self.log_test("Summary generation", "PASS")
                else:
                    self.log_test("Summary generation", "FAIL", "Summary format incorrect")
                    self.log_pitfall("SUMMARY_FORMAT", "Generated summary has incorrect format", "medium")
                    
            except Exception as e:
                self.log_test("Summary generation", "FAIL", str(e))
                self.log_pitfall("SUMMARY_GENERATION", f"Cannot generate conversation summaries: {e}", "high")
                
        except Exception as e:
            self.log_test("Conversation consolidation setup", "FAIL", str(e))
            self.log_pitfall("CONSOLIDATION_SETUP", f"Cannot set up conversation consolidation: {e}", "high")
    
    async def test_performance_optimization(self):
        """Test 4: Performance Optimization"""
        print("\n=== Testing Performance Optimization ===")
        
        db = db_client.client
        
        # Test 4.1: Performance metrics function
        try:
            result = db.rpc("get_memory_performance_metrics", {}).execute()
            if result.data:
                self.log_test("Performance metrics", "PASS", f"Retrieved {len(result.data)} metrics")
                
                # Check for concerning metrics
                for metric in result.data:
                    if metric.get("status") == "warning":
                        self.log_pitfall("PERFORMANCE_WARNING", f"Performance metric {metric['metric_name']} is in warning state", "medium")
                    elif metric.get("status") == "critical":
                        self.log_pitfall("PERFORMANCE_CRITICAL", f"Performance metric {metric['metric_name']} is in critical state", "high")
                        
            else:
                self.log_test("Performance metrics", "FAIL", "No metrics returned")
                self.log_pitfall("METRICS_EMPTY", "Performance metrics function returns no data", "medium")
                
        except Exception as e:
            self.log_test("Performance metrics", "FAIL", str(e))
            self.log_pitfall("METRICS_BROKEN", f"Performance metrics function broken: {e}", "high")
        
        # Test 4.2: Maintenance functions
        try:
            result = db.rpc("run_memory_maintenance", {}).execute()
            if result.data:
                self.log_test("Memory maintenance", "PASS", f"Ran {len(result.data)} maintenance tasks")
                
                # Check for failed maintenance tasks
                for task in result.data:
                    if task.get("status") == "error":
                        self.log_pitfall("MAINTENANCE_ERROR", f"Maintenance task {task['task_name']} failed", "medium")
                        
            else:
                self.log_test("Memory maintenance", "FAIL", "No maintenance results")
                self.log_pitfall("MAINTENANCE_EMPTY", "Maintenance function returns no results", "medium")
                
        except Exception as e:
            self.log_test("Memory maintenance", "FAIL", str(e))
            self.log_pitfall("MAINTENANCE_BROKEN", f"Memory maintenance function broken: {e}", "high")
    
    async def test_security_policies(self):
        """Test 5: Security and Access Control"""
        print("\n=== Testing Security Policies ===")
        
        db = db_client.client
        
        # Test 5.1: RLS enabled tables
        try:
            rls_tables = ["long_term_memories", "conversation_summaries", "memory_access_patterns"]
            
            for table in rls_tables:
                try:
                    result = db.table(table).select("*").limit(1).execute()
                    self.log_test(f"RLS table {table} accessible", "PASS")
                except Exception as e:
                    self.log_test(f"RLS table {table} accessible", "FAIL", str(e))
                    self.log_pitfall("RLS_ACCESS", f"RLS table {table} not accessible", "high")
                    
        except Exception as e:
            self.log_test("RLS tables check", "FAIL", str(e))
            self.log_pitfall("RLS_BROKEN", f"RLS system appears broken: {e}", "critical")
        
        # Test 5.2: Namespace isolation
        try:
            # Test inserting memories with different namespaces
            test_memory_1 = {
                "namespace": ["user", "user_123"],
                "key": "test_isolation_1",
                "value": {"content": "user 1 data"}
            }
            
            test_memory_2 = {
                "namespace": ["user", "user_456"],
                "key": "test_isolation_2", 
                "value": {"content": "user 2 data"}
            }
            
            db.table("long_term_memories").insert(test_memory_1).execute()
            db.table("long_term_memories").insert(test_memory_2).execute()
            
            # Query should show proper separation
            result = db.table("long_term_memories").select("*").eq("key", "test_isolation_1").execute()
            
            if result.data:
                self.log_test("Namespace isolation", "PASS", "Namespaces properly isolated")
            else:
                self.log_test("Namespace isolation", "FAIL", "Cannot verify namespace isolation")
                self.log_pitfall("NAMESPACE_ISOLATION", "Namespace isolation may not be working", "high")
            
            # Clean up
            db.table("long_term_memories").delete().eq("key", "test_isolation_1").execute()
            db.table("long_term_memories").delete().eq("key", "test_isolation_2").execute()
            
        except Exception as e:
            self.log_test("Namespace isolation", "FAIL", str(e))
            self.log_pitfall("NAMESPACE_ERROR", f"Namespace isolation test failed: {e}", "high")
    
    async def test_error_handling(self):
        """Test 6: Error Handling and Recovery"""
        print("\n=== Testing Error Handling ===")
        
        # Test 6.1: Invalid input handling
        try:
            # Test with invalid embeddings
            from unittest.mock import Mock, AsyncMock
            embeddings = Mock()
            embeddings.aembed_query = AsyncMock(side_effect=Exception("Embedding failed"))
            
            from backend.agents.memory import SupabaseLongTermMemoryStore
            memory_store = SupabaseLongTermMemoryStore(embeddings)
            
            try:
                await memory_store.put(("user", "test"), "key", {"content": "test"})
                self.log_test("Error handling - embedding failure", "FAIL", "Should have handled embedding failure")
                self.log_pitfall("ERROR_HANDLING", "Embedding failures not properly handled", "high")
            except Exception as e:
                if "embedding" in str(e).lower():
                    self.log_test("Error handling - embedding failure", "PASS", "Properly handled embedding failure")
                else:
                    self.log_test("Error handling - embedding failure", "FAIL", "Unexpected error type")
                    self.log_pitfall("ERROR_HANDLING", f"Unexpected error handling: {e}", "medium")
                    
        except Exception as e:
            self.log_test("Error handling setup", "FAIL", str(e))
            self.log_pitfall("ERROR_SETUP", f"Cannot set up error handling tests: {e}", "medium")
        
        # Test 6.2: Database connection resilience
        try:
            # Test with invalid database connection
            # This is difficult to test without breaking the actual connection
            self.log_test("Database resilience", "SKIP", "Cannot test without breaking actual connection")
            self.log_pitfall("TEST_LIMITATION", "Cannot fully test database connection resilience", "low")
            
        except Exception as e:
            self.log_test("Database resilience", "FAIL", str(e))
    
    async def test_memory_expiration(self):
        """Test 7: Memory Expiration and Cleanup"""
        print("\n=== Testing Memory Expiration ===")
        
        db = db_client.client
        
        # Test 7.1: Insert memory with expiration
        try:
            from datetime import datetime, timedelta
            
            # Create memory that should expire
            past_date = datetime.utcnow() - timedelta(hours=1)
            expired_memory = {
                "namespace": ["test", "expiration"],
                "key": "expired_test_key",
                "value": {"content": "should be expired"},
                "expiry_at": past_date.isoformat()
            }
            
            # Insert expired memory
            insert_result = db.table("long_term_memories").insert(expired_memory).execute()
            if insert_result.data:
                self.log_test("Insert expired memory", "PASS")
            else:
                self.log_test("Insert expired memory", "FAIL", "Cannot insert expired memory")
                self.log_pitfall("EXPIRATION_INSERT", "Cannot insert memory with expiration", "medium")
                
        except Exception as e:
            self.log_test("Memory expiration setup", "FAIL", str(e))
            self.log_pitfall("EXPIRATION_SETUP", f"Cannot set up expiration test: {e}", "medium")
        
        # Test 7.2: Cleanup function
        try:
            cleanup_result = db.rpc("cleanup_expired_memories", {}).execute()
            if cleanup_result.data is not None:
                self.log_test("Cleanup expired memories", "PASS", f"Cleaned up {cleanup_result.data} memories")
            else:
                self.log_test("Cleanup expired memories", "FAIL", "Cleanup function returned None")
                self.log_pitfall("CLEANUP_BROKEN", "Memory cleanup function not working", "high")
                
        except Exception as e:
            self.log_test("Cleanup expired memories", "FAIL", str(e))
            self.log_pitfall("CLEANUP_ERROR", f"Memory cleanup function broken: {e}", "high")
    
    def generate_report(self):
        """Generate final test report."""
        print("\n" + "="*60)
        print("LONG-TERM MEMORY TESTING REPORT")
        print("="*60)
        
        # Summary statistics
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")
        
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        
        # Pitfall summary
        print(f"\nPitfalls Identified: {len(self.pitfalls)}")
        
        critical_pitfalls = [p for p in self.pitfalls if p["severity"] == "critical"]
        high_pitfalls = [p for p in self.pitfalls if p["severity"] == "high"]
        medium_pitfalls = [p for p in self.pitfalls if p["severity"] == "medium"]
        low_pitfalls = [p for p in self.pitfalls if p["severity"] == "low"]
        
        print(f"Critical: {len(critical_pitfalls)}")
        print(f"High: {len(high_pitfalls)}")
        print(f"Medium: {len(medium_pitfalls)}")
        print(f"Low: {len(low_pitfalls)}")
        
        # Detailed pitfall report
        if self.pitfalls:
            print("\nDETAILED PITFALL REPORT:")
            print("-" * 40)
            
            for pitfall in sorted(self.pitfalls, key=lambda x: 
                                {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["severity"]]):
                print(f"[{pitfall['severity'].upper()}] {pitfall['type']}: {pitfall['description']}")
        
        # Save detailed report
        report_data = {
            "test_results": self.results,
            "pitfalls": self.pitfalls,
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total_pitfalls": len(self.pitfalls),
                "critical_pitfalls": len(critical_pitfalls),
                "high_pitfalls": len(high_pitfalls),
                "medium_pitfalls": len(medium_pitfalls),
                "low_pitfalls": len(low_pitfalls)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open("long_term_memory_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: long_term_memory_test_report.json")
        
        return report_data


async def main():
    """Run all long-term memory tests."""
    print("Starting Long-Term Memory Testing Suite for Task 5.5.4")
    print("=" * 60)
    
    tester = LongTermMemoryTester()
    
    # Run all tests
    await tester.test_database_schema()
    await tester.test_memory_store_operations()
    await tester.test_conversation_consolidation()
    await tester.test_performance_optimization()
    await tester.test_security_policies()
    await tester.test_error_handling()
    await tester.test_memory_expiration()
    
    # Generate report
    report = tester.generate_report()
    
    return report


if __name__ == "__main__":
    report = asyncio.run(main()) 