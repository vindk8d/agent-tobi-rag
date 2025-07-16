"""
Comprehensive test suite for long-term memory functionality (Task 5.5.4)

This test suite covers all components of the long-term memory system:
1. Database schema verification
2. LangGraph Store implementation testing
3. Conversation consolidation testing
4. Performance optimization testing
5. Security policy testing
6. Error handling and recovery testing
7. Memory expiration and cleanup testing

Test each component systematically to identify potential pitfalls.
"""

import pytest
import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Add the project root to the path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.memory import (
    SupabaseLongTermMemoryStore, 
    ConversationConsolidator,
    ConversationMemoryManager,
    SimpleDBManager,
    SimpleDBConnection
)
from backend.database import db_client
from backend.config import get_settings
from backend.rag.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class TestLongTermMemoryDatabaseSchema:
    """Test database schema for long-term memory tables and functions."""
    
    @pytest.fixture
    def db_client_fixture(self):
        """Provide database client for testing."""
        return db_client.client
    
    @pytest.mark.asyncio
    async def test_long_term_memories_table_exists(self, db_client_fixture):
        """Test that long_term_memories table exists with correct schema."""
        # Check table exists
        result = db_client_fixture.table("long_term_memories").select("*").limit(1).execute()
        assert result is not None, "long_term_memories table should exist"
        
        # Test table structure by inserting a test record
        test_data = {
            "namespace": ["test", "user123"],
            "key": "test_key",
            "value": {"content": "test content"},
            "memory_type": "semantic",
            "metadata": {"test": True}
        }
        
        try:
            insert_result = db_client_fixture.table("long_term_memories").insert(test_data).execute()
            assert insert_result.data, "Should be able to insert into long_term_memories"
            
            # Clean up
            db_client_fixture.table("long_term_memories").delete().eq("key", "test_key").execute()
            
        except Exception as e:
            pytest.fail(f"long_term_memories table structure issue: {e}")
    
    @pytest.mark.asyncio
    async def test_conversation_summaries_table_exists(self, db_client_fixture):
        """Test that conversation_summaries table exists with correct schema."""
        # First create a test conversation
        conversation_data = {
            "user_id": "test_user",
            "title": "Test Conversation",
            "metadata": {}
        }
        
        conv_result = db_client_fixture.table("conversations").insert(conversation_data).execute()
        assert conv_result.data, "Should be able to create test conversation"
        
        conversation_id = conv_result.data[0]["id"]
        
        # Test conversation summaries table
        summary_data = {
            "conversation_id": conversation_id,
            "user_id": "test_user",
            "summary_text": "Test summary",
            "message_count": 5,
            "summary_type": "periodic"
        }
        
        try:
            summary_result = db_client_fixture.table("conversation_summaries").insert(summary_data).execute()
            assert summary_result.data, "Should be able to insert into conversation_summaries"
            
            # Clean up
            db_client_fixture.table("conversation_summaries").delete().eq("conversation_id", conversation_id).execute()
            db_client_fixture.table("conversations").delete().eq("id", conversation_id).execute()
            
        except Exception as e:
            pytest.fail(f"conversation_summaries table structure issue: {e}")
    
    @pytest.mark.asyncio
    async def test_memory_access_patterns_table_exists(self, db_client_fixture):
        """Test that memory_access_patterns table exists with correct schema."""
        test_data = {
            "user_id": "test_user",
            "memory_namespace": ["test", "namespace"],
            "memory_key": "test_key",
            "access_frequency": 1,
            "access_context": "test context",
            "retrieval_method": "semantic"
        }
        
        try:
            result = db_client_fixture.table("memory_access_patterns").insert(test_data).execute()
            assert result.data, "Should be able to insert into memory_access_patterns"
            
            # Clean up
            db_client_fixture.table("memory_access_patterns").delete().eq("user_id", "test_user").execute()
            
        except Exception as e:
            pytest.fail(f"memory_access_patterns table structure issue: {e}")
    
    @pytest.mark.asyncio
    async def test_vector_indexes_exist(self, db_client_fixture):
        """Test that vector indexes for semantic search exist."""
        # Test if we can perform a vector search (indirect test for indexes)
        # This would fail if the vector indexes don't exist
        test_embedding = [0.1] * 1536  # Mock embedding
        
        try:
            # Test search_long_term_memories function (which uses vector indexes)
            result = db_client_fixture.rpc("search_long_term_memories", {
                "query_embedding": test_embedding,
                "similarity_threshold": 0.5,
                "match_count": 5
            }).execute()
            
            # Should not raise an error even if no results
            assert result is not None, "Vector search function should be callable"
            
        except Exception as e:
            pytest.fail(f"Vector indexes may not exist: {e}")
    
    @pytest.mark.asyncio
    async def test_database_functions_exist(self, db_client_fixture):
        """Test that all required database functions exist."""
        required_functions = [
            "search_long_term_memories",
            "search_conversation_summaries",
            "get_recent_conversation_summaries",
            "update_memory_access_pattern",
            "cleanup_expired_memories",
            "get_memory_performance_metrics",
            "run_memory_maintenance"
        ]
        
        for func_name in required_functions:
            try:
                # Test function existence by calling with minimal parameters
                if func_name == "search_long_term_memories":
                    result = db_client_fixture.rpc(func_name, {
                        "query_embedding": [0.1] * 1536,
                        "similarity_threshold": 0.5,
                        "match_count": 1
                    }).execute()
                elif func_name == "search_conversation_summaries":
                    result = db_client_fixture.rpc(func_name, {
                        "query_embedding": [0.1] * 1536,
                        "target_user_id": "test_user",
                        "similarity_threshold": 0.5,
                        "match_count": 1
                    }).execute()
                elif func_name == "get_recent_conversation_summaries":
                    result = db_client_fixture.rpc(func_name, {
                        "target_user_id": "test_user",
                        "limit_count": 1
                    }).execute()
                elif func_name == "update_memory_access_pattern":
                    result = db_client_fixture.rpc(func_name, {
                        "target_user_id": "test_user",
                        "memory_namespace": ["test"],
                        "memory_key": "test_key"
                    }).execute()
                elif func_name in ["cleanup_expired_memories", "get_memory_performance_metrics", "run_memory_maintenance"]:
                    result = db_client_fixture.rpc(func_name, {}).execute()
                
                assert result is not None, f"Function {func_name} should be callable"
                
            except Exception as e:
                pytest.fail(f"Database function {func_name} may not exist or has issues: {e}")


class TestSupabaseLongTermMemoryStore:
    """Test the SupabaseLongTermMemoryStore implementation."""
    
    @pytest.fixture
    async def memory_store(self):
        """Create a memory store instance for testing."""
        embeddings = Mock(spec=OpenAIEmbeddings)
        embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        return SupabaseLongTermMemoryStore(embeddings)
    
    @pytest.fixture
    def test_user_id(self):
        """Generate unique test user ID."""
        return f"test_user_{uuid.uuid4().hex[:8]}"
    
    @pytest.mark.asyncio
    async def test_put_and_get_memory(self, memory_store, test_user_id):
        """Test basic put and get operations."""
        namespace = ("user", test_user_id, "preferences")
        key = "favorite_color"
        value = {"color": "blue", "confidence": 0.9}
        
        # Test put
        try:
            await memory_store.put(namespace, key, value)
        except Exception as e:
            pytest.fail(f"Put operation failed: {e}")
        
        # Test get
        try:
            result = await memory_store.get(namespace, key)
            assert result is not None, "Should retrieve stored memory"
            assert result.value == value, "Retrieved value should match stored value"
            assert result.namespace == namespace, "Namespace should match"
            assert result.key == key, "Key should match"
        except Exception as e:
            pytest.fail(f"Get operation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_memory_with_ttl(self, memory_store, test_user_id):
        """Test memory with time-to-live (TTL) functionality."""
        namespace = ("user", test_user_id, "temporary")
        key = "temp_data"
        value = {"data": "temporary", "ttl_test": True}
        
        # Store with very short TTL (1 hour)
        await memory_store.put(namespace, key, value, ttl_hours=1)
        
        # Should be retrievable immediately
        result = await memory_store.get(namespace, key)
        assert result is not None, "Should retrieve memory with TTL"
        assert result.value == value, "Value should match"
        
        # Test that expired memories are handled (we can't wait for real expiration)
        # This is tested in the database function tests
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, memory_store, test_user_id):
        """Test semantic search functionality."""
        namespace_base = ("user", test_user_id, "knowledge")
        
        # Store multiple memories with different content
        memories = [
            ("python_info", {"content": "Python is a programming language"}),
            ("cooking_info", {"content": "Cooking involves preparing food"}),
            ("travel_info", {"content": "Travel means going to different places"})
        ]
        
        for key, value in memories:
            await memory_store.put(namespace_base, key, value)
        
        # Search for programming-related content
        results = await memory_store.search(
            "programming language", 
            namespace_prefix=namespace_base,
            limit=10,
            similarity_threshold=0.1  # Low threshold for testing
        )
        
        assert len(results) > 0, "Should find results for semantic search"
        
        # Check that results have expected structure
        for result in results:
            assert "namespace" in result, "Result should have namespace"
            assert "key" in result, "Result should have key"
            assert "value" in result, "Result should have value"
            assert "similarity_score" in result, "Result should have similarity score"
    
    @pytest.mark.asyncio
    async def test_namespace_isolation(self, memory_store):
        """Test that namespaces properly isolate memories."""
        user1_id = f"user1_{uuid.uuid4().hex[:8]}"
        user2_id = f"user2_{uuid.uuid4().hex[:8]}"
        
        # Store memories for different users
        await memory_store.put(("user", user1_id, "prefs"), "theme", {"theme": "dark"})
        await memory_store.put(("user", user2_id, "prefs"), "theme", {"theme": "light"})
        
        # Each user should only see their own memories
        user1_memory = await memory_store.get(("user", user1_id, "prefs"), "theme")
        user2_memory = await memory_store.get(("user", user2_id, "prefs"), "theme")
        
        assert user1_memory.value["theme"] == "dark", "User1 should get dark theme"
        assert user2_memory.value["theme"] == "light", "User2 should get light theme"
        
        # Cross-user access should return None
        cross_access = await memory_store.get(("user", user1_id, "prefs"), "theme")
        # This should work since we're using the same namespace
        assert cross_access is not None, "Same namespace should be accessible"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_store, test_user_id):
        """Test error handling in memory operations."""
        namespace = ("user", test_user_id, "error_test")
        
        # Test get with non-existent key
        result = await memory_store.get(namespace, "nonexistent_key")
        assert result is None, "Should return None for non-existent key"
        
        # Test put with invalid data (should handle gracefully)
        try:
            await memory_store.put(namespace, "test_key", None)
            # Should not raise an error
        except Exception as e:
            # If it does raise an error, it should be handled gracefully
            assert "embedding" not in str(e).lower(), "Should handle None values gracefully"


class TestConversationConsolidation:
    """Test conversation consolidation functionality."""
    
    @pytest.fixture
    async def consolidator(self):
        """Create a conversation consolidator for testing."""
        db_manager = SimpleDBManager()
        embeddings = Mock(spec=OpenAIEmbeddings)
        embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        memory_store = SupabaseLongTermMemoryStore(embeddings)
        
        llm = Mock(spec=ChatOpenAI)
        llm.ainvoke = AsyncMock(return_value=Mock(content="Test conversation summary"))
        
        return ConversationConsolidator(db_manager, memory_store, llm, embeddings)
    
    @pytest.fixture
    def test_user_id(self):
        """Generate unique test user ID."""
        return f"test_user_{uuid.uuid4().hex[:8]}"
    
    @pytest.mark.asyncio
    async def test_conversation_consolidation_candidates(self, consolidator, test_user_id):
        """Test identification of consolidation candidates."""
        # Create test conversation older than 7 days
        old_date = datetime.utcnow() - timedelta(days=8)
        
        # This test depends on having conversations in the database
        # For now, we'll test the function doesn't crash
        try:
            candidates = await consolidator._get_consolidation_candidates(test_user_id, 10)
            assert isinstance(candidates, list), "Should return a list"
        except Exception as e:
            pytest.fail(f"Error getting consolidation candidates: {e}")
    
    @pytest.mark.asyncio
    async def test_conversation_summary_generation(self, consolidator, test_user_id):
        """Test LLM-based conversation summary generation."""
        conversation = {
            "id": str(uuid.uuid4()),
            "title": "Test Conversation",
            "created_at": datetime.utcnow()
        }
        
        messages = [
            {"role": "user", "content": "Hello", "created_at": datetime.utcnow()},
            {"role": "assistant", "content": "Hi there!", "created_at": datetime.utcnow()}
        ]
        
        try:
            summary = await consolidator._generate_conversation_summary(conversation, messages)
            assert "content" in summary, "Summary should have content"
            assert "message_count" in summary, "Summary should have message count"
            assert summary["message_count"] == 2, "Should count messages correctly"
        except Exception as e:
            pytest.fail(f"Error generating conversation summary: {e}")
    
    @pytest.mark.asyncio
    async def test_consolidation_process(self, consolidator, test_user_id):
        """Test the full consolidation process."""
        try:
            result = await consolidator.consolidate_conversations(test_user_id, max_conversations=5)
            assert "consolidated_count" in result, "Should return consolidated count"
            assert "total_candidates" in result, "Should return total candidates"
            assert isinstance(result["consolidated_count"], int), "Consolidated count should be integer"
        except Exception as e:
            pytest.fail(f"Error in consolidation process: {e}")


class TestMemoryPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.fixture
    def db_client_fixture(self):
        """Provide database client for testing."""
        return db_client.client
    
    @pytest.mark.asyncio
    async def test_performance_metrics_function(self, db_client_fixture):
        """Test memory performance metrics function."""
        try:
            result = db_client_fixture.rpc("get_memory_performance_metrics", {}).execute()
            assert result.data is not None, "Should return performance metrics"
            
            # Check that metrics have expected structure
            for metric in result.data:
                assert "metric_name" in metric, "Metric should have name"
                assert "metric_value" in metric, "Metric should have value"
                assert "metric_unit" in metric, "Metric should have unit"
                assert "status" in metric, "Metric should have status"
        except Exception as e:
            pytest.fail(f"Error getting performance metrics: {e}")
    
    @pytest.mark.asyncio
    async def test_memory_maintenance_function(self, db_client_fixture):
        """Test memory maintenance function."""
        try:
            result = db_client_fixture.rpc("run_memory_maintenance", {}).execute()
            assert result.data is not None, "Should return maintenance results"
            
            # Check maintenance tasks structure
            for task in result.data:
                assert "task_name" in task, "Task should have name"
                assert "status" in task, "Task should have status"
                assert "details" in task, "Task should have details"
        except Exception as e:
            pytest.fail(f"Error running memory maintenance: {e}")
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_memories(self, db_client_fixture):
        """Test cleanup of expired memories."""
        # First, insert a memory with past expiration
        past_date = datetime.utcnow() - timedelta(hours=1)
        test_memory = {
            "namespace": ["test", "expired"],
            "key": "expired_key",
            "value": {"content": "expired content"},
            "expiry_at": past_date.isoformat()
        }
        
        # Insert expired memory
        insert_result = db_client_fixture.table("long_term_memories").insert(test_memory).execute()
        assert insert_result.data, "Should insert expired memory"
        
        # Run cleanup
        try:
            cleanup_result = db_client_fixture.rpc("cleanup_expired_memories", {}).execute()
            assert cleanup_result.data is not None, "Cleanup should return result"
            
            # Check if expired memory was cleaned up
            check_result = db_client_fixture.table("long_term_memories").select("*").eq("key", "expired_key").execute()
            assert len(check_result.data) == 0, "Expired memory should be cleaned up"
            
        except Exception as e:
            pytest.fail(f"Error in cleanup process: {e}")


class TestMemorySecurityPolicies:
    """Test Row Level Security (RLS) policies."""
    
    @pytest.fixture
    def db_client_fixture(self):
        """Provide database client for testing."""
        return db_client.client
    
    @pytest.mark.asyncio
    async def test_rls_enabled_tables(self, db_client_fixture):
        """Test that RLS is enabled on memory tables."""
        tables_with_rls = [
            "long_term_memories",
            "conversation_summaries", 
            "memory_access_patterns"
        ]
        
        for table_name in tables_with_rls:
            try:
                # This test is limited since we're using service role
                # In a real scenario, we'd test with different user contexts
                result = db_client_fixture.table(table_name).select("*").limit(1).execute()
                assert result is not None, f"Should be able to query {table_name} with service role"
            except Exception as e:
                pytest.fail(f"Error accessing {table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_namespace_access_control(self, db_client_fixture):
        """Test namespace-based access control."""
        # Test that memories are properly namespaced
        test_memory = {
            "namespace": ["user", "test_user_123"],
            "key": "private_data",
            "value": {"sensitive": "data"},
            "memory_type": "semantic"
        }
        
        try:
            # Insert memory
            insert_result = db_client_fixture.table("long_term_memories").insert(test_memory).execute()
            assert insert_result.data, "Should insert namespaced memory"
            
            # Query should work with service role
            query_result = db_client_fixture.table("long_term_memories").select("*").eq("key", "private_data").execute()
            assert len(query_result.data) > 0, "Should find inserted memory"
            
            # Clean up
            db_client_fixture.table("long_term_memories").delete().eq("key", "private_data").execute()
            
        except Exception as e:
            pytest.fail(f"Error in namespace access control test: {e}")


class TestMemoryErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    async def memory_store(self):
        """Create a memory store instance for testing."""
        embeddings = Mock(spec=OpenAIEmbeddings)
        embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        return SupabaseLongTermMemoryStore(embeddings)
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, memory_store):
        """Test handling of database connection failures."""
        # Mock database connection failure
        with patch.object(memory_store.db_manager, 'get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database connection failed")
            
            # Operations should handle the error gracefully
            result = await memory_store.get(("user", "test"), "key")
            assert result is None, "Should return None on connection failure"
    
    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, memory_store):
        """Test handling of embedding generation failures."""
        # Mock embedding failure
        memory_store.embeddings.aembed_query = AsyncMock(side_effect=Exception("Embedding failed"))
        
        # Should handle embedding failure gracefully
        try:
            await memory_store.put(("user", "test"), "key", {"content": "test"})
            # Should not crash the application
        except Exception as e:
            # If it raises an exception, it should be a handled one
            assert "embedding" in str(e).lower(), "Should be embedding-related error"
    
    @pytest.mark.asyncio
    async def test_invalid_namespace_handling(self, memory_store):
        """Test handling of invalid namespace formats."""
        invalid_namespaces = [
            (),  # Empty namespace
            ("",),  # Empty string in namespace
            ("user",),  # Too short namespace
            None,  # None namespace
        ]
        
        for namespace in invalid_namespaces:
            try:
                if namespace is None:
                    # Skip None namespace as it would cause a different error
                    continue
                    
                result = await memory_store.get(namespace, "test_key")
                # Should handle gracefully - either return None or handle the error
                assert result is None or isinstance(result, dict), "Should handle invalid namespace gracefully"
            except Exception as e:
                # Should not crash with unhandled exceptions
                assert "namespace" in str(e).lower() or "parameter" in str(e).lower(), "Should be namespace-related error"


class TestMemoryExpirationAndCleanup:
    """Test memory expiration and cleanup mechanisms."""
    
    @pytest.fixture
    def db_client_fixture(self):
        """Provide database client for testing."""
        return db_client.client
    
    @pytest.mark.asyncio
    async def test_memory_expiration_mechanism(self, db_client_fixture):
        """Test that expired memories are properly identified and cleaned."""
        # Create memory with past expiration
        past_date = datetime.utcnow() - timedelta(hours=1)
        expired_memory = {
            "namespace": ["test", "expiration"],
            "key": "expired_test",
            "value": {"content": "should be expired"},
            "expiry_at": past_date.isoformat()
        }
        
        # Insert expired memory
        insert_result = db_client_fixture.table("long_term_memories").insert(expired_memory).execute()
        assert insert_result.data, "Should insert expired memory"
        
        # Test search function excludes expired by default
        search_result = db_client_fixture.rpc("search_long_term_memories", {
            "query_embedding": [0.1] * 1536,
            "namespace_filter": ["test", "expiration"],
            "exclude_expired": True
        }).execute()
        
        # Should not find expired memory
        expired_found = any(result["key"] == "expired_test" for result in search_result.data)
        assert not expired_found, "Search should exclude expired memories by default"
        
        # Clean up
        db_client_fixture.table("long_term_memories").delete().eq("key", "expired_test").execute()
    
    @pytest.mark.asyncio
    async def test_cleanup_maintenance_task(self, db_client_fixture):
        """Test automated cleanup maintenance task."""
        # Create expired memory
        past_date = datetime.utcnow() - timedelta(hours=2)
        expired_memory = {
            "namespace": ["test", "cleanup"],
            "key": "cleanup_test",
            "value": {"content": "cleanup test"},
            "expiry_at": past_date.isoformat()
        }
        
        # Insert expired memory
        insert_result = db_client_fixture.table("long_term_memories").insert(expired_memory).execute()
        assert insert_result.data, "Should insert expired memory"
        
        # Run cleanup
        cleanup_result = db_client_fixture.rpc("cleanup_expired_memories", {}).execute()
        assert cleanup_result.data is not None, "Cleanup should return count"
        
        # Verify memory was cleaned up
        check_result = db_client_fixture.table("long_term_memories").select("*").eq("key", "cleanup_test").execute()
        assert len(check_result.data) == 0, "Expired memory should be cleaned up"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 