"""
Comprehensive Frontend Compatibility Integration Tests

Task 5.4: Test frontend compatibility with message and summary queries

This module provides complete integration testing to ensure that the streamlined
memory management system maintains full compatibility with frontend applications,
validating that:
1. All frontend API calls continue to work unchanged
2. Response formats match frontend expectations
3. Role mapping and data transformations work correctly
4. Frontend components can successfully fetch and display data
5. No breaking changes were introduced in the API layer

Test Coverage:
1. Memory Debug API endpoints used by frontend dev tools
2. Chat message loading and role normalization
3. Conversation summary fetching for watchtower components
4. User message retrieval with proper formatting
5. API response structure validation for frontend consumption
6. Error handling and fallback behavior
"""

import asyncio
import pytest
import pytest_asyncio
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

# Import test modules
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from core.database import db_client

# Skip tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)


class FrontendCompatibilityTestHelper:
    """Helper class for frontend compatibility integration testing."""
    
    def __init__(self):
        self.test_conversation_ids: List[str] = []
        self.test_user_ids: List[str] = []
        self.client: Optional[TestClient] = None
        self.base_url = "http://localhost:8000"
    
    async def setup(self):
        """Set up test environment."""
        try:
            from main import app
            self.client = TestClient(app)
        except Exception as e:
            print(f"Warning: Could not set up TestClient: {e}")
    
    async def cleanup(self):
        """Clean up test data and resources."""
        # Clean up test messages
        for conversation_id in self.test_conversation_ids:
            try:
                await asyncio.to_thread(
                    lambda cid=conversation_id: db_client.client.table("messages")
                    .delete()
                    .eq("conversation_id", cid)
                    .like("content", "Frontend Test%")
                    .execute()
                )
            except Exception as e:
                print(f"Warning: Failed to clean up test messages for conversation {conversation_id}: {e}")
    
    async def get_existing_user_with_messages(self) -> tuple[str, str]:
        """Get a user with existing messages for testing."""
        def _get_user_with_messages():
            # Get users who have messages
            return (db_client.client.table("messages")
                   .select("user_id,conversation_id")
                   .limit(1)
                   .execute())
        
        result = await asyncio.to_thread(_get_user_with_messages)
        
        if result.data and len(result.data) > 0:
            message = result.data[0]
            user_id = message["user_id"]
            conversation_id = message["conversation_id"]
            
            self.test_user_ids.append(user_id)
            self.test_conversation_ids.append(conversation_id)
            
            return user_id, conversation_id
        else:
            # Fallback
            user_id = str(uuid.uuid4())
            conversation_id = str(uuid.uuid4())
            self.test_user_ids.append(user_id)
            self.test_conversation_ids.append(conversation_id)
            return user_id, conversation_id
    
    async def make_api_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request using TestClient."""
        if self.client:
            if method.upper() == "GET":
                response = self.client.get(endpoint, **kwargs)
            elif method.upper() == "POST":
                response = self.client.post(endpoint, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return {
                "status_code": response.status_code,
                "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
                "text": response.text,
                "headers": dict(response.headers)
            }
        else:
            raise RuntimeError("TestClient not available")
    
    def validate_frontend_message_format(self, message: Dict[str, Any]) -> bool:
        """Validate that a message matches frontend expectations."""
        # Based on frontend code in memorycheck/page.tsx and dualagentdebug/page.tsx
        required_fields = ["id", "role", "content", "created_at"]
        
        for field in required_fields:
            if field not in message:
                return False
        
        # Validate role values that frontend expects
        valid_roles = ["human", "ai", "user", "assistant", "system", "tool"]
        if message["role"] not in valid_roles:
            return False
        
        # Validate data types
        if not isinstance(message["content"], str) or len(message["content"]) == 0:
            return False
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(message["created_at"].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return False
        
        return True
    
    def validate_frontend_summary_format(self, summary: Dict[str, Any]) -> bool:
        """Validate that a summary matches frontend expectations."""
        # Based on ConversationSummary.tsx
        required_fields = ["id", "conversation_id", "summary_text", "message_count", "created_at"]
        
        for field in required_fields:
            if field not in summary:
                return False
        
        # Validate data types
        if not isinstance(summary["summary_text"], str) or len(summary["summary_text"]) == 0:
            return False
        
        if not isinstance(summary["message_count"], int) or summary["message_count"] <= 0:
            return False
        
        return True
    
    def simulate_frontend_role_normalization(self, role: str) -> str:
        """Simulate the role normalization logic used in frontend."""
        # Based on normalizeRole function in dualagentdebug/page.tsx
        if role in ['human', 'user']:
            return 'user'
        elif role in ['ai', 'assistant']:
            return 'assistant'
        else:
            return role  # Keep as-is for system, tool, etc.


@pytest_asyncio.fixture
async def test_helper():
    """Fixture providing test helper with setup and cleanup."""
    helper = FrontendCompatibilityTestHelper()
    await helper.setup()
    try:
        yield helper
    finally:
        await helper.cleanup()


class TestFrontendCompatibility:
    """Test suite for frontend compatibility integration."""
    
    @pytest.mark.asyncio
    async def test_memory_debug_messages_frontend_format(self, test_helper: FrontendCompatibilityTestHelper):
        """Test that memory debug messages API returns data in frontend-compatible format."""
        print("\n🧪 Testing Memory Debug Messages Frontend Format")
        
        # Get user with existing messages
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Test the endpoint used by frontend dev tools
        endpoint = f"/api/v1/memory-debug/users/{user_id}/messages"
        response = await test_helper.make_api_request("GET", endpoint, params={"limit": 10})
        
        # Validate response structure
        assert response["status_code"] == 200, f"Expected 200, got {response['status_code']}: {response['text']}"
        
        if response["json"]:
            data = response["json"]
            
            # Validate API response structure expected by frontend
            assert "success" in data, "Frontend expects 'success' field"
            assert "data" in data, "Frontend expects 'data' field"
            assert isinstance(data["data"], list), "Frontend expects 'data' to be a list"
            
            # If messages exist, validate each message format
            if data["data"]:
                for message in data["data"]:
                    assert test_helper.validate_frontend_message_format(message), \
                        f"Message format invalid for frontend: {message}"
                    
                    # Test role normalization that frontend performs
                    normalized_role = test_helper.simulate_frontend_role_normalization(message["role"])
                    assert normalized_role in ["user", "assistant", "system", "tool"], \
                        f"Role normalization failed: {message['role']} -> {normalized_role}"
                
                print(f"✅ Validated {len(data['data'])} messages in frontend-compatible format")
        
        print("✅ Memory debug messages frontend format test passed")
    
    @pytest.mark.asyncio
    async def test_conversation_summaries_frontend_format(self, test_helper: FrontendCompatibilityTestHelper):
        """Test that conversation summaries API returns data in frontend-compatible format."""
        print("\n🧪 Testing Conversation Summaries Frontend Format")
        
        # Get user with existing data
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Test the endpoint used by frontend watchtower component
        endpoint = f"/api/v1/memory-debug/users/{user_id}/conversation-summaries"
        response = await test_helper.make_api_request("GET", endpoint)
        
        # Validate response structure
        assert response["status_code"] == 200, f"Expected 200, got {response['status_code']}: {response['text']}"
        
        if response["json"]:
            data = response["json"]
            
            # Validate API response structure expected by frontend
            assert "success" in data, "Frontend expects 'success' field"
            assert "data" in data, "Frontend expects 'data' field"
            assert isinstance(data["data"], list), "Frontend expects 'data' to be a list"
            
            # If summaries exist, validate each summary format
            if data["data"]:
                for summary in data["data"]:
                    assert test_helper.validate_frontend_summary_format(summary), \
                        f"Summary format invalid for frontend: {summary}"
                    
                    # Validate fields that frontend ConversationSummary.tsx expects
                    expected_fields = ["id", "conversation_id", "summary_text", "summary_type", 
                                     "message_count", "created_at", "consolidation_status"]
                    for field in expected_fields:
                        if field in summary:  # Some fields might be optional
                            assert summary[field] is not None, f"Field '{field}' should not be null"
                
                print(f"✅ Validated {len(data['data'])} summaries in frontend-compatible format")
        
        print("✅ Conversation summaries frontend format test passed")
    
    @pytest.mark.asyncio
    async def test_frontend_message_loading_workflow(self, test_helper: FrontendCompatibilityTestHelper):
        """Test the complete message loading workflow as used by frontend."""
        print("\n🧪 Testing Frontend Message Loading Workflow")
        
        # Get user with existing messages
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Step 1: Load messages as frontend memorycheck page does
        messages_endpoint = f"/api/v1/memory-debug/users/{user_id}/messages"
        messages_response = await test_helper.make_api_request("GET", messages_endpoint)
        
        assert messages_response["status_code"] == 200
        
        if messages_response["json"] and messages_response["json"]["data"]:
            messages = messages_response["json"]["data"]
            
            # Step 2: Simulate frontend message processing
            # Convert backend messages to frontend format (as done in memorycheck/page.tsx)
            frontend_messages = []
            for msg in messages:
                frontend_msg = {
                    "id": msg["id"],
                    "role": "user" if msg["role"] == "human" else "assistant" if msg["role"] == "ai" else msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["created_at"]  # Frontend converts this to Date object
                }
                frontend_messages.append(frontend_msg)
            
            # Step 3: Validate frontend message processing worked correctly
            assert len(frontend_messages) > 0, "Should have processed messages"
            
            for frontend_msg in frontend_messages:
                assert "id" in frontend_msg
                assert "role" in frontend_msg
                assert "content" in frontend_msg
                assert "timestamp" in frontend_msg
                assert frontend_msg["role"] in ["user", "assistant", "system", "tool"]
            
            # Step 4: Validate conversation ID extraction (as frontend does)
            if messages:
                extracted_conversation_id = messages[0]["conversation_id"]
                assert extracted_conversation_id, "Frontend needs conversation_id from messages"
            
            print(f"✅ Successfully processed {len(frontend_messages)} messages through frontend workflow")
        
        print("✅ Frontend message loading workflow test passed")
    
    @pytest.mark.asyncio
    async def test_frontend_role_handling_compatibility(self, test_helper: FrontendCompatibilityTestHelper):
        """Test that role values work correctly with frontend role normalization."""
        print("\n🧪 Testing Frontend Role Handling Compatibility")
        
        # Get user with existing messages
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Load messages
        endpoint = f"/api/v1/memory-debug/users/{user_id}/messages"
        response = await test_helper.make_api_request("GET", endpoint, params={"limit": 20})
        
        assert response["status_code"] == 200
        
        if response["json"] and response["json"]["data"]:
            messages = response["json"]["data"]
            
            # Test role normalization for each message
            role_mapping_results = {}
            for message in messages:
                original_role = message["role"]
                normalized_role = test_helper.simulate_frontend_role_normalization(original_role)
                
                if original_role not in role_mapping_results:
                    role_mapping_results[original_role] = normalized_role
            
            # Validate that all role mappings work as expected
            expected_mappings = {
                "human": "user",
                "user": "user", 
                "ai": "assistant",
                "assistant": "assistant",
                "system": "system",
                "tool": "tool"
            }
            
            for original_role, normalized_role in role_mapping_results.items():
                if original_role in expected_mappings:
                    assert normalized_role == expected_mappings[original_role], \
                        f"Role mapping failed: {original_role} -> {normalized_role} (expected {expected_mappings[original_role]})"
            
            print(f"✅ Validated role mappings: {role_mapping_results}")
        
        print("✅ Frontend role handling compatibility test passed")
    
    @pytest.mark.asyncio
    async def test_frontend_api_error_handling(self, test_helper: FrontendCompatibilityTestHelper):
        """Test that API errors are handled in a frontend-compatible way."""
        print("\n🧪 Testing Frontend API Error Handling")
        
        # Test invalid user ID (as frontend might encounter)
        invalid_user_id = "nonexistent-user-12345"
        endpoint = f"/api/v1/memory-debug/users/{invalid_user_id}/messages"
        response = await test_helper.make_api_request("GET", endpoint)
        
        # Frontend expects either success with empty data or proper error structure
        if response["status_code"] == 200:
            # Success with empty data is acceptable
            if response["json"]:
                data = response["json"]
                assert "success" in data
                assert "data" in data
                assert isinstance(data["data"], list)
                # Empty list is fine for nonexistent user
        else:
            # Error response should be structured properly
            assert response["status_code"] in [404, 422, 500], \
                f"Unexpected error status code: {response['status_code']}"
            
            if response["json"]:
                # Should have error information
                assert "detail" in response["json"] or "message" in response["json"]
        
        # Test invalid conversation ID
        invalid_conversation_id = "nonexistent-conversation-12345"
        conv_endpoint = f"/api/v1/memory-debug/conversations/{invalid_conversation_id}/messages"
        conv_response = await test_helper.make_api_request("GET", conv_endpoint)
        
        # Should handle gracefully
        assert conv_response["status_code"] in [200, 404, 422]
        
        print("✅ Frontend API error handling test passed")
    
    @pytest.mark.asyncio
    async def test_frontend_data_consistency(self, test_helper: FrontendCompatibilityTestHelper):
        """Test data consistency across multiple API calls as frontend would make."""
        print("\n🧪 Testing Frontend Data Consistency")
        
        # Get user with existing data
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Make multiple API calls that frontend components would make
        
        # 1. Get user messages (as memorycheck page does)
        messages_response = await test_helper.make_api_request(
            "GET", f"/api/v1/memory-debug/users/{user_id}/messages", 
            params={"limit": 10}
        )
        
        # 2. Get conversation summaries (as watchtower does)
        summaries_response = await test_helper.make_api_request(
            "GET", f"/api/v1/memory-debug/users/{user_id}/conversation-summaries"
        )
        
        # 3. Get conversation-specific messages (as might be used)
        conv_messages_response = await test_helper.make_api_request(
            "GET", f"/api/v1/memory-debug/conversations/{conversation_id}/messages",
            params={"limit": 10}
        )
        
        # Validate all responses are successful
        responses = [messages_response, summaries_response, conv_messages_response]
        for i, response in enumerate(responses):
            assert response["status_code"] == 200, f"Response {i+1} failed with status {response['status_code']}"
        
        # Validate data consistency between user messages and conversation messages
        if (messages_response["json"] and messages_response["json"]["data"] and
            conv_messages_response["json"] and conv_messages_response["json"]["data"]):
            
            user_messages = messages_response["json"]["data"]
            conv_messages = conv_messages_response["json"]["data"]
            
            # Messages from the same conversation should be consistent
            user_conv_messages = [m for m in user_messages if m["conversation_id"] == conversation_id]
            
            # Should have some overlap (might not be identical due to limits)
            if user_conv_messages and conv_messages:
                # At least some messages should match
                user_msg_ids = set(m["id"] for m in user_conv_messages)
                conv_msg_ids = set(m["id"] for m in conv_messages)
                overlap = user_msg_ids.intersection(conv_msg_ids)
                assert len(overlap) > 0, "Should have some message overlap between user and conversation queries"
        
        print("✅ Frontend data consistency test passed")
    
    @pytest.mark.asyncio
    async def test_frontend_response_timing(self, test_helper: FrontendCompatibilityTestHelper):
        """Test that API responses are fast enough for good frontend UX."""
        print("\n🧪 Testing Frontend Response Timing")
        
        # Get user with existing data
        user_id, conversation_id = await test_helper.get_existing_user_with_messages()
        
        # Test response times for key frontend endpoints
        endpoints_to_test = [
            f"/api/v1/memory-debug/users/{user_id}/messages",
            f"/api/v1/memory-debug/users/{user_id}/conversation-summaries",
            f"/api/v1/memory-debug/conversations/{conversation_id}/messages"
        ]
        
        for endpoint in endpoints_to_test:
            start_time = asyncio.get_event_loop().time()
            
            response = await test_helper.make_api_request("GET", endpoint, params={"limit": 10})
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # Frontend should get responses quickly for good UX
            assert response_time < 5.0, f"Endpoint {endpoint} too slow: {response_time:.2f}s"
            assert response["status_code"] == 200, f"Endpoint {endpoint} failed"
            
            print(f"✅ {endpoint}: {response_time:.2f}s")
        
        print("✅ Frontend response timing test passed")


async def run_comprehensive_frontend_compatibility_tests():
    """Run all frontend compatibility integration tests."""
    print("🚀 Starting Comprehensive Frontend Compatibility Integration Tests")
    print("=" * 80)
    
    test_helper = FrontendCompatibilityTestHelper()
    await test_helper.setup()
    
    try:
        test_suite = TestFrontendCompatibility()
        
        # Run all tests
        await test_suite.test_memory_debug_messages_frontend_format(test_helper)
        await test_suite.test_conversation_summaries_frontend_format(test_helper)
        await test_suite.test_frontend_message_loading_workflow(test_helper)
        await test_suite.test_frontend_role_handling_compatibility(test_helper)
        await test_suite.test_frontend_api_error_handling(test_helper)
        await test_suite.test_frontend_data_consistency(test_helper)
        await test_suite.test_frontend_response_timing(test_helper)
        
        print("\n" + "=" * 80)
        print("✅ ALL FRONTEND COMPATIBILITY INTEGRATION TESTS PASSED!")
        print("✅ Frontend message and summary queries work unchanged")
        print("✅ Role normalization, data formats, and API contracts validated")
        print("✅ Response timing and error handling meet frontend requirements")
        print("✅ No breaking changes detected for frontend components")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        await test_helper.cleanup()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_frontend_compatibility_tests())


