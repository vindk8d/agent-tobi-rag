#!/usr/bin/env python3
"""
Multi-Conversation Framework Test Suite
Tests the new multiple conversations per user functionality
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiConversationTester:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_user_id = "f26449e2-dce9-4b29-acd0-cb39a1f671fd"  # John Smith
        self.test_results = []
        
    async def run_all_tests(self):
        """Run comprehensive test suite for multi-conversation framework"""
        logger.info("🚀 Starting Multi-Conversation Framework Tests")
        
        tests = [
            ("Test 1: Create Multiple Conversations", self.test_create_multiple_conversations),
            ("Test 2: Verify Conversation Independence", self.test_conversation_independence),
            ("Test 3: Test Auto-Title Generation", self.test_auto_title_generation),
            ("Test 4: Test Conversation Retrieval", self.test_conversation_retrieval),
            ("Test 5: Test Conversation Title Updates", self.test_title_updates),
            ("Test 6: Test MenuWindow API Integration", self.test_menu_window_integration),
            ("Test 7: Test Chat Interface Switching", self.test_chat_interface_switching),
            ("Test 8: Verify No Empty Conversations", self.test_no_empty_conversations),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                self.test_results.append({
                    "test": test_name,
                    "status": "PASSED" if result else "FAILED",
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate test report
        await self.generate_test_report()
        
    async def test_create_multiple_conversations(self) -> bool:
        """Test creating multiple independent conversations for the same user"""
        logger.info("Creating 3 different conversations...")
        
        conversations = []
        messages = [
            "Hello, I need help with vehicle specifications",
            "Can you generate a quote for a Toyota Camry?",
            "What are your business hours and contact information?"
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, message in enumerate(messages):
                # Create new conversation (no conversation_id provided)
                payload = {
                    "message": message,
                    "user_id": self.test_user_id,
                    "include_sources": True
                }
                
                async with session.post(f"{self.api_base_url}/api/v1/chat/message", json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to create conversation {i+1}: {response.status}")
                        return False
                    
                    data = await response.json()
                    conversation_id = data.get("conversation_id")
                    
                    if not conversation_id:
                        logger.error(f"No conversation_id returned for conversation {i+1}")
                        return False
                    
                    conversations.append({
                        "id": conversation_id,
                        "message": message,
                        "response": data.get("message", "")
                    })
                    
                    logger.info(f"✅ Created conversation {i+1}: {conversation_id[:8]}...")
        
        # Verify all conversations have unique IDs
        conversation_ids = [c["id"] for c in conversations]
        if len(set(conversation_ids)) != len(conversation_ids):
            logger.error("❌ Conversations do not have unique IDs!")
            return False
        
        logger.info(f"✅ Successfully created {len(conversations)} unique conversations")
        self.test_conversations = conversations
        return True
    
    async def test_conversation_independence(self) -> bool:
        """Test that conversations are independent and don't interfere with each other"""
        if not hasattr(self, 'test_conversations'):
            logger.error("❌ No test conversations available. Run test_create_multiple_conversations first.")
            return False
        
        logger.info("Testing conversation independence...")
        
        async with aiohttp.ClientSession() as session:
            # Send follow-up messages to each conversation
            for i, conv in enumerate(self.test_conversations):
                follow_up_message = f"This is a follow-up message for conversation {i+1}"
                
                payload = {
                    "message": follow_up_message,
                    "conversation_id": conv["id"],
                    "user_id": self.test_user_id,
                    "include_sources": True
                }
                
                async with session.post(f"{self.api_base_url}/api/v1/chat/message", json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send follow-up to conversation {i+1}")
                        return False
                    
                    data = await response.json()
                    returned_conv_id = data.get("conversation_id")
                    
                    if returned_conv_id != conv["id"]:
                        logger.error(f"❌ Conversation ID mismatch! Expected: {conv['id']}, Got: {returned_conv_id}")
                        return False
                    
                    logger.info(f"✅ Conversation {i+1} maintained independence")
        
        return True
    
    async def test_auto_title_generation(self) -> bool:
        """Test that conversation titles are auto-generated from first messages"""
        logger.info("Testing auto-title generation...")
        
        test_messages = [
            "Generate an informal quote for a Honda Civic",
            "Hello, can you help me with vehicle information?",
            "What are your business hours?"
        ]
        
        expected_titles = [
            "Generate an informal quote for a Honda Civic",
            "Can you help me with vehicle information?",  # "Hello," prefix removed
            "What are your business hours?"
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, (message, expected_title) in enumerate(zip(test_messages, expected_titles)):
                # Create new conversation
                payload = {
                    "message": message,
                    "user_id": self.test_user_id,
                    "include_sources": True
                }
                
                async with session.post(f"{self.api_base_url}/api/v1/chat/message", json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to create conversation for title test {i+1}")
                        return False
                    
                    data = await response.json()
                    conversation_id = data.get("conversation_id")
                    
                    # Wait a moment for title generation
                    await asyncio.sleep(1)
                    
                    # Check conversation title via API
                    async with session.get(f"{self.api_base_url}/api/v1/chat/users/{self.test_user_id}/conversations") as conv_response:
                        if conv_response.status != 200:
                            logger.error(f"Failed to retrieve conversations for title check")
                            return False
                        
                        conv_data = await conv_response.json()
                        conversations = conv_data.get("data", [])
                        
                        # Find our conversation
                        our_conv = next((c for c in conversations if c["id"] == conversation_id), None)
                        if not our_conv:
                            logger.error(f"❌ Could not find conversation {conversation_id}")
                            return False
                        
                        actual_title = our_conv.get("title", "")
                        logger.info(f"Expected: '{expected_title}' | Actual: '{actual_title}'")
                        
                        # Check if title was generated (not default)
                        if actual_title in ["New Conversation", "Ongoing Conversation", ""]:
                            logger.error(f"❌ Title not auto-generated for conversation {i+1}")
                            return False
                        
                        logger.info(f"✅ Auto-generated title for conversation {i+1}: '{actual_title}'")
        
        return True
    
    async def test_conversation_retrieval(self) -> bool:
        """Test retrieving all conversations for a user"""
        logger.info("Testing conversation retrieval...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/api/v1/chat/users/{self.test_user_id}/conversations") as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve conversations: {response.status}")
                    return False
                
                data = await response.json()
                conversations = data.get("data", [])
                
                if len(conversations) < 3:  # We created at least 3 in previous tests
                    logger.error(f"❌ Expected at least 3 conversations, got {len(conversations)}")
                    return False
                
                # Verify conversation structure
                required_fields = ["id", "title", "created_at", "updated_at", "latest_message", "latest_message_time"]
                for conv in conversations[:3]:  # Check first 3
                    for field in required_fields:
                        if field not in conv:
                            logger.error(f"❌ Missing field '{field}' in conversation data")
                            return False
                
                logger.info(f"✅ Successfully retrieved {len(conversations)} conversations with proper structure")
                return True
    
    async def test_title_updates(self) -> bool:
        """Test updating conversation titles"""
        if not hasattr(self, 'test_conversations'):
            logger.error("❌ No test conversations available")
            return False
        
        logger.info("Testing conversation title updates...")
        
        async with aiohttp.ClientSession() as session:
            # Update title of first test conversation
            conv_id = self.test_conversations[0]["id"]
            new_title = f"Updated Title - {datetime.now().strftime('%H:%M:%S')}"
            
            payload = {"title": new_title}
            
            async with session.put(f"{self.api_base_url}/api/v1/chat/conversations/{conv_id}/title", json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to update conversation title: {response.status}")
                    return False
                
                # Verify title was updated
                async with session.get(f"{self.api_base_url}/api/v1/chat/users/{self.test_user_id}/conversations") as conv_response:
                    conv_data = await conv_response.json()
                    conversations = conv_data.get("data", [])
                    
                    updated_conv = next((c for c in conversations if c["id"] == conv_id), None)
                    if not updated_conv:
                        logger.error(f"❌ Could not find updated conversation")
                        return False
                    
                    if updated_conv["title"] != new_title:
                        logger.error(f"❌ Title not updated. Expected: '{new_title}', Got: '{updated_conv['title']}'")
                        return False
                    
                    logger.info(f"✅ Successfully updated conversation title to: '{new_title}'")
                    return True
    
    async def test_menu_window_integration(self) -> bool:
        """Test that MenuWindow can properly fetch and display conversations"""
        logger.info("Testing MenuWindow API integration...")
        
        # This simulates what MenuWindow.tsx does
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/api/v1/chat/users/{self.test_user_id}/conversations") as response:
                if response.status != 200:
                    logger.error(f"MenuWindow API call failed: {response.status}")
                    return False
                
                data = await response.json()
                
                # Check APIResponse format
                if not data.get("success"):
                    logger.error(f"❌ API response indicates failure: {data.get('message')}")
                    return False
                
                conversations = data.get("data", [])
                
                # Verify MenuWindow required fields
                menu_required_fields = ["id", "title", "created_at", "updated_at", "latest_message", "latest_message_time", "latest_message_role"]
                
                for conv in conversations[:3]:
                    for field in menu_required_fields:
                        if field not in conv:
                            logger.error(f"❌ MenuWindow missing required field: {field}")
                            return False
                
                logger.info(f"✅ MenuWindow integration test passed - {len(conversations)} conversations available")
                return True
    
    async def test_chat_interface_switching(self) -> bool:
        """Test that ChatInterface can switch between conversations"""
        if not hasattr(self, 'test_conversations'):
            logger.error("❌ No test conversations available")
            return False
        
        logger.info("Testing ChatInterface conversation switching...")
        
        async with aiohttp.ClientSession() as session:
            # Test loading messages from different conversations
            for i, conv in enumerate(self.test_conversations[:2]):  # Test first 2 conversations
                conv_id = conv["id"]
                
                # This simulates what ChatInterface does when loading a conversation
                async with session.get(f"{self.api_base_url}/api/v1/memory-debug/conversations/{conv_id}/messages") as response:
                    if response.status != 200:
                        logger.error(f"Failed to load messages for conversation {i+1}: {response.status}")
                        return False
                    
                    data = await response.json()
                    messages = data.get("data", [])
                    
                    if len(messages) < 1:
                        logger.error(f"❌ No messages found for conversation {i+1}")
                        return False
                    
                    # Verify message structure
                    first_message = messages[0]
                    required_fields = ["id", "conversation_id", "role", "content", "created_at"]
                    
                    for field in required_fields:
                        if field not in first_message:
                            logger.error(f"❌ Missing field '{field}' in message data")
                            return False
                    
                    if first_message["conversation_id"] != conv_id:
                        logger.error(f"❌ Message conversation_id mismatch")
                        return False
                    
                    logger.info(f"✅ Successfully loaded {len(messages)} messages for conversation {i+1}")
            
            return True
    
    async def test_no_empty_conversations(self) -> bool:
        """Test that no conversation exists without messages - conversations without messages are useless"""
        logger.info("Testing that all conversations have messages...")
        
        async with aiohttp.ClientSession() as session:
            # Get all conversations for the user
            async with session.get(f"{self.api_base_url}/api/v1/chat/users/{self.test_user_id}/conversations") as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve conversations: {response.status}")
                    return False
                
                data = await response.json()
                conversations = data.get("data", [])
                
                if len(conversations) == 0:
                    logger.info("✅ No conversations found - test passes")
                    return True
                
                # Check each conversation has messages
                for conv in conversations:
                    conv_id = conv["id"]
                    
                    # Get messages for this conversation
                    async with session.get(f"{self.api_base_url}/api/v1/memory-debug/conversations/{conv_id}/messages") as msg_response:
                        if msg_response.status != 200:
                            logger.error(f"❌ Failed to get messages for conversation {conv_id}")
                            return False
                        
                        msg_data = await msg_response.json()
                        messages = msg_data.get("data", [])
                        
                        if len(messages) == 0:
                            logger.error(f"❌ Found conversation {conv_id} with NO messages! Title: '{conv['title']}'")
                            return False
                        
                        logger.info(f"✅ Conversation {conv_id[:8]}... has {len(messages)} messages")
                
                logger.info(f"✅ All {len(conversations)} conversations have messages - no empty conversations found")
                return True
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 MULTI-CONVERSATION FRAMEWORK TEST REPORT")
        logger.info(f"{'='*80}")
        
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")
        total = len(self.test_results)
        
        logger.info(f"📈 SUMMARY:")
        logger.info(f"   Total Tests: {total}")
        logger.info(f"   ✅ Passed: {passed}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   🚨 Errors: {errors}")
        logger.info(f"   📊 Success Rate: {(passed/total)*100:.1f}%")
        
        logger.info(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "🚨"}[result["status"]]
            logger.info(f"   {status_emoji} {result['test']}: {result['status']}")
            if "error" in result:
                logger.info(f"      Error: {result['error']}")
        
        # Save report to file
        report_filename = f"multi_conversation_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "success_rate": (passed/total)*100
                },
                "results": self.test_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"\n💾 Test report saved to: {report_filename}")
        
        if passed == total:
            logger.info(f"\n🎉 ALL TESTS PASSED! Multi-conversation framework is working correctly!")
        else:
            logger.info(f"\n⚠️  Some tests failed. Please review the results above.")

async def main():
    """Run the multi-conversation framework tests"""
    tester = MultiConversationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
