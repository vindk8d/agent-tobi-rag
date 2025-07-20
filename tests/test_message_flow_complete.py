#!/usr/bin/env python3
"""
Complete Message Flow Test - Send actual messages and verify automatic master summary generation
"""

import asyncio
import os
import sys
import requests
import json
import uuid
from datetime import datetime, timezone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import db_client
from agents.memory import memory_manager

class MessageFlowTester:
    def __init__(self):
        self.user_id = '550e8400-e29b-41d4-a716-446655440004'  # Robert Brown
        self.api_url = os.getenv('NEXT_PUBLIC_API_URL', 'http://localhost:8000')
        self.conversation_id = None
        
    async def setup_fresh_conversation(self):
        """Create a fresh conversation for testing"""
        print(f"\n🆕 Setting up fresh conversation")
        
        # Create new conversation
        self.conversation_id = str(uuid.uuid4())
        
        try:
            # Insert conversation record
            db_client.client.table('conversations').insert({
                'id': self.conversation_id,
                'user_id': self.user_id,
                'title': 'Message Flow Test Conversation',
                'message_count': 0,
                'archival_status': 'active',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            print(f"   ✅ Created conversation: {self.conversation_id[:8]}...")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating conversation: {e}")
            return False
    
    async def send_test_message(self, message_text: str, role: str = 'human') -> bool:
        """Send a test message via API"""
        try:
            if role == 'human':
                # Send via chat API to simulate real user interaction
                response = requests.post(f'{self.api_url}/api/v1/chat/message', 
                                       json={
                                           'message': message_text,
                                           'conversation_id': self.conversation_id,
                                           'user_id': self.user_id,
                                           'include_sources': False
                                       },
                                       headers={'Content-Type': 'application/json'},
                                       timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"   ✅ Sent message: {message_text[:50]}...")
                        print(f"   🤖 Got response: {result['data']['message'][:50]}...")
                        return True
                    else:
                        print(f"   ❌ API error: {result.get('message')}")
                        return False
                else:
                    print(f"   ❌ HTTP error: {response.status_code}")
                    return False
            else:
                # Direct database insert for bot messages
                message_id = str(uuid.uuid4())
                db_client.client.table('messages').insert({
                    'id': message_id,
                    'conversation_id': self.conversation_id,
                    'user_id': self.user_id,
                    'role': role,
                    'content': message_text,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }).execute()
                
                # Update conversation message count
                conv = db_client.client.table('conversations').select('message_count').eq('id', self.conversation_id).execute()
                current_count = conv.data[0]['message_count'] if conv.data else 0
                
                db_client.client.table('conversations').update({
                    'message_count': current_count + 1,
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }).eq('id', self.conversation_id).execute()
                
                print(f"   ✅ Inserted {role} message: {message_text[:50]}...")
                return True
                
        except Exception as e:
            print(f"   ❌ Error sending message: {e}")
            return False
    
    async def get_conversation_status(self):
        """Get current conversation status"""
        try:
            conv = db_client.client.table('conversations').select('message_count,last_summarized_at').eq('id', self.conversation_id).execute()
            
            if conv.data:
                data = conv.data[0]
                message_count = data.get('message_count', 0)
                last_summarized = data.get('last_summarized_at')
                
                # Count actual messages in database  
                actual_count = db_client.client.table('messages').select('id', count='exact').eq('conversation_id', self.conversation_id).execute()
                
                return {
                    'recorded_count': message_count,
                    'actual_count': actual_count.count,
                    'last_summarized': last_summarized
                }
            return None
            
        except Exception as e:
            print(f"   ❌ Error getting status: {e}")
            return None
    
    async def check_master_summary_updated(self, before_timestamp: str) -> bool:
        """Check if master summary was updated after a given timestamp"""
        try:
            result = db_client.client.table('user_master_summaries').select('updated_at').eq('user_id', self.user_id).execute()
            
            if result.data:
                updated_at = result.data[0]['updated_at']
                return updated_at > before_timestamp
            return False
            
        except Exception as e:
            print(f"   ❌ Error checking master summary: {e}")
            return False
    
    async def test_message_counter_increases(self):
        """Test 1: Verify message counter increases with new messages"""
        print(f"\n1️⃣  TEST: Message Counter Increases")
        
        # Get initial status
        initial_status = await self.get_conversation_status()
        if not initial_status:
            print(f"   ❌ Could not get initial status")
            return False
        
        print(f"   📊 Initial: {initial_status['recorded_count']} recorded, {initial_status['actual_count']} actual")
        
        # Send a few messages and check counter increases
        test_messages = [
            "Hello, I'm testing the message counter system.",
            "This is message number 2 for the counter test.",
            "And this is message number 3 to verify counting."
        ]
        
        for i, msg in enumerate(test_messages, 1):
            success = await self.send_test_message(msg)
            if not success:
                print(f"   ❌ Failed to send message {i}")
                return False
            
            # Wait a moment for database updates
            await asyncio.sleep(1)
            
            # Check status
            status = await self.get_conversation_status()
            if status:
                expected_count = initial_status['actual_count'] + (i * 2)  # Each human message gets a bot response
                print(f"   📊 After message {i}: {status['recorded_count']} recorded, {status['actual_count']} actual (expected: {expected_count})")
                
                if status['actual_count'] <= initial_status['actual_count']:
                    print(f"   ❌ Counter did not increase!")
                    return False
            else:
                print(f"   ❌ Could not get status after message {i}")
                return False
        
        print(f"   ✅ Message counter increases correctly")
        return True
    
    async def test_automatic_summarization_trigger(self):
        """Test 2: Verify automatic summarization triggers at threshold"""
        print(f"\n2️⃣  TEST: Automatic Summarization at Threshold")
        
        # Initialize memory manager
        await memory_manager._ensure_initialized()
        consolidator = memory_manager.consolidator
        
        # Get current status
        initial_status = await self.get_conversation_status()
        if not initial_status:
            return False
        
        print(f"   📊 Starting with {initial_status['actual_count']} messages")
        print(f"   🎯 Threshold: 10 messages for auto-summarization")
        
        # Get current master summary timestamp
        master_before = db_client.client.table('user_master_summaries').select('updated_at').eq('user_id', self.user_id).execute()
        master_timestamp_before = master_before.data[0]['updated_at'] if master_before.data else None
        
        # Send messages to reach threshold (need to reach 10 total)
        messages_needed = max(0, 10 - initial_status['actual_count'])
        print(f"   📝 Need {messages_needed} more messages to reach threshold")
        
        if messages_needed <= 0:
            print(f"   ⚠️  Already at or above threshold - testing trigger directly")
            
            # Test direct trigger
            try:
                print(f"   🚀 Testing direct summarization trigger...")
                summary_result = await consolidator.check_and_trigger_summarization(self.conversation_id, self.user_id)
                
                if summary_result:
                    print(f"   ✅ Summarization triggered successfully")
                    print(f"   📝 Summary length: {len(summary_result)} chars")
                    
                    # Wait for async operations
                    await asyncio.sleep(3)
                    
                    # Check if master summary was updated
                    master_updated = await self.check_master_summary_updated(master_timestamp_before or "1970-01-01T00:00:00Z")
                    print(f"   🧠 Master summary updated: {'✅' if master_updated else '❌'}")
                    
                    return master_updated
                else:
                    print(f"   ❌ Summarization did not trigger")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error in direct trigger: {e}")
                return False
        
        # Send messages one by one and monitor for auto-trigger
        for i in range(messages_needed + 2):  # Send a few extra to ensure we cross threshold
            message = f"Test message {i+1} for threshold testing - discussing vehicle features and pricing."
            
            success = await self.send_test_message(message)
            if not success:
                continue
            
            await asyncio.sleep(2)  # Wait for processing
            
            # Check current status
            status = await self.get_conversation_status()
            if status:
                print(f"   📊 Message {i+1}: {status['actual_count']} total messages")
                
                # If we've reached threshold, test the consolidation
                if status['actual_count'] >= 10:
                    print(f"   🎯 Reached threshold! Testing consolidation...")
                    
                    try:
                        # Test consolidation trigger
                        count = await consolidator._count_conversation_messages(self.conversation_id)
                        print(f"   🔢 Consolidator counts: {count} new messages")
                        
                        if count >= 10:
                            print(f"   ✅ Should trigger automatic summarization")
                            
                            # Trigger it manually since we want to test the flow
                            summary_result = await consolidator.check_and_trigger_summarization(self.conversation_id, self.user_id)
                            
                            if summary_result:
                                print(f"   ✅ Summarization successful: {len(summary_result)} chars")
                                
                                # Wait and check master summary
                                await asyncio.sleep(3)
                                master_updated = await self.check_master_summary_updated(master_timestamp_before or "1970-01-01T00:00:00Z")
                                print(f"   🧠 Master summary auto-updated: {'✅' if master_updated else '❌'}")
                                
                                return master_updated
                            else:
                                print(f"   ❌ Summarization failed")
                                return False
                        else:
                            print(f"   ⚠️  Not enough new messages ({count}) to trigger")
                    
                    except Exception as e:
                        print(f"   ❌ Error testing consolidation: {e}")
                        return False
                        
        print(f"   ❌ Did not successfully trigger automatic summarization")
        return False
    
    async def cleanup(self):
        """Cleanup test data"""
        try:
            if self.conversation_id:
                # Delete messages
                db_client.client.table('messages').delete().eq('conversation_id', self.conversation_id).execute()
                # Delete conversation
                db_client.client.table('conversations').delete().eq('id', self.conversation_id).execute()
                print(f"   🧹 Cleaned up test conversation")
        except Exception as e:
            print(f"   ⚠️  Cleanup error: {e}")
    
    async def run_complete_test(self):
        """Run the complete message flow test"""
        print(f"🧪 COMPLETE MESSAGE FLOW TEST")
        print(f"=" * 50)
        print(f"👤 User: {self.user_id}")
        
        # Setup
        setup_success = await self.setup_fresh_conversation()
        if not setup_success:
            print(f"❌ Setup failed")
            return False
        
        try:
            # Test 1: Counter increases
            counter_test = await self.test_message_counter_increases()
            
            # Test 2: Automatic summarization
            auto_test = await self.test_automatic_summarization_trigger()
            
            # Results
            print(f"\n📋 RESULTS")
            print(f"=" * 20)
            print(f"   Counter Increases: {'✅' if counter_test else '❌'}")
            print(f"   Auto Summarization: {'✅' if auto_test else '❌'}")
            
            if counter_test and auto_test:
                print(f"\n🎉 All tests passed! Complete message flow working!")
                return True
            else:
                print(f"\n⚠️  Some issues detected")
                return False
                
        finally:
            await self.cleanup()

async def main():
    """Run the complete message flow tests"""
    tester = MessageFlowTester()
    success = await tester.run_complete_test()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 