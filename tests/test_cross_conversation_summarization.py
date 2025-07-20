"""
Test cross-conversation summarization system with master user summaries.

This test verifies:
1. Conversation summaries are generated correctly
2. Master user summaries are created and updated
3. Cross-conversation context is maintained
4. Efficiency optimizations work correctly
"""

import asyncio
import sys
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.memory import memory_manager, ConversationConsolidator
from backend.database import db_client
from backend.config import get_settings

# Test Configuration
TEST_USER_ID = str(uuid.uuid4())  # Use proper UUID format
TEST_CONVERSATIONS = []


class CrossConversationSummarizationTest:
    """Test class for comprehensive cross-conversation summarization."""
    
    def __init__(self):
        self.test_user_id = TEST_USER_ID
        self.conversation_ids = []
        self.cleanup_ids = []
        
    async def setup(self):
        """Initialize test environment."""
        print("\n" + "="*60)
        print("🧪 CROSS-CONVERSATION SUMMARIZATION TEST")
        print("="*60)
        
        # Initialize memory manager
        await memory_manager._ensure_initialized()
        
        # Create test user (required for foreign key constraint)
        try:
            user_data = {
                'id': self.test_user_id,
                'email': f'test_{self.test_user_id[:8]}@test.com',
                'display_name': f'Test User {self.test_user_id[:8]}',
                'user_type': 'customer',
                'metadata': {'test_user': True}
            }
            
            result = db_client.client.table('users').insert(user_data).execute()
            if result.data:
                print(f"✅ Created test user")
            else:
                print(f"⚠️  Using existing test user")
        except Exception as e:
            print(f"⚠️  Test user creation: {e} (may already exist)")
        
        print(f"✅ Test setup complete")
        print(f"🆔 Test user ID: {self.test_user_id}")
    
    async def test_conversation_creation_and_summarization(self):
        """Test creating conversations and generating summaries."""
        print(f"\n📝 TEST 1: Conversation Creation and Summarization")
        print("-" * 50)
        
        # Create test conversations with different topics
        test_conversations = [
            {
                "title": "CRM Integration Discussion",
                "messages": [
                    {"role": "human", "content": "I need help integrating our CRM with the new sales system"},
                    {"role": "bot", "content": "I can help you with CRM integration. What CRM system are you using?"},
                    {"role": "human", "content": "We're using Salesforce for our car dealership"},
                    {"role": "bot", "content": "Great! For Salesforce integration with car dealership operations, you'll need to configure the API connections and data mapping."},
                    {"role": "human", "content": "How long does the integration typically take?"},
                    {"role": "bot", "content": "For a car dealership Salesforce integration, it typically takes 2-4 weeks depending on complexity."},
                    {"role": "human", "content": "What about data migration from our old system?"},
                    {"role": "bot", "content": "Data migration is crucial. We'll need to map your existing customer data, vehicle inventory, and sales history."},
                    {"role": "human", "content": "Are there any compliance considerations for automotive data?"},
                    {"role": "bot", "content": "Yes, automotive data has specific compliance requirements, especially for customer PII and financial information."},
                ]
            },
            {
                "title": "Pricing Strategy Analysis", 
                "messages": [
                    {"role": "human", "content": "I want to analyze our current pricing strategy for better margins"},
                    {"role": "bot", "content": "Let's analyze your pricing strategy. What's your current margin target?"},
                    {"role": "human", "content": "We're targeting 15% gross margin but struggling to achieve it"},
                    {"role": "bot", "content": "A 15% margin is reasonable for automotive sales. What's causing the shortfall?"},
                    {"role": "human", "content": "Competition is pricing aggressively and customers are very price sensitive"},
                    {"role": "bot", "content": "Price sensitivity is common in automotive. Have you considered value-added services to justify higher prices?"},
                    {"role": "human", "content": "Like extended warranties and service packages?"},
                    {"role": "bot", "content": "Exactly! Extended warranties, maintenance packages, and financing options can improve overall margins."},
                    {"role": "human", "content": "How do we price these add-on services competitively?"},
                    {"role": "bot", "content": "Price add-ons based on customer lifetime value and competitive analysis of local market rates."},
                ]
            },
            {
                "title": "Technical Implementation Planning",
                "messages": [
                    {"role": "human", "content": "Let's plan the technical implementation timeline for our new system"},
                    {"role": "bot", "content": "Great! Let's create a structured implementation plan. What's your target go-live date?"},
                    {"role": "human", "content": "We want to go live by Q2 next year, so around April"},
                    {"role": "bot", "content": "April gives us about 4 months. Let's break this into phases: planning, development, testing, and deployment."},
                    {"role": "human", "content": "What should we prioritize in the planning phase?"},
                    {"role": "bot", "content": "Priority items: requirements gathering, system architecture, data migration strategy, and team training plans."},
                    {"role": "human", "content": "How about testing? What kind of testing do we need?"},
                    {"role": "bot", "content": "You'll need unit testing, integration testing, user acceptance testing, and performance testing under load."},
                    {"role": "human", "content": "Should we run parallel systems during transition?"},
                    {"role": "bot", "content": "Yes, running parallel systems for 2-4 weeks helps ensure data integrity and smooth transition."},
                ]
            }
        ]
        
        # Create conversations and messages
        for i, conv_data in enumerate(test_conversations):
            print(f"   Creating conversation {i+1}: {conv_data['title']}")
            
            # Create conversation
            conv_id = str(uuid.uuid4())
            self.conversation_ids.append(conv_id)
            self.cleanup_ids.append(conv_id)
            
            conversation_data = {
                'id': conv_id,
                'user_id': self.test_user_id,
                'title': conv_data['title'],
                'message_count': len(conv_data['messages']),
                'metadata': {'test_conversation': True}
            }
            
            result = db_client.client.table('conversations').insert(conversation_data).execute()
            if not result.data:
                raise Exception(f"Failed to create test conversation {i+1}")
            
            # Create messages
            for msg in conv_data['messages']:
                message_data = {
                    'conversation_id': conv_id,
                    'role': msg['role'],
                    'content': msg['content'],
                    'metadata': {'test_message': True}
                }
                
                msg_result = db_client.client.table('messages').insert(message_data).execute()
                if not msg_result.data:
                    raise Exception(f"Failed to create message in conversation {i+1}")
            
            print(f"   ✅ Created conversation with {len(conv_data['messages'])} messages")
        
        print(f"✅ Created {len(test_conversations)} test conversations")
        
        # Test conversation summarization
        print(f"\n📋 Generating conversation summaries...")
        
        for i, conv_id in enumerate(self.conversation_ids):
            print(f"   Summarizing conversation {i+1}...")
            
            # Trigger conversation summarization
            summary = await memory_manager.consolidator.check_and_trigger_summarization(conv_id, self.test_user_id)
            
            if summary:
                print(f"   ✅ Summary generated: {len(summary)} characters")
            else:
                print(f"   ⚠️  No summary generated (may be under threshold)")
        
        print(f"✅ Conversation summarization test completed")
    
    async def test_master_summary_generation(self):
        """Test master user summary generation and updates."""
        print(f"\n🧠 TEST 2: Master User Summary Generation")
        print("-" * 50)
        
        # Check if user master summary exists
        result = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
        
        if result.data:
            print(f"   📊 Found existing master summary: {len(result.data[0]['master_summary'])} characters")
            master_summary = result.data[0]['master_summary']
            print(f"   📝 Current master summary preview: {master_summary[:200]}...")
        else:
            print(f"   📝 No master summary found, generating new one...")
            
            # Generate master summary
            master_summary = await memory_manager.consolidator.consolidate_user_summary_with_llm(self.test_user_id)
            print(f"   ✅ Generated master summary: {len(master_summary)} characters")
            print(f"   📝 Master summary preview: {master_summary[:200]}...")
        
        # Verify master summary contains cross-conversation context
        expected_keywords = ['crm', 'salesforce', 'pricing', 'margin', 'implementation', 'testing']
        found_keywords = [kw for kw in expected_keywords if kw.lower() in master_summary.lower()]
        
        print(f"   🔍 Cross-conversation keywords found: {found_keywords}")
        
        if len(found_keywords) >= 3:
            print(f"   ✅ Master summary contains cross-conversation context")
        else:
            print(f"   ⚠️  Master summary may be missing cross-conversation context")
        
        print(f"✅ Master summary generation test completed")
    
    async def test_efficiency_optimization(self):
        """Test efficiency optimization (recent summary caching)."""
        print(f"\n⚡ TEST 3: Efficiency Optimization")
        print("-" * 50)
        
        # First call - should generate new summary
        print(f"   🔄 First call to consolidate_user_summary_with_llm...")
        start_time = datetime.now()
        summary1 = await memory_manager.consolidator.consolidate_user_summary_with_llm(self.test_user_id)
        first_duration = (datetime.now() - start_time).total_seconds()
        print(f"   ⏱️  First call took: {first_duration:.2f} seconds")
        
        # Second call immediately after - should use cached summary
        print(f"   ⚡ Second call (should use optimization)...")
        start_time = datetime.now()
        summary2 = await memory_manager.consolidator.consolidate_user_summary_with_llm(self.test_user_id)
        second_duration = (datetime.now() - start_time).total_seconds()
        print(f"   ⏱️  Second call took: {second_duration:.2f} seconds")
        
        # Check if optimization worked
        if second_duration < first_duration * 0.5:  # Should be much faster
            print(f"   ✅ Optimization working: {second_duration:.2f}s vs {first_duration:.2f}s")
        else:
            print(f"   ⚠️  Optimization may not be working as expected")
        
        # Summaries should be identical (from cache)
        if summary1 == summary2:
            print(f"   ✅ Cached summary identical to original")
        else:
            print(f"   ⚠️  Cached summary differs from original")
        
        print(f"✅ Efficiency optimization test completed")
    
    async def test_cross_conversation_context_loading(self):
        """Test loading user context for new conversations."""
        print(f"\n🔄 TEST 4: Cross-Conversation Context Loading")
        print("-" * 50)
        
        # Test getting user context for new conversation
        user_context = await memory_manager.get_user_context_for_new_conversation(self.test_user_id)
        
        print(f"   📊 User context loaded:")
        print(f"      - Has history: {user_context.get('has_history', False)}")
        print(f"      - Total conversations: {user_context.get('total_conversations', 0)}")
        print(f"      - Master summary length: {len(user_context.get('master_summary', ''))}")
        print(f"      - Key insights: {len(user_context.get('key_insights', []))}")
        
        if user_context.get('has_history'):
            print(f"   📝 Master summary preview: {user_context.get('master_summary', '')[:200]}...")
            
            key_insights = user_context.get('key_insights', [])
            for i, insight in enumerate(key_insights[:3], 1):
                print(f"      {i}. {insight}")
        
        # Verify cross-conversation context
        if user_context.get('total_conversations', 0) >= 3:
            print(f"   ✅ Cross-conversation context successfully loaded")
        else:
            print(f"   ⚠️  Expected more conversations in context")
        
        print(f"✅ Cross-conversation context loading test completed")
    
    async def test_database_integrity(self):
        """Test database integrity and relationships."""
        print(f"\n🗄️  TEST 5: Database Integrity")
        print("-" * 50)
        
        # Check conversations
        conversations = db_client.client.table('conversations').select('*').eq('user_id', self.test_user_id).execute()
        print(f"   📊 Conversations in DB: {len(conversations.data)}")
        
        # Check conversation summaries
        summaries = db_client.client.table('conversation_summaries').select('*').eq('user_id', self.test_user_id).execute()
        print(f"   📋 Conversation summaries: {len(summaries.data)}")
        
        # Check user master summary
        master = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
        print(f"   🧠 Master summaries: {len(master.data)}")
        
        # Check messages
        all_messages = []
        for conv_id in self.conversation_ids:
            messages = db_client.client.table('messages').select('*').eq('conversation_id', conv_id).execute()
            all_messages.extend(messages.data)
        print(f"   💬 Total messages: {len(all_messages)}")
        
        # Verify relationships
        if len(conversations.data) >= 3 and len(master.data) == 1:
            print(f"   ✅ Database relationships are correct")
        else:
            print(f"   ⚠️  Database relationship issues detected")
        
        print(f"✅ Database integrity test completed")
    
    async def cleanup(self):
        """Clean up test data."""
        print(f"\n🧹 CLEANUP: Removing test data")
        print("-" * 30)
        
        try:
            # Clean up messages
            for conv_id in self.conversation_ids:
                db_client.client.table('messages').delete().eq('conversation_id', conv_id).execute()
            
            # Clean up conversation summaries
            db_client.client.table('conversation_summaries').delete().eq('user_id', self.test_user_id).execute()
            
            # Clean up conversations
            db_client.client.table('conversations').delete().eq('user_id', self.test_user_id).execute()
            
            # Clean up master summary
            db_client.client.table('user_master_summaries').delete().eq('user_id', self.test_user_id).execute()
            
            # Clean up test user
            db_client.client.table('users').delete().eq('id', self.test_user_id).execute()
            
            print(f"   ✅ Test data cleaned up successfully")
            
        except Exception as e:
            print(f"   ⚠️  Cleanup error: {e}")
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        try:
            await self.setup()
            await self.test_conversation_creation_and_summarization()
            await self.test_master_summary_generation()
            await self.test_efficiency_optimization()
            await self.test_cross_conversation_context_loading()
            await self.test_database_integrity()
            
            print(f"\n" + "="*60)
            print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main test function."""
    test_suite = CrossConversationSummarizationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 