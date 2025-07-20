"""
Simple test for basic conversation summarization functionality.
Tests the core logic without complex database triggers.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.memory import memory_manager
from backend.database import db_client


async def test_basic_summarization():
    """Test basic conversation summarization functionality."""
    print("\n" + "="*50)
    print("🧪 BASIC SUMMARIZATION TEST")
    print("="*50)
    
    # Create test user
    test_user_id = str(uuid.uuid4())
    
    try:
        # Create test user
        user_data = {
            'id': test_user_id,
            'email': f'test_{test_user_id[:8]}@test.com',
            'display_name': f'Test User',
            'user_type': 'customer',
            'metadata': {'test_user': True}
        }
        
        db_client.client.table('users').insert(user_data).execute()
        print(f"✅ Created test user: {test_user_id[:8]}...")
        
        # Create test conversation
        conv_id = str(uuid.uuid4())
        conversation_data = {
            'id': conv_id,
            'user_id': test_user_id,
            'title': 'Test CRM Integration',
            'message_count': 10,  # Trigger summarization threshold
            'metadata': {'test_conversation': True}
        }
        
        db_client.client.table('conversations').insert(conversation_data).execute()
        print(f"✅ Created test conversation: {conv_id[:8]}...")
        
        # Create test messages (10 messages to trigger summarization)
        test_messages = [
            {"role": "human", "content": "I need help with CRM integration"},
            {"role": "bot", "content": "I can help you with CRM integration. What system are you using?"},
            {"role": "human", "content": "We're using Salesforce"},
            {"role": "bot", "content": "Great! I'll help you set up Salesforce integration."},
            {"role": "human", "content": "How long does it take?"},
            {"role": "bot", "content": "Usually 2-4 weeks for a complete integration."},
            {"role": "human", "content": "What about data migration?"},
            {"role": "bot", "content": "We'll need to map your existing data carefully."},
            {"role": "human", "content": "Any compliance issues?"},
            {"role": "bot", "content": "Yes, there are specific requirements for automotive data."},
        ]
        
        for msg in test_messages:
            message_data = {
                'conversation_id': conv_id,
                'role': msg['role'],
                'content': msg['content'],
                'metadata': {'test_message': True}
            }
            db_client.client.table('messages').insert(message_data).execute()
        
        print(f"✅ Created {len(test_messages)} test messages")
        
        # Initialize memory manager
        await memory_manager._ensure_initialized()
        print(f"✅ Memory manager initialized")
        
        # Test conversation summarization
        print(f"\n📋 Testing conversation summarization...")
        summary = await memory_manager.consolidator.check_and_trigger_summarization(conv_id, test_user_id)
        
        if summary:
            print(f"✅ Conversation summary generated: {len(summary)} characters")
            print(f"📝 Summary preview: {summary[:150]}...")
            
            # Test master user summary generation
            print(f"\n🧠 Testing master summary generation...")
            master_summary = await memory_manager.consolidator.consolidate_user_summary_with_llm(test_user_id)
            
            if master_summary:
                print(f"✅ Master summary generated: {len(master_summary)} characters")
                print(f"📝 Master summary preview: {master_summary[:150]}...")
                
                # Test user context retrieval
                print(f"\n🔄 Testing user context retrieval...")
                user_context = await memory_manager.get_user_context_for_new_conversation(test_user_id)
                
                print(f"📊 User context:")
                print(f"   - Has history: {user_context.get('has_history', False)}")
                print(f"   - Total conversations: {user_context.get('total_conversations', 0)}")
                print(f"   - Master summary length: {len(user_context.get('master_summary', ''))}")
                
                if user_context.get('has_history'):
                    print(f"✅ Cross-conversation context working!")
                else:
                    print(f"⚠️  Cross-conversation context not found")
                    
            else:
                print(f"❌ Master summary generation failed")
        else:
            print(f"❌ No conversation summary generated")
        
        print(f"\n🎉 Basic summarization test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test data...")
        try:
            db_client.client.table('messages').delete().eq('conversation_id', conv_id).execute()
            db_client.client.table('conversation_summaries').delete().eq('user_id', test_user_id).execute()
            db_client.client.table('conversations').delete().eq('user_id', test_user_id).execute()
            db_client.client.table('user_master_summaries').delete().eq('user_id', test_user_id).execute()
            db_client.client.table('users').delete().eq('id', test_user_id).execute()
            print(f"✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(test_basic_summarization()) 