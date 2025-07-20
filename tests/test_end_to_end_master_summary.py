#!/usr/bin/env python3
"""
End-to-End Test for Automatic Master Summary Generation
"""

import asyncio
import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import db_client
from agents.memory import memory_manager

async def test_automatic_master_summary_generation():
    """Test the complete flow: messages → conversation summary → master summary"""
    
    user_id = '550e8400-e29b-41d4-a716-446655440004'  # Robert Brown
    
    print(f"🚀 END-TO-END MASTER SUMMARY TEST")
    print(f"=" * 50)
    print(f"👤 User: {user_id}")
    
    # Step 1: Check initial state
    print(f"\n1️⃣  Initial State Check")
    
    conversations = db_client.client.table('conversations').select('*').eq('user_id', user_id).execute()
    summaries = db_client.client.table('conversation_summaries').select('*').eq('user_id', user_id).execute()
    master_summaries = db_client.client.table('user_master_summaries').select('*').eq('user_id', user_id).execute()
    
    print(f"   📊 Conversations: {len(conversations.data)}")
    print(f"   📝 Conversation summaries: {len(summaries.data)}")
    print(f"   🧠 Master summaries: {len(master_summaries.data)}")
    
    if len(conversations.data) == 0:
        print(f"   ❌ No conversations found - cannot test automatic flow")
        return False
    
    # Step 2: Initialize memory manager
    print(f"\n2️⃣  Initialize Memory Manager")
    try:
        await memory_manager._ensure_initialized()
        print(f"   ✅ Memory manager initialized")
    except Exception as e:
        print(f"   ❌ Error initializing: {e}")
        return False
    
    # Step 3: Get a conversation to test with
    print(f"\n3️⃣  Select Test Conversation")
    
    # Find a conversation with enough messages but no recent summary
    target_conversation = None
    for conv in conversations.data:
        message_count = conv.get('message_count', 0)
        conv_id = conv['id']
        
        print(f"   🗨️  Conv {conv_id[:8]}...: {message_count} messages")
        
        if message_count >= 10:  # Enough messages to trigger summarization
            target_conversation = conv
            break
    
    if not target_conversation:
        print(f"   ❌ No suitable conversation found (need 10+ messages)")
        return False
    
    conversation_id = target_conversation['id']
    print(f"   ✅ Selected conversation: {conversation_id[:8]}... ({target_conversation.get('message_count', 0)} messages)")
    
    # Step 4: Check current summary state for this conversation
    print(f"\n4️⃣  Pre-Test Summary State")
    
    conv_summaries_before = db_client.client.table('conversation_summaries').select('*').eq('conversation_id', conversation_id).execute()
    master_summaries_before = db_client.client.table('user_master_summaries').select('*').eq('user_id', user_id).execute()
    
    print(f"   📝 Conv summaries before: {len(conv_summaries_before.data)}")
    print(f"   🧠 Master summaries before: {len(master_summaries_before.data)}")
    
    # Step 5: Trigger auto-summarization
    print(f"\n5️⃣  Trigger Auto-Summarization")
    
    try:
        print(f"   🚀 Calling check_and_trigger_summarization...")
        
        summary_result = await memory_manager.consolidator.check_and_trigger_summarization(conversation_id, user_id)
        
        if summary_result:
            print(f"   ✅ Conversation summary generated!")
            print(f"   📝 Summary length: {len(summary_result)} chars")
            print(f"   📄 Preview: {summary_result[:150]}...")
            
            # Step 6: Check if master summary was automatically updated
            print(f"\n6️⃣  Verify Master Summary Auto-Update")
            
            # Wait a moment for async operations
            await asyncio.sleep(2)
            
            conv_summaries_after = db_client.client.table('conversation_summaries').select('*').eq('conversation_id', conversation_id).execute()
            master_summaries_after = db_client.client.table('user_master_summaries').select('*').eq('user_id', user_id).execute()
            
            print(f"   📝 Conv summaries after: {len(conv_summaries_after.data)}")
            print(f"   🧠 Master summaries after: {len(master_summaries_after.data)}")
            
            # Check if new conversation summary was created
            conv_summary_created = len(conv_summaries_after.data) > len(conv_summaries_before.data)
            print(f"   {'✅' if conv_summary_created else '❌'} Conversation summary created: {conv_summary_created}")
            
            # Check if master summary was created or updated
            master_summary_updated = len(master_summaries_after.data) > len(master_summaries_before.data)
            
            if not master_summary_updated and master_summaries_after.data and master_summaries_before.data:
                # Check if existing master summary was updated
                old_updated = master_summaries_before.data[0].get('updated_at')
                new_updated = master_summaries_after.data[0].get('updated_at') 
                master_summary_updated = new_updated != old_updated
            
            print(f"   {'✅' if master_summary_updated else '❌'} Master summary updated: {master_summary_updated}")
            
            if master_summaries_after.data:
                master_summary = master_summaries_after.data[0]
                print(f"   📊 Master summary details:")
                print(f"      - Length: {len(master_summary['master_summary'])} chars")
                print(f"      - Total conversations: {master_summary.get('total_conversations', 0)}")
                print(f"      - Total messages: {master_summary.get('total_messages', 0)}")
                print(f"      - Updated: {master_summary.get('updated_at', 'Unknown')}")
                print(f"      - Preview: {master_summary['master_summary'][:200]}...")
            
            return conv_summary_created and master_summary_updated
            
        else:
            print(f"   ⚠️  No summary generated (threshold not met or other issue)")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in auto-summarization: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the end-to-end test"""
    
    success = await test_automatic_master_summary_generation()
    
    print(f"\n📋 FINAL RESULT")
    print(f"=" * 30)
    
    if success:
        print(f"🎉 SUCCESS: End-to-end automatic master summary generation works!")
        print(f"   ✅ Conversation summary created")
        print(f"   ✅ Master summary automatically updated")
        return 0
    else:
        print(f"❌ FAILURE: End-to-end flow has issues")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 