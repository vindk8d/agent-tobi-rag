#!/usr/bin/env python3
"""
Simple Threshold Test - Force reset and test summarization
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

async def test_forced_summarization():
    """Test summarization by forcing a reset"""
    
    # Use an existing conversation with messages
    conversation_id = 'a7f7426a-000d-4f89-9a2b-57ae2cbf2a85'
    user_id = '550e8400-e29b-41d4-a716-446655440004'
    
    print(f"🧪 FORCED SUMMARIZATION TEST")
    print(f"=" * 40)
    print(f"🗨️  Conversation: {conversation_id[:8]}...")
    print(f"👤 User: {user_id}")
    
    # Step 1: Check current state
    print(f"\n1️⃣  Current State")
    
    # Get message count
    msg_count = db_client.client.table('messages').select('id', count='exact').eq('conversation_id', conversation_id).execute()
    print(f"   📊 Total messages: {msg_count.count}")
    
    # Get conversation status
    conv = db_client.client.table('conversations').select('last_summarized_at').eq('id', conversation_id).execute()
    last_summarized = conv.data[0].get('last_summarized_at') if conv.data else None
    print(f"   📅 Last summarized: {last_summarized}")
    
    # Get current master summary state
    master = db_client.client.table('user_master_summaries').select('updated_at').eq('user_id', user_id).execute()
    master_updated = master.data[0]['updated_at'] if master.data else None
    print(f"   🧠 Master summary last updated: {master_updated}")
    
    # Step 2: Reset summarization to force trigger
    print(f"\n2️⃣  Reset Summarization Status")
    
    try:
        # Reset last_summarized_at to NULL to make all messages count as "new"
        db_client.client.table('conversations').update({
            'last_summarized_at': None
        }).eq('id', conversation_id).execute()
        
        print(f"   ✅ Reset last_summarized_at to NULL")
        
        # Initialize memory manager
        await memory_manager._ensure_initialized()
        consolidator = memory_manager.consolidator
        
        # Test the count now
        count = await consolidator._count_conversation_messages(conversation_id)
        print(f"   📊 New message count: {count}")
        
        if count >= 10:
            print(f"   ✅ Should trigger summarization!")
            
            # Step 3: Trigger summarization
            print(f"\n3️⃣  Trigger Summarization")
            
            try:
                summary_result = await consolidator.check_and_trigger_summarization(conversation_id, user_id)
                
                if summary_result:
                    print(f"   ✅ Summarization successful!")
                    print(f"   📝 Summary length: {len(summary_result)} chars")
                    print(f"   📄 Preview: {summary_result[:150]}...")
                    
                    # Wait for master summary update
                    await asyncio.sleep(3)
                    
                    # Check if master summary was updated
                    master_after = db_client.client.table('user_master_summaries').select('updated_at').eq('user_id', user_id).execute()
                    new_master_updated = master_after.data[0]['updated_at'] if master_after.data else None
                    
                    print(f"   🧠 Master summary updated: {new_master_updated}")
                    
                    if new_master_updated != master_updated:
                        print(f"   🎉 SUCCESS: Master summary automatically updated after conversation summary!")
                        return True
                    else:
                        print(f"   ❌ Master summary was not updated")
                        return False
                else:
                    print(f"   ❌ Summarization failed")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error in summarization: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"   ❌ Still not enough messages ({count} < 10)")
            return False
            
    except Exception as e:
        print(f"   ❌ Error resetting: {e}")
        return False

async def main():
    success = await test_forced_summarization()
    
    print(f"\n📋 RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"🎉 Complete flow working: Messages → Conversation Summary → Master Summary")
    else:
        print(f"⚠️  Issues in the summarization flow")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 