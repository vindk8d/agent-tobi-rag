#!/usr/bin/env python3
"""
Debug Message Counting Issue
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

async def debug_message_counting():
    """Debug why message counting returns 0"""
    
    conversation_id = 'a7f7426a-000d-4f89-9a2b-57ae2cbf2a85'
    
    print(f"🔍 MESSAGE COUNTING DEBUG")
    print(f"=" * 30)
    print(f"🗨️  Conversation: {conversation_id}")
    
    # Step 1: Direct database count
    print(f"\n1️⃣  Direct Database Count")
    try:
        result = db_client.client.table("messages").select("id", count="exact").eq("conversation_id", conversation_id).execute()
        direct_count = result.count
        print(f"   📊 Direct count: {direct_count}")
        
        # Get some sample messages
        sample = db_client.client.table("messages").select("id,role,created_at").eq("conversation_id", conversation_id).limit(5).execute()
        print(f"   📝 Sample messages:")
        for msg in sample.data[:3]:
            print(f"      - {msg['id'][:8]}... [{msg['role']}] {msg['created_at']}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 2: Check conversation last_summarized_at
    print(f"\n2️⃣  Conversation Summarization Status")
    try:
        conv_result = db_client.client.table("conversations").select("last_summarized_at").eq("id", conversation_id).execute()
        
        if conv_result.data:
            last_summarized = conv_result.data[0].get('last_summarized_at')
            print(f"   📅 last_summarized_at: {last_summarized}")
            
            if last_summarized:
                print(f"   ⚠️  Conversation has been summarized before!")
                
                # Count messages since last summarization
                after_result = db_client.client.table("messages").select("id", count="exact").eq("conversation_id", conversation_id).gt("created_at", last_summarized).execute()
                print(f"   📊 Messages since last summarization: {after_result.count}")
                
                # Show what messages exist after summarization
                after_sample = db_client.client.table("messages").select("id,created_at").eq("conversation_id", conversation_id).gt("created_at", last_summarized).limit(3).execute()
                print(f"   📝 Messages after {last_summarized}:")
                for msg in after_sample.data:
                    print(f"      - {msg['id'][:8]}... {msg['created_at']}")
            else:
                print(f"   ✅ Never summarized - should count all messages")
        else:
            print(f"   ❌ Conversation not found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 3: Test the consolidator method directly  
    print(f"\n3️⃣  Test Consolidator Method")
    try:
        await memory_manager._ensure_initialized()
        consolidator = memory_manager.consolidator
        
        count = await consolidator._count_conversation_messages(conversation_id)
        print(f"   📊 Consolidator count: {count}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await debug_message_counting()

if __name__ == "__main__":
    asyncio.run(main()) 