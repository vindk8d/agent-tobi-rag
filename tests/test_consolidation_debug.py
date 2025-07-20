#!/usr/bin/env python3
"""
Debug Master Summary Consolidation Issues
"""

import asyncio
import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import db_client
from agents.memory import memory_manager

async def debug_consolidation():
    """Debug why master summary consolidation isn't working"""
    
    user_id = '550e8400-e29b-41d4-a716-446655440004'  # Robert Brown - UUID format
    
    print(f"🔍 DEBUGGING MASTER SUMMARY CONSOLIDATION")
    print(f"=" * 50)
    print(f"👤 User: {user_id}")
    
    # Step 1: Initialize memory manager
    print(f"\n1️⃣  Initializing memory manager...")
    try:
        await memory_manager._ensure_initialized()
        print(f"   ✅ Memory manager initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 2: Check database state
    print(f"\n2️⃣  Checking database state...")
    
    # Check conversation summaries
    summaries = db_client.client.table('conversation_summaries').select('*').eq('user_id', user_id).execute()
    print(f"   📝 Conversation summaries: {len(summaries.data)}")
    
    for i, summary in enumerate(summaries.data[:3]):  # Show first 3
        print(f"      #{i+1}: {summary['id'][:8]}... ({len(summary['summary_text'])} chars) - {summary['created_at']}")
    
    # Check existing master summaries
    master = db_client.client.table('user_master_summaries').select('*').eq('user_id', user_id).execute()
    print(f"   🧠 Master summaries: {len(master.data)}")
    
    if master.data:
        ms = master.data[0]
        print(f"      ID: {ms['id'][:8]}...")
        print(f"      Length: {len(ms['master_summary'])} chars")
        print(f"      Created: {ms['created_at']}")
        print(f"      Updated: {ms['updated_at']}")
    
    # Step 3: Test consolidation method directly
    print(f"\n3️⃣  Testing consolidation method directly...")
    try:
        print(f"   🚀 Calling consolidate_user_summary_with_llm...")
        result = await memory_manager.consolidator.consolidate_user_summary_with_llm(user_id=user_id)
        
        if result:
            print(f"   ✅ Consolidation returned result: {len(result)} chars")
            print(f"   📝 Preview: {result[:150]}...")
            
            # Check if it was saved to database
            master_after = db_client.client.table('user_master_summaries').select('*').eq('user_id', user_id).execute()
            print(f"   💾 Master summaries in DB after: {len(master_after.data)}")
            
            if master_after.data:
                new_ms = master_after.data[0]
                print(f"   🕒 Last updated: {new_ms['updated_at']}")
                print(f"   📊 Summary length in DB: {len(new_ms['master_summary'])} chars")
        else:
            print(f"   ❌ Consolidation returned empty result")
            
    except Exception as e:
        print(f"   ❌ Error in consolidation: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Test database functions directly
    print(f"\n4️⃣  Testing database functions...")
    
    try:
        # Test get_recent_conversation_summaries function
        print(f"   📚 Testing get_recent_conversation_summaries...")
        conn = await memory_manager.consolidator.db_manager.get_connection()
        async with conn as connection:
            cur = await connection.cursor()
            async with cur:
                query = """
                    SELECT conversation_id, summary_text, message_count, created_at, summary_type
                    FROM get_recent_conversation_summaries(%s, %s, 30)
                    ORDER BY created_at DESC
                """
                await cur.execute(query, (user_id, 5))  # limit to 5
                results = await cur.fetchall()
                
                print(f"   📊 Function returned {len(results)} summaries")
                for i, row in enumerate(results):
                    print(f"      #{i+1}: {row[0][:8]}... - {len(row[1])} chars")
                    
    except Exception as e:
        print(f"   ❌ Error testing database functions: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Test LLM connectivity
    print(f"\n5️⃣  Testing LLM connectivity...")
    try:
        test_prompt = "Say 'Hello' in exactly one word."
        response = await memory_manager.consolidator.llm.ainvoke(test_prompt)
        print(f"   ✅ LLM response: '{response.content.strip()}'")
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
    
    print(f"\n🏁 Debug complete!")

if __name__ == "__main__":
    asyncio.run(debug_consolidation()) 