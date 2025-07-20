#!/usr/bin/env python3
"""
Test Master Summary Generation - Manual and Automatic
"""

import asyncio
import os
import sys
import requests
from datetime import datetime, timedelta
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import db_client
from agents.memory import memory_manager

class TestMasterSummaryGeneration:
    def __init__(self):
        self.test_user_id = '550e8400-e29b-41d4-a716-446655440004'  # Robert Brown
        self.api_url = os.getenv('NEXT_PUBLIC_API_URL', 'http://localhost:8000')
    
    async def test_manual_master_summary_trigger(self):
        """Test 1: Manual trigger of master summary generation via API"""
        print(f"\n🧪 TEST 1: Manual Master Summary Trigger")
        print(f"   👤 User: {self.test_user_id[:8]}...")
        
        # Check current state
        print(f"   📊 Checking current state...")
        result = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
        before_count = len(result.data)
        print(f"   📈 Master summaries before: {before_count}")
        
        # Check conversation summaries available
        summaries = db_client.client.table('conversation_summaries').select('*').eq('user_id', self.test_user_id).execute()
        print(f"   📝 Conversation summaries available: {len(summaries.data)}")
        
        if len(summaries.data) == 0:
            print(f"   ⚠️  No conversation summaries found - cannot generate master summary")
            return False
        
        # Manual trigger via API
        print(f"   🚀 Triggering manual consolidation via API...")
        try:
            response = requests.post(f'{self.api_url}/api/v1/memory-debug/memory/consolidate', 
                                   json={
                                       "user_id": self.test_user_id,
                                       "force": True
                                   },
                                   timeout=30)
            
            print(f"   📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API Success: {data.get('success', False)}")
                print(f"   📋 Response: {json.dumps(data.get('data', {}), indent=2)}")
                
                # Check if master summary was created
                result = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
                after_count = len(result.data)
                print(f"   📈 Master summaries after: {after_count}")
                
                if after_count > before_count:
                    master_summary = result.data[0]['master_summary']
                    print(f"   🎉 SUCCESS: Master summary generated!")
                    print(f"   📝 Summary length: {len(master_summary)} chars")
                    print(f"   📄 Preview: {master_summary[:200]}...")
                    return True
                else:
                    print(f"   ❌ FAIL: No new master summary created")
                    return False
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    
    async def test_automatic_master_summary_after_conversation(self):
        """Test 2: Automatic master summary generation after conversation summary"""
        print(f"\n🤖 TEST 2: Automatic Master Summary After Conversation")
        print(f"   👤 User: {self.test_user_id[:8]}...")
        
        # Initialize memory manager
        print(f"   🔧 Initializing memory manager...")
        await memory_manager._ensure_initialized()
        print(f"   ✅ Memory manager initialized")
        
        # Check current conversation summaries
        summaries_before = db_client.client.table('conversation_summaries').select('*').eq('user_id', self.test_user_id).execute()
        master_before = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
        
        print(f"   📊 State before:")
        print(f"      - Conversation summaries: {len(summaries_before.data)}")
        print(f"      - Master summaries: {len(master_before.data)}")
        
        # Get an active conversation to trigger summarization
        conversations = db_client.client.table('conversations').select('*').eq('user_id', self.test_user_id).order('updated_at', desc=True).limit(1).execute()
        
        if not conversations.data:
            print(f"   ❌ No conversations found for user")
            return False
        
        conversation_id = conversations.data[0]['id']
        print(f"   🗨️  Testing with conversation: {conversation_id[:8]}...")
        
        # Try to trigger auto-summarization
        print(f"   ⚡ Attempting auto-summarization...")
        try:
            summary_result = await memory_manager.auto_summarize_conversation(conversation_id)
            
            if summary_result:
                print(f"   ✅ Conversation summary generated: {len(summary_result)} chars")
                print(f"   📝 Summary preview: {summary_result[:100]}...")
                
                # Check if master summary was updated
                master_after = db_client.client.table('user_master_summaries').select('*').eq('user_id', self.test_user_id).execute()
                print(f"   📈 Master summaries after: {len(master_after.data)}")
                
                if len(master_after.data) > len(master_before.data):
                    print(f"   🎉 SUCCESS: Master summary created automatically!")
                    master_summary = master_after.data[0]['master_summary']
                    print(f"   📝 Master summary length: {len(master_summary)} chars")
                    print(f"   📄 Preview: {master_summary[:150]}...")
                    return True
                elif len(master_after.data) > 0 and len(master_before.data) > 0:
                    # Check if existing master summary was updated
                    old_updated = datetime.fromisoformat(master_before.data[0]['updated_at'].replace('Z', '+00:00'))
                    new_updated = datetime.fromisoformat(master_after.data[0]['updated_at'].replace('Z', '+00:00'))
                    
                    if new_updated > old_updated:
                        print(f"   🎉 SUCCESS: Existing master summary updated!")
                        print(f"   🕒 Updated at: {new_updated}")
                        return True
                    else:
                        print(f"   ⚠️  Master summary exists but wasn't updated")
                        print(f"   🕒 Last updated: {old_updated}")
                        return False
                else:
                    print(f"   ❌ Master summary not created after conversation summary")
                    return False
            else:
                print(f"   ⚠️  Conversation summary not generated (threshold not met)")
                return False
                
        except Exception as e:
            print(f"   ❌ ERROR in auto-summarization: {e}")
            return False
    
    async def test_master_summary_api_endpoint(self):
        """Test 3: Master summary API endpoint"""
        print(f"\n🔗 TEST 3: Master Summary API Endpoint")
        print(f"   👤 User: {self.test_user_id[:8]}...")
        
        try:
            response = requests.get(f'{self.api_url}/api/v1/memory-debug/users/{self.test_user_id}/master-summaries', 
                                  timeout=10)
            
            print(f"   📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API Success: {data.get('success', False)}")
                
                master_summaries = data.get('data', [])
                print(f"   📈 Master summaries returned: {len(master_summaries)}")
                
                if master_summaries:
                    summary = master_summaries[0]
                    print(f"   📋 Summary details:")
                    print(f"      - ID: {summary.get('id', 'N/A')[:8]}...")
                    print(f"      - Length: {len(summary.get('master_summary', ''))} chars")
                    print(f"      - Conversations: {summary.get('total_conversations', 0)}")
                    print(f"      - Messages: {summary.get('total_messages', 0)}")
                    print(f"      - Created: {summary.get('created_at', 'N/A')}")
                    print(f"      - Updated: {summary.get('updated_at', 'N/A')}")
                    print(f"   📄 Preview: {summary.get('master_summary', '')[:150]}...")
                    return True
                else:
                    print(f"   ⚠️  No master summaries found via API")
                    return False
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    
    async def test_consolidation_trigger_conditions(self):
        """Test 4: Check conditions that should trigger consolidation"""
        print(f"\n🔍 TEST 4: Consolidation Trigger Conditions")
        print(f"   👤 User: {self.test_user_id[:8]}...")
        
        # Check message threshold setting
        from config import get_settings
        settings = get_settings()
        
        print(f"   ⚙️  Configuration:")
        print(f"      - Auto-summarize enabled: {settings.memory_auto_summarize}")
        print(f"      - Summary interval: {settings.memory_summary_interval}")
        print(f"      - Master summary conv limit: {settings.master_summary_conversation_limit}")
        
        # Check conversation data
        conversations = db_client.client.table('conversations').select('*').eq('user_id', self.test_user_id).execute()
        summaries = db_client.client.table('conversation_summaries').select('*').eq('user_id', self.test_user_id).execute()
        
        print(f"   📊 User Data:")
        print(f"      - Total conversations: {len(conversations.data)}")
        print(f"      - Total conversation summaries: {len(summaries.data)}")
        
        # Check which conversations have enough messages
        eligible_conversations = 0
        for conv in conversations.data:
            message_count = conv.get('message_count', 0)
            print(f"      - Conv {conv['id'][:8]}...: {message_count} messages {'✅' if message_count >= settings.memory_summary_interval else '❌'}")
            if message_count >= settings.memory_summary_interval:
                eligible_conversations += 1
        
        print(f"   📈 Eligible for summarization: {eligible_conversations}")
        
        # Master summary eligibility
        master_eligible = len(summaries.data) > 0  # Should generate after ANY summary
        print(f"   🧠 Master summary eligible: {'✅' if master_eligible else '❌'}")
        
        return {
            'auto_summarize_enabled': settings.memory_auto_summarize,
            'eligible_conversations': eligible_conversations,
            'conversation_summaries': len(summaries.data),
            'master_summary_eligible': master_eligible
        }
    
    async def run_all_tests(self):
        """Run all tests"""
        print(f"🚀 MASTER SUMMARY GENERATION TESTS")
        print(f"=" * 50)
        
        results = {}
        
        # Test 4 first to understand the current state
        print(f"\n📊 CHECKING SYSTEM STATE...")
        results['conditions'] = await self.test_consolidation_trigger_conditions()
        
        # Test 3: API endpoint
        results['api_endpoint'] = await self.test_master_summary_api_endpoint()
        
        # Test 1: Manual trigger
        results['manual_trigger'] = await self.test_manual_master_summary_trigger()
        
        # Test 2: Automatic generation (only if manual worked)
        if results['manual_trigger']:
            print(f"\n⏸️  Waiting 5 seconds before automatic test...")
            await asyncio.sleep(5)
            results['automatic_generation'] = await self.test_automatic_master_summary_after_conversation()
        else:
            print(f"\n⏭️  Skipping automatic generation test (manual trigger failed)")
            results['automatic_generation'] = False
        
        # Summary
        print(f"\n📋 TEST SUMMARY")
        print(f"=" * 30)
        print(f"   ✅ API Endpoint: {'PASS' if results['api_endpoint'] else 'FAIL'}")
        print(f"   ✅ Manual Trigger: {'PASS' if results['manual_trigger'] else 'FAIL'}")
        print(f"   ✅ Automatic Generation: {'PASS' if results['automatic_generation'] else 'FAIL'}")
        
        if results['conditions']:
            cond = results['conditions']
            print(f"\n⚙️  System Configuration:")
            print(f"   - Auto-summarize: {cond['auto_summarize_enabled']}")
            print(f"   - Eligible conversations: {cond['eligible_conversations']}")
            print(f"   - Conversation summaries: {cond['conversation_summaries']}")
            print(f"   - Master summary eligible: {cond['master_summary_eligible']}")
        
        return results

async def main():
    """Run the master summary tests"""
    tester = TestMasterSummaryGeneration()
    results = await tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all([results.get('api_endpoint'), results.get('manual_trigger')]):
        sys.exit(1)
    else:
        print(f"\n🎉 All critical tests passed!")

if __name__ == "__main__":
    asyncio.run(main()) 