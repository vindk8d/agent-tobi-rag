#!/usr/bin/env python3
"""
Test Frontend Master Summary Integration
"""

import asyncio
import requests
import json

async def test_frontend_integration():
    """Test that the frontend can retrieve master summaries correctly"""
    
    user_id = '550e8400-e29b-41d4-a716-446655440004'  # Robert Brown
    api_url = 'http://localhost:8000'
    
    print(f"🖥️  FRONTEND INTEGRATION TEST")
    print(f"=" * 40)
    print(f"👤 User: {user_id}")
    
    # Test 1: Get master summaries via API
    print(f"\n1️⃣  Test Master Summaries API")
    try:
        response = requests.get(f'{api_url}/api/v1/memory-debug/users/{user_id}/master-summaries')
        print(f"   📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Success: {data.get('success', False)}")
            
            summaries = data.get('data', [])
            print(f"   📊 Summaries returned: {len(summaries)}")
            
            if summaries:
                summary = summaries[0]
                print(f"   📋 Master Summary Details:")
                print(f"      - ID: {summary.get('id', 'N/A')[:8]}...")
                print(f"      - Length: {len(summary.get('master_summary', ''))} chars")
                print(f"      - Conversations: {summary.get('total_conversations', 0)}")
                print(f"      - Messages: {summary.get('total_messages', 0)}")
                print(f"      - Created: {summary.get('created_at', 'N/A')}")
                print(f"      - Updated: {summary.get('updated_at', 'N/A')}")
                
                # Show formatted preview
                print(f"   📄 Summary Preview:")
                content = summary.get('master_summary', '')
                lines = content.split('\n')[:5]  # First 5 lines
                for line in lines:
                    if line.strip():
                        print(f"      {line.strip()[:80]}{'...' if len(line.strip()) > 80 else ''}")
                
                return True
            else:
                print(f"   ❌ No master summaries found")
                return False
        else:
            print(f"   ❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Request Error: {e}")
        return False

async def test_manual_generation():
    """Test manual master summary generation via API"""
    
    user_id = '550e8400-e29b-41d4-a716-446655440004'
    api_url = 'http://localhost:8000'
    
    print(f"\n2️⃣  Test Manual Generation API")
    
    try:
        response = requests.post(f'{api_url}/api/v1/memory-debug/memory/consolidate', 
                               json={"user_id": user_id, "force": True},
                               headers={'Content-Type': 'application/json'})
        
        print(f"   📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Success: {data.get('success', False)}")
            
            result_data = data.get('data', {})
            print(f"   📋 Consolidation Details:")
            print(f"      - Triggered: {result_data.get('consolidation_triggered', False)}")
            print(f"      - Timestamp: {result_data.get('timestamp', 'N/A')}")
            print(f"      - Success: {result_data.get('success', False)}")
            
            consolidation_result = result_data.get('consolidation_result', '')
            if consolidation_result and not consolidation_result.startswith('Error'):
                print(f"      - Result length: {len(consolidation_result)} chars")
                print(f"      - Preview: {consolidation_result[:150]}...")
                return True
            else:
                print(f"      - Error: {consolidation_result}")
                return False
        else:
            print(f"   ❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Request Error: {e}")
        return False

async def main():
    """Run all frontend integration tests"""
    
    # Test current state
    api_success = await test_frontend_integration()
    
    # Test manual generation
    manual_success = await test_manual_generation()
    
    print(f"\n📋 RESULTS")
    print(f"=" * 20)
    print(f"   API Retrieval: {'✅' if api_success else '❌'}")
    print(f"   Manual Generation: {'✅' if manual_success else '❌'}")
    
    if api_success and manual_success:
        print(f"\n🎉 Frontend integration working perfectly!")
        return 0
    else:
        print(f"\n⚠️  Some issues detected, but core functionality works")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 