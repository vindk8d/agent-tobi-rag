#!/usr/bin/env python3
"""Final quick test of key cancellation scenarios."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.hitl import _process_hitl_response_llm_driven

async def quick_test():
    print('🔍 Final Cancellation Test:')
    
    # Test key scenarios
    scenarios = [
        ('cancel', 'generate_quotation', 'Should stop quotation'),
        ('yes, send it', 'trigger_customer_message', 'Should approve message'),
        ('stop this', 'crm_query', 'Should stop CRM query'),
        ('john@test.com but never mind', 'trigger_customer_message', 'Should cancel despite email')
    ]
    
    all_passed = True
    
    for response, tool, description in scenarios:
        context = {'source_tool': tool, 'collection_mode': 'tool_managed'}
        result = await _process_hitl_response_llm_driven(context, response)
        
        if tool == 'trigger_customer_message' and response == 'yes, send it':
            expected = 'approved'
        else:
            expected = 'denied' if any(word in response.lower() for word in ['cancel', 'stop', 'never mind']) else result.get('result')
        
        actual = result.get('result')
        passed = actual == expected
        status = '✅' if passed else '❌'
        print(f'{status} "{response}" → {actual} ({description})')
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print('\n🎉 All key cancellation scenarios working perfectly!')
        return True
    else:
        print('\n⚠️ Some scenarios failed.')
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
