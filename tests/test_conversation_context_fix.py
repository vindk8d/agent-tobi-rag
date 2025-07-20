"""
Test script to verify conversation context fix for CRM queries.

This test simulates the scenario where:
1. User asks about Honda Civic price
2. User asks follow-up question about discounts
3. Verifies the tool now has access to conversation context
"""

import os
import sys
import asyncio
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.tools import _get_conversation_context
from backend.agents.tools import UserContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_conversation_context_retrieval():
    """Test if we can retrieve conversation context"""
    print("🧪 Testing Conversation Context Retrieval")
    
    # Test with a mock conversation ID (replace with real one from your database)
    test_conversation_id = "c0a437cd-30bc-48f8-8599-202dfc2a1d29"  # From your logs
    
    try:
        context = await _get_conversation_context(test_conversation_id, limit=5)
        
        if context:
            print(f"✅ Context retrieved successfully!")
            print(f"📄 Context length: {len(context)} characters")
            print(f"📝 Context preview:\n{context[:500]}...")
            
            # Check if it contains relevant conversation
            if "honda civic" in context.lower() or "price" in context.lower():
                print("✅ Context contains relevant vehicle/pricing discussion!")
            else:
                print("ℹ️  Context retrieved but may not contain vehicle pricing discussion")
                
        else:
            print("⚠️  No context retrieved - conversation may be empty or ID invalid")
            
    except Exception as e:
        print(f"❌ Error testing context retrieval: {e}")

async def test_context_aware_sql_generation():
    """Test the context-aware SQL generation flow"""
    print("\n🧪 Testing Context-Aware SQL Generation")
    
    try:
        from backend.agents.tools import _generate_sql_query, _get_sql_database
        from backend.agents.tools import current_conversation_id
        
        # Set up test context
        test_conversation_id = "c0a437cd-30bc-48f8-8599-202dfc2a1d29"
        
        # Simulate the UserContext that would be set during tool execution
        with UserContext(conversation_id=test_conversation_id):
            
            # Get database connection
            db = await _get_sql_database()
            
            if db:
                # Test query that would benefit from context
                test_question = "How much discount can I give for it?"
                
                print(f"🔍 Testing question: '{test_question}'")
                print(f"🔗 With conversation context from: {test_conversation_id}")
                
                sql_query = await _generate_sql_query(test_question, db)
                
                print(f"✅ Generated SQL query:")
                print(f"📝 {sql_query}")
                
                # Check if the query seems context-aware
                if any(keyword in sql_query.lower() for keyword in ['civic', 'honda', 'vehicle', 'price', 'discount']):
                    print("✅ SQL query appears to be context-aware (contains relevant keywords)!")
                else:
                    print("⚠️  SQL query generated but may not be fully context-aware")
                    
            else:
                print("❌ Could not get database connection")
                
    except Exception as e:
        print(f"❌ Error testing context-aware SQL generation: {e}")

async def main():
    """Run all tests"""
    print("🚀 Testing Conversation Context Fix for CRM Queries\n")
    
    await test_conversation_context_retrieval()
    await test_context_aware_sql_generation()
    
    print("\n✅ Testing completed!")
    print("\n💡 Next steps:")
    print("   1. Test with real conversation by asking about Honda Civic price")  
    print("   2. Then ask 'How much discount can I give for it?'")
    print("   3. Verify the agent now provides discount information")

if __name__ == "__main__":
    asyncio.run(main()) 