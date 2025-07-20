"""
Quick Test for Simplified Tools

A streamlined test to verify our simplified tools work correctly.
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simplified_tools():
    """Test our simplified tools functionality."""
    logger.info("🚀 Testing Simplified Tools")
    logger.info("=" * 50)
    
    test_results = {"passed": 0, "total": 0}
    
    try:
        # Test 1: Import tools
        logger.info("\n📦 Testing Imports...")
        from agents.tools import (
            simple_query_crm_data,
            simple_rag,
            UserContext,
            get_current_user_id,
            _get_sql_database,
            _get_appropriate_llm
        )
        test_results["total"] += 1
        test_results["passed"] += 1
        logger.info("   ✅ All tools imported successfully")
        
        # Test 2: Database connectivity
        logger.info("\n🗄️ Testing Database...")
        db = await _get_sql_database()
        test_results["total"] += 1
        if db:
            test_results["passed"] += 1
            logger.info("   ✅ Database connected")
        else:
            logger.error("   ❌ Database connection failed")
        
        # Test 3: Model selection
        logger.info("\n🧠 Testing Model Selection...")
        llm = await _get_appropriate_llm("Test question")
        test_results["total"] += 1
        if llm:
            test_results["passed"] += 1
            logger.info(f"   ✅ Model selection works: {llm.model_name}")
        else:
            logger.error("   ❌ Model selection failed")
        
        # Test 4: User context
        logger.info("\n👤 Testing User Context...")
        test_user_id = "54394d40-ad35-4b5b-a392-1ae7c9329d11"
        test_results["total"] += 1
        
        with UserContext(user_id=test_user_id):
            current_user = get_current_user_id()
            if current_user == test_user_id:
                test_results["passed"] += 1
                logger.info("   ✅ User context works")
            else:
                logger.error(f"   ❌ User context failed: {current_user}")
        
        # Test 5: SQL Tool
        logger.info("\n🔧 Testing SQL Tool...")
        test_results["total"] += 1
        try:
            response = await simple_query_crm_data.ainvoke({"question": "How many employees are there?"})
            if response and len(response) > 10 and not response.startswith("I encountered"):
                test_results["passed"] += 1
                logger.info(f"   ✅ SQL tool works: {response[:80]}...")
            else:
                logger.warning(f"   ⚠️ SQL tool response: {response[:100] if response else 'None'}")
        except Exception as e:
            logger.error(f"   ❌ SQL tool error: {e}")
        
        # Test 6: RAG Tool
        logger.info("\n📚 Testing RAG Tool...")
        test_results["total"] += 1
        try:
            response = await simple_rag.ainvoke({"question": "What is the company policy?"})
            if response and len(response) > 10:
                test_results["passed"] += 1
                logger.info(f"   ✅ RAG tool works: {response[:80]}...")
            else:
                logger.warning(f"   ⚠️ RAG tool response: {response[:100] if response else 'None'}")
        except Exception as e:
            logger.error(f"   ❌ RAG tool error: {e}")
        
        # Test 7: SQL Tool with User Context
        logger.info("\n👤 Testing SQL Tool with Context...")
        test_results["total"] += 1
        try:
            with UserContext(user_id=test_user_id):
                response = await simple_query_crm_data.ainvoke({"question": "What are my opportunities?"})
                if response and len(response) > 10 and not response.startswith("I encountered"):
                    test_results["passed"] += 1
                    logger.info(f"   ✅ Context-aware SQL works: {response[:80]}...")
                else:
                    logger.warning(f"   ⚠️ Context SQL response: {response[:100] if response else 'None'}")
        except Exception as e:
            logger.error(f"   ❌ Context SQL error: {e}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 RESULTS: {test_results['passed']}/{test_results['total']} tests passed")
    
    if test_results["passed"] == test_results["total"]:
        logger.info("🎉 ALL TESTS PASSED! Simplified tools work perfectly!")
    elif test_results["passed"] >= test_results["total"] * 0.8:
        logger.info("✅ MOSTLY WORKING! Core functionality retained!")
    else:
        logger.warning("⚠️ Some functionality needs attention")
    
    logger.info("=" * 50)
    
    return test_results

if __name__ == "__main__":
    asyncio.run(test_simplified_tools()) 