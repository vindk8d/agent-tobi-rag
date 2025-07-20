"""
Focused Test for Simplified Tools

This test verifies that our simplified tools retain core functionality:
1. SQL tool functionality (simple_query_crm_data) 
2. User context functionality
3. Database connectivity
4. Basic error handling

Designed to run without complex agent imports to avoid module resolution issues.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSimplifiedToolsFocused:
    """Focused test suite for simplified tools core functionality."""
    
    def __init__(self):
        self.test_user_id = "54394d40-ad35-4b5b-a392-1ae7c9329d11"  # Alex Thompson
        self.test_employee_id = "db8cdcda-3c96-4e35-8fe4-860f927b9395"  # Alex's employee ID
        
    async def setup_test_environment(self):
        """Set up the test environment."""
        logger.info("🔧 Setting up focused test environment...")
        
        try:
            # Import here to avoid module resolution issues
            from config import get_settings
            
            settings = await get_settings()
            if not settings.supabase.postgresql_connection_string:
                raise Exception("No database connection string found")
            
            logger.info("✅ Settings loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_user_context_core(self):
        """Test core user context functionality."""
        logger.info("\n📋 Testing User Context Core Functionality...")
        
        test_results = {
            "context_variables_exist": False,
            "context_manager_class": False,
            "helper_functions": False
        }
        
        try:
            # Import user context components
            from agents.tools import (
                UserContext,
                get_current_user_id,
                get_current_conversation_id,
                current_user_id,
                current_conversation_id
            )
            
            # Test 1: Context variables exist
            if current_user_id and current_conversation_id:
                test_results["context_variables_exist"] = True
                logger.info("   ✅ Context variables exist")
            
            # Test 2: UserContext class works
            context_manager = UserContext(user_id=self.test_user_id)
            if context_manager:
                test_results["context_manager_class"] = True
                logger.info("   ✅ UserContext class instantiates")
            
            # Test 3: Helper functions exist
            if callable(get_current_user_id) and callable(get_current_conversation_id):
                test_results["helper_functions"] = True
                logger.info("   ✅ Helper functions exist")
            
        except Exception as e:
            logger.error(f"   ❌ User context core test failed: {e}")
        
        return test_results
    
    async def test_sql_database_connectivity(self):
        """Test SQL database connectivity."""
        logger.info("\n🗄️ Testing SQL Database Connectivity...")
        
        try:
            from agents.tools import _get_sql_database
            
            db = await _get_sql_database()
            if db is not None:
                # Try a simple query
                result = db.run("SELECT 1 as test_connection")
                if result:
                    logger.info("   ✅ Database connectivity works")
                    return {"connectivity": True, "test_query": True}
                else:
                    logger.warning("   ⚠️ Database connected but test query failed")
                    return {"connectivity": True, "test_query": False}
            else:
                logger.error("   ❌ Database connection failed")
                return {"connectivity": False, "test_query": False}
                
        except Exception as e:
            logger.error(f"   ❌ SQL database test failed: {e}")
            return {"connectivity": False, "error": str(e)}
    
    async def test_simple_sql_tool_basic(self):
        """Test simple SQL tool basic functionality."""
        logger.info("\n🔧 Testing Simple SQL Tool Basic Functionality...")
        
        try:
            from agents.tools import simple_query_crm_data
            
            # Test basic queries that should work regardless of data
            test_queries = [
                "SELECT COUNT(*) FROM employees",
                "SELECT COUNT(*) FROM customers", 
                "SELECT COUNT(*) FROM vehicles"
            ]
            
            results = {"queries_tested": len(test_queries), "successful": 0, "details": []}
            
            for i, query in enumerate(test_queries):
                try:
                    # Use the tool with a natural language question that should generate these queries
                    question_map = {
                        0: "How many employees are there?",
                        1: "How many customers do we have?", 
                        2: "How many vehicles are in our system?"
                    }
                    
                    question = question_map[i]
                    logger.info(f"   🔍 Testing: {question}")
                    
                    response = await simple_query_crm_data.ainvoke({"question": question})
                    
                    if response and len(response) > 10:
                        if not response.startswith("I encountered an issue"):
                            results["successful"] += 1
                            results["details"].append({
                                "question": question,
                                "status": "✅ SUCCESS",
                                "response_length": len(response)
                            })
                            logger.info(f"      ✅ Success - Response: {response[:100]}...")
                        else:
                            results["details"].append({
                                "question": question,
                                "status": "❌ ERROR",
                                "response": response[:200]
                            })
                            logger.warning(f"      ❌ Tool error: {response[:100]}...")
                    else:
                        results["details"].append({
                            "question": question,
                            "status": "❌ NO_RESPONSE",
                            "response": response or "None"
                        })
                        logger.warning(f"      ❌ No response")
                    
                    # Delay between queries
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    results["details"].append({
                        "question": question_map.get(i, "Unknown"),
                        "status": "❌ EXCEPTION", 
                        "error": str(e)
                    })
                    logger.error(f"      ❌ Exception: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"   ❌ Simple SQL tool test failed: {e}")
            return {"error": str(e)}
    
    async def test_sql_tool_with_context(self):
        """Test SQL tool with user context."""
        logger.info("\n👤 Testing Simple SQL Tool with User Context...")
        
        try:
            from agents.tools import simple_query_crm_data, UserContext
            
            # Test with user context
            with UserContext(user_id=self.test_user_id, conversation_id="test-context"):
                
                # Test an employee-specific query
                question = "What opportunities do I have?"
                logger.info(f"   🔍 Testing with context: {question}")
                
                response = await simple_query_crm_data(question=question)
                
                if response and len(response) > 10:
                    if not response.startswith("I encountered an issue"):
                        logger.info(f"   ✅ Context-aware query works - Response: {response[:100]}...")
                        return {"context_query": True, "response_length": len(response)}
                    else:
                        logger.warning(f"   ❌ Context query had error: {response[:100]}...")
                        return {"context_query": False, "error": response[:200]}
                else:
                    logger.warning("   ❌ No response from context query")
                    return {"context_query": False, "error": "No response"}
                    
        except Exception as e:
            logger.error(f"   ❌ SQL tool with context test failed: {e}")
            return {"error": str(e)}
    
    async def test_model_selection_simple(self):
        """Test simple model selection functionality."""
        logger.info("\n🧠 Testing Simple Model Selection...")
        
        try:
            from agents.tools import _get_appropriate_llm
            
            # Test simple question
            simple_llm = await _get_appropriate_llm("Count employees")
            
            # Test complex question
            complex_llm = await _get_appropriate_llm("Analyze the comprehensive business implications of our market strategy while considering regulatory compliance and long-term sustainability factors")
            
            if simple_llm and complex_llm:
                logger.info(f"   ✅ Model selection works - Simple: {simple_llm.model_name}, Complex: {complex_llm.model_name}")
                return {
                    "model_selection": True,
                    "simple_model": simple_llm.model_name,
                    "complex_model": complex_llm.model_name
                }
            else:
                logger.warning("   ❌ Model selection failed")
                return {"model_selection": False}
                
        except Exception as e:
            logger.error(f"   ❌ Model selection test failed: {e}")
            return {"model_selection": False, "error": str(e)}
    
    async def run_focused_tests(self):
        """Run all focused tests."""
        logger.info("🚀 Starting Focused Tests for Simplified Tools")
        logger.info("=" * 60)
        
        # Setup
        setup_success = await self.setup_test_environment()
        if not setup_success:
            logger.error("❌ Test setup failed - cannot continue")
            return {"setup_failed": True}
        
        # Run tests
        test_results = {}
        
        try:
            test_results["user_context"] = await self.test_user_context_core()
            test_results["database"] = await self.test_sql_database_connectivity() 
            test_results["sql_basic"] = await self.test_simple_sql_tool_basic()
            test_results["sql_context"] = await self.test_sql_tool_with_context()
            test_results["model_selection"] = await self.test_model_selection_simple()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            test_results["execution_error"] = str(e)
        
        # Generate summary
        self.generate_focused_summary(test_results)
        
        return test_results
    
    def generate_focused_summary(self, results: Dict[str, Any]):
        """Generate focused test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 FOCUSED TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        categories_passed = 0
        total_categories = 0
        
        for category, result in results.items():
            if isinstance(result, dict):
                total_categories += 1
                category_name = category.replace("_", " ").title()
                
                if category == "user_context":
                    passed_tests = sum(1 for v in result.values() if v is True)
                    total_tests = len([k for k in result.keys() if not k.startswith("error")])
                    if passed_tests == total_tests:
                        categories_passed += 1
                    logger.info(f"   {category_name}: {passed_tests}/{total_tests} components working")
                
                elif category == "database":
                    if result.get("connectivity", False):
                        categories_passed += 1
                    status = "✅ CONNECTED" if result.get("connectivity", False) else "❌ FAILED"
                    logger.info(f"   {category_name}: {status}")
                
                elif category == "sql_basic":
                    if "successful" in result:
                        successful = result.get("successful", 0)
                        total = result.get("queries_tested", 0)
                        if successful > 0:
                            categories_passed += 1
                        logger.info(f"   {category_name}: {successful}/{total} queries successful")
                    else:
                        logger.info(f"   {category_name}: ❌ FAILED")
                
                elif category == "sql_context":
                    if result.get("context_query", False):
                        categories_passed += 1
                    status = "✅ WORKING" if result.get("context_query", False) else "❌ FAILED"
                    logger.info(f"   {category_name}: {status}")
                
                elif category == "model_selection":
                    if result.get("model_selection", False):
                        categories_passed += 1
                    status = "✅ WORKING" if result.get("model_selection", False) else "❌ FAILED"
                    logger.info(f"   {category_name}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"🎯 OVERALL RESULT: {categories_passed}/{total_categories} categories passed")
        
        if categories_passed >= total_categories * 0.8:  # 80% pass rate
            logger.info("🎉 SIMPLIFIED TOOLS ARE WORKING! Core functionality retained!")
        else:
            logger.warning(f"⚠️  Some functionality needs attention")
        
        logger.info("=" * 60)

# Test execution
async def run_focused_test():
    """Main focused test execution."""
    tester = TestSimplifiedToolsFocused()
    return await tester.run_focused_tests()

if __name__ == "__main__":
    asyncio.run(run_focused_test()) 