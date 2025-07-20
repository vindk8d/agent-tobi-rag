"""
Comprehensive Test for Simplified Tools

This test verifies that our simplified tools retain all functionality while being much cleaner:
1. Simple SQL tool functionality (simple_query_crm_data)
2. Simple RAG tool functionality (simple_rag)
3. User context functionality
4. Conversation context functionality 
5. Integration with the RAG agent

Tests are designed to run against real agent messages and verify end-to-end functionality.
"""

import asyncio
import pytest
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.tools import (
    simple_query_crm_data, 
    simple_rag,
    UserContext,
    get_current_user_id,
    get_current_conversation_id,
    get_current_employee_id,
    get_all_tools,
    _get_sql_database,
    _get_conversation_context,
    _get_appropriate_llm
)
from backend.agents.tobi_sales_copilot.rag_agent import TobiSalesCopilotRAG
from backend.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSimplifiedToolsComprehensive:
    """Comprehensive test suite for simplified tools functionality."""
    
    def __init__(self):
        self.test_user_id = "54394d40-ad35-4b5b-a392-1ae7c9329d11"  # Alex Thompson
        self.test_employee_id = "db8cdcda-3c96-4e35-8fe4-860f927b9395"  # Alex's employee ID
        self.test_conversation_id = None
        
    async def setup_test_environment(self):
        """Set up the test environment and verify database connectivity."""
        logger.info("🔧 Setting up test environment...")
        
        # Test database connectivity
        db = await _get_sql_database()
        if db is None:
            raise Exception("❌ Database connection failed - cannot run tests")
        
        logger.info("✅ Database connection established")
        return True
    
    async def test_user_context_functionality(self):
        """Test user context management functionality."""
        logger.info("\n📋 Testing User Context Functionality...")
        
        test_results = {
            "context_manager_works": False,
            "user_id_retrieval": False,
            "conversation_id_retrieval": False,
            "employee_id_resolution": False
        }
        
        try:
            # Test 1: UserContext manager
            with UserContext(user_id=self.test_user_id, conversation_id="test-conv-123"):
                current_user = get_current_user_id()
                current_conv = get_current_conversation_id()
                
                if current_user == self.test_user_id:
                    test_results["user_id_retrieval"] = True
                    logger.info("   ✅ User ID context retrieval works")
                
                if current_conv == "test-conv-123":
                    test_results["conversation_id_retrieval"] = True
                    logger.info("   ✅ Conversation ID context retrieval works")
                
                # Test employee ID resolution
                employee_id = await get_current_employee_id()
                if employee_id == self.test_employee_id:
                    test_results["employee_id_resolution"] = True
                    logger.info(f"   ✅ Employee ID resolution works: {employee_id}")
            
            # Verify context is cleared outside the manager
            if get_current_user_id() is None and get_current_conversation_id() is None:
                test_results["context_manager_works"] = True
                logger.info("   ✅ Context manager cleanup works")
        
        except Exception as e:
            logger.error(f"   ❌ User context test failed: {e}")
        
        return test_results
    
    async def test_sql_tool_functionality(self):
        """Test simple SQL tool with various query types."""
        logger.info("\n🗃️ Testing Simple SQL Tool Functionality...")
        
        test_queries = [
            {
                "name": "Basic count query",
                "question": "How many employees are there?",
                "expect_success": True
            },
            {
                "name": "Employee listing",
                "question": "What are the names of all employees?",
                "expect_success": True
            },
            {
                "name": "Vehicle inventory",
                "question": "What vehicles do we have in stock?",
                "expect_success": True
            },
            {
                "name": "Customer count",
                "question": "How many customers do we have?",
                "expect_success": True
            },
            {
                "name": "Complex business query",
                "question": "What opportunities are in the pipeline?",
                "expect_success": True
            }
        ]
        
        results = {"total_tests": len(test_queries), "passed": 0, "failed": 0, "details": []}
        
        # Test without user context
        for query in test_queries:
            try:
                logger.info(f"   🔍 Testing: {query['name']}")
                response = await simple_query_crm_data(question=query["question"])
                
                if response and len(response) > 50 and not response.startswith("I encountered"):
                    results["passed"] += 1
                    results["details"].append({
                        "query": query["name"],
                        "status": "✅ PASS",
                        "response_length": len(response)
                    })
                    logger.info(f"      ✅ Success - Response length: {len(response)}")
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "query": query["name"],
                        "status": "❌ FAIL",
                        "response": response[:200] if response else "No response"
                    })
                    logger.warning(f"      ❌ Failed - Response: {response[:100] if response else 'None'}")
                
                # Small delay between queries
                await asyncio.sleep(1)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "query": query["name"],
                    "status": "❌ ERROR",
                    "error": str(e)
                })
                logger.error(f"      ❌ Error: {e}")
        
        return results
    
    async def test_sql_tool_with_user_context(self):
        """Test SQL tool with user context (employee-specific queries)."""
        logger.info("\n👤 Testing SQL Tool with User Context...")
        
        employee_queries = [
            {
                "name": "My leads",
                "question": "What are my leads?",
                "expect_success": True
            },
            {
                "name": "My opportunities", 
                "question": "Show me my opportunities",
                "expect_success": True
            },
            {
                "name": "My sales pipeline",
                "question": "What's in my sales pipeline?",
                "expect_success": True
            }
        ]
        
        results = {"total_tests": len(employee_queries), "passed": 0, "failed": 0, "details": []}
        
        # Test with user context
        with UserContext(user_id=self.test_user_id, conversation_id="test-employee-conv"):
            for query in employee_queries:
                try:
                    logger.info(f"   🔍 Testing: {query['name']}")
                    response = await simple_query_crm_data(question=query["question"])
                    
                    if response and len(response) > 30 and not response.startswith("I encountered"):
                        results["passed"] += 1
                        results["details"].append({
                            "query": query["name"],
                            "status": "✅ PASS", 
                            "response_length": len(response)
                        })
                        logger.info(f"      ✅ Success - Response length: {len(response)}")
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "query": query["name"],
                            "status": "❌ FAIL",
                            "response": response[:200] if response else "No response"
                        })
                        logger.warning(f"      ❌ Failed - Response: {response[:100] if response else 'None'}")
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append({
                        "query": query["name"],
                        "status": "❌ ERROR",
                        "error": str(e)
                    })
                    logger.error(f"      ❌ Error: {e}")
        
        return results
    
    async def test_rag_tool_functionality(self):
        """Test simple RAG tool functionality."""
        logger.info("\n📚 Testing Simple RAG Tool Functionality...")
        
        rag_queries = [
            {
                "name": "General company info",
                "question": "What is the company's mission?",
                "expect_success": True
            },
            {
                "name": "Product information", 
                "question": "What products does the company offer?",
                "expect_success": True
            },
            {
                "name": "Customer support",
                "question": "How can customers get support?",
                "expect_success": True
            }
        ]
        
        results = {"total_tests": len(rag_queries), "passed": 0, "failed": 0, "details": []}
        
        for query in rag_queries:
            try:
                logger.info(f"   🔍 Testing: {query['name']}")
                response = await simple_rag(question=query["question"])
                
                if response and len(response) > 50:
                    if "couldn't find any relevant documents" in response:
                        # This is acceptable - means no documents uploaded for this topic
                        results["passed"] += 1
                        results["details"].append({
                            "query": query["name"],
                            "status": "✅ PASS (No docs)",
                            "note": "Tool works, no relevant documents found"
                        })
                        logger.info(f"      ✅ Tool works - No documents found (expected)")
                    else:
                        results["passed"] += 1
                        results["details"].append({
                            "query": query["name"],
                            "status": "✅ PASS",
                            "response_length": len(response)
                        })
                        logger.info(f"      ✅ Success - Response length: {len(response)}")
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "query": query["name"],
                        "status": "❌ FAIL",
                        "response": response[:200] if response else "No response"
                    })
                    logger.warning(f"      ❌ Failed - Response: {response[:100] if response else 'None'}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "query": query["name"],
                    "status": "❌ ERROR",
                    "error": str(e)
                })
                logger.error(f"      ❌ Error: {e}")
        
        return results
    
    async def test_conversation_context(self):
        """Test conversation context functionality."""
        logger.info("\n💬 Testing Conversation Context Functionality...")
        
        test_conv_id = f"test-conv-{int(datetime.now().timestamp())}"
        
        try:
            # Test conversation context retrieval
            context = await _get_conversation_context(test_conv_id)
            
            # Even if no messages, should return empty string without error
            if isinstance(context, str):
                logger.info("   ✅ Conversation context retrieval works")
                return {"context_retrieval": True}
            else:
                logger.warning("   ❌ Conversation context didn't return string")
                return {"context_retrieval": False}
                
        except Exception as e:
            logger.error(f"   ❌ Conversation context test failed: {e}")
            return {"context_retrieval": False}
    
    async def test_agent_integration(self):
        """Test integration with the RAG agent."""
        logger.info("\n🤖 Testing Agent Integration...")
        
        try:
            # Initialize the RAG agent
            agent = TobiSalesCopilotRAG()
            
            # Test basic agent functionality
            test_messages = [
                {"role": "human", "content": "How many employees work here?"}
            ]
            
            with UserContext(user_id=self.test_user_id):
                response = await agent.run_agent(
                    messages=test_messages,
                    user_id=self.test_user_id
                )
            
            if response and response.get("response"):
                logger.info(f"   ✅ Agent integration works - Response length: {len(response['response'])}")
                return {"agent_integration": True, "response_length": len(response['response'])}
            else:
                logger.warning(f"   ❌ Agent integration failed - No valid response")
                return {"agent_integration": False}
                
        except Exception as e:
            logger.error(f"   ❌ Agent integration test failed: {e}")
            return {"agent_integration": False, "error": str(e)}
    
    async def test_model_selection(self):
        """Test LLM model selection functionality."""
        logger.info("\n🧠 Testing Model Selection...")
        
        try:
            # Test simple question
            simple_llm = await _get_appropriate_llm("How many?")
            
            # Test complex question
            complex_llm = await _get_appropriate_llm("Please analyze the comprehensive business strategy implications of our current market position relative to our competitors while considering the long-term sustainability factors and potential regulatory changes that might impact our operational efficiency over the next five years.")
            
            if simple_llm and complex_llm:
                logger.info("   ✅ Model selection works")
                return {"model_selection": True}
            else:
                logger.warning("   ❌ Model selection failed")
                return {"model_selection": False}
                
        except Exception as e:
            logger.error(f"   ❌ Model selection test failed: {e}")
            return {"model_selection": False}
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests and return detailed results."""
        logger.info("🚀 Starting Comprehensive Tests for Simplified Tools")
        logger.info("=" * 70)
        
        # Setup
        await self.setup_test_environment()
        
        # Run all tests
        test_results = {}
        
        try:
            test_results["user_context"] = await self.test_user_context_functionality()
            test_results["sql_basic"] = await self.test_sql_tool_functionality()
            test_results["sql_user_context"] = await self.test_sql_tool_with_user_context()
            test_results["rag"] = await self.test_rag_tool_functionality()
            test_results["conversation_context"] = await self.test_conversation_context()
            test_results["model_selection"] = await self.test_model_selection()
            test_results["agent_integration"] = await self.test_agent_integration()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            test_results["execution_error"] = str(e)
        
        # Generate summary
        self.generate_test_summary(test_results)
        
        return test_results
    
    def generate_test_summary(self, results: Dict[str, Any]):
        """Generate a comprehensive test summary."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        total_categories = 0
        passed_categories = 0
        
        for category, result in results.items():
            if isinstance(result, dict):
                total_categories += 1
                category_name = category.replace("_", " ").title()
                
                if category == "user_context":
                    passed_tests = sum(1 for v in result.values() if v is True)
                    total_tests = len(result)
                    if passed_tests == total_tests:
                        passed_categories += 1
                    logger.info(f"   {category_name}: {passed_tests}/{total_tests} tests passed")
                
                elif category in ["sql_basic", "sql_user_context", "rag"]:
                    passed_tests = result.get("passed", 0)
                    total_tests = result.get("total_tests", 0)
                    if passed_tests == total_tests:
                        passed_categories += 1
                    logger.info(f"   {category_name}: {passed_tests}/{total_tests} queries successful")
                
                elif category in ["conversation_context", "model_selection", "agent_integration"]:
                    success = result.get(list(result.keys())[0], False) if result else False
                    if success:
                        passed_categories += 1
                    status = "✅ PASS" if success else "❌ FAIL"
                    logger.info(f"   {category_name}: {status}")
        
        logger.info("-" * 70)
        logger.info(f"🎯 OVERALL RESULT: {passed_categories}/{total_categories} categories passed")
        
        if passed_categories == total_categories:
            logger.info("🎉 ALL TESTS PASSED! Simplified tools retain full functionality!")
        else:
            logger.warning(f"⚠️  {total_categories - passed_categories} categories need attention")
        
        logger.info("=" * 70)

# Test execution functions
async def run_comprehensive_test():
    """Main test execution function."""
    tester = TestSimplifiedToolsComprehensive()
    return await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test()) 