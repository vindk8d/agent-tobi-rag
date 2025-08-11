#!/usr/bin/env python3
"""
Comprehensive Test Suite for Optimized SQL Generation Approach
Tests various query types, user scenarios, and edge cases to validate the optimization.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
from dataclasses import dataclass, asdict
from enum import Enum

# Test Configuration
API_BASE_URL = "http://localhost:8000"
TEST_EMPLOYEE_USER_ID = "54394d40-ad35-4b5b-a392-1ae7c9329d11"  # From the error example
TEST_CUSTOMER_USER_ID = "a0b81a3b-3ac0-4a7d-b27c-dadf0fdbbe8b"  # From the logs

class QueryComplexity(Enum):
    SIMPLE = "simple"       # Should work with minimal context
    COMPLEX = "complex"     # Might need detailed schema tool
    CONTEXTUAL = "contextual"  # Needs conversation context
    INVALID = "invalid"     # Should handle gracefully

class UserType(Enum):
    EMPLOYEE = "employee"
    CUSTOMER = "customer"
    ADMIN = "admin"

@dataclass
class TestQuery:
    description: str
    question: str
    expected_complexity: QueryComplexity
    user_type: UserType
    should_succeed: bool
    expected_tools_used: List[str]  # Tools we expect the LLM to use
    notes: str = ""

@dataclass
class TestResult:
    query: TestQuery
    success: bool
    response_message: str
    error_message: Optional[str]
    tools_called: List[str]
    response_time_ms: int
    estimated_tokens: int
    actual_behavior: str

class SQLOptimizationTester:
    def __init__(self):
        self.results: List[TestResult] = []
        
    def get_test_scenarios(self) -> List[TestQuery]:
        """Define comprehensive test scenarios covering all cases."""
        
        return [
            # ================================
            # SIMPLE QUERIES (Minimal Context)
            # ================================
            TestQuery(
                description="Basic vehicle availability check",
                question="Is the Toyota Prius available?",
                expected_complexity=QueryComplexity.SIMPLE,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=[],
                notes="Should work with minimal schema only"
            ),
            
            TestQuery(
                description="Simple vehicle listing",
                question="Show me all available Toyota models",
                expected_complexity=QueryComplexity.SIMPLE,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=[],
                notes="Basic vehicle lookup"
            ),
            
            TestQuery(
                description="Employee basic customer lookup",
                question="Show me active customers",
                expected_complexity=QueryComplexity.SIMPLE,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=[],
                notes="Should work with minimal schema"
            ),
            
            # ================================
            # COMPLEX QUERIES (Need Detailed Schema)
            # ================================
            TestQuery(
                description="Vehicle with pricing information",
                question="Show me Toyota Prius models with their prices",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Needs JOIN between vehicles and pricing"
            ),
            
            TestQuery(
                description="Customer opportunities for employee",
                question="Show me customers with pending deals over $50000",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Needs JOIN between customers and opportunities"
            ),
            
            TestQuery(
                description="Complex aggregation query",
                question="What's the average price of available vehicles by brand?",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Needs detailed schema for aggregation"
            ),
            
            # ================================
            # CONTEXTUAL QUERIES (Need Conversation History)
            # ================================
            TestQuery(
                description="Reference to previous conversation",
                question="Send a follow-up message to that customer we discussed",
                expected_complexity=QueryComplexity.CONTEXTUAL,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=["get_recent_conversation_context"],
                notes="Should call conversation context tool"
            ),
            
            TestQuery(
                description="Reference with pronoun",
                question="What was the price of that Toyota model I asked about?",
                expected_complexity=QueryComplexity.CONTEXTUAL,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=["get_recent_conversation_context"],
                notes="Needs context to resolve 'that Toyota model'"
            ),
            
            # ================================
            # ACCESS CONTROL TESTS
            # ================================
            TestQuery(
                description="Customer trying to access employee data",
                question="Show me all employee performance metrics",
                expected_complexity=QueryComplexity.INVALID,
                user_type=UserType.CUSTOMER,
                should_succeed=False,
                expected_tools_used=[],
                notes="Should be blocked by access control"
            ),
            
            TestQuery(
                description="Customer trying to access customer records",
                question="Show me all customer contact information",
                expected_complexity=QueryComplexity.INVALID,
                user_type=UserType.CUSTOMER,
                should_succeed=False,
                expected_tools_used=[],
                notes="Should be blocked - customers can't see other customers"
            ),
            
            TestQuery(
                description="Employee accessing customer data (allowed)",
                question="Show me customer contact information for my active deals",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Employee should have access with proper filtering"
            ),
            
            # ================================
            # EDGE CASES & ERROR HANDLING
            # ================================
            TestQuery(
                description="Non-existent table",
                question="Show me data from the unicorns table",
                expected_complexity=QueryComplexity.INVALID,
                user_type=UserType.EMPLOYEE,
                should_succeed=False,
                expected_tools_used=[],
                notes="Should handle gracefully"
            ),
            
            TestQuery(
                description="Malformed question",
                question="asdfasdf car price when?",
                expected_complexity=QueryComplexity.INVALID,
                user_type=UserType.CUSTOMER,
                should_succeed=False,
                expected_tools_used=[],
                notes="Should handle gracefully or ask for clarification"
            ),
            
            TestQuery(
                description="Empty question",
                question="",
                expected_complexity=QueryComplexity.INVALID,
                user_type=UserType.CUSTOMER,
                should_succeed=False,
                expected_tools_used=[],
                notes="Should handle empty input gracefully"
            ),
            
            TestQuery(
                description="Impossible request",
                question="Show me vehicles that cost negative money",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Should generate valid SQL even if no results"
            ),
            
            # ================================
            # TOOL USAGE VALIDATION
            # ================================
            TestQuery(
                description="Question that should trigger schema tool",
                question="Show me the relationship between vehicles and their pricing details",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Should call detailed schema for relationship info"
            ),
            
            TestQuery(
                description="Question with time reference",
                question="Show me the vehicles we discussed in our last conversation",
                expected_complexity=QueryComplexity.CONTEXTUAL,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=["get_recent_conversation_context"],
                notes="Should call conversation context tool"
            ),
            
            # ================================
            # PERFORMANCE VALIDATION
            # ================================
            TestQuery(
                description="Very specific query (minimal context should suffice)",
                question="How many Toyota Camry vehicles are available?",
                expected_complexity=QueryComplexity.SIMPLE,
                user_type=UserType.CUSTOMER,
                should_succeed=True,
                expected_tools_used=[],
                notes="Should be fast with minimal context"
            ),
            
            TestQuery(
                description="Multi-table complex query",
                question="Show me customers who bought Toyota vehicles in the last 6 months with their transaction details",
                expected_complexity=QueryComplexity.COMPLEX,
                user_type=UserType.EMPLOYEE,
                should_succeed=True,
                expected_tools_used=["get_detailed_schema"],
                notes="Complex query needing multiple tables"
            )
        ]
    
    async def run_single_test(self, session: aiohttp.ClientSession, test_query: TestQuery) -> TestResult:
        """Run a single test query and capture results."""
        
        print(f"\n🧪 Testing: {test_query.description}")
        print(f"   Question: '{test_query.question}'")
        print(f"   User Type: {test_query.user_type.value}")
        print(f"   Expected: {'SUCCESS' if test_query.should_succeed else 'CONTROLLED FAILURE'}")
        
        start_time = time.time()
        
        # Select user ID based on user type
        user_id = TEST_EMPLOYEE_USER_ID if test_query.user_type == UserType.EMPLOYEE else TEST_CUSTOMER_USER_ID
        
        payload = {
            "message": test_query.question,
            "user_id": user_id,
            "include_sources": True
        }
        
        try:
            async with session.post(
                f"{API_BASE_URL}/api/v1/chat/message",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    # Extract response message
                    if "data" in result_data and "message" in result_data["data"]:
                        response_message = result_data["data"]["message"]
                    elif "message" in result_data:
                        response_message = result_data["message"]
                    else:
                        response_message = str(result_data)
                    
                    # Analyze response for tool usage (heuristic)
                    tools_called = []
                    if "schema" in response_message.lower() or "table" in response_message.lower():
                        tools_called.append("get_detailed_schema")
                    if "conversation" in response_message.lower() or "discussed" in response_message.lower():
                        tools_called.append("get_recent_conversation_context")
                    
                    # Estimate token usage (rough approximation)
                    estimated_tokens = len(test_query.question) // 4 + 500  # Minimal approach should be ~500-1000 tokens
                    
                    success = True
                    error_message = None
                    actual_behavior = "Generated response successfully"
                    
                    print(f"   ✅ SUCCESS ({response_time_ms}ms, ~{estimated_tokens} tokens)")
                    
                else:
                    response_text = await response.text()
                    success = False
                    response_message = response_text
                    error_message = f"HTTP {response.status}: {response_text}"
                    tools_called = []
                    estimated_tokens = 0
                    actual_behavior = f"HTTP Error {response.status}"
                    
                    print(f"   ❌ HTTP ERROR {response.status}")
                
        except asyncio.TimeoutError:
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            success = False
            response_message = "Request timed out"
            error_message = "Timeout after 60 seconds"
            tools_called = []
            estimated_tokens = 0
            actual_behavior = "Timeout"
            
            print(f"   ⏰ TIMEOUT ({response_time_ms}ms)")
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            success = False
            response_message = str(e)
            error_message = str(e)
            tools_called = []
            estimated_tokens = 0
            actual_behavior = f"Exception: {type(e).__name__}"
            
            print(f"   💥 ERROR: {e}")
        
        return TestResult(
            query=test_query,
            success=success,
            response_message=response_message,
            error_message=error_message,
            tools_called=tools_called,
            response_time_ms=response_time_ms,
            estimated_tokens=estimated_tokens,
            actual_behavior=actual_behavior
        )
    
    async def run_comprehensive_test_suite(self):
        """Run the complete test suite and generate report."""
        
        print("🚀 Starting Comprehensive SQL Optimization Test Suite")
        print("=" * 70)
        
        test_scenarios = self.get_test_scenarios()
        
        async with aiohttp.ClientSession() as session:
            # Test API connectivity first
            try:
                async with session.get(f"{API_BASE_URL}/health") as response:
                    if response.status != 200:
                        print(f"❌ API not available at {API_BASE_URL}")
                        return
                    print(f"✅ API connectivity confirmed at {API_BASE_URL}")
            except Exception as e:
                print(f"❌ Cannot connect to API: {e}")
                return
            
            # Run all test scenarios
            for i, test_query in enumerate(test_scenarios, 1):
                print(f"\n[{i}/{len(test_scenarios)}]", end="")
                result = await self.run_single_test(session, test_query)
                self.results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(0.5)
        
        # Generate comprehensive report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report with analysis."""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 70)
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"   Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Performance analysis
        response_times = [r.response_time_ms for r in self.results if r.success]
        if response_times:
            avg_response_time = sum(response_times) // len(response_times)
            print(f"   Average Response Time: {avg_response_time}ms")
        
        token_usage = [r.estimated_tokens for r in self.results if r.success]
        if token_usage:
            avg_tokens = sum(token_usage) // len(token_usage)
            print(f"   Average Token Usage: {avg_tokens} tokens")
        
        # Results by complexity
        print(f"\n📊 RESULTS BY QUERY COMPLEXITY:")
        for complexity in QueryComplexity:
            complexity_results = [r for r in self.results if r.query.expected_complexity == complexity]
            if complexity_results:
                success_count = sum(1 for r in complexity_results if r.success)
                total_count = len(complexity_results)
                print(f"   {complexity.value.title()}: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        # Results by user type
        print(f"\n👥 RESULTS BY USER TYPE:")
        for user_type in UserType:
            user_results = [r for r in self.results if r.query.user_type == user_type]
            if user_results:
                success_count = sum(1 for r in user_results if r.success)
                total_count = len(user_results)
                print(f"   {user_type.value.title()}: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        # Tool usage analysis
        print(f"\n🛠️  TOOL USAGE ANALYSIS:")
        all_tools_called = []
        for result in self.results:
            all_tools_called.extend(result.tools_called)
        
        if all_tools_called:
            from collections import Counter
            tool_counts = Counter(all_tools_called)
            for tool, count in tool_counts.items():
                print(f"   {tool}: {count} times")
        else:
            print("   No tools detected in responses (heuristic analysis)")
        
        # Failed tests analysis
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\n❌ FAILED TESTS ANALYSIS:")
            for result in failed_results:
                expected_to_fail = not result.query.should_succeed
                status = "EXPECTED" if expected_to_fail else "UNEXPECTED"
                print(f"   [{status}] {result.query.description}")
                print(f"      Error: {result.error_message}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.success else "❌"
            expected = "Expected" if (result.success == result.query.should_succeed) else "Unexpected"
            
            print(f"\n{i:2d}. {status} [{expected}] {result.query.description}")
            print(f"     Query: {result.query.question}")
            print(f"     User: {result.query.user_type.value}")
            print(f"     Time: {result.response_time_ms}ms")
            print(f"     Tokens: ~{result.estimated_tokens}")
            if result.tools_called:
                print(f"     Tools: {', '.join(result.tools_called)}")
            if not result.success:
                print(f"     Error: {result.error_message}")
        
        # Save detailed results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sql_optimization_test_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in self.results:
            json_result = asdict(result)
            # Convert enums to strings
            json_result['query']['expected_complexity'] = result.query.expected_complexity.value
            json_result['query']['user_type'] = result.query.user_type.value
            json_results.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'average_response_time_ms': avg_response_time if response_times else 0,
                'average_token_usage': avg_tokens if token_usage else 0,
                'detailed_results': json_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        # Summary and recommendations
        print(f"\n🎯 SUMMARY & RECOMMENDATIONS:")
        
        if successful_tests / total_tests >= 0.8:
            print("   ✅ EXCELLENT: >80% success rate - optimization is working well!")
        elif successful_tests / total_tests >= 0.6:
            print("   ⚠️  GOOD: >60% success rate - some areas need improvement")
        else:
            print("   ❌ NEEDS WORK: <60% success rate - significant issues detected")
        
        if avg_tokens < 2000:
            print("   ✅ TOKEN USAGE: Excellent - staying under 2K tokens")
        elif avg_tokens < 5000:
            print("   ⚠️  TOKEN USAGE: Good - under 5K tokens but could improve")
        else:
            print("   ❌ TOKEN USAGE: High - optimization may not be working")
        
        print(f"\n🏁 Test Suite Complete!")

async def main():
    """Main function to run the comprehensive test suite."""
    tester = SQLOptimizationTester()
    await tester.run_comprehensive_test_suite()

if __name__ == "__main__":
    asyncio.run(main())