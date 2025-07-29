#!/usr/bin/env python3
"""
Phase 1 Dual Agent System Test Runner

Simple test runner that validates core Phase 1 functionality:
- User verification and routing
- Tool access control
- User context management  
- Agent node validation

Run with: python tests/run_phase1_tests.py
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
    from agents.tools import UserContext, get_current_user_type, get_tools_for_user_type
    from langchain_core.messages import HumanMessage, AIMessage
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)


class TestResult:
    """Simple test result tracking."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"  ❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Test Results Summary:")
        print(f"   Total tests: {total}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")
        
        if self.failed > 0:
            print(f"\n❌ Failed tests:")
            for error in self.errors:
                print(f"   - {error}")
        
        return self.failed == 0


async def test_user_context_system():
    """Test user context management."""
    print("\n🧪 Testing User Context System...")
    results = TestResult()
    
    try:
        # Test basic context setting
        with UserContext(user_id="test.user", conversation_id="conv-123", user_type="employee"):
            if get_current_user_type() == "employee":
                results.add_pass("User context setting - employee")
            else:
                results.add_fail("User context setting - employee", f"Expected 'employee', got '{get_current_user_type()}'")
        
        # Test context clearing
        if get_current_user_type() is None:
            results.add_pass("User context clearing")
        else:
            results.add_fail("User context clearing", f"Expected None, got '{get_current_user_type()}'")
        
        # Test customer context
        with UserContext(user_id="customer.user", conversation_id="conv-456", user_type="customer"):
            if get_current_user_type() == "customer":
                results.add_pass("User context setting - customer")
            else:
                results.add_fail("User context setting - customer", f"Expected 'customer', got '{get_current_user_type()}'")
        
        # Test nested contexts
        with UserContext(user_id="user1", conversation_id="conv1", user_type="employee"):
            with UserContext(user_id="user2", conversation_id="conv2", user_type="customer"):
                if get_current_user_type() == "customer":
                    results.add_pass("Nested context override")
                else:
                    results.add_fail("Nested context override", f"Expected 'customer', got '{get_current_user_type()}'")
            
            if get_current_user_type() == "employee":
                results.add_pass("Nested context restoration")
            else:
                results.add_fail("Nested context restoration", f"Expected 'employee', got '{get_current_user_type()}'")
    
    except Exception as e:
        results.add_fail("User context system", str(e))
    
    return results


def test_tool_access_control():
    """Test tool access control by user type."""
    print("\n🔧 Testing Tool Access Control...")
    results = TestResult()
    
    try:
        # Test employee tool access
        employee_tools = get_tools_for_user_type("employee")
        if len(employee_tools) == 2:
            results.add_pass("Employee tool count")
        else:
            results.add_fail("Employee tool count", f"Expected 2 tools, got {len(employee_tools)}")
        
        employee_tool_names = [tool.name for tool in employee_tools]
        if "simple_query_crm_data" in employee_tool_names:
            results.add_pass("Employee has CRM tool")
        else:
            results.add_fail("Employee has CRM tool", f"CRM tool not found in {employee_tool_names}")
        
        if "simple_rag" in employee_tool_names:
            results.add_pass("Employee has RAG tool")
        else:
            results.add_fail("Employee has RAG tool", f"RAG tool not found in {employee_tool_names}")
        
        # Test admin tool access (should be same as employee)
        admin_tools = get_tools_for_user_type("admin")
        if len(admin_tools) == 2:
            results.add_pass("Admin tool count")
        else:
            results.add_fail("Admin tool count", f"Expected 2 tools, got {len(admin_tools)}")
        
        # Test customer tool access
        customer_tools = get_tools_for_user_type("customer")
        if len(customer_tools) == 2:
            results.add_pass("Customer tool count")
        else:
            results.add_fail("Customer tool count", f"Expected 2 tools, got {len(customer_tools)}")
        
        customer_tool_names = [tool.name for tool in customer_tools]
        if "simple_query_crm_data" in customer_tool_names:
            results.add_pass("Customer has restricted CRM tool")
        else:
            results.add_fail("Customer has restricted CRM tool", f"CRM tool not found in {customer_tool_names}")
        
        if "simple_rag" in customer_tool_names:
            results.add_pass("Customer has RAG tool")
        else:
            results.add_fail("Customer has RAG tool", f"RAG tool not found in {customer_tool_names}")
        
        # Test unknown user tool access
        unknown_tools = get_tools_for_user_type("unknown")
        if len(unknown_tools) == 0:
            results.add_pass("Unknown user has no tools")
        else:
            results.add_fail("Unknown user has no tools", f"Expected 0 tools, got {len(unknown_tools)}")
    
    except Exception as e:
        results.add_fail("Tool access control", str(e))
    
    return results


async def test_customer_crm_restrictions():
    """Test customer CRM query restrictions."""
    print("\n🛡️ Testing Customer CRM Access Restrictions...")
    results = TestResult()
    
    try:
        from agents.tools import simple_query_crm_data
        
        # Test restricted queries
        restricted_queries = [
            "show me all employees",
            "list customer data",
            "what opportunities are available?",
            "show me sales activities",
            "which branches perform best?"
        ]
        
        for query in restricted_queries:
            with patch('agents.tools.get_current_user_type', return_value='customer'):
                result = await simple_query_crm_data.ainvoke({"question": query})
                if "I apologize, but I can only help you with vehicle specifications and pricing" in result:
                    results.add_pass(f"Blocked restricted query: '{query[:30]}...'")
                else:
                    results.add_fail(f"Should block query: '{query[:30]}...'", "Query was not blocked")
        
        # Test allowed vehicle/pricing queries (mock the database execution)
        allowed_queries = [
            "show me vehicle models",
            "what's the price of a Toyota?",
            "list available inventory"
        ]
        
        for query in allowed_queries:
            with patch('agents.tools.get_current_user_type', return_value='customer'):
                # Mock the SQL generation and execution
                with patch('agents.tools._generate_sql_query_simple', return_value="SELECT * FROM vehicles"):
                    with patch('agents.tools.get_settings') as mock_settings:
                        with patch('agents.tools.create_engine') as mock_engine:
                            # Mock database setup
                            mock_settings.return_value = MagicMock()
                            mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                            mock_engine.return_value.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [("Toyota", "Camry")]
                            
                            result = await simple_query_crm_data.ainvoke({"question": query})
                            if "I apologize, but I can only help you with vehicle specifications" not in result:
                                results.add_pass(f"Allowed vehicle query: '{query[:30]}...'")
                            else:
                                results.add_fail(f"Should allow query: '{query[:30]}...'", "Valid query was blocked")
    
    except Exception as e:
        results.add_fail("Customer CRM restrictions", str(e))
    
    return results


def test_agent_routing():
    """Test agent routing logic."""
    print("\n🎯 Testing Agent Routing Logic...")
    results = TestResult()
    
    try:
        agent = UnifiedToolCallingRAGAgent()
        
        # Test employee routing
        employee_state = {
            "user_verified": True,
            "user_type": "employee",
            "user_id": "john.smith"
        }
        route = agent._route_after_user_verification(employee_state)
        if route == "employee_agent":
            results.add_pass("Employee routing")
        else:
            results.add_fail("Employee routing", f"Expected 'employee_agent', got '{route}'")
        
        # Test admin routing (should go to employee agent)
        admin_state = {
            "user_verified": True,
            "user_type": "admin",
            "user_id": "admin.user"
        }
        route = agent._route_after_user_verification(admin_state)
        if route == "employee_agent":
            results.add_pass("Admin routing")
        else:
            results.add_fail("Admin routing", f"Expected 'employee_agent', got '{route}'")
        
        # Test customer routing
        customer_state = {
            "user_verified": True,
            "user_type": "customer",
            "user_id": "jane.doe"
        }
        route = agent._route_after_user_verification(customer_state)
        if route == "customer_agent":
            results.add_pass("Customer routing")
        else:
            results.add_fail("Customer routing", f"Expected 'customer_agent', got '{route}'")
        
        # Test unverified user routing
        unverified_state = {
            "user_verified": False,
            "user_type": "unknown",
            "user_id": "unknown.user"
        }
        route = agent._route_after_user_verification(unverified_state)
        if route == "end":
            results.add_pass("Unverified user routing")
        else:
            results.add_fail("Unverified user routing", f"Expected 'end', got '{route}'")
        
        # Test unknown verified user routing (edge case)
        unknown_verified_state = {
            "user_verified": True,
            "user_type": "unknown",
            "user_id": "weird.user"
        }
        route = agent._route_after_user_verification(unknown_verified_state)
        if route == "end":
            results.add_pass("Unknown verified user routing")
        else:
            results.add_fail("Unknown verified user routing", f"Expected 'end', got '{route}'")
    
    except Exception as e:
        results.add_fail("Agent routing", str(e))
    
    return results


async def test_agent_node_access_control():
    """Test agent node access control."""
    print("\n🏢 Testing Agent Node Access Control...")
    results = TestResult()
    
    try:
        agent = UnifiedToolCallingRAGAgent()
        agent.initialized = True
        
        # Mock memory manager to avoid database calls
        agent.memory_manager = AsyncMock()
        agent.memory_manager.store_message_from_agent = AsyncMock()
        agent.memory_manager.get_user_context_for_new_conversation = AsyncMock(return_value={})
        agent.memory_manager.get_relevant_context = AsyncMock(return_value=[])
        agent.memory_manager.should_store_memory = AsyncMock(return_value=False)
        agent.memory_manager.consolidator.check_and_trigger_summarization = AsyncMock(return_value=None)
        
        # Test employee agent rejects customer
        customer_to_employee_state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "jane.doe",
            "user_type": "customer",
            "conversation_id": "conv-456",
            "user_verified": True
        }
        
        result = await agent._employee_agent_node(customer_to_employee_state)
        if "messages" in result and "routing error" in result["messages"][-1].content.lower():
            results.add_pass("Employee agent rejects customer")
        else:
            results.add_fail("Employee agent rejects customer", "Should return routing error")
        
        # Test customer agent rejects employee
        employee_to_customer_state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "john.smith",
            "user_type": "employee",
            "conversation_id": "conv-123",
            "user_verified": True
        }
        
        result = await agent._customer_agent_node(employee_to_customer_state)
        if "messages" in result and "routing error" in result["messages"][-1].content.lower():
            results.add_pass("Customer agent rejects employee")
        else:
            results.add_fail("Customer agent rejects employee", "Should return routing error")
    
    except Exception as e:
        results.add_fail("Agent node access control", str(e))
    
    return results


async def main():
    """Run all Phase 1 tests."""
    print("🧪 Phase 1 Dual Agent System Test Suite")
    print("=" * 50)
    print("Testing core functionality against PRD requirements:")
    print("• User verification and routing")
    print("• Tool access control and security")
    print("• Agent node functionality")  
    print("• Memory and context management")
    
    all_results = []
    
    # Run all test suites
    all_results.append(await test_user_context_system())
    all_results.append(test_tool_access_control())
    all_results.append(await test_customer_crm_restrictions())
    all_results.append(test_agent_routing())
    all_results.append(await test_agent_node_access_control())
    
    # Calculate overall results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print("\n" + "=" * 50)
    print(f"🏁 Final Results:")
    print(f"   Total tests run: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_failed}")
    
    if total_failed == 0:
        print(f"\n🎉 All tests passed! Phase 1 implementation is ready.")
        print(f"✅ User verification and routing: Working")
        print(f"✅ Tool access control: Working")
        print(f"✅ Customer CRM restrictions: Working")
        print(f"✅ Agent routing logic: Working") 
        print(f"✅ Agent node access control: Working")
        return True
    else:
        print(f"\n❌ {total_failed} tests failed. Phase 1 needs fixes.")
        print(f"\nFailed tests summary:")
        for i, result in enumerate(all_results):
            if result.failed > 0:
                for error in result.errors:
                    print(f"   - {error}")
        return False


if __name__ == "__main__":
    """Entry point."""
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test runner error: {e}")
        sys.exit(1) 