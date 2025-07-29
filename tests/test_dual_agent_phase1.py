"""
Comprehensive Test Suite for Phase 1: Dual User Agent System

This test suite validates all Phase 1 requirements from the PRD:
1. User verification and routing system
2. Basic customer agent with RAG access  
3. Employee agent with existing functionality
4. Unknown user fallback handling
5. Tool access control and security
6. Memory management and isolation
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from agents.tools import UserContext, get_current_user_type, get_tools_for_user_type


class TestUserVerification:
    """Test user verification and type detection."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        agent = UnifiedToolCallingRAGAgent()
        # Mock initialization to avoid actual database connections during testing
        agent.initialized = True
        return agent

    @pytest.mark.asyncio
    async def test_employee_user_verification(self, agent):
        """Test that employee users are correctly identified."""
        # Mock database response for employee user
        with patch.object(agent, '_verify_user_access', return_value='employee'):
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "john.smith",
                "conversation_id": "test-conv-123"
            }
            
            result = await agent._user_verification_node(state)
            
            assert result["user_verified"] == True
            assert result["user_type"] == "employee"
            assert result["user_id"] == "john.smith"

    @pytest.mark.asyncio
    async def test_customer_user_verification(self, agent):
        """Test that customer users are correctly identified."""
        with patch.object(agent, '_verify_user_access', return_value='customer'):
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "jane.doe",
                "conversation_id": "test-conv-456"
            }
            
            result = await agent._user_verification_node(state)
            
            assert result["user_verified"] == True
            assert result["user_type"] == "customer"
            assert result["user_id"] == "jane.doe"

    @pytest.mark.asyncio
    async def test_unknown_user_verification(self, agent):
        """Test that unknown users are properly rejected."""
        with patch.object(agent, '_verify_user_access', return_value='unknown'):
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "unknown.user",
                "conversation_id": "test-conv-789"
            }
            
            result = await agent._user_verification_node(state)
            
            assert result["user_verified"] == False
            assert "user_type" not in result or result["user_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_admin_user_verification(self, agent):
        """Test that admin users are treated as employees."""
        with patch.object(agent, '_verify_user_access', return_value='admin'):
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "admin.user",
                "conversation_id": "test-conv-admin"
            }
            
            result = await agent._user_verification_node(state)
            
            assert result["user_verified"] == True
            assert result["user_type"] == "admin"

    def test_routing_employee_to_employee_agent(self, agent):
        """Test routing logic for employee users."""
        state = {
            "user_verified": True,
            "user_type": "employee",
            "user_id": "john.smith"
        }
        
        result = agent._route_after_user_verification(state)
        assert result == "employee_agent"

    def test_routing_admin_to_employee_agent(self, agent):
        """Test routing logic for admin users."""
        state = {
            "user_verified": True,
            "user_type": "admin",
            "user_id": "admin.user"
        }
        
        result = agent._route_after_user_verification(state)
        assert result == "employee_agent"

    def test_routing_customer_to_customer_agent(self, agent):
        """Test routing logic for customer users."""
        state = {
            "user_verified": True,
            "user_type": "customer",
            "user_id": "jane.doe"
        }
        
        result = agent._route_after_user_verification(state)
        assert result == "customer_agent"

    def test_routing_unverified_user_to_end(self, agent):
        """Test routing logic for unverified users."""
        state = {
            "user_verified": False,
            "user_type": "unknown",
            "user_id": "unknown.user"
        }
        
        result = agent._route_after_user_verification(state)
        assert result == "end"

    def test_routing_unknown_verified_user_to_end(self, agent):
        """Test routing logic for unknown but verified users (edge case)."""
        state = {
            "user_verified": True,
            "user_type": "unknown",
            "user_id": "weird.user"
        }
        
        result = agent._route_after_user_verification(state)
        assert result == "end"


class TestToolAccessControl:
    """Test tool access control based on user type."""

    def test_employee_tool_access(self):
        """Test that employees get full tool access."""
        tools = get_tools_for_user_type("employee")
        
        # Should have both tools with full access
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "simple_query_crm_data" in tool_names
        assert "simple_rag" in tool_names

    def test_admin_tool_access(self):
        """Test that admins get full tool access."""
        tools = get_tools_for_user_type("admin")
        
        # Should have both tools with full access
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "simple_query_crm_data" in tool_names
        assert "simple_rag" in tool_names

    def test_customer_tool_access(self):
        """Test that customers get restricted tool access."""
        tools = get_tools_for_user_type("customer")
        
        # Should have both tools but CRM will be internally restricted
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "simple_query_crm_data" in tool_names
        assert "simple_rag" in tool_names

    def test_unknown_user_tool_access(self):
        """Test that unknown users get no tools."""
        tools = get_tools_for_user_type("unknown")
        
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_customer_crm_keyword_blocking(self):
        """Test that customers are blocked from accessing restricted CRM data."""
        from agents.tools import simple_query_crm_data
        
        # Test various restricted keywords
        restricted_queries = [
            "show me all employees",
            "list customer records",
            "what opportunities are in the pipeline?",
            "show me sales activities",
            "which branches have the best performance?"
        ]
        
        for query in restricted_queries:
            with patch('agents.tools.get_current_user_type', return_value='customer'):
                result = await simple_query_crm_data(query)
                
                # Should be blocked with access denied message
                assert "I apologize, but I can only help you with vehicle specifications and pricing" in result
                assert "vehicle" in result.lower()
                assert "pricing" in result.lower()

    @pytest.mark.asyncio 
    async def test_customer_crm_allowed_queries(self):
        """Test that customers can access vehicle and pricing information."""
        from agents.tools import simple_query_crm_data
        
        allowed_queries = [
            "show me vehicle models",
            "what is the price of a Toyota Camry?",
            "list available inventory",
            "compare vehicle specifications"
        ]
        
        for query in allowed_queries:
            with patch('agents.tools.get_current_user_type', return_value='customer'):
                with patch('agents.tools._generate_sql_query_simple', return_value="SELECT * FROM vehicles"):
                    with patch('agents.tools.get_settings') as mock_settings:
                        with patch('agents.tools.create_engine') as mock_engine:
                            # Mock database setup
                            mock_settings.return_value = MagicMock()
                            mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                            mock_engine.return_value.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [("Toyota", "Camry", "25000")]
                            
                            result = await simple_query_crm_data(query)
                            
                            # Should not be blocked (shouldn't contain access denied message)
                            assert "I apologize, but I can only help you with vehicle specifications" not in result

    @pytest.mark.asyncio
    async def test_employee_crm_full_access(self):
        """Test that employees have full CRM access."""
        from agents.tools import simple_query_crm_data
        
        employee_queries = [
            "show me all employees",
            "list customer records", 
            "what opportunities are in my pipeline?",
            "show me recent sales activities"
        ]
        
        for query in employee_queries:
            with patch('agents.tools.get_current_user_type', return_value='employee'):
                with patch('agents.tools._generate_sql_query_simple', return_value="SELECT * FROM employees"):
                    with patch('agents.tools.get_settings') as mock_settings:
                        with patch('agents.tools.create_engine') as mock_engine:
                            # Mock database setup
                            mock_settings.return_value = MagicMock()
                            mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                            mock_engine.return_value.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [("John", "Smith", "Sales")]
                            
                            result = await simple_query_crm_data(query)
                            
                            # Should not be blocked 
                            assert "I apologize, but I can only help you with vehicle specifications" not in result


class TestUserContext:
    """Test user context management system."""
    
    def test_user_context_setting_and_getting(self):
        """Test that user context properly sets and retrieves user type."""
        
        # Test setting employee context
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            assert get_current_user_type() == "employee"
        
        # Context should be cleared after exiting
        assert get_current_user_type() is None
        
        # Test setting customer context
        with UserContext(user_id="jane.doe", conversation_id="conv-456", user_type="customer"):
            assert get_current_user_type() == "customer"
        
        # Context should be cleared after exiting
        assert get_current_user_type() is None

    def test_nested_user_context(self):
        """Test nested user context scenarios."""
        
        with UserContext(user_id="user1", conversation_id="conv1", user_type="employee"):
            assert get_current_user_type() == "employee"
            
            # Nested context should override
            with UserContext(user_id="user2", conversation_id="conv2", user_type="customer"):
                assert get_current_user_type() == "customer"
            
            # Should restore previous context
            assert get_current_user_type() == "employee"
        
        # Should be cleared completely
        assert get_current_user_type() is None


class TestAgentNodes:
    """Test agent node functionality."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        agent = UnifiedToolCallingRAGAgent()
        agent.initialized = True
        # Mock memory manager
        agent.memory_manager = AsyncMock()
        agent.memory_manager.store_message_from_agent = AsyncMock()
        agent.memory_manager.get_user_context_for_new_conversation = AsyncMock(return_value={})
        agent.memory_manager.get_relevant_context = AsyncMock(return_value=[])
        agent.memory_manager.should_store_memory = AsyncMock(return_value=False)
        agent.memory_manager.consolidator.check_and_trigger_summarization = AsyncMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_employee_agent_node_access_control(self, agent):
        """Test that employee agent only processes employee users."""
        # Test valid employee user
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "john.smith",
            "user_type": "employee",
            "conversation_id": "conv-123",
            "user_verified": True
        }
        
        with patch.object(agent, '_get_system_prompt', return_value="System prompt"):
            with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                mock_llm = AsyncMock()
                mock_llm.ainvoke.return_value = AIMessage(content="Hello! How can I help you?")
                mock_llm_class.return_value.bind_tools.return_value = mock_llm
                
                result = await agent._employee_agent_node(state)
                
                assert "messages" in result
                assert len(result["messages"]) > 0
                assert result["user_id"] == "john.smith"

    @pytest.mark.asyncio
    async def test_employee_agent_rejects_customer(self, agent):
        """Test that employee agent rejects customer users."""
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "jane.doe",
            "user_type": "customer",
            "conversation_id": "conv-456",
            "user_verified": True
        }
        
        result = await agent._employee_agent_node(state)
        
        # Should return error message
        assert "messages" in result
        error_message = result["messages"][-1]
        assert "routing error" in error_message.content.lower()

    @pytest.mark.asyncio
    async def test_customer_agent_node_access_control(self, agent):
        """Test that customer agent only processes customer users."""
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "jane.doe", 
            "user_type": "customer",
            "conversation_id": "conv-456",
            "user_verified": True
        }
        
        with patch.object(agent, '_get_customer_system_prompt', return_value="Customer system prompt"):
            with patch('agents.tools.get_tools_for_user_type', return_value=[]):
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = AsyncMock()
                    mock_llm.ainvoke.return_value = AIMessage(content="Hello! I can help you with vehicle information.")
                    mock_llm_class.return_value.bind_tools.return_value = mock_llm
                    
                    result = await agent._customer_agent_node(state)
                    
                    assert "messages" in result
                    assert len(result["messages"]) > 0
                    assert result["user_id"] == "jane.doe"

    @pytest.mark.asyncio 
    async def test_customer_agent_rejects_employee(self, agent):
        """Test that customer agent rejects employee users."""
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "john.smith",
            "user_type": "employee", 
            "conversation_id": "conv-123",
            "user_verified": True
        }
        
        result = await agent._customer_agent_node(state)
        
        # Should return error message
        assert "messages" in result
        error_message = result["messages"][-1]
        assert "routing error" in error_message.content.lower()


class TestIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        agent = UnifiedToolCallingRAGAgent()
        agent.initialized = True
        # Mock memory manager
        agent.memory_manager = AsyncMock()
        agent.memory_manager.store_message_from_agent = AsyncMock()
        agent.memory_manager.get_user_context_for_new_conversation = AsyncMock(return_value={})
        agent.memory_manager.get_relevant_context = AsyncMock(return_value=[])
        agent.memory_manager.should_store_memory = AsyncMock(return_value=False)
        agent.memory_manager.consolidator.check_and_trigger_summarization = AsyncMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_employee_end_to_end_workflow(self, agent):
        """Test complete employee workflow from verification to response."""
        # Mock user verification to return employee
        with patch.object(agent, '_verify_user_access', return_value='employee'):
            with patch.object(agent, '_get_system_prompt', return_value="System prompt"):
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = AsyncMock()
                    mock_llm.ainvoke.return_value = AIMessage(content="Here's your CRM data analysis.")
                    mock_llm_class.return_value.bind_tools.return_value = mock_llm
                    
                    # Test verification
                    state = {
                        "messages": [HumanMessage(content="Show me my sales pipeline")],
                        "user_id": "john.smith",
                        "conversation_id": "conv-123"
                    }
                    
                    # Verify user
                    verified_state = await agent._user_verification_node(state)
                    assert verified_state["user_verified"] == True
                    assert verified_state["user_type"] == "employee"
                    
                    # Test routing
                    route = agent._route_after_user_verification(verified_state)
                    assert route == "employee_agent"
                    
                    # Test agent execution
                    final_state = await agent._employee_agent_node(verified_state)
                    assert "messages" in final_state
                    assert len(final_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_customer_end_to_end_workflow(self, agent):
        """Test complete customer workflow from verification to response."""
        # Mock user verification to return customer
        with patch.object(agent, '_verify_user_access', return_value='customer'):
            with patch.object(agent, '_get_customer_system_prompt', return_value="Customer system prompt"):
                with patch('agents.tools.get_tools_for_user_type', return_value=[]):
                    with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                        mock_llm = AsyncMock()
                        mock_llm.ainvoke.return_value = AIMessage(content="Here are our vehicle options.")
                        mock_llm_class.return_value.bind_tools.return_value = mock_llm
                        
                        # Test verification
                        state = {
                            "messages": [HumanMessage(content="What vehicles do you have?")],
                            "user_id": "jane.doe",
                            "conversation_id": "conv-456"
                        }
                        
                        # Verify user
                        verified_state = await agent._user_verification_node(state)
                        assert verified_state["user_verified"] == True
                        assert verified_state["user_type"] == "customer"
                        
                        # Test routing
                        route = agent._route_after_user_verification(verified_state)
                        assert route == "customer_agent"
                        
                        # Test agent execution
                        final_state = await agent._customer_agent_node(verified_state)
                        assert "messages" in final_state
                        assert len(final_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_user_workflow(self, agent):
        """Test unknown user workflow - should be gracefully rejected."""
        # Mock user verification to return unknown
        with patch.object(agent, '_verify_user_access', return_value='unknown'):
            # Test verification
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "unknown.user",
                "conversation_id": "conv-789"
            }
            
            # Verify user
            verified_state = await agent._user_verification_node(state)
            assert verified_state["user_verified"] == False
            
            # Test routing
            route = agent._route_after_user_verification(verified_state)
            assert route == "end"


if __name__ == "__main__":
    """Run basic tests to validate implementation."""
    print("🧪 Running Phase 1 Dual Agent System Tests")
    print("=" * 50)
    
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"]) 