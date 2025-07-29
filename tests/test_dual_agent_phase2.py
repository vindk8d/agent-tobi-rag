"""
Comprehensive Test Suite for Phase 2: Dual User Agent System - Customer Messaging

This test suite validates all Phase 2 requirements from the PRD:
1. Customer messaging tool functionality and validation
2. Interrupt-based confirmation workflow
3. Message delivery simulation and error handling
4. User experience flows for employee and customer perspectives  
5. Integration with Phase 1 foundation
6. Security and access control for messaging features
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call
from langchain_core.messages import HumanMessage, AIMessage

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from agents.tobi_sales_copilot.state import AgentState
from agents.tools import (
    trigger_customer_message,
    _validate_message_content,
    _lookup_customer,
    _format_message_by_type,
    _list_active_customers,
    UserContext,
    get_current_user_type,
    get_tools_for_user_type
)


class TestCustomerMessageTool:
    """Test customer messaging tool functionality."""
    
    @pytest.mark.asyncio
    async def test_employee_can_access_customer_message_tool(self):
        """Test that employee users can access the customer messaging tool."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Jane",
                        "last_name": "Doe", 
                        "email": "jane.doe@example.com",
                        "phone": "555-0123"
                    }
                    
                    # Mock interrupt to return approval
                    mock_interrupt.return_value = "APPROVE"
                    
                    result = await trigger_customer_message.ainvoke({
                        "customer_id": "CUST001",
                        "message_content": "Thank you for your recent inquiry. We have some updates for you.",
                        "message_type": "follow_up"
                    })
                    
                    # Should succeed and indicate delivery
                    assert "Message Delivered Successfully" in result
                    assert "Jane Doe" in result
                    mock_lookup.assert_called_once_with("CUST001")
                    mock_interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_customer_cannot_access_customer_message_tool(self):
        """Test that customer users cannot access the customer messaging tool."""
        with UserContext(user_id="jane.doe", conversation_id="conv-456", user_type="customer"):
            result = await trigger_customer_message(
                customer_id="CUST001",
                message_content="Test message",
                message_type="follow_up"
            )
            
            # Should be denied with appropriate message
            assert "customer messaging is only available to employees" in result
            assert "administrator" in result

    @pytest.mark.asyncio
    async def test_admin_can_access_customer_message_tool(self):
        """Test that admin users can access the customer messaging tool."""
        with UserContext(user_id="admin.user", conversation_id="conv-admin", user_type="admin"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "John",
                        "last_name": "Smith",
                        "email": "john.smith@example.com",
                        "phone": "555-0456"
                    }
                    
                    # Mock interrupt to return approval
                    mock_interrupt.return_value = "APPROVE"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Administrative notice regarding your account.",
                        message_type="information"
                    )
                    
                    # Should succeed
                    assert "Message Delivered Successfully" in result
                    assert "John Smith" in result

    @pytest.mark.asyncio
    async def test_unknown_user_cannot_access_customer_message_tool(self):
        """Test that unknown users cannot access the customer messaging tool."""
        with UserContext(user_id="unknown.user", conversation_id="conv-unknown", user_type="unknown"):
            result = await trigger_customer_message(
                customer_id="CUST001",
                message_content="Test message",
                message_type="follow_up"
            )
            
            # Should be denied
            assert "customer messaging is only available to employees" in result

    def test_customer_message_tool_in_employee_tool_list(self):
        """Test that customer messaging tool appears in employee tool list."""
        employee_tools = get_tools_for_user_type("employee")
        admin_tools = get_tools_for_user_type("admin")
        customer_tools = get_tools_for_user_type("customer")
        
        # Should be in employee and admin tools
        employee_tool_names = [tool.name for tool in employee_tools]
        admin_tool_names = [tool.name for tool in admin_tools]
        customer_tool_names = [tool.name for tool in customer_tools]
        
        assert "trigger_customer_message" in employee_tool_names
        assert "trigger_customer_message" in admin_tool_names
        assert "trigger_customer_message" not in customer_tool_names

    @pytest.mark.asyncio
    async def test_invalid_customer_id_handling(self):
        """Test handling of invalid customer IDs."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer', return_value=None):
                result = await trigger_customer_message(
                    customer_id="INVALID123",
                    message_content="Test message",
                    message_type="follow_up"
                )
                
                # Should return error message
                assert "Customer with ID 'INVALID123' not found" in result
                assert "verify the customer ID" in result

    @pytest.mark.asyncio
    async def test_empty_customer_id_validation(self):
        """Test validation of empty customer ID."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            result = await trigger_customer_message(
                customer_id="",
                message_content="Test message",
                message_type="follow_up"
            )
            
            # Should return validation error
            assert "Customer ID is required" in result

    @pytest.mark.asyncio
    async def test_invalid_message_type_handling(self):
        """Test handling of invalid message types with fallback."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "email": "jane.doe@example.com",
                        "phone": "555-0123"
                    }
                    
                    # Mock interrupt to return approval
                    mock_interrupt.return_value = "APPROVE"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Test message",
                        message_type="invalid_type"  # Should fallback to "follow_up"
                    )
                    
                    # Should succeed with fallback
                    assert "Message Delivered Successfully" in result
                    # Check interrupt call contains follow_up formatting
                    mock_interrupt.assert_called_once()


class TestMessageValidation:
    """Test message content validation functionality."""
    
    def test_valid_message_content(self):
        """Test validation of valid message content."""
        valid_messages = [
            ("Thank you for your inquiry about our services. We'll follow up soon.", "follow_up"),
            ("Here's the information you requested about our new product line.", "information"),
            ("Special offer: 20% off your next purchase this month only!", "promotional"),
            ("I'm here to help resolve your technical issue. Let me know the details.", "support")
        ]
        
        for message, msg_type in valid_messages:
            result = _validate_message_content(message, msg_type)
            assert result["valid"] == True
            assert len(result["errors"]) == 0
            assert result["character_count"] == len(message)
            assert result["word_count"] > 0

    def test_empty_message_validation(self):
        """Test validation of empty messages."""
        result = _validate_message_content("", "follow_up")
        assert result["valid"] == False
        assert "Message content cannot be empty" in result["errors"]

    def test_message_too_long_validation(self):
        """Test validation of overly long messages."""
        long_message = "x" * 2500  # Exceeds all type limits
        result = _validate_message_content(long_message, "follow_up")
        assert result["valid"] == False
        assert "Message too long" in result["errors"][0]
        assert "1500 characters" in result["errors"][0]

    def test_message_length_by_type(self):
        """Test message length limits by type."""
        test_cases = [
            ("follow_up", 1500),
            ("information", 2000),
            ("promotional", 1200),
            ("support", 2000)
        ]
        
        for msg_type, limit in test_cases:
            # Test message at limit (should pass)
            message_at_limit = "x" * limit
            result = _validate_message_content(message_at_limit, msg_type)
            assert result["valid"] == True
            
            # Test message over limit (should fail)
            message_over_limit = "x" * (limit + 1)
            result = _validate_message_content(message_over_limit, msg_type)
            assert result["valid"] == False
            assert f"Maximum {limit} characters" in result["errors"][0]

    def test_message_quality_warnings(self):
        """Test message quality warnings."""
        test_cases = [
            ("HELLO THIS IS ALL CAPS!!!", "ALL CAPS"),
            ("Very short", "very short"),
            ("Contact me URGENT!!! ASAP!!!", "professional language"),
            ("Check your email at user@company.com", "@ symbol")
        ]
        
        for message, expected_warning in test_cases:
            result = _validate_message_content(message, "follow_up")
            # Should still be valid but have warnings
            assert result["valid"] == True
            warning_text = " ".join(result["warnings"]).lower()
            assert expected_warning.lower() in warning_text

    def test_inappropriate_content_detection(self):
        """Test detection of inappropriate content."""
        inappropriate_messages = [
            "This is damn difficult to explain",
            "What the hell is going on?",
            "This is crap quality"
        ]
        
        for message in inappropriate_messages:
            result = _validate_message_content(message, "follow_up")
            assert result["valid"] == False
            assert "professional language" in result["errors"][0]

    def test_message_type_specific_validation(self):
        """Test message type specific validation suggestions."""
        # Promotional without offer terms
        result = _validate_message_content("Buy our product today", "promotional")
        assert result["valid"] == True
        warning_text = " ".join(result["warnings"])
        assert "special offers" in warning_text or "value propositions" in warning_text

        # Support without questions
        result = _validate_message_content("We can help you", "support")
        assert result["valid"] == True
        warning_text = " ".join(result["warnings"])
        assert "questions" in warning_text or "next steps" in warning_text

        # Follow-up without gratitude
        result = _validate_message_content("Here is the information", "follow_up")
        assert result["valid"] == True
        warning_text = " ".join(result["warnings"])
        assert "gratitude" in warning_text or "previous interaction" in warning_text


class TestCustomerLookup:
    """Test customer lookup functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_customer_lookup(self):
        """Test successful customer lookup."""
        with patch('agents.tools.get_settings') as mock_settings:
            with patch('agents.tools.create_engine') as mock_engine:
                # Mock database setup
                mock_settings.return_value = MagicMock()
                mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                
                # Mock database response
                mock_conn = MagicMock()
                mock_result = MagicMock()
                mock_result.fetchone.return_value = (
                    "CUST001", "John", "Smith", "john@example.com", "555-0123", 
                    "active", "2024-01-01", "2024-01-01"
                )
                mock_conn.execute.return_value = mock_result
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
                
                result = await _lookup_customer("CUST001")
                
                assert result is not None
                assert result["customer_id"] == "CUST001"
                assert result["first_name"] == "John"
                assert result["last_name"] == "Smith"
                assert result["email"] == "john@example.com"
                assert result["phone"] == "555-0123"

    @pytest.mark.asyncio
    async def test_customer_not_found_lookup(self):
        """Test customer lookup when customer doesn't exist."""
        with patch('agents.tools.get_settings') as mock_settings:
            with patch('agents.tools.create_engine') as mock_engine:
                # Mock database setup
                mock_settings.return_value = MagicMock()
                mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                
                # Mock no result found
                mock_conn = MagicMock()
                mock_result = MagicMock()
                mock_result.fetchone.return_value = None
                mock_conn.execute.return_value = mock_result
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
                
                result = await _lookup_customer("NONEXISTENT")
                
                assert result is None

    @pytest.mark.asyncio
    async def test_customer_lookup_database_error(self):
        """Test customer lookup with database error."""
        with patch('agents.tools.get_settings') as mock_settings:
            with patch('agents.tools.create_engine') as mock_engine:
                # Mock database setup
                mock_settings.return_value = MagicMock()
                mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                
                # Mock database error
                mock_engine.side_effect = Exception("Database connection failed")
                
                result = await _lookup_customer("CUST001")
                
                assert result is None

    @pytest.mark.asyncio
    async def test_list_active_customers(self):
        """Test listing active customers."""
        with patch('agents.tools.get_settings') as mock_settings:
            with patch('agents.tools.create_engine') as mock_engine:
                # Mock database setup
                mock_settings.return_value = MagicMock()
                mock_settings.return_value.supabase.postgresql_connection_string = "postgresql://test"
                
                # Mock database response with multiple customers
                mock_conn = MagicMock()
                mock_result = MagicMock()
                mock_result.__iter__ = lambda self: iter([
                    ("CUST001", "John", "Smith", "john@example.com", "555-0123"),
                    ("CUST002", "Jane", "Doe", "jane@example.com", "555-0456")
                ])
                mock_conn.execute.return_value = mock_result
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
                
                result = await _list_active_customers(limit=10)
                
                assert len(result) == 2
                assert result[0]["customer_id"] == "CUST001"
                assert result[0]["display_name"] == "John Smith"
                assert result[1]["customer_id"] == "CUST002"
                assert result[1]["display_name"] == "Jane Doe"


class TestMessageFormatting:
    """Test message formatting functionality."""
    
    def test_follow_up_message_formatting(self):
        """Test formatting of follow-up messages."""
        customer_info = {
            "first_name": "John",
            "last_name": "Smith",
            "email": "john@example.com"
        }
        
        message = "We have updates on your recent inquiry."
        formatted = _format_message_by_type(message, "follow_up", customer_info)
        
        assert "Dear John Smith," in formatted
        assert "following up on our recent interaction" in formatted
        assert message in formatted
        assert "Best regards" in formatted

    def test_information_message_formatting(self):
        """Test formatting of information messages."""
        customer_info = {
            "first_name": "Jane",
            "last_name": "Doe",
            "email": "jane@example.com"
        }
        
        message = "Here are the product specifications you requested."
        formatted = _format_message_by_type(message, "information", customer_info)
        
        assert "Dear Jane Doe," in formatted
        assert "important information" in formatted
        assert message in formatted
        assert "questions about this information" in formatted

    def test_promotional_message_formatting(self):
        """Test formatting of promotional messages."""
        customer_info = {
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice@example.com"
        }
        
        message = "Get 25% off your next purchase this month!"
        formatted = _format_message_by_type(message, "promotional", customer_info)
        
        assert "Dear Alice Johnson," in formatted
        assert "exciting opportunity" in formatted
        assert message in formatted
        assert "limited time" in formatted

    def test_support_message_formatting(self):
        """Test formatting of support messages."""
        customer_info = {
            "first_name": "Bob",
            "last_name": "Wilson",
            "email": "bob@example.com"
        }
        
        message = "I'll help you resolve the login issue you're experiencing."
        formatted = _format_message_by_type(message, "support", customer_info)
        
        assert "Dear Bob Wilson," in formatted
        assert "Thank you for reaching out" in formatted
        assert message in formatted
        assert "additional questions" in formatted

    def test_unknown_message_type_formatting(self):
        """Test formatting with unknown message type (default format)."""
        customer_info = {
            "first_name": "Carol",
            "last_name": "Davis",
            "email": "carol@example.com"
        }
        
        message = "This is a general message."
        formatted = _format_message_by_type(message, "unknown_type", customer_info)
        
        assert "Dear Carol Davis," in formatted
        assert message in formatted
        assert "Best regards" in formatted

    def test_formatting_with_missing_customer_name(self):
        """Test formatting when customer name is missing."""
        customer_info = {
            "first_name": "",
            "last_name": "",
            "email": "unknown@example.com"
        }
        
        message = "Test message content."
        formatted = _format_message_by_type(message, "follow_up", customer_info)
        
        assert "Dear Valued Customer," in formatted
        assert message in formatted


class TestInterruptWorkflow:
    """Test interrupt mechanisms and confirmation workflows."""
    
    @pytest.mark.asyncio
    async def test_interrupt_approval_workflow(self):
        """Test interrupt workflow with approval."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "email": "jane.doe@example.com",
                        "phone": "555-0123"
                    }
                    
                    # Mock interrupt to return approval
                    mock_interrupt.return_value = "APPROVE"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Thank you for your inquiry.",
                        message_type="follow_up"
                    )
                    
                    # Verify interrupt was called with proper confirmation details
                    mock_interrupt.assert_called_once()
                    interrupt_args = mock_interrupt.call_args[0][0]  # Get first argument
                    
                    assert "CUSTOMER MESSAGE CONFIRMATION REQUIRED" in interrupt_args
                    assert "Jane Doe" in interrupt_args
                    assert "jane.doe@example.com" in interrupt_args
                    assert "APPROVE" in interrupt_args
                    assert "CANCEL" in interrupt_args
                    assert "MODIFY" in interrupt_args
                    
                    # Verify successful delivery response
                    assert "Message Delivered Successfully" in result
                    assert "Jane Doe" in result

    @pytest.mark.asyncio
    async def test_interrupt_cancellation_workflow(self):
        """Test interrupt workflow with cancellation."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "John",
                        "last_name": "Smith",
                        "email": "john.smith@example.com",
                        "phone": "555-0456"
                    }
                    
                    # Mock interrupt to return cancellation
                    mock_interrupt.return_value = "CANCEL"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Information about our services.",
                        message_type="information"
                    )
                    
                    # Verify cancellation response
                    assert "Message Cancelled" in result
                    assert "John Smith" in result
                    assert "was not sent" in result

    @pytest.mark.asyncio
    async def test_interrupt_modification_workflow(self):
        """Test interrupt workflow with modification request."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Alice",
                        "last_name": "Brown",
                        "email": "alice.brown@example.com",
                        "phone": "555-0789"
                    }
                    
                    # Mock interrupt to return modification request
                    mock_interrupt.return_value = "MODIFY: Please make the tone more formal"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Quick update on your order.",
                        message_type="follow_up"
                    )
                    
                    # Verify modification response
                    assert "Modification Requested" in result
                    assert "Please make the tone more formal" in result
                    assert "use the customer messaging tool again" in result

    @pytest.mark.asyncio
    async def test_interrupt_invalid_response_handling(self):
        """Test interrupt workflow with invalid response."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Bob",
                        "last_name": "Green",
                        "email": "bob.green@example.com",
                        "phone": "555-0101"
                    }
                    
                    # Mock interrupt to return invalid response
                    mock_interrupt.return_value = "INVALID_RESPONSE"
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Updates on your account.",
                        message_type="information"
                    )
                    
                    # Verify invalid response handling
                    assert "Invalid Response" in result
                    assert "INVALID_RESPONSE" in result
                    assert "APPROVE" in result
                    assert "CANCEL" in result
                    assert "MODIFY" in result
                    assert "was not sent" in result

    @pytest.mark.asyncio
    async def test_interrupt_no_response_handling(self):
        """Test interrupt workflow with no response."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Carol",
                        "last_name": "White",
                        "email": "carol.white@example.com",
                        "phone": "555-0202"
                    }
                    
                    # Mock interrupt to return no response (None or empty)
                    mock_interrupt.return_value = None
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Promotional offer details.",
                        message_type="promotional"
                    )
                    
                    # Verify no response handling
                    assert "Message Cancelled" in result
                    assert "No confirmation response received" in result


class TestMessageDeliverySimulation:
    """Test message delivery simulation and error handling."""
    
    @pytest.mark.asyncio
    async def test_successful_message_delivery(self):
        """Test successful message delivery simulation."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    with patch('random.random', return_value=0.9):  # Force success (< 0.95)
                        # Setup mock customer
                        mock_lookup.return_value = {
                            "customer_id": "CUST001",
                            "first_name": "David",
                            "last_name": "Johnson",
                            "email": "david.johnson@example.com",
                            "phone": "555-0303"
                        }
                        
                        # Mock interrupt to approve
                        mock_interrupt.return_value = "APPROVE"
                        
                        result = await trigger_customer_message(
                            customer_id="CUST001",
                            message_content="Your order has been processed.",
                            message_type="information"
                        )
                        
                        # Verify successful delivery
                        assert "Message Delivered Successfully" in result
                        assert "David Johnson" in result
                        assert "david.johnson@example.com" in result
                        assert "Delivered at:" in result

    @pytest.mark.asyncio
    async def test_failed_message_delivery(self):
        """Test failed message delivery simulation."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    with patch('random.random', return_value=0.98):  # Force failure (> 0.95)
                        # Setup mock customer
                        mock_lookup.return_value = {
                            "customer_id": "CUST001",
                            "first_name": "Emma",
                            "last_name": "Wilson",
                            "email": "emma.wilson@example.com",
                            "phone": "555-0404"
                        }
                        
                        # Mock interrupt to approve
                        mock_interrupt.return_value = "APPROVE"
                        
                        result = await trigger_customer_message(
                            customer_id="CUST001",
                            message_content="Service maintenance notification.",
                            message_type="information"
                        )
                        
                        # Verify failed delivery
                        assert "Message Delivery Failed" in result
                        assert "Emma Wilson" in result
                        assert "delivery service unavailable" in result
                        assert "Failed at:" in result

    @pytest.mark.asyncio
    async def test_delivery_simulation_timing(self):
        """Test delivery simulation includes appropriate delays."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    with patch('asyncio.sleep') as mock_sleep:
                        # Setup mock customer
                        mock_lookup.return_value = {
                            "customer_id": "CUST001",
                            "first_name": "Frank",
                            "last_name": "Brown",
                            "email": "frank.brown@example.com",
                            "phone": "555-0505"
                        }
                        
                        # Mock interrupt to approve
                        mock_interrupt.return_value = "APPROVE"
                        
                        result = await trigger_customer_message(
                            customer_id="CUST001",
                            message_content="Account update information.",
                            message_type="information"
                        )
                        
                        # Verify delivery delay was simulated
                        mock_sleep.assert_called_once_with(1)


class TestPhase2Integration:
    """Test integration of Phase 2 features with Phase 1 foundation."""
    
    @pytest_asyncio.fixture
    async def agent(self):
        """Create agent instance for integration testing."""
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
    async def test_employee_agent_has_access_to_customer_messaging(self, agent):
        """Test that employee agents can access customer messaging tool."""
        # Test employee agent tool access includes customer messaging
        employee_state = {
            "messages": [HumanMessage(content="I need to send a follow-up to customer CUST001")],
            "user_id": "john.smith",
            "user_type": "employee",
            "conversation_id": "conv-123",
            "user_verified": True
        }
        
        with patch.object(agent, '_get_system_prompt', return_value="System prompt"):
            with patch('agents.tools.get_tools_for_user_type') as mock_get_tools:
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    # Mock tools to include customer messaging
                    mock_tools = [MagicMock()]
                    mock_tools[0].name = "trigger_customer_message"
                    mock_get_tools.return_value = mock_tools
                    
                    mock_llm = AsyncMock()
                    mock_llm.ainvoke.return_value = AIMessage(content="I can help you send a customer message.")
                    mock_llm_class.return_value.bind_tools.return_value = mock_llm
                    
                    result = await agent._employee_agent_node(employee_state)
                    
                    # Verify tools were requested for employee type
                    mock_get_tools.assert_called_with("employee")

    @pytest.mark.asyncio
    async def test_customer_agent_does_not_have_messaging_access(self, agent):
        """Test that customer agents do not have access to customer messaging tool."""
        customer_state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "jane.doe",
            "user_type": "customer",
            "conversation_id": "conv-456",
            "user_verified": True
        }
        
        with patch.object(agent, '_get_customer_system_prompt', return_value="Customer system prompt"):
            with patch('agents.tools.get_tools_for_user_type') as mock_get_tools:
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    # Mock tools to exclude customer messaging
                    mock_tools = [MagicMock(), MagicMock()]
                    mock_tools[0].name = "simple_query_crm_data"
                    mock_tools[1].name = "simple_rag"
                    mock_get_tools.return_value = mock_tools
                    
                    mock_llm = AsyncMock()
                    mock_llm.ainvoke.return_value = AIMessage(content="I can help with information.")
                    mock_llm_class.return_value.bind_tools.return_value = mock_llm
                    
                    result = await agent._customer_agent_node(customer_state)
                    
                    # Verify tools were requested for customer type
                    mock_get_tools.assert_called_with("customer")
                    
                    # Verify no customer messaging tool was included
                    tool_names = [tool.name for tool in mock_get_tools.return_value]
                    assert "trigger_customer_message" not in tool_names

    @pytest.mark.asyncio
    async def test_phase1_functionality_preserved_with_phase2(self, agent):
        """Test that Phase 1 functionality is preserved after Phase 2 implementation."""
        # Test that basic routing and user verification still work
        with patch.object(agent, '_verify_user_access', return_value='employee'):
            # User verification
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "john.smith",
                "conversation_id": "conv-123"
            }
            
            verified_state = await agent._user_verification_node(state)
            assert verified_state["user_verified"] == True
            assert verified_state["user_type"] == "employee"
            
            # Routing
            route = agent._route_after_user_verification(verified_state)
            assert route == "employee_agent"

        # Test customer routing still works
        with patch.object(agent, '_verify_user_access', return_value='customer'):
            state = {
                "messages": [HumanMessage(content="Hello")],
                "user_id": "jane.doe",
                "conversation_id": "conv-456"
            }
            
            verified_state = await agent._user_verification_node(state)
            assert verified_state["user_verified"] == True
            assert verified_state["user_type"] == "customer"
            
            route = agent._route_after_user_verification(verified_state)
            assert route == "customer_agent"

    @pytest.mark.asyncio
    async def test_phase1_tools_still_available_with_phase2(self):
        """Test that Phase 1 tools are still available after Phase 2 implementation."""
        # Employee tools
        employee_tools = get_tools_for_user_type("employee")
        employee_tool_names = [tool.name for tool in employee_tools]
        
        # Phase 1 tools should still be there
        assert "simple_query_crm_data" in employee_tool_names
        assert "simple_rag" in employee_tool_names
        # Phase 2 addition
        assert "trigger_customer_message" in employee_tool_names
        
        # Customer tools  
        customer_tools = get_tools_for_user_type("customer")
        customer_tool_names = [tool.name for tool in customer_tools]
        
        # Phase 1 tools should still be there
        assert "simple_query_crm_data" in customer_tool_names
        assert "simple_rag" in customer_tool_names
        # Phase 2 tool should not be available
        assert "trigger_customer_message" not in customer_tool_names


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_customer_message_tool_database_error(self):
        """Test customer messaging tool with database connection error."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                # Mock database error
                mock_lookup.side_effect = Exception("Database connection failed")
                
                result = await trigger_customer_message(
                    customer_id="CUST001",
                    message_content="Test message",
                    message_type="follow_up"
                )
                
                # Should handle error gracefully
                assert "encountered an error" in result
                assert "try again" in result

    @pytest.mark.asyncio  
    async def test_customer_message_tool_interrupt_error(self):
        """Test customer messaging tool with interrupt system error."""
        with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
            with patch('agents.tools._lookup_customer') as mock_lookup:
                with patch('agents.tools.interrupt') as mock_interrupt:
                    # Setup mock customer
                    mock_lookup.return_value = {
                        "customer_id": "CUST001",
                        "first_name": "Test",
                        "last_name": "User",
                        "email": "test@example.com",
                        "phone": "555-0000"
                    }
                    
                    # Mock interrupt error
                    mock_interrupt.side_effect = Exception("Interrupt system failed")
                    
                    result = await trigger_customer_message(
                        customer_id="CUST001",
                        message_content="Test message",
                        message_type="follow_up"
                    )
                    
                    # Should handle error gracefully
                    assert "encountered an error" in result

    @pytest.mark.asyncio
    async def test_message_validation_edge_cases(self):
        """Test message validation with edge cases."""
        edge_cases = [
            ("   ", "follow_up"),  # Whitespace only
            ("a", "follow_up"),    # Single character
            (None, "follow_up")    # None input
        ]
        
        for message, msg_type in edge_cases:
            with UserContext(user_id="john.smith", conversation_id="conv-123", user_type="employee"):
                result = await trigger_customer_message(
                    customer_id="CUST001",
                    message_content=message,
                    message_type=msg_type
                )
                
                # Should handle validation gracefully
                if message is None or not str(message).strip():
                    assert "Message Validation Failed" in result or "encountered an error" in result

    def test_formatting_with_none_customer_info(self):
        """Test message formatting with None customer info."""
        result = _format_message_by_type("Test message", "follow_up", {})
        
        # Should handle missing info gracefully
        assert "Dear Valued Customer," in result
        assert "Test message" in result


if __name__ == "__main__":
    """Run Phase 2 tests to validate customer messaging implementation."""
    print("🧪 Running Phase 2 Dual Agent System Tests - Customer Messaging")
    print("=" * 65)
    
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"]) 