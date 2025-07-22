"""
Comprehensive tests for customer message functionality.
Tests the trigger_customer_message tool, delivery functions, confirmation flows, and error handling.
"""

import pytest
import asyncio
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import the functions we're testing
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from agents.tools import (
    trigger_customer_message,
    _deliver_message_via_chat,
    _track_message_delivery,
    _handle_customer_message_confirmation,
    _lookup_customer,
    _validate_message_content,
    _format_message_by_type,
    UserContext
)


class TestCustomerMessageDelivery:
    """Test message delivery via chat functionality."""

    @pytest.fixture
    def sample_customer_info(self):
        """Sample customer information for testing."""
        return {
            "customer_id": str(uuid.uuid4()),
            "id": str(uuid.uuid4()),
            "name": "John Smith",
            "first_name": "John",
            "last_name": "Smith",
            "email": "john.smith@example.com",
            "phone": "555-1234",
            "company": "Test Company"
        }

    @pytest.fixture
    def sample_confirmation_data(self, sample_customer_info):
        """Sample confirmation data for testing."""
        return {
            "requires_human_confirmation": True,
            "confirmation_type": "customer_message",
            "customer_info": sample_customer_info,
            "message_content": "Thank you for your interest in our vehicles. I wanted to follow up on our recent conversation.",
            "formatted_message": "Dear John Smith,\n\nThank you for your interest in our vehicles. I wanted to follow up on our recent conversation.\n\nBest regards",
            "message_type": "follow_up",
            "customer_id": "john.smith@example.com",
            "sender_employee_id": str(uuid.uuid4()),
            "requested_by": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }

    @pytest.mark.asyncio
    async def test_deliver_message_via_chat_success(self, sample_customer_info):
        """Test successful message delivery via chat."""
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect().__enter__.return_value = mock_conn
            
            # Mock user lookup (customer has active chat account)
            mock_conn.execute.side_effect = [
                MagicMock(fetchone=lambda: (str(uuid.uuid4()),)),  # User found
                MagicMock(fetchone=lambda: None),  # No existing conversation
                None,  # Insert conversation
                None,  # Insert message
                None   # Update conversation
            ]
            
            result = await _deliver_message_via_chat(
                customer_info=sample_customer_info,
                formatted_message="Test message content",
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["success"] is True
            assert "conversation_id" in result
            assert "message_id" in result
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_deliver_message_via_chat_no_user_account(self, sample_customer_info):
        """Test message delivery failure when customer has no active user account."""
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect().__enter__.return_value = mock_conn
            
            # Mock user lookup (no user found)
            mock_conn.execute.side_effect = [
                MagicMock(fetchone=lambda: None),  # No user found
            ]
            
            result = await _deliver_message_via_chat(
                customer_info=sample_customer_info,
                formatted_message="Test message content",
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["success"] is False
            assert result["conversation_id"] is None
            assert "does not have an active chat account" in result["error"]

    @pytest.mark.asyncio
    async def test_deliver_message_via_chat_existing_conversation(self, sample_customer_info):
        """Test message delivery to existing conversation."""
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect().__enter__.return_value = mock_conn
            
            existing_conv_id = str(uuid.uuid4())
            
            # Mock user lookup and existing conversation
            mock_conn.execute.side_effect = [
                MagicMock(fetchone=lambda: (str(uuid.uuid4()),)),  # User found
                MagicMock(fetchone=lambda: (existing_conv_id,)),  # Existing conversation found
                None,  # Insert message
                None   # Update conversation
            ]
            
            result = await _deliver_message_via_chat(
                customer_info=sample_customer_info,
                formatted_message="Test message content",
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["success"] is True
            assert result["conversation_id"] == existing_conv_id


class TestMessageDeliveryTracking:
    """Test message delivery tracking functionality."""

    @pytest.fixture
    def delivery_result_success(self):
        """Sample successful delivery result."""
        return {
            "success": True,
            "conversation_id": str(uuid.uuid4()),
            "message_id": str(uuid.uuid4()),
            "error": None
        }

    @pytest.fixture
    def delivery_result_failure(self):
        """Sample failed delivery result."""
        return {
            "success": False,
            "conversation_id": None,
            "error": "Customer does not have an active chat account"
        }

    @pytest.mark.asyncio
    async def test_track_message_delivery_success(self, delivery_result_success):
        """Test tracking successful message delivery."""
        customer_info = {
            "customer_id": str(uuid.uuid4()),
            "name": "John Smith"
        }
        
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect().__enter__.return_value = mock_conn
            mock_conn.execute.return_value = None
            
            result = await _track_message_delivery(
                customer_info=customer_info,
                delivery_result=delivery_result_success,
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["tracked"] is True
            assert result["delivery_status"] == "success"
            assert "tracking_id" in result

    @pytest.mark.asyncio
    async def test_track_message_delivery_failure(self, delivery_result_failure):
        """Test tracking failed message delivery."""
        customer_info = {
            "customer_id": str(uuid.uuid4()),
            "name": "John Smith"
        }
        
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect().__enter__.return_value = mock_conn
            mock_conn.execute.return_value = None
            
            result = await _track_message_delivery(
                customer_info=customer_info,
                delivery_result=delivery_result_failure,
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["tracked"] is True
            assert result["delivery_status"] == "failed"
            assert "tracking_id" in result


class TestConfirmationFlow:
    """Test confirmation flow handling."""

    @pytest.fixture
    def confirmation_data(self):
        """Sample confirmation data."""
        return {
            "customer_info": {
                "customer_id": str(uuid.uuid4()),
                "name": "John Smith",
                "email": "john.smith@example.com"
            },
            "formatted_message": "Dear John Smith,\n\nTest message content\n\nBest regards",
            "message_type": "follow_up",
            "sender_employee_id": str(uuid.uuid4()),
            "message_content": "Test message content"
        }

    @pytest.mark.asyncio
    async def test_handle_confirmation_approve_success(self, confirmation_data):
        """Test handling approved confirmation with successful delivery."""
        with patch('agents.tools._deliver_message_via_chat') as mock_deliver, \
             patch('agents.tools._track_message_delivery') as mock_track:
            
            mock_deliver.return_value = {
                "success": True,
                "conversation_id": str(uuid.uuid4()),
                "message_id": str(uuid.uuid4()),
                "error": None
            }
            
            mock_track.return_value = {
                "tracking_id": str(uuid.uuid4()),
                "tracked": True,
                "delivery_status": "success"
            }
            
            result = await _handle_customer_message_confirmation(
                confirmation_data, "I approve this message"
            )
            
            assert "Message Sent Successfully" in result
            assert "John Smith" in result
            assert "Follow_Up" in result
            assert "Chat message" in result
            
            # Verify functions were called
            mock_deliver.assert_called_once()
            mock_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_confirmation_approve_delivery_failure(self, confirmation_data):
        """Test handling approved confirmation with delivery failure."""
        with patch('agents.tools._deliver_message_via_chat') as mock_deliver, \
             patch('agents.tools._track_message_delivery') as mock_track:
            
            mock_deliver.return_value = {
                "success": False,
                "conversation_id": None,
                "error": "Customer does not have an active chat account"
            }
            
            mock_track.return_value = {
                "tracking_id": str(uuid.uuid4()),
                "tracked": True,
                "delivery_status": "failed"
            }
            
            result = await _handle_customer_message_confirmation(
                confirmation_data, "I approve this message"
            )
            
            assert "Message Delivery Failed" in result
            assert "does not have an active chat account" in result
            assert "John Smith" in result

    @pytest.mark.asyncio
    async def test_handle_confirmation_deny(self, confirmation_data):
        """Test handling denied confirmation."""
        result = await _handle_customer_message_confirmation(
            confirmation_data, "I deny this request"
        )
        
        assert "Message Cancelled" in result
        assert "John Smith" in result
        assert "was not sent" in result

    @pytest.mark.asyncio
    async def test_handle_confirmation_invalid_response(self, confirmation_data):
        """Test handling invalid/unclear confirmation response."""
        result = await _handle_customer_message_confirmation(
            confirmation_data, "Maybe later"
        )
        
        assert "Message Cancelled" in result
        assert "was not sent" in result


class TestTriggerCustomerMessage:
    """Test the main trigger_customer_message tool."""

    @pytest.mark.asyncio
    async def test_trigger_customer_message_employee_success(self):
        """Test triggering customer message as employee with valid inputs."""
        with patch('agents.tools.get_current_user_type') as mock_user_type, \
             patch('agents.tools.get_current_employee_id') as mock_employee_id, \
             patch('agents.tools._lookup_customer') as mock_lookup, \
             patch('agents.tools._validate_message_content') as mock_validate, \
             patch('agents.tools._format_message_by_type') as mock_format:
            
            mock_user_type.return_value = "employee"
            mock_employee_id.return_value = str(uuid.uuid4())
            
            mock_lookup.return_value = {
                "customer_id": str(uuid.uuid4()),
                "name": "John Smith",
                "email": "john.smith@example.com"
            }
            
            mock_validate.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            mock_format.return_value = "Formatted message content"
            
            result = await trigger_customer_message(
                customer_id="john.smith@example.com",
                message_content="Thank you for your interest in our vehicles.",
                message_type="follow_up"
            )
            
            assert "CUSTOMER_MESSAGE_CONFIRMATION_REQUIRED:" in result
            
            # Parse the JSON data
            prefix = "CUSTOMER_MESSAGE_CONFIRMATION_REQUIRED: "
            data_str = result[len(prefix):]
            confirmation_data = json.loads(data_str)
            
            assert confirmation_data["requires_human_confirmation"] is True
            assert confirmation_data["confirmation_type"] == "customer_message"
            assert confirmation_data["message_type"] == "follow_up"
            assert "John Smith" == confirmation_data["customer_info"]["name"]

    @pytest.mark.asyncio
    async def test_trigger_customer_message_customer_denied(self):
        """Test that customers cannot use the customer messaging tool."""
        with patch('agents.tools.get_current_user_type') as mock_user_type:
            mock_user_type.return_value = "customer"
            
            result = await trigger_customer_message(
                customer_id="john.smith@example.com",
                message_content="Test message",
                message_type="follow_up"
            )
            
            assert "customer messaging is only available to employees" in result

    @pytest.mark.asyncio
    async def test_trigger_customer_message_invalid_inputs(self):
        """Test trigger_customer_message with invalid inputs."""
        with patch('agents.tools.get_current_user_type') as mock_user_type, \
             patch('agents.tools.get_current_employee_id') as mock_employee_id:
            
            mock_user_type.return_value = "employee"
            mock_employee_id.return_value = str(uuid.uuid4())
            
            # Test empty customer_id
            result = await trigger_customer_message(
                customer_id="",
                message_content="Test message",
                message_type="follow_up"
            )
            
            assert "Customer identifier is required" in result

    @pytest.mark.asyncio
    async def test_trigger_customer_message_customer_not_found(self):
        """Test trigger_customer_message with non-existent customer."""
        with patch('agents.tools.get_current_user_type') as mock_user_type, \
             patch('agents.tools.get_current_employee_id') as mock_employee_id, \
             patch('agents.tools._lookup_customer') as mock_lookup, \
             patch('agents.tools._validate_message_content') as mock_validate:
            
            mock_user_type.return_value = "employee"
            mock_employee_id.return_value = str(uuid.uuid4())
            mock_lookup.return_value = None
            
            mock_validate.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            result = await trigger_customer_message(
                customer_id="nonexistent@example.com",
                message_content="Test message",
                message_type="follow_up"
            )
            
            assert "Customer 'nonexistent@example.com' not found" in result

    @pytest.mark.asyncio
    async def test_trigger_customer_message_validation_failure(self):
        """Test trigger_customer_message with message validation failure."""
        with patch('agents.tools.get_current_user_type') as mock_user_type, \
             patch('agents.tools.get_current_employee_id') as mock_employee_id, \
             patch('agents.tools._validate_message_content') as mock_validate:
            
            mock_user_type.return_value = "employee"
            mock_employee_id.return_value = str(uuid.uuid4())
            
            mock_validate.return_value = {
                "valid": False,
                "errors": ["Message too short", "Missing professional greeting"],
                "warnings": ["Consider adding more context"]
            }
            
            result = await trigger_customer_message(
                customer_id="john.smith@example.com",
                message_content="hi",
                message_type="follow_up"
            )
            
            assert "Message Validation Failed" in result
            assert "Message too short" in result
            assert "Missing professional greeting" in result
            assert "Consider adding more context" in result


class TestMessageValidationAndFormatting:
    """Test message validation and formatting functions."""

    def test_validate_message_content_valid(self):
        """Test message validation with valid content."""
        result = _validate_message_content(
            "Thank you for your interest in our vehicles. I wanted to follow up on our recent conversation about the Toyota Camry.",
            "follow_up"
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_message_content_too_short(self):
        """Test message validation with too short content."""
        result = _validate_message_content("Hi", "follow_up")
        
        assert len(result["warnings"]) > 0
        assert any("very short" in warning for warning in result["warnings"])

    def test_validate_message_content_too_long(self):
        """Test message validation with too long content."""
        long_message = "A" * 2000  # Exceeds follow_up limit of 1500
        result = _validate_message_content(long_message, "follow_up")
        
        assert result["valid"] is False
        assert any("too long" in error for error in result["errors"])

    def test_validate_message_content_unprofessional(self):
        """Test message validation with unprofessional content."""
        result = _validate_message_content("URGENT!!! BUY NOW!!!", "promotional")
        
        assert len(result["warnings"]) > 0
        assert any("professional language" in warning for warning in result["warnings"])

    def test_format_message_by_type_follow_up(self):
        """Test message formatting for follow_up type."""
        customer_info = {
            "first_name": "John",
            "last_name": "Smith"
        }
        
        result = _format_message_by_type(
            "Thank you for your interest in our vehicles.",
            "follow_up",
            customer_info
        )
        
        assert result.startswith("Dear John Smith,")
        assert "following up" in result
        assert "Best regards" in result

    def test_format_message_by_type_promotional(self):
        """Test message formatting for promotional type."""
        customer_info = {
            "first_name": "John",
            "last_name": "Smith"
        }
        
        result = _format_message_by_type(
            "We have a special discount on Toyota Camry this month!",
            "promotional",
            customer_info
        )
        
        assert result.startswith("Dear John Smith,")
        assert "exciting opportunity" in result
        assert "limited time" in result
        assert "Best regards" in result


class TestUserContextIntegration:
    """Test integration with user context system."""

    @pytest.mark.asyncio
    async def test_user_context_during_message_delivery(self):
        """Test that user context is properly maintained during message operations."""
        user_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        
        with UserContext(user_id=user_id, conversation_id=conversation_id, user_type="employee"):
            # Test that context is accessible within the context manager
            from agents.tools import get_current_user_id, get_current_user_type, get_current_conversation_id
            
            assert get_current_user_id() == user_id
            assert get_current_user_type() == "employee"
            assert get_current_conversation_id() == conversation_id


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test proper error handling when database operations fail."""
        customer_info = {
            "customer_id": str(uuid.uuid4()),
            "name": "John Smith"
        }
        
        with patch('agents.tools._get_sql_engine') as mock_engine:
            mock_engine.side_effect = Exception("Database connection failed")
            
            result = await _deliver_message_via_chat(
                customer_info=customer_info,
                formatted_message="Test message",
                message_type="follow_up",
                sender_employee_id=str(uuid.uuid4())
            )
            
            assert result["success"] is False
            assert "Database connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_confirmation_handler_error_handling(self):
        """Test error handling in confirmation handler."""
        invalid_confirmation_data = {}  # Empty/invalid data
        
        result = await _handle_customer_message_confirmation(
            invalid_confirmation_data, "approve"
        )
        
        assert "Confirmation Processing Error" in result


if __name__ == "__main__":
    # Run specific test categories
    import pytest
    
    print("Running Customer Message Functionality Tests...")
    pytest.main([__file__, "-v", "--tb=short"]) 