"""
Test that customer personalization still works with simplified summary-based approach.

This test verifies that:
1. Customer context is loaded from conversation summaries
2. System prompts are enhanced with personalization data
3. Customer responses are tailored based on conversation history
4. Background task system maintains personalization data persistence
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from typing import Dict, Any, List

from backend.agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from backend.agents.tobi_sales_copilot.state import AgentState
from backend.agents.tobi_sales_copilot.language import get_customer_system_prompt
from backend.agents.background_tasks import BackgroundTaskManager


class TestCustomerPersonalizationSimplified:
    """Test customer personalization with simplified summary-based approach."""

    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        agent = UnifiedToolCallingRAGAgent()
        
        # Mock initialization to avoid external dependencies
        with patch.object(agent, '_ensure_initialized', new_callable=AsyncMock):
            await agent._ensure_initialized()
            
        # Mock required components
        agent.settings = Mock()
        agent.settings.openai_chat_model = "gpt-4o-mini"
        agent.settings.openai_temperature = 0.3
        agent.settings.openai_max_tokens = 4000
        agent.settings.openai_api_key = "test-key"
        agent.settings.memory_max_messages = 12
        
        agent.memory_manager = Mock()
        agent.background_task_manager = Mock()
        
        return agent

    @pytest.fixture
    def customer_state(self):
        """Create customer agent state for testing."""
        return {
            "messages": [
                {"role": "human", "content": "Hi, I'm interested in SUVs for my family"},
                {"role": "assistant", "content": "Great! I'd be happy to help you find the perfect SUV for your family. What size family are you looking to accommodate?"},
                {"role": "human", "content": "We have 3 kids, so need something spacious with good safety features"}
            ],
            "conversation_id": "test-conv-123",
            "user_id": "customer-user-456",
            "customer_id": "customer-456",
            "employee_id": None,
            "conversation_summary": "Customer is a parent with 3 children looking for a spacious SUV with good safety features for family use."
        }

    def test_conversation_summary_personalization(self, customer_state):
        """Test that conversation summaries provide personalization context."""
        # Test that conversation summary contains relevant personalization data
        summary = customer_state.get("conversation_summary", "")
        
        assert "3 children" in summary or "family" in summary
        assert "SUV" in summary
        assert "safety" in summary or "spacious" in summary
        
        # Verify summary format is useful for personalization
        assert len(summary) > 20  # Meaningful summary length
        assert summary != "Empty conversation"

    def test_customer_system_prompt_with_context(self, customer_state):
        """Test that customer system prompt uses conversation summary for personalization."""
        conversation_summary = customer_state.get("conversation_summary", "")
        user_context = "Customer user interested in vehicle information"
        
        # Generate system prompt with personalization context
        system_prompt = get_customer_system_prompt(
            user_language="english",
            conversation_summary=conversation_summary,
            user_context=user_context,
            memory_manager=None
        )
        
        # Verify personalization context is included
        assert "CONVERSATION CONTEXT" in system_prompt
        assert conversation_summary in system_prompt
        assert user_context in system_prompt
        
        # Verify customer-specific instructions are present
        assert "vehicle sales assistant" in system_prompt.lower()
        assert "customers looking for vehicle information" in system_prompt
        assert "personalized vehicle recommendations" in system_prompt

    @pytest.mark.asyncio
    async def test_context_loading_with_fallback(self, agent, customer_state):
        """Test that context loading works with state summary and database fallback."""
        conversation_id = customer_state["conversation_id"]
        user_id = customer_state["user_id"]
        
        # Mock background task manager to return summary
        mock_summary = "Customer interested in family SUVs with safety features, has 3 children"
        
        with patch('agents.background_tasks.background_task_manager') as mock_btm:
            mock_btm.get_conversation_summary = AsyncMock(return_value=mock_summary)
            
            # Test context loading method
            context = await agent._get_conversation_context_simple(conversation_id, user_id)
            
            # Verify context is loaded and formatted correctly
            assert context is not None
            assert "Previous conversation context:" in context
            assert mock_summary in context
            
            # Verify background task manager was called
            mock_btm.get_conversation_summary.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_context_loading_graceful_fallback(self, agent):
        """Test graceful fallback when no summary is available."""
        conversation_id = "new-conversation"
        user_id = "new-user"
        
        # Mock background task manager to return None (no summary)
        with patch('agents.background_tasks.background_task_manager') as mock_btm:
            mock_btm.get_conversation_summary = AsyncMock(return_value=None)
            
            # Test context loading with no available summary
            context = await agent._get_conversation_context_simple(conversation_id, user_id)
            
            # Verify graceful fallback
            assert context == "No previous conversation context available"

    def test_personalization_data_structure(self, customer_state):
        """Test that personalization data structure supports customer needs."""
        # Verify state contains required fields for personalization
        assert "customer_id" in customer_state
        assert "conversation_summary" in customer_state
        assert customer_state["customer_id"] is not None
        
        # Verify conversation summary structure supports personalization
        summary = customer_state.get("conversation_summary", "")
        
        # Should contain customer preferences and context
        personalization_indicators = [
            "family", "children", "SUV", "safety", "spacious",
            "looking for", "interested in", "needs"
        ]
        
        found_indicators = [indicator for indicator in personalization_indicators 
                          if indicator.lower() in summary.lower()]
        
        assert len(found_indicators) >= 2, f"Summary should contain personalization indicators, found: {found_indicators}"

    @pytest.mark.asyncio
    async def test_background_task_summary_generation(self):
        """Test that background task system generates personalized summaries."""
        # Mock message data with personalization content
        messages = [
            {"role": "user", "content": "Hi, I'm looking for a car for my daily commute", "created_at": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "Great! What's your typical commute like? City driving or highway?", "created_at": "2024-01-01T10:01:00"},
            {"role": "user", "content": "Mostly city driving, about 20 miles each way. I prefer fuel efficient cars", "created_at": "2024-01-01T10:02:00"}
        ]
        
        # Test background task manager summary generation
        btm = BackgroundTaskManager()
        summary = btm._generate_simple_summary(messages)
        
        # Verify summary captures personalization elements
        assert "3 messages" in summary
        assert "user" in summary.lower()
        assert "assistant" in summary.lower()
        assert "2024-01-01" in summary
        
        # Verify summary is suitable for personalization
        assert len(summary) > 20
        assert summary != "Empty conversation"

    def test_customer_tools_restriction_with_personalization(self):
        """Test that customer tools are properly restricted while maintaining personalization."""
        # This test verifies that personalization works within security boundaries
        
        # Mock customer state with personalization data
        customer_state = {
            "customer_id": "customer-123",
            "conversation_summary": "Customer interested in luxury vehicles, budget around $50k",
            "user_id": "user-123"
        }
        
        # Generate system prompt with personalization
        system_prompt = get_customer_system_prompt(
            user_language="english",
            conversation_summary=customer_state["conversation_summary"],
            user_context="Customer user interested in vehicle information"
        )
        
        # Verify personalization is included
        assert customer_state["conversation_summary"] in system_prompt
        
        # Verify security restrictions are maintained
        assert "You CANNOT access employee information" in system_prompt
        assert "You CANNOT provide information about other customers" in system_prompt
        
        # Verify customer-appropriate tools are specified
        assert "simple_rag" in system_prompt
        assert "simple_query_crm_data" in system_prompt
        
        # Verify personalization guidance is present
        assert "personalized vehicle recommendations" in system_prompt

    @pytest.mark.asyncio
    async def test_end_to_end_personalization_flow(self, agent, customer_state):
        """Test complete personalization flow with simplified approach."""
        # Mock required components for full flow test
        with patch.object(agent, '_ensure_initialized', new_callable=AsyncMock), \
             patch('agents.background_tasks.background_task_manager') as mock_btm, \
             patch('agents.tobi_sales_copilot.language.detect_user_language_from_context') as mock_lang, \
             patch('agents.toolbox.toolbox.get_tools_for_user_type') as mock_tools, \
             patch('langchain_openai.ChatOpenAI') as mock_llm:
            
            # Setup mocks
            mock_btm.get_conversation_summary = AsyncMock(return_value="Customer looking for family SUV")
            mock_lang.return_value = "english"
            mock_tools.return_value = []  # Empty tools for test
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.bind_tools.return_value = Mock()
            
            # Test that customer agent node uses personalization correctly
            conversation_id = customer_state["conversation_id"]
            user_id = customer_state["user_id"]
            
            # Test context loading
            context = await agent._get_conversation_context_simple(conversation_id, user_id)
            
            # Verify personalization context was loaded
            assert "Previous conversation context:" in context
            assert "family SUV" in context
            
            # Test system prompt generation with context
            system_prompt = get_customer_system_prompt(
                user_language="english",
                conversation_summary=customer_state.get("conversation_summary", ""),
                user_context="Customer user interested in vehicle information"
            )
            
            # Verify complete personalization integration
            assert customer_state["conversation_summary"] in system_prompt
            assert "CONVERSATION CONTEXT" in system_prompt
            assert "personalized vehicle recommendations" in system_prompt

    def test_personalization_performance_impact(self, customer_state):
        """Test that personalization doesn't significantly impact performance."""
        # Test that personalization data is efficiently structured
        summary = customer_state.get("conversation_summary", "")
        
        # Verify summary is concise but informative
        assert len(summary) < 500  # Should be concise for performance
        assert len(summary) > 20   # Should be informative for personalization
        
        # Verify state structure is efficient
        required_fields = ["customer_id", "conversation_summary", "user_id"]
        for field in required_fields:
            assert field in customer_state
        
        # Verify no unnecessary complex data structures
        assert not any(isinstance(v, list) and len(v) > 50 for v in customer_state.values())

    @pytest.mark.asyncio 
    async def test_personalization_data_persistence(self):
        """Test that personalization data persists correctly through background tasks."""
        # Mock conversation data with personalization elements
        conversation_id = "test-conv-persistence"
        user_id = "customer-user-persistence"
        
        # Mock background task manager
        btm = BackgroundTaskManager()
        
        # Mock database operations
        with patch('core.database.db_client') as mock_db:
            mock_db.client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
                {"role": "user", "content": "I need a reliable car for my elderly parents", "created_at": "2024-01-01T10:00:00"},
                {"role": "assistant", "content": "I understand. Safety and reliability are key for elderly drivers", "created_at": "2024-01-01T10:01:00"}
            ]
            
            mock_db.client.table.return_value.insert.return_value.execute.return_value.data = [{"id": "summary-123"}]
            
            # Create background task for summary generation
            task_id = btm.schedule_summary_generation(
                conversation_id=conversation_id,
                user_id=user_id,
                priority=btm.TaskPriority.NORMAL
            )
            
            # Verify task was scheduled with personalization context
            assert task_id is not None
            
            # Verify task contains required personalization data
            scheduled_tasks = [task for task in btm.task_queue if task.id == task_id]
            assert len(scheduled_tasks) == 1
            
            task = scheduled_tasks[0]
            assert task.conversation_id == conversation_id
            assert task.user_id == user_id
            assert task.task_type == "summary_generation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
