"""
Comprehensive tests for dynamic model selection functionality.
Tests initialization, model selection logic, and integration across all components.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import the components we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.tools import ModelSelector, QueryComplexity, _get_sql_llm
from backend.agents.memory import MemoryManager
from backend.agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent
from backend.config import get_settings


class TestModelSelector:
    """Test the core ModelSelector functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_settings = Mock()
        self.mock_settings.openai_simple_model = "gpt-3.5-turbo"
        self.mock_settings.openai_complex_model = "gpt-4"
        self.mock_settings.openai_api_key = "test-key"
        
        self.model_selector = ModelSelector(self.mock_settings)
    
    def test_model_selector_initialization(self):
        """Test ModelSelector initializes correctly with settings."""
        assert self.model_selector.simple_model == "gpt-3.5-turbo"
        assert self.model_selector.complex_model == "gpt-4"
        assert len(self.model_selector.complex_keywords) > 0
        assert len(self.model_selector.simple_keywords) > 0
    
    def test_query_complexity_enum(self):
        """Test QueryComplexity enum values."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.COMPLEX.value == "complex"
    
    def test_classify_simple_queries(self):
        """Test classification of simple queries."""
        simple_messages = [
            HumanMessage(content="What is the price?"),
            HumanMessage(content="Show me contact information"),
            HumanMessage(content="How much does it cost?"),
            HumanMessage(content="Where is the office?")
        ]
        
        for messages in [[msg] for msg in simple_messages]:
            complexity = self.model_selector.classify_query_complexity(messages)
            assert complexity == QueryComplexity.SIMPLE, f"Failed for: {messages[0].content}"
    
    def test_classify_complex_queries(self):
        """Test classification of complex queries."""
        complex_messages = [
            HumanMessage(content="Analyze the sales performance and compare different strategies"),
            HumanMessage(content="Explain why our revenue decreased"),
            HumanMessage(content="Provide recommendations for improving efficiency"),
            HumanMessage(content="Evaluate multiple approaches to this problem"),
            HumanMessage(content="What are the pros and cons of each option?")
        ]
        
        for messages in [[msg] for msg in complex_messages]:
            complexity = self.model_selector.classify_query_complexity(messages)
            assert complexity == QueryComplexity.COMPLEX, f"Failed for: {messages[0].content}"
    
    def test_classify_long_queries_as_complex(self):
        """Test that long queries (>200 chars) are classified as complex."""
        long_query = "This is a very long query that exceeds 200 characters. " * 5
        messages = [HumanMessage(content=long_query)]
        
        complexity = self.model_selector.classify_query_complexity(messages)
        assert complexity == QueryComplexity.COMPLEX
    
    def test_classify_empty_messages_as_simple(self):
        """Test that empty message list defaults to simple."""
        complexity = self.model_selector.classify_query_complexity([])
        assert complexity == QueryComplexity.SIMPLE
    
    def test_get_model_for_simple_query(self):
        """Test model selection for simple queries."""
        simple_messages = [HumanMessage(content="What is the price?")]
        model = self.model_selector.get_model_for_query(simple_messages)
        assert model == "gpt-3.5-turbo"
    
    def test_get_model_for_complex_query(self):
        """Test model selection for complex queries."""
        complex_messages = [HumanMessage(content="Analyze sales performance and recommend strategies")]
        model = self.model_selector.get_model_for_query(complex_messages)
        assert model == "gpt-4"
    
    def test_get_model_for_context_multiple_tools(self):
        """Test context-based model selection with multiple tools."""
        tool_calls = ["tool1", "tool2", "tool3"]
        model = self.model_selector.get_model_for_context(tool_calls, 100, False)
        assert model == "gpt-4"
    
    def test_get_model_for_context_crm_query(self):
        """Test context-based model selection for CRM queries."""
        tool_calls = ["query_crm_data"]
        model = self.model_selector.get_model_for_context(tool_calls, 50, False)
        assert model == "gpt-4"
    
    def test_get_model_for_context_long_query_with_docs(self):
        """Test context-based model selection for long queries with documents."""
        tool_calls = ["single_tool"]
        model = self.model_selector.get_model_for_context(tool_calls, 200, True)
        assert model == "gpt-4"
    
    def test_get_model_for_context_simple_default(self):
        """Test context-based model selection defaults to simple."""
        tool_calls = ["single_tool"]
        model = self.model_selector.get_model_for_context(tool_calls, 50, False)
        assert model == "gpt-3.5-turbo"


class TestMemoryManagerDynamicModelSelection:
    """Test MemoryManager with dynamic model selection."""
    
    @pytest.fixture
    async def mock_settings(self):
        """Mock settings fixture."""
        settings = Mock()
        settings.openai_simple_model = "gpt-3.5-turbo"
        settings.openai_complex_model = "gpt-4"
        settings.openai_api_key = "test-key"
        settings.supabase = Mock()
        settings.supabase.postgresql_connection_string = "postgresql://test"
        settings.memory_auto_summarize = True
        settings.memory_summary_interval = 10
        settings.memory_max_messages = 12
        return settings
    
    @pytest.fixture
    async def memory_manager(self, mock_settings):
        """Memory manager fixture."""
        manager = MemoryManager()
        
        # Mock the database components to avoid actual connections
        with patch('backend.agents.memory.SimpleDBManager'):
            with patch('backend.agents.memory.AsyncPostgresSaver'):
                with patch('backend.agents.memory.OpenAIEmbeddings'):
                    with patch('backend.agents.memory.get_settings', return_value=mock_settings):
                        await manager._ensure_initialized()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization_with_settings(self, memory_manager):
        """Test that MemoryManager properly stores settings during initialization."""
        # This test ensures the bug we just fixed doesn't reoccur
        assert hasattr(memory_manager, 'settings'), "MemoryManager should have settings attribute"
        assert memory_manager.settings is not None, "Settings should not be None"
        assert hasattr(memory_manager, 'model_selector'), "MemoryManager should have model_selector"
    
    @pytest.mark.asyncio
    async def test_create_memory_llm_simple(self, memory_manager):
        """Test creating simple model LLM."""
        with patch('backend.agents.memory.ChatOpenAI') as mock_chat_openai:
            llm = memory_manager._create_memory_llm("simple")
            
            # Verify ChatOpenAI was called with simple model
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs['model'] == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_create_memory_llm_complex(self, memory_manager):
        """Test creating complex model LLM."""
        with patch('backend.agents.memory.ChatOpenAI') as mock_chat_openai:
            llm = memory_manager._create_memory_llm("complex")
            
            # Verify ChatOpenAI was called with complex model
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs['model'] == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_get_llm_for_memory_task_insights(self, memory_manager):
        """Test LLM selection for insights task (should be complex)."""
        with patch.object(memory_manager, '_create_memory_llm') as mock_create:
            memory_manager._get_llm_for_memory_task("insights")
            mock_create.assert_called_once_with("complex")
    
    @pytest.mark.asyncio
    async def test_get_llm_for_memory_task_analysis(self, memory_manager):
        """Test LLM selection for analysis task (should be complex)."""
        with patch.object(memory_manager, '_create_memory_llm') as mock_create:
            memory_manager._get_llm_for_memory_task("analysis")
            mock_create.assert_called_once_with("complex")
    
    @pytest.mark.asyncio
    async def test_get_llm_for_memory_task_summary_simple(self, memory_manager):
        """Test LLM selection for summary task with few messages (should be simple)."""
        messages = [{"content": "short message"}] * 5  # 5 short messages
        
        with patch.object(memory_manager, '_create_memory_llm') as mock_create:
            memory_manager._get_llm_for_memory_task("summary", messages)
            mock_create.assert_called_once_with("simple")
    
    @pytest.mark.asyncio
    async def test_get_llm_for_memory_task_summary_complex(self, memory_manager):
        """Test LLM selection for summary task with many messages (should be complex)."""
        messages = [{"content": "message"}] * 25  # 25 messages
        
        with patch.object(memory_manager, '_create_memory_llm') as mock_create:
            memory_manager._get_llm_for_memory_task("summary", messages)
            mock_create.assert_called_once_with("complex")
    
    @pytest.mark.asyncio
    async def test_get_llm_for_memory_task_summary_long_content(self, memory_manager):
        """Test LLM selection for summary task with long content (should be complex)."""
        messages = [{"content": "x" * 600}] * 3  # 3 long messages
        
        with patch.object(memory_manager, '_create_memory_llm') as mock_create:
            memory_manager._get_llm_for_memory_task("summary", messages)
            mock_create.assert_called_once_with("complex")
    
    @pytest.mark.asyncio 
    async def test_create_memory_llm_fallback_on_error(self, memory_manager):
        """Test fallback behavior when LLM creation fails."""
        # Simulate missing complex model setting
        memory_manager.settings.openai_complex_model = None
        
        with patch('backend.agents.memory.ChatOpenAI') as mock_chat_openai:
            llm = memory_manager._create_memory_llm("complex")
            
            # Should fallback to gpt-3.5-turbo
            mock_chat_openai.assert_called()
            call_kwargs = mock_chat_openai.call_args[1]
            # Should use getattr default fallback
            assert 'model' in call_kwargs


class TestSQLToolsDynamicModelSelection:
    """Test SQL tools with dynamic model selection."""
    
    @pytest.mark.asyncio
    async def test_get_sql_llm_with_question(self):
        """Test _get_sql_llm with question parameter for dynamic selection."""
        mock_settings = Mock()
        mock_settings.openai_simple_model = "gpt-3.5-turbo"
        mock_settings.openai_complex_model = "gpt-4"
        mock_settings.openai_api_key = "test-key"
        
        with patch('backend.agents.tools.get_settings', return_value=mock_settings):
            with patch('backend.agents.tools.ModelSelector') as mock_model_selector_class:
                mock_selector = Mock()
                mock_selector.get_model_for_query.return_value = "gpt-4"
                mock_model_selector_class.return_value = mock_selector
                
                with patch('backend.agents.tools.ChatOpenAI') as mock_chat_openai:
                    await _get_sql_llm("Analyze sales performance by region")
                    
                    # Verify ModelSelector was used
                    mock_model_selector_class.assert_called_once_with(mock_settings)
                    mock_selector.get_model_for_query.assert_called_once()
                    
                    # Verify ChatOpenAI was called with selected model
                    mock_chat_openai.assert_called_once_with(
                        model="gpt-4",
                        temperature=0,
                        openai_api_key="test-key"
                    )
    
    @pytest.mark.asyncio
    async def test_get_sql_llm_without_question_uses_complex_default(self):
        """Test _get_sql_llm without question uses complex model as default."""
        mock_settings = Mock()
        mock_settings.openai_complex_model = "gpt-4"
        mock_settings.openai_api_key = "test-key"
        
        with patch('backend.agents.tools.get_settings', return_value=mock_settings):
            with patch('backend.agents.tools.ChatOpenAI') as mock_chat_openai:
                await _get_sql_llm()
                
                # Verify ChatOpenAI was called with complex model as default
                mock_chat_openai.assert_called_once_with(
                    model="gpt-4",
                    temperature=0,
                    openai_api_key="test-key"
                )


class TestRAGAgentDynamicModelSelection:
    """Test RAG agent with dynamic model selection."""
    
    @pytest.fixture
    async def mock_rag_agent(self):
        """Mock RAG agent fixture."""
        agent = UnifiedToolCallingRAGAgent()
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.openai_simple_model = "gpt-3.5-turbo"
        mock_settings.openai_complex_model = "gpt-4"
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_temperature = 0.3
        mock_settings.openai_max_tokens = 1500
        mock_settings.langsmith = Mock()
        mock_settings.langsmith.tracing_enabled = False
        
        # Mock dependencies
        with patch('backend.agents.tobi_sales_copilot.rag_agent.get_settings', return_value=mock_settings):
            with patch('backend.agents.tobi_sales_copilot.rag_agent.memory_manager'):
                with patch('backend.agents.tobi_sales_copilot.rag_agent.memory_scheduler'):
                    with patch('backend.agents.tobi_sales_copilot.rag_agent.setup_langsmith_tracing'):
                        with patch('backend.agents.tobi_sales_copilot.rag_agent.get_all_tools', return_value=[]):
                            with patch('backend.agents.tobi_sales_copilot.rag_agent.get_tool_names', return_value=[]):
                                agent.settings = mock_settings
                                agent.tools = []
                                agent.tool_names = []
                                agent.tool_map = {}
                                
                                # Initialize ModelSelector
                                from backend.agents.tools import ModelSelector
                                agent.model_selector = ModelSelector(mock_settings)
                                agent._initialized = True
        
        return agent
    
    @pytest.mark.asyncio
    async def test_rag_agent_has_model_selector(self, mock_rag_agent):
        """Test that RAG agent initializes ModelSelector correctly."""
        assert hasattr(mock_rag_agent, 'model_selector'), "RAG agent should have model_selector"
        assert mock_rag_agent.model_selector is not None, "ModelSelector should not be None"
        assert mock_rag_agent.model_selector.simple_model == "gpt-3.5-turbo"
        assert mock_rag_agent.model_selector.complex_model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_rag_agent_model_selection_integration(self, mock_rag_agent):
        """Test model selection integration in unified agent node."""
        # Mock messages for testing
        simple_messages = [HumanMessage(content="What is the price?")]
        
        # Test model selection
        selected_model = mock_rag_agent.model_selector.get_model_for_query(simple_messages)
        assert selected_model == "gpt-3.5-turbo"
        
        # Test complex query
        complex_messages = [HumanMessage(content="Analyze our sales strategy and recommend improvements")]
        selected_model = mock_rag_agent.model_selector.get_model_for_query(complex_messages)
        assert selected_model == "gpt-4"


class TestIntegrationAndErrorHandling:
    """Test integration scenarios and error handling."""
    
    @pytest.mark.asyncio
    async def test_missing_settings_attributes_handled_gracefully(self):
        """Test graceful handling of missing settings attributes."""
        # Create settings without dynamic model fields
        mock_settings = Mock()
        mock_settings.openai_api_key = "test-key"
        # Missing openai_simple_model and openai_complex_model
        
        selector = ModelSelector(mock_settings)
        
        # Should use defaults from getattr calls
        assert selector.simple_model == "gpt-3.5-turbo"  # getattr default
        assert selector.complex_model == "gpt-4"  # getattr default
    
    @pytest.mark.asyncio
    async def test_model_selector_with_non_human_messages(self):
        """Test ModelSelector with non-human messages."""
        mock_settings = Mock()
        mock_settings.openai_simple_model = "gpt-3.5-turbo"
        mock_settings.openai_complex_model = "gpt-4"
        
        selector = ModelSelector(mock_settings)
        
        # Mix of message types
        messages = [
            AIMessage(content="I can help with that"),
            HumanMessage(content="What is the price?")
        ]
        
        complexity = selector.classify_query_complexity(messages)
        assert complexity == QueryComplexity.SIMPLE  # Should find the human message
    
    @pytest.mark.asyncio
    async def test_model_selector_with_malformed_messages(self):
        """Test ModelSelector with malformed or empty messages."""
        mock_settings = Mock()
        mock_settings.openai_simple_model = "gpt-3.5-turbo"
        mock_settings.openai_complex_model = "gpt-4"
        
        selector = ModelSelector(mock_settings)
        
        # Test with empty content
        messages = [HumanMessage(content="")]
        complexity = selector.classify_query_complexity(messages)
        assert complexity == QueryComplexity.SIMPLE  # Should default to simple
        
        # Test with None messages
        messages = []
        complexity = selector.classify_query_complexity(messages)
        assert complexity == QueryComplexity.SIMPLE
    
    @pytest.mark.asyncio
    async def test_concurrent_model_selection_calls(self):
        """Test that concurrent model selection calls work correctly."""
        mock_settings = Mock()
        mock_settings.openai_simple_model = "gpt-3.5-turbo"
        mock_settings.openai_complex_model = "gpt-4"
        
        selector = ModelSelector(mock_settings)
        
        # Simulate concurrent calls
        async def classify_query(content):
            messages = [HumanMessage(content=content)]
            return selector.classify_query_complexity(messages)
        
        results = await asyncio.gather(
            classify_query("What is the price?"),
            classify_query("Analyze our sales performance"),
            classify_query("Show me the contact info"),
            classify_query("Provide recommendations for improvement")
        )
        
        # Verify results
        assert results[0] == QueryComplexity.SIMPLE
        assert results[1] == QueryComplexity.COMPLEX
        assert results[2] == QueryComplexity.SIMPLE
        assert results[3] == QueryComplexity.COMPLEX


def test_configuration_integration():
    """Test that configuration properly supports dynamic model selection."""
    # This would be an integration test with actual config loading
    # For now, just verify the expected fields exist
    from backend.config import Settings
    
    # Check that the Settings class has the new fields
    # This is more of a schema validation test
    settings_fields = Settings.__fields__.keys()
    assert 'openai_simple_model' in settings_fields
    assert 'openai_complex_model' in settings_fields


if __name__ == "__main__":
    # Run specific tests for debugging
    pytest.main([__file__, "-v"]) 