"""
Fast unit tests for vehicle search system - no LLM calls.

This module contains fast-running tests that don't make external API calls
and focus on core functionality validation.
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import os

# Import the functions we're testing
from backend.agents.tools import (
    VehicleSearchResult,
    _execute_vehicle_sql_safely
)


class TestVehicleSearchFast:
    """Fast unit tests for vehicle search functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up test environment and skip if credentials are missing."""
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
            pytest.skip("Supabase credentials not available for testing")
    
    def test_vehicle_search_result_model(self):
        """Test VehicleSearchResult model validation."""
        # Valid data
        valid_data = {
            "sql_query": "SELECT v.id, v.make FROM vehicles v WHERE v.is_available = true",
            "reasoning": "Test reasoning",
            "search_intent": "test_intent",
            "semantic_understanding": {"brand": "Toyota"}
        }
        result = VehicleSearchResult(**valid_data)
        assert result.sql_query == valid_data["sql_query"]
        assert result.reasoning == valid_data["reasoning"]
        assert result.search_intent == valid_data["search_intent"]
        assert result.semantic_understanding == {"brand": "Toyota"}
    
    def test_vehicle_search_result_defaults(self):
        """Test VehicleSearchResult with default values."""
        minimal_data = {
            "sql_query": "SELECT * FROM vehicles",
            "reasoning": "Test",
            "search_intent": "test"
        }
        result = VehicleSearchResult(**minimal_data)
        assert result.semantic_understanding == {}  # Should default to empty dict
    
    def test_vehicle_search_result_validation_errors(self):
        """Test VehicleSearchResult validation errors."""
        # Missing required fields
        with pytest.raises(ValueError):
            VehicleSearchResult()
        
        with pytest.raises(ValueError):
            VehicleSearchResult(sql_query="SELECT * FROM vehicles")
    
    @pytest.mark.asyncio
    async def test_sql_safety_basic(self):
        """Test basic SQL safety validation."""
        # Safe query
        safe_query = """
            SELECT v.id, v.make, v.model 
            FROM vehicles v 
            WHERE v.is_available = true 
            AND v.stock_quantity > 0
        """
        result = await _execute_vehicle_sql_safely(safe_query, "Safe test query")
        assert isinstance(result, list)
        
        # Dangerous query
        dangerous_query = "DROP TABLE vehicles"
        result = await _execute_vehicle_sql_safely(dangerous_query, "Dangerous query")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        injection_queries = [
            "SELECT * FROM vehicles; DROP TABLE vehicles;",
            "SELECT * FROM vehicles WHERE make = 'Toyota'; DELETE FROM pricing;",
            "UPDATE vehicles SET make = 'malicious'",
            "INSERT INTO vehicles VALUES ('test')"
        ]
        
        for query in injection_queries:
            result = await _execute_vehicle_sql_safely(query, "Injection test")
            assert result == [], f"Injection should be blocked: {query[:30]}..."
    
    @pytest.mark.asyncio 
    async def test_required_safety_conditions(self):
        """Test that required safety conditions are enforced."""
        # Query missing required conditions
        unsafe_queries = [
            "SELECT * FROM vehicles WHERE make = 'Toyota'",  # Missing safety conditions
            "SELECT * FROM vehicles WHERE v.stock_quantity > 0",  # Missing is_available
            "SELECT * FROM vehicles WHERE v.is_available = true"   # Missing stock_quantity
        ]
        
        for query in unsafe_queries:
            result = await _execute_vehicle_sql_safely(query, "Safety test")
            assert result == [], f"Unsafe query should be rejected: {query[:30]}..."


class TestErrorHandlingAndFallbackBehavior:
    """Test suite for error handling and fallback behavior (Task 7.1.3)."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up test environment and skip if credentials are missing."""
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
            pytest.skip("Supabase credentials not available for testing")
    
    @pytest.mark.asyncio
    async def test_llm_api_failure_handling(self, sample_vehicle_results):
        """Test LLM API failures and timeout handling."""
        from backend.agents.tools import _search_vehicles_with_llm, _fallback_vehicle_search
        
        # Test various LLM failure scenarios
        with patch('backend.agents.tools.get_settings') as mock_settings, \
             patch('backend.agents.tools._fallback_vehicle_search') as mock_fallback:
            
            # Mock fallback to return sample results
            mock_fallback.return_value = sample_vehicle_results
            
            # Scenario 1: Settings failure
            mock_settings.side_effect = Exception("Settings API unavailable")
            result = await _search_vehicles_with_llm("Toyota Prius")
            assert isinstance(result, list)
            mock_fallback.assert_called_with("Toyota Prius", 20)
            
            # Reset mock
            mock_fallback.reset_mock()
            
            # Scenario 2: LLM initialization failure
            mock_settings.side_effect = None
            mock_settings.return_value = MagicMock(openai_simple_model="invalid-model")
            
            with patch('backend.agents.tools.ChatOpenAI') as mock_llm_class:
                mock_llm_class.side_effect = Exception("LLM API unavailable")
                
                result = await _search_vehicles_with_llm("Honda Civic")
                assert isinstance(result, list)
                mock_fallback.assert_called_with("Honda Civic", 20)
    
    @pytest.mark.asyncio
    async def test_llm_timeout_simulation(self, sample_vehicle_results):
        """Test LLM timeout handling."""
        from backend.agents.tools import _search_vehicles_with_llm
        
        with patch('backend.agents.tools._fallback_vehicle_search') as mock_fallback:
            mock_fallback.return_value = sample_vehicle_results
            
            # Simulate LLM chain timeout
            with patch('backend.agents.tools.get_settings') as mock_settings, \
                 patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
                 patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
                 patch('backend.agents.tools.StrOutputParser') as mock_parser:
                
                mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
                mock_template.from_template.return_value = MagicMock()
                mock_llm_class.return_value = MagicMock()
                mock_parser.return_value = MagicMock()
                
                # Mock chain to raise timeout
                mock_chain = AsyncMock()
                mock_chain.ainvoke.side_effect = asyncio.TimeoutError("LLM request timed out")
                mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                result = await _search_vehicles_with_llm("BMW luxury car")
                assert isinstance(result, list)
                mock_fallback.assert_called_with("BMW luxury car", 20)
    
    @pytest.mark.asyncio
    async def test_database_connection_errors(self):
        """Test database connection errors during SQL execution."""
        from backend.agents.tools import _execute_vehicle_sql_safely
        
        # Test with database connection failure
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            mock_engine.side_effect = Exception("Database connection failed")
            
            safe_query = """
                SELECT v.id, v.make, v.model 
                FROM vehicles v 
                WHERE v.is_available = true 
                AND v.stock_quantity > 0
                LIMIT 10
            """
            
            result = await _execute_vehicle_sql_safely(safe_query, "Test query")
            assert result == []  # Should return empty list on DB error
    
    @pytest.mark.asyncio
    async def test_database_query_execution_errors(self):
        """Test database query execution errors."""
        from backend.agents.tools import _execute_vehicle_sql_safely
        
        # Test with query execution failure
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            mock_connection = MagicMock()
            mock_connection.execute.side_effect = Exception("Query execution failed")
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
            
            safe_query = """
                SELECT v.id, v.make, v.model 
                FROM vehicles v 
                WHERE v.is_available = true 
                AND v.stock_quantity > 0
                LIMIT 10
            """
            
            result = await _execute_vehicle_sql_safely(safe_query, "Test query")
            assert result == []  # Should return empty list on execution error
    
    @pytest.mark.asyncio
    async def test_invalid_sql_generation_recovery(self, sample_vehicle_results):
        """Test invalid SQL generation recovery."""
        from backend.agents.tools import _search_vehicles_with_llm
        
        with patch('backend.agents.tools._fallback_vehicle_search') as mock_fallback:
            mock_fallback.return_value = sample_vehicle_results
            
            # Mock LLM to return invalid JSON
            with patch('backend.agents.tools.get_settings') as mock_settings, \
                 patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
                 patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
                 patch('backend.agents.tools.StrOutputParser') as mock_parser:
                
                mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
                mock_template.from_template.return_value = MagicMock()
                mock_llm_class.return_value = MagicMock()
                mock_parser.return_value = MagicMock()
                
                # Mock chain to return invalid JSON
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = "Invalid JSON response {malformed"
                mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                result = await _search_vehicles_with_llm("Nissan SUV")
                assert isinstance(result, list)
                mock_fallback.assert_called_with("Nissan SUV", 20)
    
    @pytest.mark.asyncio
    async def test_invalid_vehicle_search_result_structure(self, sample_vehicle_results):
        """Test handling of invalid VehicleSearchResult structure."""
        from backend.agents.tools import _search_vehicles_with_llm
        
        with patch('backend.agents.tools._fallback_vehicle_search') as mock_fallback:
            mock_fallback.return_value = sample_vehicle_results
            
            # Mock LLM to return JSON with missing required fields
            with patch('backend.agents.tools.get_settings') as mock_settings, \
                 patch('backend.agents.tools.ChatPromptTemplate') as mock_template, \
                 patch('backend.agents.tools.ChatOpenAI') as mock_llm_class, \
                 patch('backend.agents.tools.StrOutputParser') as mock_parser:
                
                mock_settings.return_value = MagicMock(openai_simple_model="gpt-3.5-turbo")
                mock_template.from_template.return_value = MagicMock()
                mock_llm_class.return_value = MagicMock()
                mock_parser.return_value = MagicMock()
                
                # Mock chain to return JSON missing required fields
                invalid_response = json.dumps({
                    "sql_query": "SELECT * FROM vehicles",
                    # Missing "reasoning" and "search_intent" fields
                })
                
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = invalid_response
                mock_template.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                result = await _search_vehicles_with_llm("Mazda CX-5")
                assert isinstance(result, list)
                mock_fallback.assert_called_with("Mazda CX-5", 20)
    
    @pytest.mark.asyncio
    async def test_empty_result_set_handling(self):
        """Test empty result set handling."""
        from backend.agents.tools import _execute_vehicle_sql_safely
        
        # Test with query that returns no results
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_result.__iter__.return_value = iter([])  # Empty result set
            mock_connection.execute.return_value = mock_result
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
            
            safe_query = """
                SELECT v.id, v.make, v.model 
                FROM vehicles v 
                WHERE v.is_available = true 
                AND v.stock_quantity > 0
                AND v.make = 'NonExistentBrand'
                LIMIT 10
            """
            
            result = await _execute_vehicle_sql_safely(safe_query, "Empty result test")
            assert result == []  # Should return empty list
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_fallback_search_error_handling(self):
        """Test fallback search error handling."""
        from backend.agents.tools import _fallback_vehicle_search
        
        # Test fallback search with database error
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            mock_engine.side_effect = Exception("Database unavailable")
            
            result = await _fallback_vehicle_search("Toyota Camry")
            assert result == []  # Should return empty list on fallback error
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_fallback_search_query_construction_error(self):
        """Test fallback search query construction errors."""
        from backend.agents.tools import _fallback_vehicle_search
        
        # Test with malformed requirements that might break query construction
        malformed_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # This would cause an error in real usage
            "'; DROP TABLE vehicles; --",  # Injection attempt
        ]
        
        for malformed_input in malformed_inputs:
            try:
                # Convert None to empty string to avoid TypeError
                search_input = malformed_input if malformed_input is not None else ""
                result = await _fallback_vehicle_search(search_input)
                assert isinstance(result, list)  # Should handle gracefully
            except Exception as e:
                # If an exception occurs, it should be handled gracefully
                # by the calling function
                assert isinstance(e, (TypeError, AttributeError))
    
    @pytest.mark.asyncio
    async def test_data_serialization_errors(self):
        """Test data serialization errors in result processing."""
        from backend.agents.tools import _execute_vehicle_sql_safely
        
        # Test with data that might cause serialization issues
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            mock_connection = MagicMock()
            
            # Mock a row with problematic data types
            mock_row = MagicMock()
            mock_row._mapping = {
                'id': 'test-uuid',
                'make': 'Toyota',
                'model': 'Prius',
                'problematic_field': object(),  # Non-serializable object
            }
            
            mock_result = MagicMock()
            mock_result.__iter__.return_value = iter([mock_row])
            mock_connection.execute.return_value = mock_result
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
            
            safe_query = """
                SELECT v.id, v.make, v.model 
                FROM vehicles v 
                WHERE v.is_available = true 
                AND v.stock_quantity > 0
                LIMIT 1
            """
            
            result = await _execute_vehicle_sql_safely(safe_query, "Serialization test")
            assert isinstance(result, list)
            # Should handle serialization gracefully
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_cascade(self, sample_vehicle_results):
        """Test comprehensive error cascade - LLM fails, SQL fails, fallback succeeds."""
        from backend.agents.tools import _search_vehicles_with_llm
        
        with patch('backend.agents.tools._fallback_vehicle_search') as mock_fallback:
            mock_fallback.return_value = sample_vehicle_results
            
            # Simulate complete LLM failure
            with patch('backend.agents.tools.get_settings') as mock_settings:
                mock_settings.side_effect = Exception("Complete system failure")
                
                result = await _search_vehicles_with_llm("emergency vehicle search")
                assert isinstance(result, list)
                assert len(result) == 2  # Should get fallback results
                mock_fallback.assert_called_once_with("emergency vehicle search", 20)
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, sample_vehicle_results):
        """Test partial failure recovery scenarios."""
        from backend.agents.tools import _execute_vehicle_sql_safely
        
        # Test that SQL execution handles errors gracefully
        with patch('backend.agents.tools._get_sql_engine') as mock_engine:
            # Simulate engine connection failure
            mock_engine.side_effect = Exception("Engine connection failed")
            
            safe_query = """
                SELECT v.id, v.make, v.model 
                FROM vehicles v 
                WHERE v.is_available = true 
                AND v.stock_quantity > 0
                AND v.make = 'Subaru'
                LIMIT 10
            """
            
            result = await _execute_vehicle_sql_safely(safe_query, "Partial failure test")
            assert isinstance(result, list)
            assert result == []  # Should return empty list on engine failure
            
            # Verify the engine was called
            mock_engine.assert_called_once()


@pytest.fixture
def sample_vehicle_results():
    """Sample vehicle search results."""
    return [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "make": "Toyota",
            "model": "Prius",
            "year": 2023,
            "color": "Silver",
            "mileage": 0,
            "stock_quantity": 3,
            "base_price": 1800000,
            "discounted_price": 1750000
        },
        {
            "id": "123e4567-e89b-12d3-a456-426614174001",
            "make": "Toyota",
            "model": "Prius",
            "year": 2022,
            "color": "White",
            "mileage": 15000,
            "stock_quantity": 2,
            "base_price": 1650000,
            "discounted_price": 1600000
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
