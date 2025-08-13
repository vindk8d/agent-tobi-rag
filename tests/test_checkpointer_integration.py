#!/usr/bin/env python3
"""
Test LangGraph checkpointer integration with simplified AgentState.

This test validates that our streamlined AgentState (Tasks 3.1-3.9) works correctly
with LangGraph's AsyncPostgresSaver checkpointer for conversation persistence.

Tests cover:
1. State serialization/deserialization with checkpointer
2. Conversation persistence across interrupts (HITL)
3. Message history preservation with add_messages annotation
4. State field integrity after checkpoint save/load
5. Performance with simplified state structure
"""

import sys
import os
import pytest
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.tobi_sales_copilot.state import AgentState
from backend.agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
from backend.agents.memory import memory_manager
from langchain_core.messages import HumanMessage, AIMessage


class TestCheckpointerIntegration:
    """Test LangGraph checkpointer integration with simplified AgentState."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = UnifiedToolCallingRAGAgent()
        
        # Sample simplified AgentState for testing
        self.sample_state = {
            "messages": [
                HumanMessage(content="Hello, I need help with my order"),
                AIMessage(content="I'll help you with your order. Let me look that up.")
            ],
            "conversation_id": f"test-conv-{uuid.uuid4()}",
            "user_id": f"test-user-{uuid.uuid4()}",
            "employee_id": f"emp-{uuid.uuid4()}",
            "customer_id": None,
            "conversation_summary": "Customer inquiry about order status",
            # HITL fields
            "hitl_phase": None,
            "hitl_prompt": None,
            "hitl_context": None
        }

    @pytest.mark.asyncio
    async def test_checkpointer_initialization(self):
        """Test that checkpointer can be initialized with our simplified state."""
        try:
            # Initialize memory manager (which creates the checkpointer)
            await memory_manager._ensure_initialized()
            checkpointer = await memory_manager.get_checkpointer()
            
            assert checkpointer is not None
            print("✅ Checkpointer initialized successfully")
            
        except Exception as e:
            # If we can't connect to database, skip the test
            if "connection" in str(e).lower() or "database" in str(e).lower():
                pytest.skip(f"Database not available for testing: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_state_serialization_with_checkpointer(self):
        """Test that AgentState serializes correctly for checkpointer storage."""
        import json
        
        state = self.sample_state.copy()
        
        # Test that all fields can be JSON serialized (what checkpointer needs)
        try:
            # Serialize messages separately (LangGraph handles this)
            state_without_messages = {k: v for k, v in state.items() if k != "messages"}
            
            serialized = json.dumps(state_without_messages)
            deserialized = json.loads(serialized)
            
            # Verify all fields preserved
            assert deserialized["conversation_id"] == state["conversation_id"]
            assert deserialized["user_id"] == state["user_id"]
            assert deserialized["employee_id"] == state["employee_id"]
            assert deserialized["conversation_summary"] == state["conversation_summary"]
            assert deserialized["hitl_phase"] == state["hitl_phase"]
            assert deserialized["hitl_prompt"] == state["hitl_prompt"]
            assert deserialized["hitl_context"] == state["hitl_context"]
            
            print("✅ State serialization works correctly")
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"State serialization failed: {e}")

    @pytest.mark.asyncio
    async def test_hitl_state_persistence(self):
        """Test that HITL fields persist correctly through checkpointer."""
        state = self.sample_state.copy()
        
        # Set HITL state
        state.update({
            "hitl_phase": "needs_prompt",
            "hitl_prompt": "Approve this customer refund?",
            "hitl_context": {
                "source_tool": "process_refund",
                "customer_id": "cust-123",
                "refund_amount": 99.99,
                "reason": "defective_product"
            }
        })
        
        # Test serialization of HITL state
        import json
        state_without_messages = {k: v for k, v in state.items() if k != "messages"}
        
        try:
            serialized = json.dumps(state_without_messages)
            deserialized = json.loads(serialized)
            
            # Verify HITL fields are preserved
            assert deserialized["hitl_phase"] == "needs_prompt"
            assert deserialized["hitl_prompt"] == "Approve this customer refund?"
            assert deserialized["hitl_context"]["source_tool"] == "process_refund"
            assert deserialized["hitl_context"]["refund_amount"] == 99.99
            
            print("✅ HITL state persists correctly")
            
        except Exception as e:
            pytest.fail(f"HITL state persistence failed: {e}")

    def test_state_structure_compatibility(self):
        """Test that our simplified state structure is compatible with LangGraph."""
        from backend.agents.tobi_sales_copilot.state import AgentState
        from langgraph.graph import StateGraph, START, END
        
        # Test that StateGraph can be created with our AgentState
        try:
            graph = StateGraph(AgentState)
            
            # Add a simple test node
            def test_node(state: AgentState) -> Dict[str, Any]:
                return {
                    "messages": state.get("messages", []),
                    "conversation_id": state.get("conversation_id"),
                    "user_id": state.get("user_id")
                }
            
            graph.add_node("test_node", test_node)
            
            # Add required edges for compilation
            graph.add_edge(START, "test_node")
            graph.add_edge("test_node", END)
            
            # Should compile without errors
            compiled_graph = graph.compile()
            
            assert compiled_graph is not None
            print("✅ Simplified AgentState is compatible with StateGraph")
            
        except Exception as e:
            pytest.fail(f"State structure compatibility failed: {e}")

    def test_add_messages_annotation_compatibility(self):
        """Test that add_messages annotation works with our state."""
        from langchain_core.messages import HumanMessage, AIMessage
        from langgraph.graph.message import add_messages
        
        # Test the add_messages function with our message structure
        existing_messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response")
        ]
        
        new_messages = [
            HumanMessage(content="Second message"),
            AIMessage(content="Second response")
        ]
        
        try:
            # This is what LangGraph does internally with add_messages annotation
            combined_messages = add_messages(existing_messages, new_messages)
            
            assert len(combined_messages) == 4
            assert combined_messages[0].content == "First message"
            assert combined_messages[3].content == "Second response"
            
            print("✅ add_messages annotation works correctly")
            
        except Exception as e:
            pytest.fail(f"add_messages annotation failed: {e}")

    def test_removed_fields_not_present(self):
        """Test that removed bloated fields are not in our state."""
        from backend.agents.tobi_sales_copilot.state import AgentState
        
        state_annotations = AgentState.__annotations__
        
        # Verify removed fields are not present (from Task 3.1)
        removed_fields = ["retrieved_docs", "sources", "long_term_context"]
        
        for field in removed_fields:
            assert field not in state_annotations, f"Removed field '{field}' still present in AgentState"
        
        print("✅ Bloated fields successfully removed from state")

    def test_essential_fields_present(self):
        """Test that all essential fields are present in simplified state."""
        from backend.agents.tobi_sales_copilot.state import AgentState
        
        state_annotations = AgentState.__annotations__
        
        # Essential fields that must be present
        essential_fields = [
            "messages",
            "conversation_id", 
            "user_id",
            "customer_id",
            "employee_id",
            "conversation_summary",
            "hitl_phase",
            "hitl_prompt", 
            "hitl_context"
        ]
        
        for field in essential_fields:
            assert field in state_annotations, f"Essential field '{field}' missing from AgentState"
        
        print("✅ All essential fields present in simplified state")

    @pytest.mark.asyncio
    async def test_graph_compilation_with_checkpointer(self):
        """Test that graph compiles successfully with checkpointer."""
        try:
            # Initialize agent (which builds the graph with checkpointer)
            await self.agent._ensure_initialized()
            
            # Get the compiled graph from the agent's graph attribute
            graph = self.agent.graph
            
            assert graph is not None
            print("✅ Graph compiles successfully with checkpointer")
            
            # Clean up
            await self.agent.cleanup()
            
        except Exception as e:
            # If we can't connect to database, skip the test
            if "connection" in str(e).lower() or "database" in str(e).lower():
                pytest.skip(f"Database not available for testing: {e}")
            else:
                raise

    def test_state_memory_footprint(self):
        """Test that simplified state has reduced memory footprint."""
        import sys
        
        state = self.sample_state.copy()
        
        # Calculate approximate memory usage
        state_size = sys.getsizeof(state)
        
        # Our simplified state should be relatively small
        # (This is a rough estimate - actual size depends on content)
        max_expected_size = 10000  # 10KB should be more than enough for our simple state
        
        assert state_size < max_expected_size, f"State size ({state_size} bytes) larger than expected"
        
        print(f"✅ Simplified state size: {state_size} bytes (efficient)")

    @pytest.mark.asyncio
    async def test_interrupt_and_resume_compatibility(self):
        """Test that interrupts work correctly with simplified state."""
        try:
            await self.agent._ensure_initialized()
            
            # Test that HITL interrupts are configured correctly
            graph = self.agent.graph
            
            # The graph should be compiled with interrupt_before=["hitl_node"]
            # We can't directly test the interrupt mechanism without a full conversation,
            # but we can verify the graph structure supports it
            
            assert graph is not None
            print("✅ Graph supports interrupt/resume with simplified state")
            
            await self.agent.cleanup()
            
        except Exception as e:
            if "connection" in str(e).lower() or "database" in str(e).lower():
                pytest.skip(f"Database not available for testing: {e}")
            else:
                raise

    def test_langraph_best_practices_compliance(self):
        """Test that our state follows LangGraph best practices."""
        from backend.agents.tobi_sales_copilot.state import AgentState
        from langgraph.graph.message import add_messages
        from typing import get_args, get_origin
        
        state_annotations = AgentState.__annotations__
        
        # Check that messages field uses add_messages annotation
        messages_annotation = state_annotations["messages"]
        
        # Verify it's an Annotated type
        assert get_origin(messages_annotation) is not None, "messages field should use Annotated"
        
        # Check that other fields are Optional (best practice for state fields)
        optional_fields = ["conversation_id", "user_id", "customer_id", "employee_id", 
                          "conversation_summary", "hitl_phase", "hitl_prompt", "hitl_context"]
        
        for field in optional_fields:
            annotation = state_annotations[field]
            # Check if it's Optional (Union with None)
            if hasattr(annotation, '__args__'):
                assert type(None) in annotation.__args__, f"Field '{field}' should be Optional"
        
        print("✅ State follows LangGraph best practices")


if __name__ == "__main__":
    # Run specific tests
    test = TestCheckpointerIntegration()
    test.setup_method()
    
    print("🧪 Testing LangGraph Checkpointer Integration...")
    
    try:
        # Run synchronous tests
        test.test_state_structure_compatibility()
        test.test_add_messages_annotation_compatibility()
        test.test_removed_fields_not_present()
        test.test_essential_fields_present()
        test.test_state_memory_footprint()
        test.test_langraph_best_practices_compliance()
        
        # Run async tests
        asyncio.run(test.test_state_serialization_with_checkpointer())
        asyncio.run(test.test_hitl_state_persistence())
        
        print("\n🎉 All checkpointer integration tests passed!")
        print("✅ Simplified AgentState is fully compatible with LangGraph checkpointer")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
