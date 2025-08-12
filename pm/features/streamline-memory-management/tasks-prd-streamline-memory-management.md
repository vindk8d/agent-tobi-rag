# Tasks: Streamline Memory Management System

Based on the PRD for streamlining the memory management system to align with LangGraph best practices.

## Relevant Files

- `backend/agents/background_tasks.py` - Core background task management system for non-blocking operations
- `backend/agents/background_tasks.test.py` - Unit tests for background task manager
- `backend/agents/background_message_store.py` - Handles message persistence to database in background
- `backend/agents/background_message_store.test.py` - Unit tests for background message storage
- `backend/agents/background_summary_manager.py` - Manages conversation summarization in background
- `backend/agents/background_summary_manager.test.py` - Unit tests for background summary manager
- `backend/agents/tobi_sales_copilot/agent.py` - Updated agent nodes with simplified memory management
- `backend/agents/tobi_sales_copilot/state.py` - Simplified AgentState following LangGraph best practices
- `backend/agents/memory.py` - Updated memory manager with background processing integration
- `backend/agents/memory.test.py` - Unit tests for updated memory management
- `tests/test_streamlined_memory_integration.py` - End-to-end integration tests for the streamlined system
- `tests/test_performance_benchmarks.py` - Performance testing to verify improvements
- `tests/test_data_persistence.py` - Tests to ensure data persistence requirements are met

### Notes

- Follow LangGraph best practices for state management and persistence
- Use AsyncPostgresSaver as the primary persistence mechanism
- Implement comprehensive error handling and retry logic for background tasks
- Maintain 100% backward compatibility with existing APIs
- Use `pytest backend/agents/` to run unit tests for agent components
- Use `pytest tests/test_streamlined_memory_integration.py` to run integration tests

## Tasks

- [ ] 1.0 Implement Background Task Infrastructure
  - [ ] 1.1 Create BackgroundTaskManager class with asyncio queue and worker processes
  - [ ] 1.2 Implement task scheduling, prioritization, and retry logic (max 3 retries)
  - [ ] 1.3 Add comprehensive logging and error handling for task execution
  - [ ] 1.4 Create BackgroundMessageStore for database message persistence
  - [ ] 1.5 Implement role mapping (human→user, ai→assistant) and metadata preservation
  - [ ] 1.6 Create BackgroundSummaryManager for conversation summarization
  - [ ] 1.7 Implement threshold-based summary generation and LangGraph state updates
  - [ ] 1.8 Add background task monitoring and health checks
  - [ ] 1.9 Write comprehensive unit tests for all background components
  - [ ] 1.10 Test background processing with 1000+ concurrent tasks

- [ ] 2.0 Create Simplified Agent Nodes with Context Management
  - [ ] 2.1 Update _employee_agent_node to handle context management internally
  - [ ] 2.2 Implement in-node message count limits (12 for employees, 15 for customers)
  - [ ] 2.3 Add conversation summary generation when message limits exceeded
  - [ ] 2.4 Implement lazy user context loading within agent nodes
  - [ ] 2.5 Update _customer_agent_node with customer-specific context management
  - [ ] 2.6 Add background task scheduling calls to agent nodes (non-blocking)
  - [ ] 2.7 Remove all memory prep and storage node dependencies from agent nodes
  - [ ] 2.8 Implement system prompt enhancement with loaded context
  - [ ] 2.9 Add performance monitoring and response time tracking
  - [ ] 2.10 Test agent nodes achieve 60-80% response time improvement

- [ ] 3.0 Simplify AgentState and Implement Lazy Context Loading
  - [ ] 3.1 Update AgentState to remove retrieved_docs, sources, long_term_context
  - [ ] 3.2 Ensure conversation_summary field follows LangGraph patterns
  - [ ] 3.3 Maintain essential fields: messages, conversation_id, user_id, customer_id, employee_id
  - [ ] 3.4 Preserve HITL fields: hitl_phase, hitl_prompt, hitl_context
  - [ ] 3.5 Implement lazy context loading methods in memory manager
  - [ ] 3.6 Add context caching for frequently accessed user data
  - [ ] 3.7 Update state serialization/deserialization for LangGraph checkpointer
  - [ ] 3.8 Verify state size reduction of 70% achieved
  - [ ] 3.9 Test state compatibility with existing HITL functionality
  - [ ] 3.10 Validate LangGraph checkpointer integration works correctly

- [ ] 4.0 Update Graph Structure and Remove Memory Nodes
  - [ ] 4.1 Remove ea_memory_prep and ca_memory_prep nodes from graph
  - [ ] 4.2 Remove ea_memory_store and ca_memory_store nodes from graph
  - [ ] 4.3 Update graph routing to go directly from user_verification to agent nodes
  - [ ] 4.4 Implement direct routing from agent nodes to END (no memory storage)
  - [ ] 4.5 Update _route_to_agent method to handle simplified routing
  - [ ] 4.6 Remove all memory node routing logic and conditionals
  - [ ] 4.7 Update graph creation in _create_graph method
  - [ ] 4.8 Test all user workflows (employee and customer) work without memory nodes
  - [ ] 4.9 Verify HITL routing and functionality remains unchanged
  - [ ] 4.10 Validate graph simplification reduces complexity by 50%

- [ ] 5.0 Validate Data Persistence and Performance
  - [ ] 5.1 Create comprehensive integration tests for message storage
  - [ ] 5.2 Test conversation summary generation and storage to database
  - [ ] 5.3 Verify all existing API endpoints return correct data unchanged
  - [ ] 5.4 Test frontend compatibility with message and summary queries
  - [ ] 5.5 Implement performance benchmarking for response times
  - [ ] 5.6 Validate 60-80% response time improvement achieved
  - [ ] 5.7 Test concurrent conversation handling capacity increase
  - [ ] 5.8 Verify background task reliability with comprehensive retry testing
  - [ ] 5.9 Test data integrity under failure scenarios (database errors, task failures)
  - [ ] 5.10 Create rollback procedures and test migration safety
  - [ ] 5.11 Document performance metrics and system behavior changes
  - [ ] 5.12 Conduct end-to-end user journey testing for both employee and customer workflows
