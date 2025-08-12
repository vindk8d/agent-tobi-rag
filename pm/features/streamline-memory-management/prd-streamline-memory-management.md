# PRD: Streamline Memory Management System

## Introduction/Overview

The current memory management system uses a complex dual-layer architecture with dedicated memory preparation and storage nodes that block user responses while performing database operations. This PRD outlines the streamlining of the memory management system to align with LangGraph best practices, eliminate blocking operations, and simplify the codebase while maintaining all data persistence requirements.

**Problem Statement**: The current system has multiple inefficiencies:
- Memory prep nodes add 200-500ms overhead per request
- Complex dual-layer memory architecture (short-term + long-term)
- Context added as SystemMessages bloats state size
- Synchronous memory operations block user responses
- Deviation from LangGraph recommended patterns

**Goal**: Implement a simplified memory management system that follows LangGraph best practices, improves response times by 60-80%, reduces codebase complexity by 50%, while maintaining all data persistence and functionality.

## Goals

1. **Performance**: Reduce agent response times from 500-800ms to 200-300ms by eliminating blocking memory operations
2. **Simplification**: Remove complex memory prep and storage nodes, reducing memory management code by 50%
3. **LangGraph Alignment**: Adopt LangGraph's native persistence patterns using checkpointer + conversation_summary
4. **Data Integrity**: Maintain 100% data persistence for messages, summaries, and user context via background processing
5. **Maintainability**: Simplify codebase to standard LangGraph patterns for easier debugging and maintenance
6. **Backward Compatibility**: Ensure frontend APIs and data access remain unchanged

## User Stories

### As a User (Customer/Employee)
- I want to receive faster responses from the agent so that my conversations feel more natural and responsive
- I want my conversation history to be preserved so that the agent remembers our previous interactions
- I want conversation summaries to be available so that I can review past interactions

### As a Frontend Developer
- I want the existing API endpoints to continue working without changes so that the frontend doesn't break
- I want message and summary data to be available in the same database tables so that queries remain unchanged
- I want conversation data to be reliably stored so that users can access their history

### As a Backend Developer
- I want simpler memory management code so that it's easier to debug and maintain
- I want to follow LangGraph best practices so that the system is more reliable and performant
- I want background processing for non-critical operations so that user responses aren't blocked

## Functional Requirements

### Core Architecture Changes

1. **The system must eliminate dedicated memory preparation nodes** (`ea_memory_prep`, `ca_memory_prep`)
2. **The system must eliminate dedicated memory storage nodes** (`ea_memory_store`, `ca_memory_store`)
3. **The system must use LangGraph's native checkpointer for state persistence**
4. **The system must implement background task processing for database operations**
5. **The system must use the conversation_summary field in AgentState instead of SystemMessage context**

### Context Window Management

6. **The system must implement context management within agent nodes** (not separate prep nodes)
7. **The system must apply message count limits** (default: 12 messages for employees, 15 for customers)
8. **The system must generate conversation summaries when message limits are exceeded**
9. **The system must store summaries in the conversation_summary field of AgentState**
10. **The system must preserve recent messages while summarizing older ones**

### Background Processing

11. **The system must implement a BackgroundTaskManager for non-blocking operations**
12. **The system must store messages to the database via background tasks**
13. **The system must generate and store conversation summaries via background tasks**
14. **The system must extract conversation insights via background tasks**
15. **The system must include retry logic for failed background tasks**

### Data Persistence

16. **The system must continue storing all messages in the messages table**
17. **The system must continue storing summaries in the conversation_summaries table**
18. **The system must maintain conversation metadata in the conversations table**
19. **The system must preserve message role mapping** (human→user, ai→assistant)
20. **The system must maintain all metadata and timestamps**

### State Management

21. **The system must simplify AgentState to essential fields only**
22. **The system must remove retrieved_docs, sources, and long_term_context from state**
23. **The system must load context lazily in agent nodes when needed**
24. **The system must use add_messages annotation for automatic message handling**

### API Compatibility

25. **The system must maintain all existing API endpoints unchanged**
26. **The system must preserve frontend data access patterns**
27. **The system must maintain response formats for message and summary queries**

## Non-Goals (Out of Scope)

1. **Changing database schema**: Existing tables (messages, conversations, conversation_summaries) remain unchanged
2. **Modifying frontend code**: All frontend components and queries work without changes
3. **Altering HITL functionality**: HITL system remains unchanged with 3-field architecture
4. **Changing user-facing features**: All user functionality remains identical
5. **Modifying tool calling patterns**: Agent tools continue to work as before
6. **Changing authentication/authorization**: User verification and access control unchanged

## Technical Considerations

### LangGraph Integration
- Use AsyncPostgresSaver as primary persistence mechanism
- Leverage LangGraph's built-in message handling with add_messages annotation
- Implement conversation_summary field following LangGraph patterns
- Use LangGraph's native context management recommendations

### Background Processing Architecture
- Implement asyncio-based task queue for background operations
- Use execution-scoped connection management for database operations
- Include comprehensive error handling and retry logic
- Maintain background task monitoring and logging

### Performance Optimizations
- Eliminate blocking database operations from request path
- Reduce state object size by removing unnecessary fields
- Implement lazy loading for context and user data
- Use efficient token counting with tiktoken

### Migration Strategy
- Implement dual-track approach during transition
- Maintain backward compatibility throughout migration
- Include comprehensive testing at each phase
- Provide rollback capability if needed

## Success Metrics

### Performance Metrics
- **Response Time**: Reduce average agent response time by 60-80% (target: 200-300ms)
- **Memory Usage**: Reduce state object size by 70%
- **Throughput**: Increase concurrent conversation handling capacity by 50%

### Code Quality Metrics
- **Code Reduction**: Reduce memory management code by 50%
- **Complexity**: Eliminate 4 complex nodes (2 prep + 2 storage nodes)
- **Maintainability**: Achieve 100% LangGraph best practice compliance

### Reliability Metrics
- **Data Integrity**: Maintain 100% message and summary persistence
- **API Compatibility**: Zero breaking changes to existing endpoints
- **Error Rate**: Maintain or improve current error rates

### User Experience Metrics
- **Conversation Flow**: Maintain smooth conversation experience
- **Feature Parity**: Preserve 100% of existing functionality
- **Frontend Performance**: No degradation in frontend load times

## Implementation Phases

### Phase 1: Background Task Infrastructure (2 days)
- Implement BackgroundTaskManager
- Create BackgroundMessageStore
- Create BackgroundSummaryManager
- Add comprehensive testing

### Phase 2: Simplified Agent Nodes (2 days)
- Update agent nodes with context management
- Remove memory prep/storage calls
- Add background task scheduling
- Test response times and functionality

### Phase 3: State Simplification (1 day)
- Update AgentState structure
- Remove unnecessary fields
- Implement lazy context loading
- Update state serialization

### Phase 4: Graph Structure Update (1 day)
- Remove memory prep and storage nodes
- Update graph routing
- Simplify workflow paths
- Test end-to-end functionality

### Phase 5: Validation and Cleanup (1 day)
- Verify data persistence
- Test frontend integration
- Performance benchmarking
- Code cleanup and documentation

## Open Questions

1. **Background Task Monitoring**: Should we implement a dashboard for monitoring background task health and performance?

2. **Summary Generation Timing**: Should summary generation be triggered by message count, time intervals, or both?

3. **Error Handling Strategy**: How should the system handle cases where background tasks consistently fail?

4. **Migration Rollback**: What specific rollback mechanisms should be in place during the migration?

5. **Performance Monitoring**: What specific metrics should be tracked to ensure the simplified system maintains performance gains?

6. **Context Loading Strategy**: Should user context be cached in memory or loaded fresh for each conversation?

## Dependencies

- LangGraph framework and AsyncPostgresSaver
- Existing database schema (messages, conversations, conversation_summaries)
- Current agent tooling and HITL system
- Frontend API contracts and data access patterns
- Supabase client and connection management

## Risk Mitigation

- **Data Loss Risk**: Comprehensive background task retry logic and monitoring
- **Performance Regression**: Extensive performance testing at each phase
- **Frontend Breaking Changes**: Maintain API compatibility throughout migration
- **Migration Complexity**: Phased approach with rollback capabilities at each step
