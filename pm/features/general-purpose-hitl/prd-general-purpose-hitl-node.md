# Product Requirements Document: General-Purpose HITL Node

## Introduction/Overview

The General-Purpose Human-in-the-Loop (HITL) Node is a unified architecture component that standardizes all human interactions within the LangGraph agent system. Currently, different tools implement their own custom HITL mechanisms, leading to inconsistent user experiences, code duplication, and agent workflow errors when information is missing or unclear.

This feature creates a single, reusable node that handles all types of human interactions - confirmations, selections, input requests, and multi-step inputs - providing a consistent interface for both tool developers and end users.

**Problem Statement**: Agent workflows frequently fail or provide poor user experiences when information is missing, ambiguous, or requires human confirmation. Users become frustrated when they cannot make the agent perform needed actions due to unclear information requirements.

## Goals

1. **Minimize Agent Errors**: Reduce agent workflow failures caused by missing or ambiguous information by providing clear, structured ways to gather needed data
2. **Standardize HITL Interactions**: Create a unified interface for all human interactions across the system
3. **Improve Developer Experience**: Enable developers to easily add HITL capabilities to any tool without building custom confirmation flows
4. **Enhance User Experience**: Provide consistent, clear prompts that help users understand exactly what information the agent needs
5. **Reduce Code Duplication**: Eliminate the need for multiple HITL implementations across different tools

## User Stories

### Primary User Stories

1. **As an employee**, I want to get confirmation before sending customer messages so that I don't send incorrect communications
2. **As an employee**, I want to provide missing customer information when the system can't find a customer so that I can complete my task without frustration
3. **As a developer**, I want to easily add HITL interactions to any tool so that I don't have to build custom confirmation flows and can focus on tool logic

### Secondary User Stories

4. **As an employee**, I want clear prompts about what information is needed so that I can quickly provide the right details
5. **As an employee**, I want to select from provided options when multiple choices are available so that I can disambiguate my request efficiently
6. **As a developer**, I want to migrate existing tools to use the standard HITL approach so that the system has consistent behavior

## Functional Requirements

### Core HITL Interaction Types

1. **The system must support confirmation interactions** that allow users to approve or deny actions with clear yes/no responses
2. **The system must support selection interactions** that present numbered options for users to choose from
3. **The system must support input request interactions** that gather free-form text input from users
4. **The system must support multi-step input interactions** that can collect multiple pieces of information sequentially

### Tool Integration Requirements

5. **Tools must be able to trigger HITL interactions** by returning standardized response formats (`HITL_REQUIRED:{type}:{data}`)
6. **The system must parse tool responses** and automatically route to the HITL node when HITL interactions are requested
7. **The system must preserve tool context** during HITL interactions so that processing can continue after human input is received

### User Experience Requirements

8. **The system must provide clear, formatted prompts** for each interaction type with specific instructions on how to respond
9. **The system must handle invalid user responses** by re-prompting with clarification rather than failing
10. **The system must allow users to cancel operations** at any HITL interaction point
11. **The system must maintain conversation context** throughout multi-step interactions

### Developer Experience Requirements

12. **The system must provide a standardized HITLRequest class** with helper methods for creating each interaction type
13. **Tools must be able to specify custom prompts and validation rules** for their specific use cases
14. **The system must maintain tool-specific context** that gets passed back to tools after HITL completion

### LangGraph Integration Requirements

15. **The HITL node must integrate seamlessly** with existing LangGraph interrupt mechanisms
16. **The system must use a single HITL node** that handles both prompting and response processing
17. **The system must maintain state consistency** across interrupt/resume cycles
18. **The graph routing must automatically direct** HITL-required responses to the HITL node

## Non-Goals (Out of Scope)

1. **Advanced input validation** (e.g., email format checking, data type validation) - tools should handle their own validation
2. **Timeout handling** for unresponsive users - this will be handled at the infrastructure level
3. **Complex multi-turn conversations** - focus on simple, direct information gathering
4. **Backward compatibility** with existing `confirmation_data` approach - all tools will be migrated
5. **User interface enhancements** beyond text-based interactions - this is focused on the backend logic
6. **User authentication or permission checking** - this is handled by existing user context systems

## Design Considerations

### State Management
- Use simplified `AgentState` with single `hitl_data` field containing all HITL state
- `hitl_data` structure includes type, prompt, options, context, and internal tracking
- Atomic state operations: HITL active when `hitl_data` exists, normal flow when `None`
- Self-contained interactions with all necessary data embedded in single field

### Response Format Standardization
- Tools return `HITL_REQUIRED:{interaction_type}:{json_data}` format
- Agent parses this and creates single `hitl_data` field with structure:
  ```
  hitl_data = {
    "type": "confirmation",
    "prompt": "User prompt text",
    "options": {...},
    "awaiting_response": False/True,
    "context": {"source_tool": ..., "original_args": ...}
  }
  ```
- Consistent field naming and atomic state management across all interaction types

### Error Handling
- Invalid responses trigger re-prompts rather than errors
- Clear error messages guide users toward correct response format
- Graceful degradation for malformed HITL requests

## Technical Considerations

### Dependencies
- Requires LangGraph with interrupt functionality
- Depends on existing AgentState and tool infrastructure
- Uses JSON serialization for context preservation

### Migration Requirements
- All existing tools using custom HITL must be migrated to new format
- Existing `trigger_customer_message` tool serves as primary migration example
- `confirmation_data` field can be deprecated after migration

### Performance Considerations
- Single HITL node reduces graph complexity
- State serialization must handle datetime objects and complex data types
- Interrupt/resume cycles should be efficient

## Success Metrics

### Primary Success Metrics
1. **Reduction in Agent Logic Errors**: Measure decrease in workflow failures due to missing information
2. **User Task Completion Rate**: Track percentage of user requests that complete successfully
3. **Developer Adoption**: Measure how quickly new tools adopt the HITL standard

### Secondary Success Metrics
4. **Code Reduction**: Measure lines of code eliminated through HITL standardization
5. **User Satisfaction**: Track user feedback on clarity of information requests
6. **Response Time**: Measure time for users to provide requested information

## Open Questions

1. **Should we implement a fallback mechanism** if the HITL node fails or becomes unresponsive?
2. **How should we handle concurrent HITL requests** in multi-user scenarios?
3. **Should there be logging/analytics** for HITL interaction patterns to improve prompts over time?
4. **What is the priority order** for migrating existing tools to the new HITL system?
5. **Should we provide development tools** (like testing utilities) for developers building HITL-enabled tools?

## Implementation Priority

### Phase 1: Core Infrastructure
- Implement single HITL node with basic interaction types
- Create HITLRequest standardized response format
- Update AgentState schema

### Phase 2: Tool Migration
- Migrate `trigger_customer_message` tool as primary example
- Create `gather_further_details` tool for missing information scenarios
- Update graph routing logic

### Phase 3: Developer Experience
- Create documentation and examples
- Add error handling and edge case management
- Implement remaining interaction types as needed 