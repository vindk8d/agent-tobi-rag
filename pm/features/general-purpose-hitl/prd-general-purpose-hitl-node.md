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

1. **The system must support natural language interactions** where users can respond in any way they prefer (approve/deny, selections, input) and the LLM interprets intent
2. **The system must eliminate rigid interaction types** and let tools define their own prompting style using dedicated HITL request tools
3. **The system must support tool-managed recursive collection** where tools control their own multi-step data gathering by re-calling themselves with updated context

### Tool Integration Requirements

4. **Tools must be able to trigger HITL interactions** using dedicated HITL request tools (e.g., `request_approval`, `request_input`, `request_selection`)
5. **The system must parse tool responses** and automatically route to the HITL node when HITL interactions are requested
6. **The agent must re-call tools with updated context** after HITL completion to continue tool-managed recursive collection
7. **Tools must signal completion** by returning normal results (not HITL_REQUIRED) when all required information has been gathered

### User Experience Requirements

8. **The system must provide clear, formatted prompts** that tools can customize without rigid format constraints
9. **The system must use LLM interpretation** to understand user responses naturally instead of rigid validation rules
10. **The system must allow users to cancel operations** using natural language (e.g., "cancel", "never mind", "abort")
11. **The system must maintain conversation context** throughout tool-managed multi-step interactions

### Developer Experience Requirements

12. **The system must provide dedicated HITL request tools** (e.g., `request_approval`, `request_input`) instead of a complex HITLRequest class
13. **Tools must be able to specify completely custom prompts** without being constrained by rigid interaction types
14. **Tools must manage their own collection state** through serialized context parameters passed between tool calls
15. **The agent must automatically re-call tools** with user responses integrated into the tool's context parameters

### LangGraph Integration Requirements

16. **The HITL node must integrate seamlessly** with existing LangGraph interrupt mechanisms
17. **The system must use a single ultra-simple HITL node** that handles prompting and LLM-driven response interpretation only
18. **The HITL node must never route back to itself** - always return to agent node after processing user response
19. **The agent node must detect tool-managed collection mode** and automatically re-call tools with updated context
20. **The system must maintain state consistency** across interrupt/resume cycles and tool re-calling loops

### State Management Requirements

21. **The system must use ultra-minimal HITL fields** (`hitl_phase`, `hitl_prompt`, `hitl_context`) - only 3 fields total
22. **The system must eliminate `hitl_type` and `hitl_result`** as they are no longer needed with LLM interpretation and tool-managed collection
23. **The system must replace all legacy HITL fields** (`hitl_data`, `confirmation_data`, `execution_data`, `confirmation_result`) with 3 simple flat fields  
24. **The system must use explicit phase enumeration** with clear transitions between needs_prompt, awaiting_response, approved, denied states
25. **The system must support direct field access** without requiring JSON parsing or navigation (`state.get("hitl_phase")`)
26. **The system must provide crystal clear routing logic** where HITL never routes to itself, always back to agent
27. **The system must support graceful migration** allowing coexistence of legacy and new fields during transition

### Tool-Managed Recursive Collection Requirements

28. **Tools must manage their own collection state** through serialized context parameters passed between tool calls
29. **Tools must determine what information is still needed** and generate appropriate HITL requests for each missing piece
30. **The agent must re-call tools with user responses** integrated into the tool's context parameters after each HITL completion
31. **Tools must signal collection completion** by returning normal results (not HITL_REQUIRED) when all data has been gathered
32. **The system must support tool-managed iteration limits** to prevent infinite collection loops through tool-controlled logic

## Non-Goals (Out of Scope)

1. **Advanced input validation** (e.g., email format checking, data type validation) - tools should handle their own validation
2. **Timeout handling** for unresponsive users - this will be handled at the infrastructure level
3. **Complex multi-turn conversations** - focus on simple, direct information gathering
4. **Backward compatibility** with existing `confirmation_data` approach - all tools will be migrated
5. **User interface enhancements** beyond text-based interactions - this is focused on the backend logic
6. **User authentication or permission checking** - this is handled by existing user context systems

## Design Considerations

### State Management - Ultra-Minimal 3-Field Design for HITL interactions
- **Revolutionary Simplicity**: Only 3 fields total for HITL - `hitl_phase`, `hitl_prompt`, `hitl_context`
- **Eliminated Complexity**: No `hitl_type` (tools define their own style), no `hitl_result` (use phase + context)
- **Replaces All Legacy Fields**: Eliminates `hitl_data`, `confirmation_data`, `execution_data`, `confirmation_result`
- **Easy Access**: Direct field access without JSON parsing (`state.get("hitl_phase")`)
- **LLM-Native**: Designed for natural language interpretation instead of rigid validation
- **Migration Friendly**: Can coexist with legacy fields during transition period

### Tool-Based HITL Request Format
- Tools use dedicated HITL request tools instead of generic HITLRequest class
- Agent parses tool responses and creates ultra-minimal state:
  ```python
  # Ultra-minimal HITL state - only 3 fields!
  hitl_phase: Optional[str] = "needs_prompt"       # Lifecycle phase
  hitl_prompt: Optional[str] = "User prompt text"  # What to show user  
  hitl_context: Optional[Dict] = {"source_tool": ..., "args": ...}  # Execution context
  ```
- **Revolutionary Simplicity**: Eliminated `hitl_type` (tools define style) and `hitl_result` (use phase + context)
- **Ultra-Simple Access**: `state.get("hitl_phase")` - no JSON parsing needed
- **Clear Lifecycle**: Phase progresses from needs_prompt → awaiting_response → approved/denied
- **LLM Interpretation**: No rigid validation - LLM understands user intent naturally

### Tool-Managed Recursive Collection Pattern
- **Tool-Controlled Flow**: Tools manage their own multi-step collection logic without HITL complexity
- **Agent Coordination**: Agent detects tool-managed collection mode and re-calls tools with updated context
- **Simple HITL Role**: HITL only handles single interactions, never manages collection state or routing
- **Natural Completion**: Tools return normal results (not HITL_REQUIRED) when collection is complete
- **Clear Separation**: HITL handles user interaction, tools handle collection logic, agent coordinates between them
- **No HITL Recursion**: HITL node never routes back to itself, always returns to agent node

### Comprehensive State Consolidation Analysis
**Current State Redundancy Issues:**
- `hitl_data` + `confirmation_data` → Both hold interaction data
- `execution_data` + `confirmation_result` → Both hold results/execution context  
- Complex nested JSON structures require parsing and validation
- Multiple fields serving overlapping purposes creates confusion and maintenance burden

**Recommended Ultra-Minimal 3-Field Design:**
```python
# Revolutionary simplicity - only 3 fields total!
hitl_phase: Optional[str] = None        # "needs_prompt" | "awaiting_response" | "approved" | "denied"
hitl_prompt: Optional[str] = None       # The actual text to show the user
hitl_context: Optional[Dict] = None     # Minimal execution context when needed

# ELIMINATED: hitl_type (tools define their own style)
# ELIMINATED: hitl_result (use hitl_phase + hitl_context instead)
```

**Benefits of Ultra-Minimal 3-Field Design:**
- **Maximum Simplicity**: Only 3 fields - impossible to over-engineer
- **No Type Dispatch**: Eliminated complex handler routing based on interaction type
- **LLM-Native**: Designed for natural language interpretation, not rigid validation
- **Tool Freedom**: Tools can define completely custom interaction styles
- **Easy Routing**: `if state.get("hitl_phase") == "approved"` - crystal clear
- **Zero Maintenance**: No complex validation logic to maintain or debug

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

### Phase 1: Ultra-Minimal 3-Field Architecture (Revolutionary)
- Update AgentState schema with only 3 HITL fields (`hitl_phase`, `hitl_prompt`, `hitl_context`)
- Create HITLPhase enum with explicit phase definitions  
- Eliminate `hitl_type` and `hitl_result` fields completely
- Plan migration strategy from legacy fields to ultra-minimal fields

### Phase 2: LLM-Native HITL Node (Revolutionary)
- Replace complex HITL node with ultra-simple LLM-driven response interpretation
- Eliminate HITLRequest class - replace with dedicated HITL request tools
- Remove all type-based dispatch logic and rigid validation
- Add LLM-powered natural language response understanding

### Phase 3: Tool-Managed Recursive Collection Implementation (Revolutionary)
- Update agent node to detect tool-managed collection mode and automatically re-call tools with updated context
- Eliminate all HITL recursion logic - HITL never routes back to itself, always returns to agent
- Implement tool collection pattern where tools manage their own state through serialized parameters
- Test multi-information collection scenarios with tool-managed approach

### Phase 4: Tool Migration and Routing Enhancement (Revolutionary)
- Replace HITLRequest usage with dedicated HITL request tools (`request_approval`, `request_input`, etc.)
- Migrate existing tools to use ultra-minimal 3-field structure  
- Update graph routing to handle `hitl_phase` values like "approved" and "denied" for routing decisions
- Implement recursive collection examples using 3-field approach

### Phase 5: Testing and Documentation (Revolutionary)  
- Create comprehensive 3-field phase transition tests
- Add recursive collection integration tests with ultra-minimal state
- Document new LLM-native HITL patterns and dedicated request tools
- Provide migration guide from legacy fields to revolutionary 3-field approach