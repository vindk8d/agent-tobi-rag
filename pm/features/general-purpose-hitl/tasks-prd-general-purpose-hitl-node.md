## Relevant Files

- `backend/agents/tobi_sales_copilot/state.py` - ✅ Enhanced AgentState schema with single hitl_data field containing all HITL interaction state **→ 🔄 WILL BE REVOLUTIONIZED** to use ultra-minimal 3 HITL fields (`hitl_phase`, `hitl_prompt`, `hitl_context`) eliminating `hitl_type` and `hitl_result` for maximum simplicity
- `backend/agents/hitl.py` - ✅ New module containing HITLRequest class, interaction utilities, _format_hitl_prompt(), and _process_hitl_response() functions (standardization system + main HITL node completed) **→ 🔄 WILL BE REVOLUTIONIZED** by eliminating HITLRequest class, removing complex dispatch logic, and implementing LLM-driven natural language response interpretation
- `backend/agents/tools.py` - ✅ Updated trigger_customer_message to use HITLRequest.confirmation() and HITLRequest.input_request() when customer not found, added gather_further_details() tool for generic information gathering, plus handler functions with enhanced customer lookup and delivery functions. Removed legacy functions and unused interrupt imports - all tools now use standardized HITL patterns. **→ 🔄 WILL BE REVOLUTIONIZED** by replacing HITLRequest usage with dedicated HITL request tools (`request_approval`, `request_input`) and implementing 3-field structure
- `backend/agents/tobi_sales_copilot/agent.py` - ✅ **SIMPLIFIED ARCHITECTURE**: Updated agent node with HITL response parsing and graph structure. Removed legacy _confirmation_node() and replaced with imported hitl_node. Added parse_tool_response() integration in _employee_agent_node() only (customers don't use HITL). Implemented simplified routing: route_from_employee_agent() and route_from_hitl() with clean separation between employee and customer workflows. Employee workflow: ea_memory_prep → employee_agent → (hitl_node or ea_memory_store). Customer workflow: ca_memory_prep → customer_agent → ca_memory_store (simple, no HITL). HITL workflow: employee_agent ↔ hitl_node loop only. **→ 🔄 WILL BE REVOLUTIONIZED** with ultra-simple 3-field routing logic using direct field access (`state.get("hitl_phase")`) and elimination of type-based parsing
- `tests/test_hitl_end_to_end_simple.py` - ✅ Comprehensive end-to-end tests for simplified HITL architecture covering employee workflows with/without HITL, customer workflows (no HITL), routing edge cases, tool-HITL integration, user verification routing, and graph structure validation. All tests PASS ✅ **→ 🔄 WILL BE UPDATED** to test revolutionary 3-field structure and LLM-driven response interpretation
- `tests/test_hitl_langgraph_integration.py` - ✅ Comprehensive LangGraph integration tests verifying HITL interactions work correctly with the LangGraph interrupt mechanism. Tests graph compilation with interrupts, execution pausing at HITL node, state persistence during interrupts, human response processing, Command-based resumption, error handling, and complete end-to-end interrupt simulation. All tests PASS ✅ **→ 🔄 WILL BE ENHANCED** with 3-field phase transition and recursive collection tests
- `tests/test_recursive_collection.py` - **→ 🆕 NEW FILE** for testing recursive collection pattern with multi-step information gathering scenarios using ultra-minimal 3-field approach
- `tests/test_phase_transitions.py` - **→ 🆕 NEW FILE** for testing 3-field HITL phase transitions and ultra-minimal state management validation
- `tests/test_state_migration.py` - **→ 🆕 NEW FILE** for testing migration from legacy fields to revolutionary 3-field structure

### Notes

- Unit tests should typically be placed alongside the code files they are testing
- Use `npx jest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the Jest configuration.

## Tasks

- [x] 1.0 Create Core HITL Infrastructure
  - [x] 1.1 Update AgentState schema in `backend/agents/tobi_sales_copilot/state.py` to include single `hitl_data` field containing all HITL state
  - [x] 1.2 Create `backend/agents/hitl.py` module with base imports and logging setup
  - [x] 1.3 Define HITLInteractionType literal type with values: "confirmation", "selection", "input_request", "multi_step_input"
  - [x] 1.4 Create custom JSON encoder class to handle datetime objects and complex data types in HITL context
  - [x] 1.5 Add HITL-specific error handling classes and logging utilities

- [x] 2.0 Implement HITLRequest Standardization System
  - [x] 2.1 Create HITLRequest class in `backend/agents/hitl.py` with static methods for each interaction type
  - [x] 2.2 Implement HITLRequest.confirmation() method that formats confirmation requests with approve/deny options
  - [x] 2.3 Implement HITLRequest.selection() method that formats selection requests with numbered options and optional cancel
  - [x] 2.4 Implement HITLRequest.input_request() method that formats input requests with validation hints
  - [x] 2.5 Implement HITLRequest.multi_step_input() method for sequential input collection
  - [x] 2.6 Create parse_tool_response() utility function to extract HITL requirements from tool responses
  - [x] 2.7 Add comprehensive validation for HITLRequest data structure and required fields

- [x] 3.0 Develop Single-Node HITL Handler
  - [x] 3.1 Create hitl_node() function in `backend/agents/hitl.py` that uses hitl_data.awaiting_response to determine if showing prompt or processing response
  - [x] 3.2 Implement _format_hitl_prompt() helper function that creates user-friendly prompts for each interaction type
  - [x] 3.3 Implement _has_new_human_message() utility to detect new human responses in conversation state
  - [x] 3.4 Create _process_hitl_response() function to handle response validation and routing for each interaction type
  - [x] 3.5 Implement confirmation response processing with approval/denial logic
  - [x] 3.6 Implement selection response processing with number validation and option selection
  - [x] 3.7 Implement input request response processing with cancellation handling
  - [x] 3.8 Add error handling for invalid responses that re-prompts with clarification instead of failing
  - [x] 3.9 Implement multi-step input handling for sequential information gathering

- [x] 4.0 Migrate Existing Tools to New HITL Format ✅ **COMPLETE**
  - [x] 4.1 Update trigger_customer_message() in `backend/agents/tools.py` to use HITLRequest.confirmation() for message confirmation
  - [x] 4.2 Update trigger_customer_message() to use HITLRequest.input_request() when customer is not found
  - [x] 4.3 Create _handle_confirmation_approved() function to process approved customer messages
  - [x] 4.4 Create _handle_confirmation_denied() function to handle denied customer message requests
  - [x] 4.5 Create _handle_input_received() function to process customer identifier inputs and retry lookup
  - [x] 4.6 Create gather_further_details() tool using HITLRequest.input_request() for generic information gathering
  - [x] 4.7 Remove legacy confirmation_data handling from trigger_customer_message() after migration
  - [x] 4.8 Update any other tools currently using custom HITL mechanisms to use new standardized format

- [x] 5.0 Update LangGraph Architecture and Routing ✅ **COMPLETE**
  - [x] 5.1 Update agent_node() in `backend/agents/tobi_sales_copilot/rag_agent.py` to parse tool responses using parse_tool_response()
  - [x] 5.2 Add HITL detection logic to agent_node() that populates hitl_data field when HITL_REQUIRED responses are detected
  - [x] 5.3 Create route_from_agent() conditional edge function that routes to HITL node when hitl_data is present
  - [x] 5.4 Create route_from_hitl() conditional edge function that routes back to agent when hitl_data is None or continues HITL for re-prompts
  - [x] 5.5 Update graph structure in _build_graph() to include hitl_node and conditional routing edges
  - [x] 5.6 Configure interrupt_before=["hitl_node"] in graph compilation to enable human interaction
- [x] 5.6.5 Remove _confirmation_node() method from rag_agent.py and replace with imported hitl_node from hitl.py module
- [x] 5.7 Update existing confirmation node references to use new hitl_node throughout the codebase
  - [x] 5.8 Test complete end-to-end flow from tool HITL request through human response back to agent processing
  - [x] 5.9 Create comprehensive integration tests that verify HITL interactions work correctly with LangGraph interrupt mechanism

- [ ] 6.0 Revolutionary Ultra-Minimal 3-Field Architecture Implementation  
  - [ ] 6.1 Update AgentState schema in `backend/agents/tobi_sales_copilot/state.py` to replace all HITL fields with ONLY 3 fields: `hitl_phase`, `hitl_prompt`, `hitl_context`
  - [ ] 6.2 Create HITLPhase enum in `backend/agents/hitl.py` with values: "needs_prompt", "awaiting_response", "approved", "denied"
  - [ ] 6.3 **ELIMINATE** `hitl_type` completely (tools will define their own interaction style)
  - [ ] 6.4 **ELIMINATE** `hitl_result` completely (use `hitl_phase` + `hitl_context` instead)
  - [ ] 6.5 Update parse_tool_response() to create ultra-minimal 3-field assignments only
  - [ ] 6.6 Plan migration strategy from legacy fields to revolutionary 3-field approach

- [ ] 7.0 Revolutionary LLM-Native HITL Node Implementation
  - [ ] 7.1 **ELIMINATE** HITLRequest class completely - replace with dedicated HITL request tools
  - [ ] 7.2 **ELIMINATE** complex type-based dispatch logic (_process_confirmation_response, _process_selection_response, etc.)
  - [ ] 7.3 **IMPLEMENT** LLM-driven response interpretation that understands user intent naturally
  - [ ] 7.4 Update hitl_node() to use ultra-simple 3-field flow control (`hitl_phase`, `hitl_prompt`, `hitl_context`)
  - [ ] 7.5 **ELIMINATE** rigid validation - let LLM interpret "yes", "approve", "send it", "go ahead", etc. as approval
  - [ ] 7.6 Add 3-field transition logging and debugging support for better troubleshooting

- [ ] 8.0 Ultra-Simple Non-Recursive Routing Implementation  
  - [ ] 8.1 Update route_from_employee_agent() in `backend/agents/tobi_sales_copilot/agent.py` to use ultra-simple `state.get("hitl_phase")` for routing decisions
  - [ ] 8.2 **SIMPLIFY** route_from_hitl() to NEVER route back to itself - always route to "employee_agent" after processing user response
  - [ ] 8.3 **ELIMINATE** all HITL recursion logic from routing functions - HITL always returns to agent
  - [ ] 8.4 Add agent node logic to detect tool-managed collection mode and automatically re-call tools with updated context
  - [ ] 8.5 Update employee_agent_node() to handle tool re-calling loop for recursive collection

- [ ] 9.0 Revolutionary Tool-Managed Recursive Collection Implementation
  - [ ] 9.1 Create example tool-managed recursive collection tool in `backend/agents/tools.py` that manages its own collection state through parameters
  - [ ] 9.2 Implement tool-controlled collection completion logic where tools determine what information is still needed
  - [ ] 9.3 **ELIMINATE** HITL-managed collection logic - tools generate HITL requests for each missing piece individually
  - [ ] 9.4 Add agent node logic to detect `collection_mode=tool_managed` in execution context and re-call tools with user responses
  - [ ] 9.5 Create helper functions for serializing/deserializing tool collection state between tool calls

- [ ] 10.0 Revolutionary Tool Migration to Dedicated HITL Request Tools
  - [ ] 10.1 **REPLACE** trigger_customer_message() HITLRequest usage with dedicated `request_approval` tool calls
  - [ ] 10.2 **REPLACE** gather_further_details() HITLRequest usage with dedicated `request_input` tool calls  
  - [ ] 10.3 **CREATE** dedicated HITL request tools: `request_approval`, `request_input`, `request_selection`
  - [ ] 10.4 **ELIMINATE** all HITLRequest class usage throughout the codebase
  - [ ] 10.5 Remove all legacy HITL field usage (`hitl_data`, `confirmation_data`, `execution_data`, `confirmation_result`) throughout the codebase
  - [ ] 10.6 Update tool execution logic to use ultra-minimal `hitl_context` field for approved action context
  - [ ] 10.7 Implement backward compatibility layer during migration period to support both legacy and revolutionary 3-field approaches

- [ ] 11.0 Revolutionary Testing and Validation
  - [ ] 11.1 Create 3-field phase transition tests that verify HITL never routes back to itself
  - [ ] 11.2 Add tool-managed recursive collection integration tests with multi-step information gathering scenarios
  - [ ] 11.3 Test agent node tool re-calling logic with `collection_mode=tool_managed` detection
  - [ ] 11.4 Test LLM-driven response interpretation with various natural language inputs ("yes", "approve", "send it", "go ahead", etc.)
  - [ ] 11.5 Create performance tests to demonstrate 3-field approach provides maximum performance vs. nested JSON
  - [ ] 11.6 Add edge case tests for tool re-calling loops and collection completion detection
  - [ ] 11.7 Test dedicated HITL request tools (`request_approval`, `request_input`, `request_selection`) end-to-end
  - [ ] 11.8 Create migration validation tests to ensure legacy and revolutionary 3-field coexistence works correctly

- [ ] 12.0 Revolutionary Documentation and Developer Experience
  - [ ] 12.1 Update existing HITL documentation to reflect revolutionary 3-field management approach and elimination of HITL recursion
  - [ ] 12.2 Create migration guide for developers updating existing tools from legacy fields to ultra-minimal 3-field structure
  - [ ] 12.3 Document dedicated HITL request tools (`request_approval`, `request_input`, `request_selection`) with examples
  - [ ] 12.4 Add code examples demonstrating tool-managed recursive collection pattern with agent coordination
  - [ ] 12.5 Document agent node tool re-calling logic and `collection_mode=tool_managed` detection
  - [ ] 12.6 Create debugging guide for troubleshooting 3-field phase transitions and tool re-calling loops
  - [ ] 12.7 Update API documentation to reflect revolutionary 3-field definitions and tool-managed collection patterns
  - [ ] 12.8 Document the elimination of HITLRequest class, type-based dispatch complexity, and HITL recursion
  - [ ] 12.9 Document migration strategy and backward compatibility considerations for seamless transition from HITL-managed to tool-managed collection

## Functions to Change/Eliminate Based on Tool-Managed Recursion

### Functions to ELIMINATE (No Longer Needed)
- **All HITL recursion routing logic** - HITL never routes back to itself
- **Complex collection state management in HITL** - tools manage their own state
- **Multi-step HITL processing functions** - replaced by tool re-calling
- **HITL completion checking logic** - tools determine completion themselves
- **Dynamic `hitl_prompt` updating in HITL** - tools generate prompts per request

### Functions to MODIFY
- **hitl_node()** - Simplify to only handle single interactions, always return to agent
- **route_from_hitl()** - Always route to "employee_agent", never back to "hitl_node"
- **employee_agent_node()** - Add tool re-calling logic for `collection_mode=tool_managed`
- **parse_tool_response()** - Detect tool-managed collection mode from tool responses
- **Tool execution logic** - Handle tool re-calling with updated context parameters

### Functions to ADD
- **detect_tool_managed_collection()** - Identify when tools are managing recursive collection
- **recall_tool_with_user_response()** - Agent logic to re-call tools with user input integrated
- **serialize_tool_collection_context()** - Helper for tool state management between calls
- **is_tool_collection_complete()** - Detect when tools return normal results vs. HITL_REQUIRED