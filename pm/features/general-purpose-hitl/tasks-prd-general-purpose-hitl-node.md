## Relevant Files

- `backend/agents/tobi_sales_copilot/state.py` - ✅ Enhanced AgentState schema with single hitl_data field containing all HITL interaction state
- `backend/agents/hitl.py` - ✅ New module containing HITLRequest class, interaction utilities, _format_hitl_prompt(), and _process_hitl_response() functions (standardization system + main HITL node completed)
- `backend/agents/tools.py` - ✅ Updated trigger_customer_message to use HITLRequest.confirmation() and HITLRequest.input_request() when customer not found, added gather_further_details() tool for generic information gathering, plus _handle_confirmation_approved(), _handle_confirmation_denied(), _handle_input_received(), and _handle_information_gathered() with enhanced customer lookup and delivery functions. Removed legacy _deliver_message_via_chat, _track_message_delivery, and _handle_customer_message_confirmation functions. Removed unused interrupt imports - all tools now use standardized HITL patterns.
- `backend/agents/tobi_sales_copilot/rag_agent.py` - ✅ **SIMPLIFIED ARCHITECTURE**: Updated agent node with HITL response parsing and graph structure. Removed legacy _confirmation_node() and replaced with imported hitl_node. Added parse_tool_response() integration in _employee_agent_node() only (customers don't use HITL). Implemented simplified routing: route_from_employee_agent() and route_from_hitl() with clean separation between employee and customer workflows. Employee workflow: ea_memory_prep → employee_agent → (hitl_node or ea_memory_store). Customer workflow: ca_memory_prep → customer_agent → ca_memory_store (simple, no HITL). HITL workflow: employee_agent ↔ hitl_node loop only.
- `tests/test_hitl_end_to_end_simple.py` - ✅ Comprehensive end-to-end tests for simplified HITL architecture covering employee workflows with/without HITL, customer workflows (no HITL), routing edge cases, tool-HITL integration, user verification routing, and graph structure validation. All tests PASS ✅
- `tests/test_hitl_langgraph_integration.py` - ✅ Comprehensive LangGraph integration tests verifying HITL interactions work correctly with the LangGraph interrupt mechanism. Tests graph compilation with interrupts, execution pausing at HITL node, state persistence during interrupts, human response processing, Command-based resumption, error handling, and complete end-to-end interrupt simulation. All tests PASS ✅

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