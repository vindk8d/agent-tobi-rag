# HITL Robustness Evaluation: Test Scenarios and Expected State Variables

This document provides explicit specifications for testing the General-Purpose HITL Node implementation, detailing every scenario and the exact state variable transitions expected through each node.

## Core State Variables Being Tracked

### Primary HITL State (3-Field Architecture)
- **`hitl_phase`**: `None` | `"needs_prompt"` | `"awaiting_response"` | `"approved"` | `"denied"`
- **`hitl_prompt`**: `None` | `"prompt text shown to user"`
- **`hitl_context`**: `None` | `{"source_tool": "...", "contextual_data": "..."}`

### Supporting State Variables
- **`messages`**: List of conversation messages (grows with prompts and responses)
- **`employee_id`**: Employee identifier for access control
- **`conversation_id`**: Conversation persistence identifier
- **Node routing decisions**: Which node to transition to next

### Node Flow Pattern
```
employee_agent → hitl_node → employee_agent → ea_memory_store
       ↑              ↓              ↑
    (never routes back to hitl_node - no recursion)
```

---

## Test Scenario 1: Customer Message Approval Flow

### Scenario Description
Employee asks to send a message to a customer. The system requires approval before sending.

### User Story
"As an employee, I want to send a follow-up message to John Smith about tomorrow's meeting, and I want the system to confirm before sending."

### Step-by-Step Flow and State Variables

| Step | Node | Action | hitl_phase | hitl_prompt | hitl_context | messages_count | Route Decision |
|------|------|--------|------------|-------------|--------------|----------------|----------------|
| 1 | start | Employee sends request | `None` | `None` | `None` | 1 | → employee_agent |
| 2 | employee_agent | Calls trigger_customer_message tool | `None` | `None` | `None` | 1 | (processing) |
| 3 | employee_agent | Tool returns request_approval() | `"needs_prompt"` | `"Send this message to John Smith?\n\nMessage: Following up..."` | `{"source_tool": "trigger_customer_message", "customer_id": "john_smith_123", "message_content": "...", "message_type": "follow_up"}` | 1 | → hitl_node |
| 4 | hitl_node | Shows prompt to user, interrupts | `"awaiting_response"` | `"Send this message to John Smith?..."` | `{...preserved...}` | 2 | (interrupt) |
| 5 | hitl_node | User responds "send it" | `"awaiting_response"` | `"Send this message to John Smith?..."` | `{...preserved...}` | 3 | (processing) |
| 6 | hitl_node | Processes approval | `None` | `None` | `None` | 4 | → employee_agent |
| 7 | employee_agent | Detects approved action, re-calls tool | `None` | `None` | `None` | 4 | (processing) |
| 8 | employee_agent | Tool executes successfully | `None` | `None` | `None` | 5 | → ea_memory_store |
| 9 | ea_memory_store | Stores conversation | `None` | `None` | `None` | 5 | END |

### Expected State Progression
```
hitl_phase: [None] → [needs_prompt] → [awaiting_response] → [None] → [None] → [None]
```

### Denial Variant
If user responds "no, cancel that" at step 5:
- Step 6: `hitl_phase` becomes `None`, `hitl_context` becomes `None` (cleared)
- Step 7: Agent does not re-call tool, routes directly to memory store
- Step 8: `ea_memory_store` with cancellation message

---

## Test Scenario 2: Multi-Step Input Collection Flow

### Scenario Description  
Employee asks to collect sales requirements for a customer. Tool needs multiple pieces of information through successive HITL input requests.

### User Story
"As an employee, I want to collect complete sales requirements for john@example.com, and the system should guide me through gathering all necessary information step by step."

### Step-by-Step Flow and State Variables

| Step | Node | Action | hitl_phase | hitl_prompt | hitl_context | Collection Progress | Route Decision |
|------|------|--------|------------|-------------|--------------|-------------------|----------------|
| 1 | start | Employee requests collection | `None` | `None` | `None` | {} | → employee_agent |
| 2 | employee_agent | Calls collect_sales_requirements | `None` | `None` | `None` | {} | (processing) |
| 3 | employee_agent | Tool needs vehicle_type | `"needs_prompt"` | `"What type of vehicle...?"` | `{"source_tool": "collect_sales_requirements", "customer_identifier": "john@example.com", "collected_data": {}, "missing_fields": ["vehicle_type", "budget", "timeline"], "collection_step": 1}` | {} | → hitl_node |
| 4 | hitl_node | Shows vehicle prompt, interrupts | `"awaiting_response"` | `"What type of vehicle...?"` | `{...preserved...}` | {} | (interrupt) |
| 5 | hitl_node | User responds "SUV" | `"awaiting_response"` | `"What type of vehicle...?"` | `{...preserved...}` | {} | (processing) |
| 6 | hitl_node | Processes input | `None` | `None` | `None` | {} | → employee_agent |
| 7 | employee_agent | Re-calls tool with vehicle_type | `None` | `None` | `None` | {"vehicle_type": "SUV"} | (processing) |
| 8 | employee_agent | Tool needs budget | `"needs_prompt"` | `"What's your budget range...?"` | `{"source_tool": "collect_sales_requirements", "customer_identifier": "john@example.com", "collected_data": {"vehicle_type": "SUV"}, "missing_fields": ["budget", "timeline"], "collection_step": 2}` | {"vehicle_type": "SUV"} | → hitl_node |
| 9 | hitl_node | Shows budget prompt, interrupts | `"awaiting_response"` | `"What's your budget range...?"` | `{...preserved...}` | {"vehicle_type": "SUV"} | (interrupt) |
| 10 | hitl_node | User responds "$40k-$50k" | `"awaiting_response"` | `"What's your budget range...?"` | `{...preserved...}` | {"vehicle_type": "SUV"} | (processing) |
| 11 | hitl_node | Processes input | `None` | `None` | `None` | {"vehicle_type": "SUV"} | → employee_agent |
| 12 | employee_agent | Re-calls tool with budget | `None` | `None` | `None` | {"vehicle_type": "SUV", "budget": "$40k-$50k"} | (processing) |
| 13 | employee_agent | Tool needs timeline | `"needs_prompt"` | `"When do you need it by...?"` | `{"source_tool": "collect_sales_requirements", "customer_identifier": "john@example.com", "collected_data": {"vehicle_type": "SUV", "budget": "$40k-$50k"}, "missing_fields": ["timeline"], "collection_step": 3}` | {"vehicle_type": "SUV", "budget": "$40k-$50k"} | → hitl_node |
| 14 | hitl_node | Shows timeline prompt, interrupts | `"awaiting_response"` | `"When do you need it by...?"` | `{...preserved...}` | {"vehicle_type": "SUV", "budget": "$40k-$50k"} | (interrupt) |
| 15 | hitl_node | User responds "2 weeks" | `"awaiting_response"` | `"When do you need it by...?"` | `{...preserved...}` | {"vehicle_type": "SUV", "budget": "$40k-$50k"} | (processing) |
| 16 | hitl_node | Processes input | `None` | `None` | `None` | {"vehicle_type": "SUV", "budget": "$40k-$50k"} | → employee_agent |
| 17 | employee_agent | Re-calls tool with complete data | `None` | `None` | `None` | {"vehicle_type": "SUV", "budget": "$40k-$50k", "timeline": "2 weeks"} | (processing) |
| 18 | employee_agent | Tool returns success (no HITL) | `None` | `None` | `None` | Complete | → ea_memory_store |
| 19 | ea_memory_store | Stores conversation | `None` | `None` | `None` | Complete | END |

### Expected State Progression
```
hitl_phase: [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [None]
```

### Key Validation Points
- Each HITL cycle completely clears state before the next
- Tool-managed collection controls the iteration, not HITL node
- `hitl_context` preserves collection state between iterations
- No HITL recursion - always routes back to employee_agent

---

## Test Scenario 3: Customer Selection Flow

### Scenario Description
Employee request results in multiple matching options requiring user disambiguation.

### User Story
"As an employee, when I look up 'John' and multiple customers match, I want to clearly select which John I meant."

### Step-by-Step Flow and State Variables

| Step | Node | Action | hitl_phase | hitl_prompt | hitl_context | Found Options | Route Decision |
|------|------|--------|------------|-------------|--------------|---------------|----------------|
| 1 | start | Employee: "Look up customer John" | `None` | `None` | `None` | None | → employee_agent |
| 2 | employee_agent | Calls customer_lookup tool | `None` | `None` | `None` | None | (processing) |
| 3 | employee_agent | Tool finds multiple matches | `"needs_prompt"` | `"I found multiple customers named John. Which one did you mean?\n\n1. John Smith (john.smith@email.com) - Premium Account\n2. John Doe (johndoe@company.com) - Business Account\n3. John Wilson (j.wilson@personal.com) - Standard Account\n\nPlease select 1, 2, or 3:"` | `{"source_tool": "customer_lookup", "original_query": "John", "options": [...], "selection_step": 1}` | 3 options | → hitl_node |
| 4 | hitl_node | Shows options, interrupts | `"awaiting_response"` | `"I found multiple customers..."` | `{...preserved...}` | 3 options | (interrupt) |
| 5 | hitl_node | User responds "option 2" | `"awaiting_response"` | `"I found multiple customers..."` | `{...preserved...}` | 3 options | (processing) |
| 6 | hitl_node | Processes selection | `None` | `None` | `None` | 3 options | → employee_agent |
| 7 | employee_agent | Re-calls tool with selection | `None` | `None` | `None` | John Doe selected | (processing) |
| 8 | employee_agent | Tool returns specific customer | `None` | `None` | `None` | Customer data | → ea_memory_store |
| 9 | ea_memory_store | Stores conversation | `None` | `None` | `None` | Complete | END |

### Expected State Progression
```
hitl_phase: [None] → [needs_prompt] → [awaiting_response] → [None] → [None]
```

---

## Test Scenario 4: Edge Cases and Error Handling

### Scenario 4a: Invalid User Responses

| Response Type | User Input | Expected LLM Interpretation | Expected hitl_phase | Expected Outcome |
|---------------|------------|---------------------------|-------------------|------------------|
| Empty | `""` | `"input"` | `None` | Treated as input, state cleared |
| Ambiguous | `"maybe"` | `"input"` | `None` | Treated as input, state cleared |
| Gibberish | `"asdfgh"` | `"input"` | `None` | Treated as input, state cleared |
| Conflicting | `"yes no maybe"` | `"input"` | `None` | Treated as input, state cleared |
| Emoji only | `"🎉🎊"` | `"input"` | `None` | Treated as input, state cleared |

### Scenario 4b: LLM Interpretation Failure

| Step | Node | Event | hitl_phase | hitl_prompt | hitl_context | Route Decision |
|------|------|-------|------------|-------------|--------------|----------------|
| 1 | hitl_node | User responds "yes, do it" | `"awaiting_response"` | `"Approve action?"` | `{"source_tool": "test"}` | (processing) |
| 2 | hitl_node | LLM API fails/times out | `"awaiting_response"` | `"Approve action?"` | `{"source_tool": "test"}` | (error handling) |
| 3 | hitl_node | Fallback processing | `None` | `None` | `None` | → employee_agent |
| 4 | employee_agent | Error message added | `None` | `None` | `None` | → ea_memory_store |

### Scenario 4c: Malformed HITL Context

| Context Type | hitl_context Value | Expected Behavior | Route Decision |
|--------------|-------------------|-------------------|----------------|
| Missing Context | `None` | Route to HITL, handle gracefully | → hitl_node |
| Empty Context | `{}` | Route to HITL, process normally | → hitl_node |
| Missing source_tool | `{"data": "value"}` | Route to HITL, log warning | → hitl_node |
| Invalid JSON | `{"invalid": object()}` | Handle serialization error | → hitl_node |

---

## Test Scenario 5: Natural Language Interpretation Matrix

### Approval Phrases Test Matrix

| Category | Phrases | Expected LLM Output | Expected hitl_phase After Processing |
|----------|---------|-------------------|-----------------------------------|
| Direct | `"yes"`, `"ok"`, `"sure"` | `"approval"` | `None` (cleared) |
| Action-oriented | `"send it"`, `"do it"`, `"go ahead"` | `"approval"` | `None` (cleared) |
| Confirmatory | `"confirm"`, `"proceed"`, `"approve"` | `"approval"` | `None` (cleared) |
| Casual | `"sounds good"`, `"let's do this"`, `"yep"` | `"approval"` | `None` (cleared) |
| Emoji | `"👍"`, `"✅"` | `"approval"` | `None` (cleared) |

### Denial Phrases Test Matrix

| Category | Phrases | Expected LLM Output | Expected hitl_phase After Processing |
|----------|---------|-------------------|-----------------------------------|
| Direct | `"no"`, `"nope"`, `"nah"` | `"denial"` | `None` (cleared) |
| Action-oriented | `"cancel"`, `"abort"`, `"stop"` | `"denial"` | `None` (cleared) |
| Polite | `"not now"`, `"maybe later"`, `"skip this"` | `"denial"` | `None` (cleared) |
| Definitive | `"don't send it"`, `"never mind"` | `"denial"` | `None` (cleared) |
| Emoji | `"❌"`, `"🚫"` | `"denial"` | `None` (cleared) |

### Input Data Test Matrix

| Data Type | Example Input | Expected LLM Output | Expected Processing |
|-----------|---------------|-------------------|-------------------|
| Email | `"john@example.com"` | `"input"` | Store as input data |
| Name | `"Customer ABC Corp"` | `"input"` | Store as input data |
| Selection | `"option 2"` | `"input"` | Process as selection |
| Time | `"tomorrow at 3pm"` | `"input"` | Store as timeline data |
| Money | `"budget is $50,000"` | `"input"` | Store as budget data |
| Phone | `"phone: 555-123-4567"` | `"input"` | Store as contact data |

---

## Test Scenario 6: Tool-Managed Recursive Collection

### Scenario Description
The `collect_sales_requirements` tool demonstrates tool-managed recursive collection where the tool controls its own multi-step data gathering.

### Collection State Evolution

| Iteration | Missing Fields | User Input | Collected Data | Next HITL Request |
|-----------|----------------|------------|----------------|-------------------|
| 1 | `["vehicle_type", "budget", "timeline"]` | N/A | `{}` | Vehicle type request |
| 2 | `["budget", "timeline"]` | `"SUV"` | `{"vehicle_type": "SUV"}` | Budget request |
| 3 | `["timeline"]` | `"$40k-$50k"` | `{"vehicle_type": "SUV", "budget": "$40k-$50k"}` | Timeline request |
| 4 | `[]` | `"2 weeks"` | `{"vehicle_type": "SUV", "budget": "$40k-$50k", "timeline": "2 weeks"}` | Collection complete |

### State Variable Progression for Each Iteration

#### Iteration 1 (Vehicle Type)
| Step | hitl_phase | hitl_context.missing_fields | hitl_context.collected_data |
|------|------------|---------------------------|---------------------------|
| Before HITL | `"needs_prompt"` | `["vehicle_type", "budget", "timeline"]` | `{}` |
| After HITL | `None` | N/A (cleared) | N/A (cleared) |
| Tool Re-call | N/A | `["budget", "timeline"]` | `{"vehicle_type": "SUV"}` |

#### Iteration 2 (Budget)
| Step | hitl_phase | hitl_context.missing_fields | hitl_context.collected_data |
|------|------------|---------------------------|---------------------------|
| Before HITL | `"needs_prompt"` | `["budget", "timeline"]` | `{"vehicle_type": "SUV"}` |
| After HITL | `None` | N/A (cleared) | N/A (cleared) |
| Tool Re-call | N/A | `["timeline"]` | `{"vehicle_type": "SUV", "budget": "$40k-$50k"}` |

#### Iteration 3 (Timeline)
| Step | hitl_phase | hitl_context.missing_fields | hitl_context.collected_data |
|------|------------|---------------------------|---------------------------|
| Before HITL | `"needs_prompt"` | `["timeline"]` | `{"vehicle_type": "SUV", "budget": "$40k-$50k"}` |
| After HITL | `None` | N/A (cleared) | N/A (cleared) |
| Tool Re-call | N/A | `[]` | `{"vehicle_type": "SUV", "budget": "$40k-$50k", "timeline": "2 weeks"}` |

### Complete Multi-Step State Progression
```
hitl_phase: [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [None]
```

---

## Test Scenario 7: Complex Multi-Tool Workflow

### Scenario Description
Employee performs a complex workflow involving customer lookup (selection), requirements collection (multi-input), and message sending (approval).

### Workflow State Progression

| Phase | Tool | HITL Type | hitl_phase Progression | Key Context Data |
|-------|------|-----------|----------------------|------------------|
| **Phase 1: Customer Lookup** | `customer_lookup` | Selection | `None → needs_prompt → awaiting_response → None` | `{"source_tool": "customer_lookup", "options": [...]}` |
| **Phase 2a: Requirements - Vehicle** | `collect_sales_requirements` | Input | `None → needs_prompt → awaiting_response → None` | `{"source_tool": "collect_sales_requirements", "missing_fields": ["vehicle_type", "budget"]}` |
| **Phase 2b: Requirements - Budget** | `collect_sales_requirements` | Input | `None → needs_prompt → awaiting_response → None` | `{"source_tool": "collect_sales_requirements", "missing_fields": ["budget"]}` |
| **Phase 3: Message Sending** | `trigger_customer_message` | Approval | `None → needs_prompt → awaiting_response → None` | `{"source_tool": "trigger_customer_message", "customer_id": "...", "message_content": "..."}` |

### Complete Workflow State Progression
```
hitl_phase: [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [needs_prompt] → [awaiting_response] → [None] → [None]
```

**Total HITL Interactions**: 4 (1 selection + 2 inputs + 1 approval)
**Total State Transitions**: 13 steps
**Final State**: All HITL variables cleared, conversation stored

---

## Test Scenario 8: Performance and Stress Testing

### Rapid Sequential Requests
Test 5 rapid HITL requests in sequence to ensure no state bleeding:

| Request # | hitl_context.source_tool | User Response | Expected Phase Clearing |
|-----------|-------------------------|---------------|------------------------|
| 1 | `"tool_1"` | `"yes"` | `hitl_phase: None` |
| 2 | `"tool_2"` | `"no"` | `hitl_phase: None` |
| 3 | `"tool_3"` | `"approve"` | `hitl_phase: None` |
| 4 | `"tool_4"` | `"cancel"` | `hitl_phase: None` |
| 5 | `"tool_5"` | `"proceed"` | `hitl_phase: None` |

**Validation**: Each request should start and end with clean state, no cross-contamination.

### Concurrent Context Testing
Test different HITL contexts processed independently:

| Context Type | source_tool | Expected Routing | State Isolation |
|--------------|------------|------------------|-----------------|
| Message Context | `"trigger_customer_message"` | `→ hitl_node` | Isolated |
| Collection Context | `"collect_sales_requirements"` | `→ hitl_node` | Isolated |
| Lookup Context | `"customer_lookup"` | `→ hitl_node` | Isolated |

---

## Critical Validation Points

### State Field Consistency
1. **hitl_phase transitions**: Must follow exact progression without skipping phases
2. **hitl_prompt handling**: Must be set when needed, preserved during processing, cleared after
3. **hitl_context preservation**: Must maintain tool context through HITL cycles, clear appropriately
4. **Message history**: Must grow with prompts and responses, maintain conversation flow

### Routing Consistency  
1. **From employee_agent**: Routes to `hitl_node` when prompt exists, `ea_memory_store` otherwise
2. **From hitl_node**: ALWAYS routes to `employee_agent` (never self-routes)
3. **Tool re-calling**: Agent must detect approved contexts and re-call tools with user data

### Error Handling Robustness
1. **Invalid responses**: Must handle gracefully without breaking state
2. **LLM failures**: Must have fallback behavior that clears state safely
3. **Malformed context**: Must process without errors and route correctly
4. **Empty/missing data**: Must handle edge cases without system failure

### Performance Requirements
1. **State isolation**: Rapid sequential requests must not bleed state between interactions
2. **Memory efficiency**: Large context objects must be handled without memory leaks
3. **Response time**: LLM interpretation should complete within reasonable time bounds
4. **Interrupt handling**: Must handle interrupts and resumes cleanly without state corruption

---

## Test Execution Guide

### Running the Comprehensive Evaluation
```bash
# Activate virtual environment
source venv/bin/activate

# Run the comprehensive test suite
python tests/test_comprehensive_hitl_robustness_eval.py

# Expected output: Detailed state transition logs and pass/fail results
```

### Success Criteria
- **All 8 test scenarios pass completely**
- **State transitions match expected progressions exactly**
- **No HITL recursion occurs (always routes back to employee_agent)**
- **LLM interpretation handles diverse natural language correctly**
- **Error scenarios are handled gracefully**
- **Performance tests show no state bleeding or memory issues**

### Monitoring During Tests
Watch for these key indicators:
- ✅ Clean state transitions (no unexpected phases)
- ✅ Proper routing decisions (no infinite loops)
- ✅ Context preservation and clearing at right times
- ✅ Message history grows correctly
- ✅ Tool re-calling works with user responses integrated
- ✅ Final state is always clean (all HITL fields cleared)

This comprehensive evaluation validates that the General-Purpose HITL Node implementation is robust, user-friendly, and production-ready according to the PRD specifications.