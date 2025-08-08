# HITL Flow Evaluation: Generate Quotation Tool

## Architecture Compliance Analysis

### Current HITL Pattern (Established)
1. **Tool calls `request_input()` or `request_approval()`** → Returns formatted string
2. **Agent node receives tool result** → Calls `parse_tool_response()`
3. **HITL node gets control** → Sets `hitl_phase="needs_prompt"`, shows prompt, calls `interrupt()`
4. **User responds** → HITL node processes response, sets `hitl_phase="approved"/"denied"`
5. **Agent node resumes** → Tool execution continues or terminates

### Our Generate Quotation Tool Analysis

## Scenario 1: Complete Information Available
```
Input: customer_identifier="john.smith@email.com", vehicle_requirements="Toyota Camry 2023"
```

**State Flow:**
1. **Agent State Initial**: `hitl_phase=None, hitl_context=None`
2. **Tool Execution**: 
   - ✅ Extracts context successfully
   - ✅ Finds customer in CRM
   - ✅ Parses vehicle requirements with LLM
   - ✅ Finds vehicles and pricing
   - ✅ No missing information detected
   - ✅ Creates quotation preview
3. **HITL Trigger**: `request_approval()` called with final confirmation
4. **Expected State**: 
   ```python
   {
     "hitl_phase": "needs_prompt",
     "hitl_prompt": "📄 **Quotation Ready for Generation**\n...",
     "hitl_context": {
       "tool": "generate_quotation",
       "customer_data": {...},
       "vehicle_pricing": [...],
       "step": "final_approval"
     }
   }
   ```
5. **User Approves**: State becomes `hitl_phase="approved"`
6. **Tool Resumes**: Should generate PDF and return shareable link

**✅ COMPLIANCE STATUS**: COMPLIANT - Follows established pattern

## Scenario 2: Customer Not Found
```
Input: customer_identifier="unknown customer", vehicle_requirements="Toyota Camry"
```

**State Flow:**
1. **Agent State Initial**: `hitl_phase=None`
2. **Tool Execution**: 
   - ✅ Extracts context
   - ❌ Customer lookup fails
   - ❌ Alternative lookups from context fail
3. **HITL Trigger**: `request_input()` called for customer information
4. **Expected State**:
   ```python
   {
     "hitl_phase": "needs_prompt", 
     "hitl_prompt": "📋 **Customer Information Needed**\n...",
     "hitl_context": {
       "tool": "generate_quotation",
       "customer_identifier": "unknown customer",
       "extracted_context": {...},
       "step": "customer_lookup_with_context"
     }
   }
   ```
5. **User Provides Info**: State becomes `hitl_phase="approved"` with user input
6. **Tool Should Resume**: But our tool doesn't handle resume logic!

**❌ ISSUE IDENTIFIED**: Missing resume/continuation logic

## Scenario 3: Vehicle Requirements Unclear  
```
Input: customer_identifier="john@email.com", vehicle_requirements="something reliable"
```

**State Flow:**
1. **Tool Execution**:
   - ✅ Finds customer
   - ❌ LLM parsing fails to extract specific criteria
   - ❌ No vehicles found with vague criteria
3. **HITL Trigger**: `request_input()` for detailed vehicle requirements
4. **Expected State**: Similar to Scenario 2
5. **User Provides Details**: `hitl_phase="approved"` with detailed requirements
6. **Tool Should Resume**: Again, missing resume logic

**❌ ISSUE IDENTIFIED**: Same resume logic problem

## Scenario 4: Missing Critical Information
```
Input: Complete customer/vehicle info, but missing email for quotation delivery
```

**State Flow:**
1. **Tool Execution**:
   - ✅ Customer found, vehicles found, pricing retrieved
   - ❌ Missing critical info detected (email)
3. **HITL Trigger**: `_request_missing_information_via_hitl()` 
4. **Expected State**: Standard HITL state
5. **User Provides Missing Info**: `hitl_phase="approved"`
6. **Tool Should Resume**: Missing resume logic again

**❌ ISSUE IDENTIFIED**: Same resume logic problem

## Critical Issues Identified

### 1. ❌ **Missing Resume/Continuation Logic**
Our tool is designed as a **single-shot execution** but HITL flows require **resumable execution**. When the tool returns a HITL request, it terminates. When the user responds and the system resumes, the tool needs to:

- **Continue from where it left off**
- **Use the user's input to proceed**
- **Maintain all previous context**

### 2. ❌ **State Context Incomplete**
Our HITL contexts don't preserve enough state for resumption:
- Missing intermediate results (customer_data, vehicle_criteria, etc.)
- Missing execution step tracking
- Missing error recovery information

### 3. ❌ **No Multi-Step HITL Support**
Our tool might need multiple HITL interactions (customer lookup → vehicle clarification → missing info), but we don't handle chaining.

## Required Fixes

### Fix 1: Add Resume Logic Pattern
Need to implement the established pattern:
```python
# Check if this is a resume from HITL
if state.get("hitl_phase") == "approved" and state.get("hitl_context", {}).get("tool") == "generate_quotation":
    return await _resume_quotation_generation(state)
else:
    return await _start_quotation_generation(...)
```

### Fix 2: Enhanced Context Preservation
HITL contexts need to preserve ALL intermediate state:
```python
context = {
    "tool": "generate_quotation",
    "step": "customer_lookup_with_context",
    "original_params": {...},
    "extracted_context": {...},
    "intermediate_results": {
        "customer_data": customer_data,
        "vehicle_criteria": vehicle_criteria,
        # ... all computed data
    }
}
```

### Fix 3: Multi-Step HITL Chain Support
Need to handle sequential HITL interactions gracefully.

## Recommendation: Implement Resume Logic
The generate_quotation tool needs to be refactored to support resumable execution following the established patterns used by other tools in the system.

