# Dynamic Model Selection Tests

This directory contains tests for the dynamic model selection functionality implemented in Task 6.6.

## Bug Fix

### Issue
The agent was failing with:
```
MemoryManager' object has no attribute 'settings'
```

### Root Cause
In `backend/agents/memory.py`, the `_ensure_initialized()` method was not storing the settings object, causing `_create_memory_llm()` to fail when trying to access `self.settings`.

### Fix Applied
Added `self.settings = settings` before ModelSelector initialization in the MemoryManager.

## Test Files

### `test_settings_bug_fix.py`
Focused test that verifies:
- ✅ ModelSelector basic functionality
- ✅ MemoryManager settings bug is fixed
- ✅ Configuration supports new model fields

### `test_dynamic_model_selection.py` 
Comprehensive test suite covering:
- ModelSelector query complexity classification
- Memory system dynamic model selection
- SQL tools dynamic model selection  
- RAG agent integration
- Error handling and edge cases

### `run_dynamic_model_tests.py`
Test runner script (may have import issues in complex environments)

## Running Tests

**Quick verification:**
```bash
python tests/test_settings_bug_fix.py
```

**Full test suite (if environment supports it):**
```bash
pytest tests/test_dynamic_model_selection.py -v
```

## Implementation Coverage

Our tests verify that dynamic model selection works across:

1. **RAG Agent** (`rag_agent.py`)
   - Per-query model selection
   - Simple queries → GPT-3.5-turbo
   - Complex queries → GPT-4

2. **SQL Tools** (`tools.py`)
   - CRM query complexity assessment  
   - Business analysis → GPT-4
   - Simple lookups → GPT-3.5-turbo

3. **Memory System** (`memory.py`)
   - Summary tasks with few messages → GPT-3.5-turbo
   - Summary tasks with many/long messages → GPT-4
   - Insights extraction → GPT-4 (always complex)
   - Analysis tasks → GPT-4 (always complex)

4. **Configuration** (`config.py`)
   - `OPENAI_SIMPLE_MODEL` environment variable
   - `OPENAI_COMPLEX_MODEL` environment variable
   - Proper fallback to defaults

## Architecture Verification

Tests confirm that separation of concerns is maintained:
- **Agent-specific code** in `rag_agent.py` (agent folder)
- **Reusable components** in `tools.py` and `memory.py` (shared folder)
- **Clean import boundaries** with no circular dependencies
- **Proper initialization order** to prevent attribute errors

## Key Benefits Verified

1. **Cost Optimization**: Simple queries use cheaper GPT-3.5-turbo
2. **Quality Enhancement**: Complex analysis uses GPT-4 for better reasoning
3. **System Reliability**: Proper error handling and fallbacks
4. **Maintainability**: Clean, testable architecture with good separation of concerns 