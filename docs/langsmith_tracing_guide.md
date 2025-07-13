# LangSmith Tracing Integration Guide

## Overview

This guide explains how LangSmith tracing has been integrated into the RAG agent system to provide comprehensive monitoring and debugging capabilities for all agent components and conversation flows.

## What is LangSmith Tracing?

LangSmith is LangChain's observability platform that provides:
- **Trace Visualization**: See the complete execution flow of your LangChain applications
- **Performance Monitoring**: Track latency, token usage, and costs
- **Debugging Tools**: Inspect inputs, outputs, and intermediate steps
- **Error Tracking**: Monitor and diagnose failures in your chains

## Integration Points

### 1. Configuration Setup

The LangSmith configuration is handled in `backend/config.py`:

```python
class LangSmithConfig(BaseSettings):
    """LangSmith tracing configuration"""
    tracing_enabled: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    project: str = Field(default="salesperson-copilot-rag", env="LANGCHAIN_PROJECT")
```

### 2. Environment Variables

Add these variables to your `.env` file:

```bash
# LangSmith Configuration (for monitoring - optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=salesperson-copilot-rag
```

### 3. Agent Integration

The `UnifiedToolCallingRAGAgent` initializes tracing in its constructor:

```python
def __init__(self):
    self.settings = get_settings()
    
    # Initialize LangSmith tracing
    setup_langsmith_tracing()
    
    # ... rest of initialization
```

### 4. Tracing Decorators

Key methods are decorated with `@traceable` for monitoring:

#### Agent Methods:
- `@traceable(name="unified_agent_node")` - Main agent processing node
- `@traceable(name="rag_agent_invoke")` - Main entry point for agent invocation

#### Tool Methods:
- `@traceable(name="semantic_search_tool")` - Document search functionality
- `@traceable(name="format_sources_tool")` - Source formatting
- `@traceable(name="build_context_tool")` - Context building
- `@traceable(name="get_conversation_summary_tool")` - Conversation summarization

## What Gets Traced

### 1. Conversation Flows
- **User Query Processing**: Complete flow from user input to final response
- **Tool Execution**: Each tool call with inputs and outputs
- **LLM Interactions**: All calls to ChatOpenAI with token usage
- **State Transitions**: Changes in agent state throughout the conversation

### 2. Performance Metrics
- **Latency**: Time taken for each component
- **Token Usage**: Input and output tokens for LLM calls
- **Tool Performance**: Execution time for each tool
- **End-to-End Timing**: Total conversation response time

### 3. Error Tracking
- **Tool Failures**: Exceptions during tool execution
- **LLM Errors**: API failures or rate limiting
- **Agent Errors**: Issues in agent logic or state management

## Using the Traces

### 1. Viewing Traces

1. Go to [LangSmith](https://smith.langchain.com)
2. Navigate to your project (`salesperson-copilot-rag`)
3. View traces in real-time as conversations happen

### 2. Trace Structure

Each conversation creates a trace tree:
```
RAG Agent Invoke
├── Unified Agent Node
│   ├── Semantic Search Tool
│   ├── Format Sources Tool
│   ├── Build Context Tool
│   └── LLM Call (Final Response)
└── Response Processing
```

### 3. Debugging Features

- **Input/Output Inspection**: See exact data passed between components
- **Error Analysis**: Detailed error messages and stack traces
- **Performance Bottlenecks**: Identify slow components
- **Token Usage**: Monitor costs and optimize prompts

## Testing the Integration

Run the test script to verify tracing is working:

```bash
cd backend
python -m agents.test_langsmith_tracing
```

This will test:
- Configuration loading
- Tracing decorator functionality
- Agent initialization
- Tool tracing
- Environment variable setup

## Best Practices

### 1. Naming Conventions
- Use descriptive names for `@traceable` decorators
- Follow the pattern: `{component}_{action}` (e.g., `semantic_search_tool`)

### 2. Data Privacy
- Ensure sensitive information is not logged in traces
- Use LangSmith's data filtering features if needed

### 3. Performance
- Tracing adds minimal overhead (~1-2ms per trace)
- Disable tracing in production if extreme performance is required

### 4. Error Handling
- Tracing failures should not break the agent
- The system gracefully degrades if LangSmith is unavailable

## Troubleshooting

### Common Issues:

1. **"Tracing not configured"**
   - Check that `LANGCHAIN_API_KEY` is set in your environment
   - Verify your LangSmith API key is valid

2. **Traces not appearing**
   - Ensure `LANGCHAIN_TRACING_V2=true`
   - Check your project name matches the configuration

3. **Import errors**
   - Verify `langsmith` package is installed
   - Check that all dependencies are properly imported

### Debug Commands:

```bash
# Check environment variables
python -c "from backend.config import get_settings; s = get_settings(); print(f'Tracing: {s.langsmith.tracing_enabled}, Project: {s.langsmith.project}')"

# Test configuration
python -c "from backend.config import setup_langsmith_tracing; setup_langsmith_tracing()"
```

## Integration Status

✅ **Completed:**
- Configuration setup
- Agent method tracing
- Tool method tracing
- Environment variable configuration
- Test script creation

🔄 **Next Steps:**
- Performance optimization
- Custom metadata addition
- Advanced filtering setup
- Production monitoring setup

## Related Documentation

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Tracing Guide](https://python.langchain.com/docs/langsmith/tracing)
- [Agent Configuration Guide](../config.py)
- [Tool Development Guide](./tools.py) 