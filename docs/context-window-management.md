# Context Window Management

## Overview

This document explains the context window management system implemented to prevent token overflow and rate limiting issues in the RAG agent.

## Problems Addressed

1. **Context Length Exceeded**: Messages totaling 19,762 tokens exceeded the 16,385 token limit for gpt-3.5-turbo
2. **Rate Limiting**: GPT-4 hitting 429 errors due to high token usage from large SQL results (42,644 bytes)
3. **No Message History Trimming**: Agent preserved entire conversation history without truncation
4. **Large SQL Results**: Raw SQL results being sent directly to LLM for formatting

## Solutions Implemented

### 1. Context Window Manager (`backend/agents/context_manager.py`)

A comprehensive context management system that:
- **Counts tokens accurately** using tiktoken for precise token calculation
- **Trims message history** intelligently while preserving conversation flow
- **Respects model limits** with specific buffers for each model:
  - `gpt-3.5-turbo`: 14K tokens (16K total - 2K buffer)
  - `gpt-4`: 6K tokens (8K total - 2K buffer)
  - `gpt-4o/gpt-4o-mini`: 126K tokens (128K total - 2K buffer)

#### Key Features:
- **Message count limiting**: Respects `MEMORY_MAX_MESSAGES=12` configuration
- **Token-based trimming**: Removes oldest messages when token limit approached
- **System message preservation**: Always keeps system prompts intact
- **Conversation continuity**: Maintains recent context for coherent responses

### 2. SQL Result Size Limiting (`backend/agents/tools.py`)

Enhanced SQL result processing with aggressive size limits:

```python
# Configuration (from env variables)
OPENAI_MAX_RESULT_SIZE_SIMPLE_MODEL=20000  # Down from 50K
OPENAI_FORCE_COMPLEX_MODEL_SIZE=10000      # Down from 20K  
OPENAI_MAX_DISPLAY_LENGTH=5000             # Down from 10K
```

#### Fallback Strategy:
1. **Size Check**: Results >20KB trigger immediate fallback (no LLM formatting)
2. **Truncation**: Large results truncated to 5KB with clear indication
3. **User Guidance**: Provides tips for getting smaller, more targeted results

### 3. Rate Limit Handling

Implemented exponential backoff retry logic for OpenAI API calls:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((openai.RateLimitError,))
)
```

#### Error Recovery:
- **429 Rate Limit**: Automatic retry with backoff, fallback to non-LLM formatting
- **Context Length Exceeded**: Automatic fallback to truncated display
- **General API Errors**: Graceful degradation with informative messages

### 4. Dynamic Model Selection

Smart model selection based on:
- **Query complexity**: Simple queries → gpt-3.5-turbo, Complex queries → gpt-4
- **Result size**: Large results (>10KB) → Force GPT-4 for better context handling
- **Tool usage**: Multiple tools → GPT-4 for complex reasoning

## Configuration

### Environment Variables
```bash
# Memory Management
MEMORY_MAX_MESSAGES=12              # Limit conversation history
MEMORY_SUMMARY_INTERVAL=10          # Summary creation interval

# Token Limits  
OPENAI_MAX_TOKENS=1500              # Max response tokens
OPENAI_MAX_RESULT_SIZE_SIMPLE_MODEL=20000
OPENAI_FORCE_COMPLEX_MODEL_SIZE=10000
OPENAI_MAX_DISPLAY_LENGTH=5000

# Model Selection
OPENAI_SIMPLE_MODEL=gpt-3.5-turbo
OPENAI_COMPLEX_MODEL=gpt-4
```

### Usage in Agent
The agent automatically applies context management:

```python
# Automatic context trimming
trimmed_messages, trim_stats = await context_manager.trim_messages_for_context(
    messages=original_messages,
    model=selected_model,
    system_prompt=system_prompt,
    max_messages=settings.memory.max_messages
)
```

## Monitoring

### Logging
- **Context trimming**: Logs when messages are removed and token savings
- **Model selection**: Logs which model is selected and why  
- **Size limits**: Logs when SQL results exceed limits
- **Rate limits**: Logs rate limit encounters and retry attempts

### Statistics
Trim operations provide detailed stats:
- Original message count and token count
- Final message count and token count  
- Number of messages trimmed
- Tokens saved through trimming

## Benefits

1. **Prevents Context Overflow**: No more "context_length_exceeded" errors
2. **Reduces Rate Limiting**: Smaller payloads mean fewer 429 errors
3. **Maintains Performance**: Fast responses through intelligent model selection
4. **Preserves UX**: Users get results even when systems hit limits
5. **Cost Optimization**: Smaller contexts = lower API costs

## Best Practices

1. **Keep conversations focused**: Longer conversations will be automatically trimmed
2. **Be specific with queries**: Reduces SQL result sizes and improves response quality
3. **Monitor logs**: Watch for frequent trimming as a sign of overly long conversations
4. **Tune limits**: Adjust environment variables based on usage patterns

## Future Improvements

1. **Semantic trimming**: Remove less important messages based on relevance
2. **Conversation summarization**: Replace old messages with summaries
3. **Progressive disclosure**: Show abbreviated results with option to expand
4. **Caching**: Cache formatted results to avoid re-processing 