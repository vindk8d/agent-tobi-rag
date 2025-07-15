# Moving Context Window Memory Management

## Overview

The Moving Context Window approach is a memory management strategy that appends previous conversation messages to the current message content, ensuring the agent always has access to the full conversation context. When the conversation reaches a specified message limit (default: 12 messages), the system generates a summary and resets the conversation counter while preserving context.

## Key Features

### 1. **Context Appending**
- Previous messages are appended to the current user message
- The agent receives the full conversation context in a single message
- Context is clearly structured with labels like "Previous conversation:" and "Current request:"

### 2. **Automatic Summarization and Reset**
- When messages reach the limit (12 by default), the system:
  - Generates a comprehensive summary of the entire conversation
  - Resets the message counter to 1
  - Appends both the summary and recent context to the current message

### 3. **LangGraph Persistence Integration**
- Works seamlessly with LangGraph's checkpointing system
- Maintains conversation state across sessions
- Follows LangGraph best practices for state management

## How It Works

### Phase 1: Context Appending (Messages 1-11)
```
Message 1: "Hello, can you help me with Python?"
→ No previous context, message unchanged

Message 2: "I want to learn about list comprehensions"
→ Enhanced message:
   Previous conversation:
   User: Hello, can you help me with Python?
   Assistant: Of course! I'd be happy to help...
   
   Current request: I want to learn about list comprehensions
```

### Phase 2: Summary and Reset (Message 12+)
```
Message 12: "What about decorators?"
→ Enhanced message:
   Previous conversation summary:
   The user has been learning Python, specifically about list comprehensions...
   
   Recent conversation context:
   User: Can you give me an example?
   Assistant: Sure! Here's an example...
   
   Current request: What about decorators?
```

## Configuration

```python
from backend.agents.memory_manager import ConversationMemoryManager

# Default configuration (12 messages)
memory_manager = ConversationMemoryManager()

# Custom configuration
memory_manager = ConversationMemoryManager(max_messages=8)
```

## Usage Example

```python
from langchain_core.messages import HumanMessage, AIMessage
from backend.agents.memory_manager import memory_manager

# Sample conversation
messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help you?"),
    HumanMessage(content="Tell me about Python"),
]

# Get effective context
context = await memory_manager.get_effective_context(messages, None)
processed_messages = context["messages"]

# The last message now contains appended context
enhanced_message = processed_messages[-1]
print(enhanced_message.content)
```

## Benefits

### 1. **Continuous Context**
- Agent always has access to the full conversation history
- No loss of context during the conversation
- Better understanding of user intent and conversation flow

### 2. **Efficient Memory Usage**
- Periodic summarization prevents unlimited memory growth
- Structured context appending is more efficient than maintaining large message lists
- Automatic cleanup with reset mechanism

### 3. **LangGraph Compatibility**
- Follows LangGraph persistence best practices
- Works with checkpointing and state management
- Maintains thread-based conversation tracking

## Comparison with Previous Sliding Window

| Feature | Sliding Window | Moving Context Window |
|---------|----------------|----------------------|
| **Context Preservation** | Limited to N recent messages | Full conversation history |
| **Memory Usage** | Fixed window size | Dynamic with periodic reset |
| **Context Access** | Separate messages | Appended to current message |
| **Summarization** | Periodic (every N messages) | On reset (at message limit) |
| **Agent Context** | Partial conversation | Complete conversation |

## Migration Guide

The new system is backward compatible. Existing code will continue to work, but you can migrate to the new approach:

### Before (Sliding Window)
```python
# Old approach - used sliding window
effective_context = await memory_manager.get_effective_context(messages, summary)
windowed_messages = effective_context["messages"]  # Limited to window size
```

### After (Moving Context Window)
```python
# New approach - uses moving context window
effective_context = await memory_manager.get_effective_context(messages, summary)
processed_messages = effective_context["messages"]  # Full context appended
was_reset = effective_context.get("was_reset", False)  # Check if conversation reset
```

## Testing

Run the test script to see the moving context window in action:

```bash
python tests/test_moving_context_window.py
```

This will demonstrate:
- How context is appended to messages
- When conversations are summarized and reset
- The structure of enhanced messages

## Advanced Configuration

### Custom Summary Generation
```python
# Override summary generation for specific use cases
class CustomMemoryManager(ConversationMemoryManager):
    async def generate_conversation_summary(self, messages):
        # Custom summary logic
        return "Custom summary based on specific criteria"
```

### Integration with RAG Agent
The moving context window integrates seamlessly with the RAG agent:

```python
# In the RAG agent, the memory preparation node handles context automatically
async def _memory_preparation_node(self, state, config):
    effective_context = await self.memory_manager.get_effective_context(
        state["messages"], 
        state.get("conversation_summary")
    )
    return {
        "messages": effective_context["messages"],  # Enhanced with context
        "conversation_summary": effective_context["conversation_summary"]
    }
```

## Best Practices

1. **Choose Appropriate Message Limit**: 12 messages is a good default, but adjust based on your use case
2. **Monitor Token Usage**: Context appending increases token usage, monitor for cost implications
3. **Test with Real Conversations**: Use the test script to understand behavior with your specific use cases
4. **Consider Domain-Specific Summaries**: Customize summarization for better domain-specific context preservation

## Troubleshooting

### Common Issues

1. **Large Token Usage**: If context appending creates very long messages, consider reducing `max_messages`
2. **Summary Quality**: Ensure your OpenAI API key is properly configured for summary generation
3. **Context Overflow**: Very long conversations may need custom chunking strategies

### Debugging

Enable debug logging to see the context window in action:

```python
import logging
logging.getLogger("backend.agents.memory_manager").setLevel(logging.DEBUG)
```

This will show:
- When context is appended
- When conversations are reset
- Summary generation details
- Message processing statistics 