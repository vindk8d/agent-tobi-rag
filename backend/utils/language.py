"""
Universal Language Detection and Processing Utilities.

These utilities can be used by ANY agent that needs multilingual support.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- Language detection from single messages
- Context-based language detection from conversation history
- Support for English, Filipino, and Taglish
- Business-context aware language patterns
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.language import detect_user_language, detect_user_language_from_context

# Any agent can use these utilities
class MyAgent:
    async def process_message(self, message: str, conversation_history: list):
        # Detect language from single message
        message_language = detect_user_language(message)
        
        # Detect language from conversation context (more accurate)
        context_language = detect_user_language_from_context(conversation_history)
        
        # Use detected language for response formatting
        if context_language == 'taglish':
            # Format response in Taglish
            pass
        elif context_language == 'filipino':
            # Format response in Filipino
            pass
        else:
            # Format response in English
            pass
```

This ensures consistent language detection across all agents without code duplication.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def detect_user_language(message: str) -> str:
    """
    PORTABLE: Detect user's language preference from their message.
    
    Can be used by any agent that needs language detection.
    Returns: 'taglish', 'english', or 'filipino'
    
    Args:
        message: User's input message
        
    Returns:
        Language preference as string
    """
    if not message:
        return 'english'
    
    message_lower = message.lower()
    
    # Filipino words commonly used in Taglish
    filipino_indicators = [
        # Common Filipino words
        'ano', 'saan', 'kelan', 'bakit', 'paano', 'sino', 'alin', 'ilan',
        'opo', 'hindi', 'oo', 'tama', 'mali', 'pwede', 'kaya', 'siguro',
        'talaga', 'ganun', 'ganyan', 'dito', 'doon', 'diyan', 'naman',
        'lang', 'nalang', 'na lang', 'po', 'ho', 'kuya', 'ate', 'tito', 'tita',
        'salamat', 'pasensya', 'sige', 'okay lang', 'ayos lang', 'walang anuman',
        'kumusta', 'musta', 'kamusta', 'baka', 'kasi', 'kaya nga', 'eh',
        'pala', 'daw', 'raw', 'diba', 'di ba', 'yung', 'yun', 'yan',
        'mahal', 'mura', 'presyo', 'bayad', 'bili', 'benta',
        'gusto', 'ayaw', 'trip', 'type', 'bet', 'solid', 'sulit',
        'magkano', 'ilang', 'marami', 'konti', 'sobra', 'kulang',
        'may', 'wala', 'meron', 'walang', 'mayron'
    ]
    
    # Count Filipino indicators
    filipino_count = sum(1 for word in filipino_indicators if word in message_lower)
    
    # Exclude common car model acronyms that might trigger false positives
    car_models = ['crv', 'cr-v', 'suv', 'rav4', 'civic', 'camry', 'f-150']
    car_model_matches = sum(1 for model in car_models if model in message_lower)
    
    # Adjust Filipino count to account for car models
    if car_model_matches > 0 and filipino_count <= car_model_matches:
        # If Filipino indicators are only car models, don't count them as strong Filipino signals
        pass
    
    # Check for mixed language patterns (Taglish indicators)
    taglish_patterns = [
        r'\b(yung|yun)\s+(the|a|an)\b',  # "yung the"
        r'\b(mga)\s+[a-z]+(?:s|es|ies)\b',  # "mga" + English plural
        r'\b(may|meron)\s+[a-z]+(?:ing|ed|er)\b',  # "may" + English verb forms
        r'\b(hindi|di)\s+(naman|kaya|pwede)\b',  # Filipino negation patterns
        r'\b(sana|hope)\s+(ma|na)\w+\b',  # Mixed hope expressions
        r'\b(send|follow-up|about)\s+(ka|kay|sa)\b',  # English verbs + Filipino particles
    ]
    
    taglish_pattern_matches = sum(1 for pattern in taglish_patterns if re.search(pattern, message_lower))
    
    # Check for English dominance
    words = message_lower.split()
    total_words = len(words)
    
    if total_words == 0:
        return 'english'
    
    filipino_ratio = filipino_count / total_words
    
    # Decision logic for business context
    # In business settings, pure Filipino is rare - most Filipino speakers code-switch
    if taglish_pattern_matches > 0:
        return 'taglish'
    elif filipino_count > 0:
        # Check if it's predominantly Filipino (90%+ Filipino words)
        if filipino_ratio >= 0.9 and total_words >= 5:
            return 'filipino'
        else:
            # Business context: any Filipino presence suggests Taglish tendency
            return 'taglish'
    else:
        return 'english'


def detect_user_language_from_context(messages, max_messages: int = 10) -> str:
    """
    PORTABLE: Detect user's language preference from conversation context (multiple messages).
    
    Can be used by any agent to analyze conversation history for language detection.
    This analyzes the recent conversation history to determine predominant language preference,
    preventing abrupt language switching based on single messages.
    
    Args:
        messages: List of conversation messages (from any agent's message format)
        max_messages: Maximum number of recent human messages to analyze
        
    Returns:
        Language preference as string: 'taglish', 'english', or 'filipino'
    """
    if not messages:
        return 'english'
    
    # Extract human messages from the conversation context
    human_messages = []
    message_count = 0
    
    # Look through messages in reverse order (most recent first)
    for msg in reversed(messages):
        if message_count >= max_messages:
            break
            
        # Extract content from different message formats (portable across agents)
        content = ""
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
        elif isinstance(msg, dict) and msg.get("role") == "human":
            content = msg.get("content", "")
        elif hasattr(msg, "role") and getattr(msg, "role", None) == "human":
            content = getattr(msg, "content", "")
        
        if content and content.strip():
            human_messages.append(content.strip())
            message_count += 1
    
    if not human_messages:
        return 'english'
    
    # Analyze all human messages to get language scores
    language_scores = {'english': 0.0, 'taglish': 0.0, 'filipino': 0.0}
    total_weight = 0
    
    for i, message in enumerate(human_messages):
        # Give more weight to recent messages (exponential decay)
        weight = 2 ** (len(human_messages) - i - 1)  # Recent messages have higher weight
        detected_lang = detect_user_language(message)
        language_scores[detected_lang] += weight
        total_weight += weight
    
    # Normalize scores
    if total_weight > 0:
        for lang in language_scores:
            language_scores[lang] /= total_weight
    
    # Determine predominant language with some smoothing
    # If there's a clear winner (>50%), use it
    # Otherwise, prefer taglish as it's the most flexible for business context
    max_score = max(language_scores.values()) if language_scores else 0
    predominant_lang = max(language_scores.keys(), key=lambda k: language_scores[k]) if language_scores else 'taglish'
    
    # Add some stability - don't switch unless there's a clear preference
    if max_score >= 0.5:
        result = predominant_lang
    elif language_scores['taglish'] >= 0.3:
        # If taglish has decent presence, prefer it for business flexibility
        result = 'taglish'
    elif language_scores['english'] > language_scores['filipino']:
        result = 'english'
    else:
        result = 'taglish'  # Default to taglish for business context
    
    logger.info(f"[PORTABLE_LANG_DETECT] Analyzed {len(human_messages)} messages. Scores: {language_scores}. Result: {result}")
    return result
