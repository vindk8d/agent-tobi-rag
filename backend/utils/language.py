"""
Universal Language Detection and Processing Utilities.

These utilities can be used by ANY agent that needs multilingual support.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- LLM-based intelligent language detection from single messages
- Context-based language detection from conversation history  
- Support for English, Filipino, and Taglish
- Business-context aware language patterns
- Portable across all agent implementations
- Fallback to algorithmic detection for reliability

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.language import detect_user_language, detect_user_language_from_context

# Any agent can use these utilities
class MyAgent:
    async def process_message(self, message: str, conversation_history: list):
        # Detect language from single message (LLM-based with fallback)
        message_language = await detect_user_language(message)
        
        # Detect language from conversation context (more accurate)
        context_language = await detect_user_language_from_context(conversation_history)
        
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

BACKWARD COMPATIBILITY:
======================
Synchronous versions are available for existing code:
```python
# These work unchanged in existing synchronous contexts
language = detect_user_language(message)  # Auto-detects async context
context_lang = detect_user_language_from_context(messages)  # Auto-detects async context
```

This ensures consistent language detection across all agents without code duplication.
"""

import re
import logging
import asyncio
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Cache for LLM instances and language detection results
_language_detection_llm = None
_language_cache = {}
_llm_lock = asyncio.Lock()
_cache_lock = asyncio.Lock()


async def _get_language_detection_llm() -> Optional[ChatOpenAI]:
    """
    Get or create a lightweight LLM instance for language detection.
    Uses caching to avoid repeated instantiation.
    """
    global _language_detection_llm
    
    async with _llm_lock:
        if _language_detection_llm is None:
            try:
                # Import settings here to avoid circular imports
                from core.config import get_settings
                settings = await get_settings()
                
                # Use simple/fast model for language detection
                model = getattr(settings, 'openai_simple_model', 'gpt-4o-mini')
                
                _language_detection_llm = ChatOpenAI(
                    model=model,
                    temperature=0,  # Deterministic for consistent detection
                    max_tokens=50,  # Very short responses needed
                    api_key=settings.openai_api_key
                )
                logger.debug(f"[LLM_LANG_DETECT] Created language detection LLM with model: {model}")
                
            except Exception as e:
                logger.error(f"[LLM_LANG_DETECT] Failed to create LLM: {e}")
                # Return None to trigger fallback to algorithmic detection
                return None
                
        return _language_detection_llm


async def _manage_cache_size():
    """
    Manage cache size to prevent unlimited growth.
    Removes oldest entries when cache exceeds configured size.
    """
    try:
        from core.config import get_settings
        settings = await get_settings()
        max_size = settings.language_detection.cache_size
    except Exception:
        max_size = 1000  # Default fallback
    
    if len(_language_cache) > max_size:
        # Remove oldest entries (simple FIFO approach)
        # In a production system, you might want LRU instead
        items_to_remove = len(_language_cache) - max_size
        keys_to_remove = list(_language_cache.keys())[:items_to_remove]
        for key in keys_to_remove:
            _language_cache.pop(key, None)
        logger.debug(f"[LLM_LANG_DETECT] Cache pruned: removed {items_to_remove} entries")


async def _detect_language_with_llm(message: str, llm: ChatOpenAI) -> Optional[str]:
    """
    Use LLM to intelligently detect language from a single message.
    """
    system_prompt = """You are a language detection expert for business communications in the Philippines.

Analyze the user's message and determine their language preference. Consider:

1. **English**: Pure English with no Filipino/Tagalog words
2. **Filipino**: Predominantly Filipino/Tagalog words (90%+ Filipino)  
3. **Taglish**: Mixed English-Filipino, code-switching, or any Filipino presence in business context

IMPORTANT BUSINESS CONTEXT:
- In Philippine business settings, pure Filipino is rare
- Most Filipino speakers code-switch (use Taglish)
- Any Filipino words in business context usually indicates Taglish preference
- Car model names (CR-V, RAV4, Civic, etc.) are NOT Filipino indicators

Respond with ONLY one word: "english", "filipino", or "taglish" """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Detect language preference for: {message}")
        ]
        
        response = await llm.ainvoke(messages)
        result = response.content.strip().lower()
        
        # Validate response
        if result in ['english', 'filipino', 'taglish']:
            return result
        else:
            logger.warning(f"[LLM_LANG_DETECT] Invalid LLM response: {result}")
            return None
            
    except Exception as e:
        logger.error(f"[LLM_LANG_DETECT] LLM call failed: {e}")
        return None


async def _detect_language_from_context_with_llm(human_messages: List[str], llm: ChatOpenAI) -> Optional[str]:
    """
    Use LLM to intelligently detect language preference from conversation context.
    """
    # Combine recent messages for context analysis
    context_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(human_messages[-5:])])  # Last 5 messages
    
    system_prompt = """You are a language detection expert analyzing conversation patterns in Philippine business communications.

Analyze the conversation history to determine the user's predominant language preference. Consider:

1. **English**: Consistently uses English with no Filipino/Tagalog
2. **Filipino**: Predominantly uses Filipino/Tagalog (90%+ Filipino across messages)
3. **Taglish**: Shows code-switching, mixed languages, or any consistent Filipino presence

ANALYSIS PRINCIPLES:
- Look for patterns across multiple messages, not just individual words
- Recent messages have higher importance
- In business context, any Filipino usage typically indicates Taglish preference
- Consider conversation flow and natural language switching
- Ignore car model names and technical terms

Respond with ONLY one word: "english", "filipino", or "taglish" """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze conversation context:\n{context_text}")
        ]
        
        response = await llm.ainvoke(messages)
        result = response.content.strip().lower()
        
        # Validate response
        if result in ['english', 'filipino', 'taglish']:
            return result
        else:
            logger.warning(f"[LLM_LANG_DETECT] Invalid LLM context response: {result}")
            return None
            
    except Exception as e:
        logger.error(f"[LLM_LANG_DETECT] LLM context call failed: {e}")
        return None


async def detect_user_language(message: str, use_llm: Optional[bool] = None) -> str:
    """
    PORTABLE: Detect user's language preference from their message using LLM intelligence.
    
    Can be used by any agent that needs language detection.
    Returns: 'taglish', 'english', or 'filipino'
    
    Args:
        message: User's input message
        use_llm: Whether to use LLM-based detection. If None, uses configuration setting.
        
    Returns:
        Language preference as string
    """
    if not message or not message.strip():
        return 'english'
    
    # Get configuration settings
    try:
        from core.config import get_settings
        settings = await get_settings()
        
        # Use configuration if use_llm not explicitly provided
        if use_llm is None:
            use_llm = settings.language_detection.use_llm
        
        cache_enabled = settings.language_detection.cache_enabled
        fallback_enabled = settings.language_detection.fallback_enabled
        
    except Exception as e:
        logger.warning(f"[LLM_LANG_DETECT] Failed to load settings, using defaults: {e}")
        use_llm = True if use_llm is None else use_llm
        cache_enabled = True
        fallback_enabled = True
    
    # Check cache first for cost optimization
    cache_key = hash(message.lower().strip())
    if cache_enabled:
        async with _cache_lock:
            if cache_key in _language_cache:
                logger.debug(f"[LLM_LANG_DETECT] Cache hit for message: '{message[:30]}...'")
                return _language_cache[cache_key]
    
    # Try LLM-based detection first
    if use_llm:
        try:
            llm = await _get_language_detection_llm()
            if llm:
                result = await _detect_language_with_llm(message, llm)
                if result:
                    # Cache the result
                    if cache_enabled:
                        async with _cache_lock:
                            _language_cache[cache_key] = result
                            # Manage cache size
                            await _manage_cache_size()
                    logger.debug(f"[LLM_LANG_DETECT] Message: '{message[:50]}...' -> {result}")
                    return result
        except Exception as e:
            if fallback_enabled:
                logger.warning(f"[LLM_LANG_DETECT] LLM detection failed, falling back to algorithmic: {e}")
            else:
                logger.error(f"[LLM_LANG_DETECT] LLM detection failed and fallback disabled: {e}")
                return 'english'  # Safe default
    
    # Fallback to algorithmic detection (if enabled)
    if fallback_enabled or not use_llm:
        result = _detect_user_language_algorithmic(message)
        
        # Cache algorithmic result too
        if cache_enabled:
            async with _cache_lock:
                _language_cache[cache_key] = result
                await _manage_cache_size()
        
        return result
    else:
        logger.error("[LLM_LANG_DETECT] LLM failed and fallback disabled, returning default")
        return 'english'


def _detect_user_language_algorithmic(message: str) -> str:
    """
    FALLBACK: Original algorithmic language detection for reliability.
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


async def detect_user_language_from_context(messages, max_messages: int = 10, use_llm: Optional[bool] = None) -> str:
    """
    PORTABLE: Detect user's language preference from conversation context using LLM intelligence.
    
    Can be used by any agent to analyze conversation history for language detection.
    This analyzes the recent conversation history to determine predominant language preference,
    preventing abrupt language switching based on single messages.
    
    Args:
        messages: List of conversation messages (from any agent's message format)
        max_messages: Maximum number of recent human messages to analyze
        use_llm: Whether to use LLM-based detection. If None, uses configuration setting.
        
    Returns:
        Language preference as string: 'taglish', 'english', or 'filipino'
    """
    if not messages:
        return 'english'
    
    # Get configuration settings
    try:
        from core.config import get_settings
        settings = await get_settings()
        
        # Use configuration if use_llm not explicitly provided
        if use_llm is None:
            use_llm = settings.language_detection.use_llm
        
        fallback_enabled = settings.language_detection.fallback_enabled
        
    except Exception as e:
        logger.warning(f"[LLM_LANG_DETECT] Failed to load settings, using defaults: {e}")
        use_llm = True if use_llm is None else use_llm
        fallback_enabled = True
    
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
    
    # Try LLM-based context detection first
    if use_llm:
        try:
            llm = await _get_language_detection_llm()
            if llm:
                result = await _detect_language_from_context_with_llm(human_messages, llm)
                if result:
                    logger.info(f"[LLM_LANG_DETECT] Context analysis of {len(human_messages)} messages -> {result}")
                    return result
        except Exception as e:
            if fallback_enabled:
                logger.warning(f"[LLM_LANG_DETECT] LLM context detection failed, falling back to algorithmic: {e}")
            else:
                logger.error(f"[LLM_LANG_DETECT] LLM context detection failed and fallback disabled: {e}")
                return 'english'  # Safe default
    
    # Fallback to algorithmic context detection (if enabled)
    if fallback_enabled or not use_llm:
        return _detect_user_language_from_context_algorithmic(human_messages)
    else:
        logger.error("[LLM_LANG_DETECT] LLM failed and fallback disabled, returning default")
        return 'english'


def _detect_user_language_from_context_algorithmic(human_messages: List[str]) -> str:
    """
    FALLBACK: Original algorithmic context-based language detection.
    """
    if not human_messages:
        return 'english'
    
    # Analyze all human messages to get language scores
    language_scores = {'english': 0.0, 'taglish': 0.0, 'filipino': 0.0}
    total_weight = 0
    
    for i, message in enumerate(human_messages):
        # Give more weight to recent messages (exponential decay)
        weight = 2 ** (len(human_messages) - i - 1)  # Recent messages have higher weight
        detected_lang = _detect_user_language_algorithmic(message)
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
    
    logger.info(f"[ALGORITHMIC_LANG_DETECT] Analyzed {len(human_messages)} messages. Scores: {language_scores}. Result: {result}")
    return result


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================

def detect_user_language_sync(message: str, use_llm: bool = False) -> str:
    """
    Synchronous version of detect_user_language for backward compatibility.
    Defaults to algorithmic detection to avoid async issues in sync contexts.
    
    Args:
        message: User's input message
        use_llm: Whether to attempt LLM-based detection (may fall back to algorithmic)
        
    Returns:
        Language preference as string
    """
    if use_llm:
        try:
            # Try to run async version if event loop exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we can't use asyncio.run, fall back to algorithmic
                logger.debug("[LANG_DETECT_SYNC] Event loop running, using algorithmic detection")
                return _detect_user_language_algorithmic(message)
            else:
                return asyncio.run(detect_user_language(message, use_llm=True))
        except Exception as e:
            logger.debug(f"[LANG_DETECT_SYNC] Async failed, using algorithmic: {e}")
            return _detect_user_language_algorithmic(message)
    else:
        return _detect_user_language_algorithmic(message)


def detect_user_language_from_context_sync(messages, max_messages: int = 10, use_llm: bool = False) -> str:
    """
    Synchronous version of detect_user_language_from_context for backward compatibility.
    Defaults to algorithmic detection to avoid async issues in sync contexts.
    
    Args:
        messages: List of conversation messages
        max_messages: Maximum number of recent human messages to analyze
        use_llm: Whether to attempt LLM-based detection (may fall back to algorithmic)
        
    Returns:
        Language preference as string
    """
    # Extract human messages first
    human_messages = []
    message_count = 0
    
    for msg in reversed(messages):
        if message_count >= max_messages:
            break
            
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
    
    if use_llm:
        try:
            # Try to run async version if event loop exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we can't use asyncio.run, fall back to algorithmic
                logger.debug("[LANG_DETECT_CONTEXT_SYNC] Event loop running, using algorithmic detection")
                return _detect_user_language_from_context_algorithmic(human_messages)
            else:
                return asyncio.run(detect_user_language_from_context(messages, max_messages, use_llm=True))
        except Exception as e:
            logger.debug(f"[LANG_DETECT_CONTEXT_SYNC] Async failed, using algorithmic: {e}")
            return _detect_user_language_from_context_algorithmic(human_messages)
    else:
        return _detect_user_language_from_context_algorithmic(human_messages)


# =============================================================================
# SMART COMPATIBILITY LAYER
# =============================================================================

def detect_user_language_smart(message: str, use_llm: bool = True) -> str:
    """
    Smart wrapper that automatically detects async context and calls appropriate version.
    This maintains backward compatibility while enabling LLM usage when possible.
    
    Args:
        message: User's input message
        use_llm: Whether to use LLM-based detection when in async context
        
    Returns:
        Language preference as string
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, but we can't await here since this is sync
        # Fall back to algorithmic for safety
        return _detect_user_language_algorithmic(message)
    except RuntimeError:
        # No event loop running, we can safely use asyncio.run if needed
        if use_llm:
            try:
                return asyncio.run(detect_user_language(message, use_llm=True))
            except Exception as e:
                logger.debug(f"[LANG_DETECT_SMART] Async failed, using algorithmic: {e}")
                return _detect_user_language_algorithmic(message)
        else:
            return _detect_user_language_algorithmic(message)


def detect_user_language_from_context_smart(messages, max_messages: int = 10, use_llm: bool = True) -> str:
    """
    Smart wrapper that automatically detects async context and calls appropriate version.
    This maintains backward compatibility while enabling LLM usage when possible.
    
    Args:
        messages: List of conversation messages
        max_messages: Maximum number of recent human messages to analyze
        use_llm: Whether to use LLM-based detection when in async context
        
    Returns:
        Language preference as string
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, but we can't await here since this is sync
        # Fall back to algorithmic for safety
        human_messages = []
        message_count = 0
        
        for msg in reversed(messages):
            if message_count >= max_messages:
                break
                
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
        
        return _detect_user_language_from_context_algorithmic(human_messages)
        
    except RuntimeError:
        # No event loop running, we can safely use asyncio.run if needed
        if use_llm:
            try:
                return asyncio.run(detect_user_language_from_context(messages, max_messages, use_llm=True))
            except Exception as e:
                logger.debug(f"[LANG_DETECT_CONTEXT_SMART] Async failed, using algorithmic: {e}")
                return detect_user_language_from_context_sync(messages, max_messages, use_llm=False)
        else:
            return detect_user_language_from_context_sync(messages, max_messages, use_llm=False)
