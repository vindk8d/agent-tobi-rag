"""
Universal Source Processing Utilities.

These utilities can be used by ANY agent that needs document source extraction.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- Source extraction from tool call results in messages
- Support for multiple RAG tool formats (LCEL, simple_rag, etc.)
- Source deduplication and formatting
- Similarity score handling
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.source_processing import extract_sources_from_messages, format_sources_for_display

class MyAgent:
    async def process_rag_response(self, messages: List):
        # Extract sources from tool responses - works with any agent
        sources = extract_sources_from_messages(messages)
        
        # Format sources for user display
        formatted_sources = format_sources_for_display(sources)
        
        return {
            "sources": sources,
            "formatted_sources": formatted_sources
        }
```

This ensures consistent source processing across all agents without code duplication.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


def extract_sources_from_messages(messages: List) -> List[Dict[str, Any]]:
    """
    PORTABLE: Extract sources from tool call results in messages.
    
    Can be used by any agent to extract document sources from RAG tool responses.
    Supports multiple RAG tool formats and provides consistent source formatting.
    
    Args:
        messages: List of messages (any message format)
        
    Returns:
        List of source dictionaries with 'source' and 'similarity' keys
    """
    sources = []
    
    try:
        for message in messages:
            # Extract sources from LCEL-based RAG tools
            if isinstance(message, ToolMessage) and message.name in ["simple_rag"]:
                try:
                    # LCEL tools return formatted text with source information
                    # Extract sources from the formatted response
                    content = message.content
                    
                    if "Sources (" in content:
                        # Parse sources from lcel_rag output format
                        sources_section = (
                            content.split("Sources (")[1].split("):")[1]
                            if "):" in content
                            else ""
                        )
                        for line in sources_section.split("\n"):
                            if line.strip().startswith("•"):
                                source = line.strip()[1:].strip()
                                if source:
                                    sources.append({
                                        "source": source,
                                        "similarity": 0.0,  # LCEL tools don't expose individual similarity scores
                                    })
                                    
                    elif "**Retrieved Documents (" in content:
                        # Parse sources from lcel_retrieval output format
                        lines = content.split("\n")
                        for line in lines:
                            if line.startswith("Source: "):
                                source = line.replace("Source: ", "").strip()
                                if source:
                                    sources.append({
                                        "source": source,
                                        "similarity": 0.0
                                    })
                                    
                except Exception as e:
                    logger.debug(f"[PORTABLE_SOURCE_EXTRACTION] Error extracting sources from message: {e}")
                    continue
            
            # Extract sources from other RAG tool formats
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                try:
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, 'name') and 'rag' in tool_call.name.lower():
                            # Extract sources from tool call results
                            if hasattr(tool_call, 'args') and isinstance(tool_call.args, dict):
                                if 'sources' in tool_call.args:
                                    tool_sources = tool_call.args['sources']
                                    if isinstance(tool_sources, list):
                                        sources.extend(tool_sources)
                except Exception as e:
                    logger.debug(f"[PORTABLE_SOURCE_EXTRACTION] Error extracting from tool calls: {e}")
                    continue
            
            # Extract sources from message metadata
            elif hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                try:
                    metadata = message.additional_kwargs
                    if 'sources' in metadata and isinstance(metadata['sources'], list):
                        sources.extend(metadata['sources'])
                except Exception as e:
                    logger.debug(f"[PORTABLE_SOURCE_EXTRACTION] Error extracting from metadata: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"[PORTABLE_SOURCE_EXTRACTION] Error in source extraction: {e}")
    
    # Deduplicate sources based on source name
    unique_sources = []
    seen_sources = set()
    
    for source in sources:
        if isinstance(source, dict) and 'source' in source:
            source_name = source['source']
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                unique_sources.append(source)
        elif isinstance(source, str):
            # Handle string sources
            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append({
                    "source": source,
                    "similarity": 0.0
                })
    
    logger.info(f"[PORTABLE_SOURCE_EXTRACTION] Extracted {len(unique_sources)} unique sources from {len(messages)} messages")
    return unique_sources


def format_sources_for_display(sources: List[Dict[str, Any]], max_sources: int = 5) -> List[Dict[str, Any]]:
    """
    PORTABLE: Format sources for user display with consistent formatting.
    
    Can be used by any agent to format extracted sources for user presentation.
    
    Args:
        sources: List of source dictionaries
        max_sources: Maximum number of sources to return
        
    Returns:
        List of formatted source dictionaries ready for display
    """
    try:
        if not sources:
            return []
        
        # Sort sources by similarity if available
        sorted_sources = sorted(
            sources, 
            key=lambda x: x.get('similarity', 0.0), 
            reverse=True
        )
        
        # Limit to max_sources
        limited_sources = sorted_sources[:max_sources]
        
        # Format for display
        formatted_sources = []
        for i, source in enumerate(limited_sources, 1):
            formatted_source = {
                "id": i,
                "source": source.get("source", "Unknown source"),
                "similarity": source.get("similarity", 0.0),
                "display_name": extract_display_name_from_source(source.get("source", "")),
                "type": determine_source_type(source.get("source", ""))
            }
            formatted_sources.append(formatted_source)
        
        logger.info(f"[PORTABLE_SOURCE_FORMATTING] Formatted {len(formatted_sources)} sources for display")
        return formatted_sources
        
    except Exception as e:
        logger.error(f"[PORTABLE_SOURCE_FORMATTING] Error formatting sources: {e}")
        return []


def extract_display_name_from_source(source: str) -> str:
    """
    PORTABLE: Extract a user-friendly display name from a source path or URL.
    
    Can be used by any agent to create readable source names.
    
    Args:
        source: Source path, URL, or identifier
        
    Returns:
        User-friendly display name
    """
    try:
        if not source:
            return "Unknown source"
        
        # Handle file paths
        if "/" in source:
            # Extract filename from path
            filename = source.split("/")[-1]
            # Remove file extension for display
            if "." in filename:
                return filename.rsplit(".", 1)[0]
            return filename
        
        # Handle URLs
        if source.startswith(("http://", "https://")):
            # Extract domain or meaningful part
            try:
                from urllib.parse import urlparse
                parsed = urlparse(source)
                if parsed.path:
                    path_parts = parsed.path.strip("/").split("/")
                    if path_parts and path_parts[-1]:
                        return path_parts[-1]
                return parsed.netloc or source
            except:
                return source
        
        # Handle document identifiers
        if source.startswith("doc_") or source.startswith("id_"):
            return source.replace("_", " ").title()
        
        # Default: return source as-is but cleaned up
        return source.strip()
        
    except Exception as e:
        logger.debug(f"[PORTABLE_SOURCE_DISPLAY] Error extracting display name: {e}")
        return source or "Unknown source"


def determine_source_type(source: str) -> str:
    """
    PORTABLE: Determine the type of source based on its format.
    
    Can be used by any agent to categorize sources for display or processing.
    
    Args:
        source: Source path, URL, or identifier
        
    Returns:
        Source type string (document, webpage, database, etc.)
    """
    try:
        if not source:
            return "unknown"
        
        source_lower = source.lower()
        
        # Web sources
        if source_lower.startswith(("http://", "https://")):
            return "webpage"
        
        # Document files
        if any(ext in source_lower for ext in [".pdf", ".doc", ".docx", ".txt", ".md"]):
            return "document"
        
        # Spreadsheets
        if any(ext in source_lower for ext in [".xls", ".xlsx", ".csv"]):
            return "spreadsheet"
        
        # Presentations
        if any(ext in source_lower for ext in [".ppt", ".pptx"]):
            return "presentation"
        
        # Database or API sources
        if any(keyword in source_lower for keyword in ["api", "db", "database", "table"]):
            return "database"
        
        # Knowledge base
        if any(keyword in source_lower for keyword in ["kb", "knowledge", "wiki"]):
            return "knowledge_base"
        
        # Default
        return "document"
        
    except Exception as e:
        logger.debug(f"[PORTABLE_SOURCE_TYPE] Error determining source type: {e}")
        return "unknown"


def filter_sources_by_relevance(sources: List[Dict[str, Any]], min_similarity: float = 0.1) -> List[Dict[str, Any]]:
    """
    PORTABLE: Filter sources by relevance threshold.
    
    Can be used by any agent to filter out low-relevance sources.
    
    Args:
        sources: List of source dictionaries
        min_similarity: Minimum similarity threshold
        
    Returns:
        List of filtered sources above the threshold
    """
    try:
        if not sources:
            return []
        
        filtered_sources = [
            source for source in sources
            if source.get("similarity", 0.0) >= min_similarity
        ]
        
        logger.info(f"[PORTABLE_SOURCE_FILTERING] Filtered {len(sources)} sources to {len(filtered_sources)} above threshold {min_similarity}")
        return filtered_sources
        
    except Exception as e:
        logger.error(f"[PORTABLE_SOURCE_FILTERING] Error filtering sources: {e}")
        return sources  # Return original sources on error


def merge_duplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    PORTABLE: Merge duplicate sources and keep the highest similarity score.
    
    Can be used by any agent to clean up duplicate sources from multiple tools.
    
    Args:
        sources: List of source dictionaries that may contain duplicates
        
    Returns:
        List of deduplicated sources with highest similarity scores
    """
    try:
        if not sources:
            return []
        
        # Group sources by source name
        source_groups = {}
        for source in sources:
            source_name = source.get("source", "")
            if source_name:
                if source_name not in source_groups:
                    source_groups[source_name] = []
                source_groups[source_name].append(source)
        
        # Keep the source with highest similarity for each group
        merged_sources = []
        for source_name, group in source_groups.items():
            best_source = max(group, key=lambda x: x.get("similarity", 0.0))
            merged_sources.append(best_source)
        
        logger.info(f"[PORTABLE_SOURCE_MERGING] Merged {len(sources)} sources to {len(merged_sources)} unique sources")
        return merged_sources
        
    except Exception as e:
        logger.error(f"[PORTABLE_SOURCE_MERGING] Error merging sources: {e}")
        return sources  # Return original sources on error
