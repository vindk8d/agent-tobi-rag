"""
Serialization utility functions for the backend.
Enhanced for LangGraph checkpointer compatibility.
"""

import json
from datetime import datetime, date
from typing import Any, Dict, List, Union
from uuid import UUID


# NOTE: AgentStateEncoder removed - LangGraph's JsonPlusSerializer handles our 
# simplified AgentState natively. Our state only contains:
# - strings (conversation_id, user_id, etc.) 
# - Optional[str] fields
# - messages (handled by add_messages annotation)
# - Dict[str, Any] (hitl_context with basic JSON data)
# All of these are natively supported by LangGraph's serialization.


class DateTimeEncoder(json.JSONEncoder):
    """Simple JSON encoder to handle datetime objects (backward compatibility)."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def convert_datetime_to_iso(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_datetime_to_iso(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_iso(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_datetime_to_iso(item) for item in obj)
    else:
        return obj


# NOTE: Custom AgentState serialization functions removed.
# LangGraph's JsonPlusSerializer handles our simplified AgentState automatically.
# No custom serialization needed for basic JSON types (strings, dicts, lists).


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize objects to JSON, handling datetime objects (backward compatibility)."""
    try:
        return json.dumps(obj, cls=DateTimeEncoder)
    except TypeError:
        # Fallback: recursively convert datetime objects to strings
        converted_obj = convert_datetime_to_iso(obj)
        return json.dumps(converted_obj, cls=DateTimeEncoder)


# NOTE: AgentState validation removed - LangGraph handles this automatically.
# Our simplified state (strings, Optional[str], Dict[str, Any]) is natively JSON-serializable. 