"""
Utility functions for the backend.
"""

import json
from datetime import datetime
from typing import Any


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
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


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize objects to JSON, handling datetime objects."""
    try:
        return json.dumps(obj, cls=DateTimeEncoder)
    except TypeError:
        # Fallback: recursively convert datetime objects to strings
        converted_obj = convert_datetime_to_iso(obj)
        return json.dumps(converted_obj, cls=DateTimeEncoder) 