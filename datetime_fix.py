#!/usr/bin/env python3
"""
Patch script to fix datetime serialization issues in memory.py
Run this script from the project root to apply the fix.
"""

import fileinput
import sys
import os

def apply_datetime_fix():
    """Apply the datetime serialization fix to memory.py"""
    
    memory_file = "backend/agents/memory.py"
    
    if not os.path.exists(memory_file):
        print(f"Error: {memory_file} not found")
        return False
    
    print(f"Applying datetime serialization fix to {memory_file}")
    
    # Create backup
    backup_file = memory_file + ".backup"
    with open(memory_file, 'r') as original:
        with open(backup_file, 'w') as backup:
            backup.write(original.read())
    
    print(f"Created backup: {backup_file}")
    
    # Read the file
    with open(memory_file, 'r') as f:
        content = f.read()
    
    # Add the DateTimeEncoder class after imports
    datetime_encoder = '''
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def convert_datetime_to_iso(obj):
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

'''
    
    # Find the location after imports to insert the encoder
    import_end = content.find('\nlogger = logging.getLogger(__name__)')
    if import_end == -1:
        # Fallback: find after datetime import
        import_end = content.find('from datetime import datetime, timedelta')
        if import_end != -1:
            import_end = content.find('\n', import_end) + 1
    
    if import_end != -1:
        content = content[:import_end] + datetime_encoder + content[import_end:]
    
    # Fix the JSON serialization line
    old_json_line = "value_json = json.dumps(value) if not isinstance(value, str) else value"
    new_json_code = '''if not isinstance(value, str):
                        try:
                            value_json = json.dumps(value, cls=DateTimeEncoder)
                        except TypeError as e:
                            logger.warning(f"JSON serialization failed with DateTimeEncoder, using fallback: {e}")
                            # Fallback: recursively convert datetime objects to strings
                            converted_value = convert_datetime_to_iso(value)
                            value_json = json.dumps(converted_value)
                    else:
                        value_json = value'''
    
    content = content.replace(old_json_line, new_json_code)
    
    # Write the fixed content back
    with open(memory_file, 'w') as f:
        f.write(content)
    
    print("✅ Applied datetime serialization fix successfully!")
    print("The fix includes:")
    print("  - Added DateTimeEncoder class for handling datetime objects in JSON")
    print("  - Added convert_datetime_to_iso utility function")
    print("  - Updated JSON serialization to use the encoder with fallback")
    print(f"\nBackup saved as: {backup_file}")
    
    return True

if __name__ == "__main__":
    if apply_datetime_fix():
        print("\n🎉 DateTime serialization fix applied successfully!")
        print("Please restart your backend service to apply the changes.")
    else:
        print("\n❌ Failed to apply the fix. Please check the error messages above.")
        sys.exit(1) 