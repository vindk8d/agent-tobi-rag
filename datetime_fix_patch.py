#!/usr/bin/env python3
"""
Quick patch script to fix datetime serialization in memory.py
Usage: python datetime_fix_patch.py
"""

import os
import re

def apply_fix():
    memory_file = "backend/agents/memory.py"
    
    if not os.path.exists(memory_file):
        print(f"❌ File not found: {memory_file}")
        return False
    
    # Create backup
    backup_file = memory_file + ".backup"
    with open(memory_file, 'r') as f:
        original_content = f.read()
    
    with open(backup_file, 'w') as f:
        f.write(original_content)
    print(f"✅ Created backup: {backup_file}")
    
    content = original_content
    
    # 1. Add DateTimeEncoder after imports
    encoder_code = '''
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
    
    # Find location after logger definition
    logger_pattern = r'logger = logging\.getLogger\(__name__\)\n'
    if re.search(logger_pattern, content):
        content = re.sub(logger_pattern, r'logger = logging.getLogger(__name__)\n' + encoder_code, content)
        print("✅ Added DateTimeEncoder class")
    
    # 2. Fix JSON serialization
    old_pattern = r'value_json = json\.dumps\(value\) if not isinstance\(value, str\) else value'
    new_code = '''# Serialize value to JSON for storage with datetime handling
                    if not isinstance(value, str):
                        try:
                            value_json = json.dumps(value, cls=DateTimeEncoder)
                        except TypeError as e:
                            logger.warning(f"JSON serialization failed with DateTimeEncoder, using fallback: {e}")
                            # Fallback: recursively convert datetime objects to strings
                            converted_value = convert_datetime_to_iso(value)
                            value_json = json.dumps(converted_value)
                    else:
                        value_json = value'''
    
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_code, content)
        print("✅ Fixed JSON serialization")
    else:
        print("⚠️ JSON serialization pattern not found - manual fix required")
    
    # Write fixed content
    with open(memory_file, 'w') as f:
        f.write(content)
    
    print("✅ Datetime serialization fix applied successfully!")
    return True

if __name__ == "__main__":
    if apply_fix():
        print("\n🎉 Fix completed! Restart your backend service.")
    else:
        print("\n❌ Fix failed. Please apply manually.")
