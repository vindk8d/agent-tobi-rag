#!/usr/bin/env python3
"""
Script to update integration tests to use the new simplified vehicle search system.
"""

import re

def update_test_file(file_path):
    """Update the test file to use new vehicle search functions."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace old vehicle search function calls with new ones
    replacements = [
        # Replace the old complex vehicle search functions
        (r"patch\('backend\.agents\.tools\._lookup_vehicle_by_criteria'[^)]*\),\s*\\?\n\s*", ""),
        (r"patch\('backend\.agents\.tools\._parse_vehicle_requirements_with_llm'[^)]*\),\s*\\?\n\s*", ""),
        (r"patch\('backend\.agents\.tools\._get_available_makes_and_models'[^)]*\),\s*\\?\n\s*", ""),
        (r"patch\('backend\.agents\.tools\._enhance_vehicle_criteria_with_fuzzy_matching'[^)]*\),\s*\\?\n\s*", ""),
        (r"patch\('backend\.agents\.tools\._generate_inventory_suggestions'[^)]*\),\s*\\?\n\s*", ""),
    ]
    
    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Add the new vehicle search function where needed
    # Find patterns where we had _lookup_customer and add _search_vehicles_with_llm
    patterns_to_add = [
        (r"(patch\('backend\.agents\.tools\._lookup_customer', return_value=sample_customer_data\),\s*\\?\n\s*)",
         r"\1patch('backend.agents.tools._search_vehicles_with_llm', return_value=sample_vehicle_data), \\\n             "),
    ]
    
    for pattern, replacement in patterns_to_add:
        content = re.sub(pattern, replacement, content)
    
    # Special case for empty vehicle results (vehicle not found scenarios)
    content = re.sub(
        r"patch\('backend\.agents\.tools\._search_vehicles_with_llm', return_value=sample_vehicle_data\), \\\n\s*patch\('backend\.agents\.tools\._lookup_current_pricing'",
        "patch('backend.agents.tools._search_vehicles_with_llm', return_value=[]), \\\n             patch('backend.agents.tools._lookup_current_pricing'",
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

if __name__ == "__main__":
    update_test_file("tests/test_quotation_generation.py")
