#!/usr/bin/env python3
"""
Quick Employee Verification Test Tool

Use this tool for rapid testing of employee verification with correct UUIDs.
Based on verified database schema.

Usage: python tests/employee_verification_quick_test.py [user_id]
"""

import asyncio
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.agents.tobi_sales_copilot.rag_agent import UnifiedToolCallingRAGAgent


# Verified user data from database
VERIFIED_USERS = {
    # Valid Employees (should pass)
    "alex": "54394d40-ad35-4b5b-a392-1ae7c9329d11",  # Alex Thompson
    "carlos": "492dbd86-ddc9-4d52-9208-b578e0fccc93",  # Carlos Rodriguez  
    "john": "f26449e2-dce9-4b29-acd0-cb39a1f671fd",  # John Smith
    "lisa": "343565b9-2e7c-405a-b8b1-3d9d4d275531",  # Lisa Wang
    "emma": "896d2df2-b35d-41c9-93cc-e2d5400d0eba",  # Emma Wilson
    
    # Customers (should fail)
    "alice": "a0b81a3b-3ac0-4a7d-b27c-dadf0fdbbe8b",  # Alice Johnson (Customer)
    "bob": "e8a13560-3b49-4495-bb47-c95c30de5165",    # Bob Smith (Customer)
    "carol": "95b070ff-e574-40c5-89f2-331cbf9ecdbb",  # Carol Thompson (Customer)
}


async def quick_test(user_input=None):
    """Quick test of employee verification."""
    
    print("🔧 Employee Verification Quick Test")
    print("=" * 40)
    
    if user_input and user_input in VERIFIED_USERS:
        user_id = VERIFIED_USERS[user_input]
        print(f"Using shorthand '{user_input}' → {user_id}")
    elif user_input:
        user_id = user_input
        print(f"Testing provided UUID: {user_id}")
    else:
        print("Available shortcuts:")
        print("Employees: alex, carlos, john, lisa, emma")  
        print("Customers: alice, bob, carol")
        print()
        user_input = input("Enter shortcut or UUID: ").strip()
        
        if user_input in VERIFIED_USERS:
            user_id = VERIFIED_USERS[user_input]
            print(f"Using '{user_input}' → {user_id}")
        else:
            user_id = user_input
    
    print()
    print(f"Testing user_id: {user_id}")
    print("-" * 40)
    
    try:
        agent = UnifiedToolCallingRAGAgent()
        result = await agent._verify_employee_access(user_id)
        
        if result:
            print("✅ PASS - Employee access verified")
        else:
            print("❌ FAIL - Access denied (likely customer or invalid user)")
            
        return result
        
    except Exception as e:
        print(f"❌ ERROR - Exception occurred: {str(e)}")
        return False


def print_help():
    """Print help information."""
    print("Employee Verification Quick Test Tool")
    print()
    print("Usage:")
    print("  python tests/employee_verification_quick_test.py          # Interactive mode")
    print("  python tests/employee_verification_quick_test.py alex     # Test Alex Thompson")
    print("  python tests/employee_verification_quick_test.py alice    # Test Alice Johnson (customer)")
    print("  python tests/employee_verification_quick_test.py [UUID]   # Test specific UUID")
    print()
    print("Shortcuts:")
    print("  Employees: alex, carlos, john, lisa, emma")
    print("  Customers: alice, bob, carol")
    print()
    print("Examples:")
    print("  python tests/employee_verification_quick_test.py alex     # Should pass ✅")
    print("  python tests/employee_verification_quick_test.py alice    # Should fail ❌")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print_help()
            sys.exit(0)
        else:
            user_input = sys.argv[1]
            result = asyncio.run(quick_test(user_input))
    else:
        result = asyncio.run(quick_test())
    
    sys.exit(0 if result else 1) 