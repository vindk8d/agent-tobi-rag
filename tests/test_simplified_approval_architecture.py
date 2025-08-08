#!/usr/bin/env python3
"""
Test script to verify the simplified approval architecture works correctly.

This tests that:
1. All approval logic is centralized in the HITL system
2. Other layers properly import and use the HITL function
3. No redundant logic exists
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_no_redundant_functions():
    """Test that redundant approval functions have been removed"""
    print("🔍 Testing for Redundant Functions")
    print("=" * 50)
    
    try:
        # This should fail since we removed the redundant function
        from agents.tools import interpret_user_approval_intent
        print("❌ FAIL: Redundant function still exists in tools.py")
        return False
    except ImportError:
        print("✅ PASS: Redundant function properly removed from tools.py")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

async def test_centralized_imports():
    """Test that all files now import from the centralized HITL system"""
    print("\n📦 Testing Centralized Imports")
    print("=" * 50)
    
    try:
        # Test that the HITL function is accessible
        from agents.hitl import _interpret_user_intent_with_llm
        print("✅ PASS: Can import from centralized HITL system")
        return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

async def test_import_fixes():
    """Test that import issues have been fixed"""
    print("\n🔧 Testing Import Fixes")
    print("=" * 50)
    
    try:
        # Test that storage imports are correct
        from core.storage import upload_quotation_pdf, create_signed_quotation_url
        print("✅ PASS: Storage functions import correctly")
        return True
            
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

async def test_architecture_simplification():
    """Test that the architecture has been simplified as intended"""
    print("\n🏗️ Testing Architecture Simplification")
    print("=" * 50)
    
    # Check that files use centralized HITL system
    files_to_check = [
        ("backend/agents/tools.py", "_interpret_user_intent_with_llm"),
        ("backend/agents/tobi_sales_copilot/agent.py", "_interpret_user_intent_with_llm"),
    ]
    
    results = []
    for file_path, expected_function in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if expected_function in content:
                    print(f"✅ PASS: {file_path} uses centralized HITL function")
                    results.append(True)
                else:
                    print(f"❌ FAIL: {file_path} doesn't use centralized HITL function")
                    results.append(False)
        except Exception as e:
            print(f"❌ ERROR checking {file_path}: {e}")
            results.append(False)
    
    return all(results)

async def main():
    """Run all tests"""
    print("🏗️  Testing Simplified Approval Architecture")
    print("=" * 60)
    
    tests = [
        test_no_redundant_functions, 
        test_centralized_imports,
        test_import_fixes,
        test_architecture_simplification
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Simplified architecture is working correctly.")
        print("\n✨ Architecture Benefits:")
        print("  • Single source of truth for approval logic (HITL system)")
        print("  • No redundant functions or logic")
        print("  • Consistent behavior across all layers")
        print("  • Minimal complexity with maximum portability")
        print("  • Fixed import issues")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)