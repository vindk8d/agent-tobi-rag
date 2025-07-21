"""
Phase 2 Test Runner: Comprehensive Testing for Dual User Agent System

This script runs comprehensive Phase 2 tests and generates detailed reports
on the customer messaging functionality and interrupt workflows.
"""

import pytest
import sys
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def generate_test_report():
    """Generate detailed test report for Phase 2 features."""
    
    report = {
        "test_run": {
            "phase": "Phase 2 - Customer Messaging",
            "timestamp": datetime.now().isoformat(),
            "description": "Comprehensive testing of customer messaging tool, interrupt workflows, and Phase 2 integration"
        },
        "test_categories": {
            "customer_message_tool": {
                "description": "Testing customer messaging tool functionality and access control",
                "test_count": 8,
                "key_features": [
                    "Employee/admin access to customer messaging tool",
                    "Customer/unknown user access denial",
                    "Invalid customer ID handling",
                    "Message type validation and fallbacks",
                    "Tool availability in user tool lists"
                ]
            },
            "message_validation": {
                "description": "Testing message content validation and quality checks",
                "test_count": 8,
                "key_features": [
                    "Valid message content acceptance",
                    "Empty message rejection",
                    "Message length limits by type",
                    "Quality warnings (caps, profanity, etc.)",
                    "Message type specific suggestions",
                    "Inappropriate content detection"
                ]
            },
            "customer_lookup": {
                "description": "Testing customer lookup and database integration",
                "test_count": 4,
                "key_features": [
                    "Successful customer lookup from database",
                    "Non-existent customer handling",
                    "Database error resilience",
                    "Active customer listing functionality"
                ]
            },
            "message_formatting": {
                "description": "Testing message formatting by type and personalization",
                "test_count": 6,
                "key_features": [
                    "Follow-up message professional formatting",
                    "Information message structure",
                    "Promotional message templates", 
                    "Support message formatting",
                    "Unknown type fallback formatting",
                    "Missing customer name handling"
                ]
            },
            "interrupt_workflow": {
                "description": "Testing LangGraph interrupt mechanisms and confirmation flows",
                "test_count": 5,
                "key_features": [
                    "Approval workflow (APPROVE response)",
                    "Cancellation workflow (CANCEL response)",
                    "Modification request workflow (MODIFY response)",
                    "Invalid response handling",
                    "No response timeout handling"
                ]
            },
            "message_delivery": {
                "description": "Testing message delivery simulation and error handling",
                "test_count": 3,
                "key_features": [
                    "Successful delivery simulation (95% success rate)",
                    "Failed delivery handling (5% failure rate)",
                    "Delivery timing and delay simulation"
                ]
            },
            "phase2_integration": {
                "description": "Testing integration with Phase 1 foundation",
                "test_count": 4,
                "key_features": [
                    "Employee agent access to customer messaging",
                    "Customer agent exclusion from messaging",
                    "Phase 1 functionality preservation",
                    "Combined tool availability validation"
                ]
            },
            "error_handling": {
                "description": "Testing error handling and edge cases",
                "test_count": 4,
                "key_features": [
                    "Database connection error handling",
                    "Interrupt system error recovery",
                    "Message validation edge cases",
                    "None/empty data resilience"
                ]
            }
        }
    }
    
    return report

def run_phase2_tests():
    """Run Phase 2 tests with detailed reporting."""
    
    print("🚀 Starting Phase 2: Customer Messaging Tests")
    print("=" * 70)
    
    # Generate pre-test report
    report = generate_test_report()
    
    print(f"📋 Test Plan: {report['test_run']['description']}")
    print(f"📅 Started: {report['test_run']['timestamp']}")
    print()
    
    # Show test categories
    total_tests = sum(cat["test_count"] for cat in report["test_categories"].values())
    print(f"📊 Test Categories ({len(report['test_categories'])} categories, {total_tests} total tests):")
    
    for category, info in report["test_categories"].items():
        print(f"  • {category.replace('_', ' ').title()}: {info['test_count']} tests")
        print(f"    {info['description']}")
    
    print()
    print("🧪 Running Tests...")
    print("-" * 70)
    
    # Run the actual tests
    test_file = Path(__file__).parent / "test_dual_agent_phase2.py"
    
    # Run pytest with detailed output
    result = pytest.main([
        str(test_file),
        "-v",
        "--tb=short",
        "--durations=10",
        f"--junitxml=tests/phase2_test_results.xml"
    ])
    
    print()
    print("-" * 70)
    
    if result == 0:
        print("✅ Phase 2 Tests: ALL PASSED!")
        print()
        print("🎯 Success Criteria Validation:")
        print("  ✅ Customer messaging tool works for employees/admins")
        print("  ✅ Customer messaging denied for customers/unknown users")
        print("  ✅ Interrupt workflow handles all response types")
        print("  ✅ Message validation prevents inappropriate content")
        print("  ✅ Message delivery simulation works with error handling")
        print("  ✅ Phase 1 functionality preserved with Phase 2 additions")
        print("  ✅ Database integration robust with error handling")
        print("  ✅ Message formatting professional and type-appropriate")
        
        print()
        print("📈 Phase 2 Implementation Status:")
        print("  ✅ Task 2.1: Customer Messaging Tool Development - COMPLETE")
        print("  ✅ Task 2.2: Interrupt-Based Confirmation System - COMPLETE") 
        print("  ✅ Task 2.3: Message Delivery Infrastructure - COMPLETE")
        print("  ⏳ Task 2.2.4: Comprehensive Phase 2 Testing - COMPLETE")
        print("  📝 Task 2.2.5: Dual Agent Debug Frontend - PENDING")
        
        print()
        print("🚀 Ready for Task 2.2.5: Building dual agent debug frontend interface!")
        
    else:
        print("❌ Phase 2 Tests: SOME FAILED")
        print()
        print("🔧 Please review test failures above and fix issues before proceeding.")
        print("   Common issues:")
        print("   • Database connection mocking")
        print("   • Import path issues")  
        print("   • Missing dependencies")
        print("   • LangGraph interrupt functionality")
        
    return result

def run_combined_tests():
    """Run both Phase 1 and Phase 2 tests together."""
    
    print("🔄 Running Combined Phase 1 + Phase 2 Tests")
    print("=" * 70)
    
    # Run Phase 1 tests first
    print("1️⃣ Running Phase 1 Tests...")
    phase1_file = Path(__file__).parent / "test_dual_agent_phase1.py"
    phase1_result = pytest.main([
        str(phase1_file),
        "-v",
        "--tb=short",
        "--durations=5"
    ])
    
    print()
    print("2️⃣ Running Phase 2 Tests...")
    phase2_file = Path(__file__).parent / "test_dual_agent_phase2.py"
    phase2_result = pytest.main([
        str(phase2_file),
        "-v", 
        "--tb=short",
        "--durations=5"
    ])
    
    print()
    print("-" * 70)
    
    if phase1_result == 0 and phase2_result == 0:
        print("🎉 ALL TESTS PASSED! Phase 1 + Phase 2 Complete!")
        print()
        print("✅ Dual User Agent System Status:")
        print("  ✅ Phase 1: Core functionality (user verification, routing, tool access)")
        print("  ✅ Phase 2: Customer messaging with interrupt confirmation")
        print("  ✅ Integration: Both phases work together seamlessly")
        print("  ✅ Security: Access control working for all user types")
        print("  ✅ Error handling: Robust error recovery implemented")
        
    elif phase1_result != 0:
        print("❌ Phase 1 Tests Failed - Please fix Phase 1 issues first")
        
    elif phase2_result != 0:
        print("❌ Phase 2 Tests Failed - Please fix Phase 2 issues")
        
    return phase1_result == 0 and phase2_result == 0

if __name__ == "__main__":
    """Main test runner entry point."""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--combined":
            success = run_combined_tests()
        elif sys.argv[1] == "--report-only":
            report = generate_test_report()
            print(json.dumps(report, indent=2))
            success = True
        else:
            print("Usage: python run_phase2_tests.py [--combined|--report-only]")
            success = False
    else:
        # Default: run just Phase 2 tests
        success = run_phase2_tests() == 0
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 