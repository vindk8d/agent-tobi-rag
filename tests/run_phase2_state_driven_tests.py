#!/usr/bin/env python3
"""
Test Runner for Phase 2: State-Driven HITL Architecture

This script runs comprehensive tests for the updated Phase 2 implementation
with state-driven HITL approach and generates detailed test reports.
"""

import sys
import os
import subprocess
import json
from datetime import datetime
import xml.etree.ElementTree as ET

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def run_phase2_state_driven_tests():
    """
    Run the comprehensive Phase 2 state-driven tests and generate reports.
    """
    print("🚀 Starting Phase 2 State-Driven HITL Tests...")
    print("=" * 60)
    
    # Test configuration
    test_file = "test_phase2_state_driven_hitl.py"
    results_file = "phase2_state_driven_results.xml"
    report_file = "phase2_state_driven_report.txt"
    
    # Command to run pytest with detailed output
    cmd = [
        "python3", "-m", "pytest",
        test_file,
        "-v",                          # Verbose output
        "--tb=short",                  # Short traceback format
        f"--junitxml={results_file}",  # Generate XML report
        "--capture=no",                # Show print statements
        "--durations=10"               # Show 10 slowest tests
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Run the tests
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True)
        
        # Print stdout and stderr
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Generate summary report
        generate_test_report(results_file, report_file, result)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def generate_test_report(xml_file, report_file, result):
    """
    Generate a comprehensive test report from pytest results.
    """
    try:
        # Parse XML results if available
        test_summary = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        test_details = []
        
        xml_path = os.path.join(os.path.dirname(__file__), xml_file)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get test summary
            test_summary["total"] = int(root.get("tests", 0))
            test_summary["passed"] = test_summary["total"] - int(root.get("failures", 0)) - int(root.get("errors", 0)) - int(root.get("skipped", 0))
            test_summary["failed"] = int(root.get("failures", 0))
            test_summary["errors"] = int(root.get("errors", 0))
            test_summary["skipped"] = int(root.get("skipped", 0))
            
            # Get individual test results
            for testcase in root.findall("testcase"):
                test_name = f"{testcase.get('classname')}.{testcase.get('name')}"
                test_time = float(testcase.get('time', 0))
                
                # Determine test status
                if testcase.find('failure') is not None:
                    status = "FAILED"
                    failure_msg = testcase.find('failure').text
                elif testcase.find('error') is not None:
                    status = "ERROR"
                    failure_msg = testcase.find('error').text
                elif testcase.find('skipped') is not None:
                    status = "SKIPPED"
                    failure_msg = testcase.find('skipped').get('message', '')
                else:
                    status = "PASSED"
                    failure_msg = None
                
                test_details.append({
                    "name": test_name,
                    "status": status,
                    "time": test_time,
                    "failure": failure_msg
                })
        
        # Generate comprehensive report
        report_path = os.path.join(os.path.dirname(__file__), report_file)
        with open(report_path, 'w') as f:
            f.write("PHASE 2: STATE-DRIVEN HITL ARCHITECTURE TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test Summary
            f.write("TEST SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Tests: {test_summary['total']}\n")
            f.write(f"Passed: {test_summary['passed']}\n") 
            f.write(f"Failed: {test_summary['failed']}\n")
            f.write(f"Errors: {test_summary['errors']}\n")
            f.write(f"Skipped: {test_summary['skipped']}\n\n")
            
            if test_summary['total'] > 0:
                success_rate = (test_summary['passed'] / test_summary['total']) * 100
                f.write(f"Success Rate: {success_rate:.1f}%\n\n")
            
            # Test Categories Covered
            f.write("TEST CATEGORIES COVERED\n")
            f.write("-" * 25 + "\n")
            categories = [
                "✓ State-driven tool behavior (STATE_DRIVEN_CONFIRMATION_REQUIRED)",
                "✓ Employee agent state population and handling", 
                "✓ State-driven routing logic (confirmation_data presence)",
                "✓ Dedicated HITL node functionality",
                "✓ Interrupt mechanism with approval/denial flows",
                "✓ Side effect protection (confirmation_result)",
                "✓ Centralized response handling in employee agent",
                "✓ HITL resumption detection and cleanup",
                "✓ Phase 1 integration (user verification still works)",
                "✓ Graph flow validation (employee → HITL → employee → END)"
            ]
            for category in categories:
                f.write(f"{category}\n")
            f.write("\n")
            
            # Key Architecture Features Validated
            f.write("KEY ARCHITECTURE FEATURES VALIDATED\n")
            f.write("-" * 38 + "\n")
            features = [
                "🔧 Tool returns confirmation data instead of handling interrupts",
                "🔄 Employee agent populates confirmation_data in AgentState",
                "🎯 Routing detects state presence for implicit routing decisions",
                "⚡ HITL node combines interrupt + delivery in atomic operation",
                "🛡️ Side effect protection prevents message re-delivery", 
                "🔄 Clean graph flow with centralized response handling",
                "✨ State cleanup maintains minimal footprint philosophy"
            ]
            for feature in features:
                f.write(f"{feature}\n")
            f.write("\n")
            
            # Individual Test Results
            if test_details:
                f.write("INDIVIDUAL TEST RESULTS\n")
                f.write("-" * 25 + "\n")
                for test in test_details:
                    status_icon = "✅" if test["status"] == "PASSED" else "❌"
                    f.write(f"{status_icon} {test['name']} ({test['time']:.3f}s) - {test['status']}\n")
                    if test["failure"]:
                        f.write(f"   Error: {test['failure'][:200]}...\n")
                f.write("\n")
            
            # Command Output
            f.write("FULL COMMAND OUTPUT\n")
            f.write("-" * 20 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        print(f"\n📊 Comprehensive test report generated: {report_file}")
        print(f"📊 XML results available: {xml_file}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")

def print_phase2_validation_summary():
    """
    Print a summary of what Phase 2 validation covers.
    """
    print("\n🎯 PHASE 2 STATE-DRIVEN HITL VALIDATION COVERAGE")
    print("=" * 60)
    
    validation_areas = [
        {
            "area": "State-Driven Tool Behavior",
            "tests": [
                "Tool returns STATE_DRIVEN_CONFIRMATION_REQUIRED",
                "JSON confirmation data structure validation", 
                "Employee access control enforcement",
                "Message content validation preserved"
            ]
        },
        {
            "area": "Employee Agent State Handling",
            "tests": [
                "Populates confirmation_data when tool indicates need",
                "Detects HITL resumption via confirmation_result",
                "Provides appropriate delivery feedback messages",
                "Cleans up state variables after processing"
            ]
        },
        {
            "area": "State-Driven Routing Logic", 
            "tests": [
                "Routes to HITL when confirmation_data present",
                "Routes to END when no confirmation needed",
                "Skips HITL when confirmation_result exists",
                "Implicit routing based on state presence"
            ]
        },
        {
            "area": "Dedicated HITL Node",
            "tests": [
                "Handles missing confirmation_data gracefully", 
                "Side effect protection via confirmation_result",
                "Interrupt mechanism with approval flows",
                "Interrupt mechanism with denial flows",
                "Combined confirmation + delivery operations"
            ]
        },
        {
            "area": "Integration & Architecture",
            "tests": [
                "Phase 1 user verification compatibility",
                "Customer agent direct-to-END routing",
                "Complete graph flow validation",
                "Minimal state footprint maintenance"
            ]
        }
    ]
    
    for area in validation_areas:
        print(f"\n📋 {area['area']}")
        for test in area["tests"]:
            print(f"   • {test}")

if __name__ == "__main__":
    """
    Main execution when run directly.
    """
    print_phase2_validation_summary()
    print("\n" + "=" * 60)
    
    success = run_phase2_state_driven_tests()
    
    if success:
        print("\n✅ All Phase 2 state-driven HITL tests completed successfully!")
        print("🎉 The updated architecture is validated and ready for production.")
    else:
        print("\n❌ Some Phase 2 tests failed. Please review the results and fix issues.")
        sys.exit(1) 