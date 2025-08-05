#!/usr/bin/env python3
"""
Comprehensive HITL Evaluation Test Runner

This script executes the complete HITL robustness evaluation suite and generates
a detailed report of results, state transitions, and performance metrics.

Usage:
    python tests/run_comprehensive_hitl_evaluation.py [--verbose] [--save-report]
    
Options:
    --verbose: Show detailed state transition logs during execution
    --save-report: Save test results to a timestamped report file
"""

import asyncio
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add backend to path for agent imports
backend_path = Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import test module with explicit path handling
import importlib.util
test_module_path = Path(__file__).parent / "test_comprehensive_hitl_robustness_eval.py"
spec = importlib.util.spec_from_file_location("test_comprehensive_hitl_robustness_eval", test_module_path)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
run_comprehensive_hitl_evaluation = test_module.run_comprehensive_hitl_evaluation


class EvaluationReporter:
    """Handles test result reporting and analysis."""
    
    def __init__(self, verbose: bool = False, save_report: bool = False):
        self.verbose = verbose
        self.save_report = save_report
        self.start_time = time.time()
        self.results = {}
        self.summary = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamps."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "ERROR": "❌",
            "WARNING": "⚠️"
        }.get(level, "📝")
        
        print(f"[{timestamp}] {prefix} {message}")
        
    def analyze_results(self, test_results: dict):
        """Analyze test results and generate summary."""
        self.results = test_results
        
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results.values() if r.startswith("✅")])
        failed_tests = total_tests - passed_tests
        
        execution_time = time.time() - self.start_time
        
        self.summary = {
            "execution_time_seconds": round(execution_time, 2),
            "total_test_suites": total_tests,
            "passed_test_suites": passed_tests,
            "failed_test_suites": failed_tests,
            "success_rate_percent": round((passed_tests / total_tests) * 100, 1) if total_tests > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "detailed_results": test_results
        }
        
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "="*120)
        print("COMPREHENSIVE HITL ROBUSTNESS EVALUATION - FINAL REPORT")
        print("="*120)
        
        # Executive Summary
        print(f"\n📊 EXECUTIVE SUMMARY")
        print(f"   Execution Time: {self.summary['execution_time_seconds']}s")
        print(f"   Test Suites: {self.summary['passed_test_suites']}/{self.summary['total_test_suites']} passed")
        print(f"   Success Rate: {self.summary['success_rate_percent']}%")
        print(f"   Timestamp: {self.summary['timestamp']}")
        
        # Detailed Results
        print(f"\n📋 DETAILED TEST RESULTS")
        print("-" * 80)
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result.startswith("✅") else "❌"
            test_display_name = test_name.replace('_', ' ').title()
            
            if result.startswith("✅"):
                print(f"   {status_icon} {test_display_name}")
            else:
                print(f"   {status_icon} {test_display_name}")
                print(f"       └─ {result}")
        
        # Test Coverage Analysis
        print(f"\n🔍 TEST COVERAGE ANALYSIS")
        print("-" * 80)
        
        coverage_areas = {
            "approval_flow": "Customer Message Approval (trigger_customer_message)",
            "input_collection": "Multi-Step Input Collection (collect_sales_requirements)", 
            "selection_flow": "Option Selection (customer_lookup)",
            "edge_cases": "Edge Cases and Error Handling",
            "nlp_interpretation": "Natural Language Interpretation (LLM-driven)",
            "state_consistency": "State Consistency and Routing Logic",
            "performance": "Performance and Stress Testing",
            "complex_integration": "Complex Multi-Tool Workflow Integration"
        }
        
        for test_key, description in coverage_areas.items():
            status = "✅ COVERED" if test_key in self.results and self.results[test_key].startswith("✅") else "❌ FAILED"
            print(f"   {status} {description}")
        
        # Critical Features Validation
        print(f"\n🎯 CRITICAL FEATURES VALIDATION")
        print("-" * 80)
        
        critical_features = [
            ("3-Field Architecture", ["approval_flow", "input_collection", "selection_flow"]),
            ("LLM-Driven Interpretation", ["nlp_interpretation"]),
            ("Tool-Managed Collection", ["input_collection", "complex_integration"]),
            ("Non-Recursive Routing", ["state_consistency", "complex_integration"]),
            ("Error Handling", ["edge_cases"]),
            ("Performance Robustness", ["performance"])
        ]
        
        for feature_name, related_tests in critical_features:
            feature_status = all(
                test_name in self.results and self.results[test_name].startswith("✅")
                for test_name in related_tests
            )
            status_icon = "✅" if feature_status else "❌"
            print(f"   {status_icon} {feature_name}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        if self.summary['success_rate_percent'] == 100:
            print("   🎉 ALL TESTS PASSED! The HITL implementation is robust and production-ready.")
            print("   ✅ Ready for deployment")
            print("   ✅ All PRD requirements validated")
            print("   ✅ State management is consistent")
            print("   ✅ Error handling is comprehensive") 
        elif self.summary['success_rate_percent'] >= 80:
            print("   ⚠️  Most tests passed, but some issues need attention.")
            print("   📝 Review failed test details above")
            print("   🔧 Fix failing scenarios before production deployment")
        else:
            print("   ❌ SIGNIFICANT ISSUES DETECTED")
            print("   🚨 Multiple critical failures - thorough review required")
            print("   🛑 DO NOT deploy until all issues are resolved")
        
        print("\n" + "="*120)
        
    def save_report_to_file(self):
        """Save test results to a JSON report file."""
        if not self.save_report:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(__file__).parent / f"hitl_evaluation_report_{timestamp}.json"
        
        report_data = {
            "evaluation_metadata": {
                "test_suite": "Comprehensive HITL Robustness Evaluation",
                "prd_reference": "prd-general-purpose-hitl-node.md",
                "execution_timestamp": self.summary['timestamp'],
                "execution_time_seconds": self.summary['execution_time_seconds']
            },
            "summary": self.summary,
            "detailed_results": self.results,
            "test_scenarios_covered": [
                "Customer Message Approval Flow (trigger_customer_message)",
                "Multi-Step Input Collection (collect_sales_requirements)",
                "Customer Selection Flow (customer_lookup)",
                "Edge Cases and Error Handling",
                "Natural Language Interpretation (LLM-driven)",
                "State Consistency and Routing Logic", 
                "Performance and Stress Testing",
                "Complex Multi-Tool Workflow Integration"
            ],
            "state_variables_validated": [
                "hitl_phase transitions (None → needs_prompt → awaiting_response → approved/denied → None)",
                "hitl_prompt lifecycle (None → prompt_text → preserved → None)",
                "hitl_context management (None → tool_context → preserved/cleared → None)",
                "Message history growth and conversation flow",
                "Node routing decisions (employee_agent ↔ hitl_node → ea_memory_store)"
            ],
            "critical_features_tested": [
                "3-Field Architecture (hitl_phase, hitl_prompt, hitl_context)",
                "LLM-Driven Natural Language Interpretation",
                "Tool-Managed Recursive Collection Pattern",
                "Non-Recursive HITL Routing (no self-loops)",
                "Dedicated HITL Request Tools (request_approval, request_input, request_selection)",
                "Interrupt/Resume State Preservation",
                "Error Handling and Edge Cases",
                "Performance and Stress Robustness"
            ]
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.log(f"Test report saved to: {report_file}", "SUCCESS")
            print(f"\n📄 Detailed test report available at: {report_file}")
            
        except Exception as e:
            self.log(f"Failed to save report: {e}", "ERROR")


async def main():
    """Main test runner function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive HITL evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show detailed state transition logs")
    parser.add_argument("--save-report", action="store_true", help="Save results to JSON report file")
    args = parser.parse_args()
    
    # Initialize reporter
    reporter = EvaluationReporter(verbose=args.verbose, save_report=args.save_report)
    
    # Start evaluation
    reporter.log("Starting Comprehensive HITL Robustness Evaluation", "INFO")
    reporter.log("Testing General-Purpose HITL Node implementation from PRD", "INFO")
    
    if args.verbose:
        reporter.log("Verbose mode enabled - detailed state transitions will be shown", "INFO")
    
    try:
        # Run the comprehensive evaluation
        test_results = await run_comprehensive_hitl_evaluation()
        
        # Analyze and report results
        reporter.analyze_results(test_results)
        reporter.print_summary_report()
        reporter.save_report_to_file()
        
        # Exit with appropriate code
        if reporter.summary['success_rate_percent'] == 100:
            reporter.log("All tests passed - HITL implementation validated", "SUCCESS")
            sys.exit(0)
        else:
            reporter.log(f"Some tests failed - see report above", "ERROR")
            sys.exit(1)
            
    except KeyboardInterrupt:
        reporter.log("Evaluation interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        reporter.log(f"Evaluation failed with error: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for this evaluation suite")
        sys.exit(1)
    
    # Run the evaluation
    asyncio.run(main())