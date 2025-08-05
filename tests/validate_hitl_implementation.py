#!/usr/bin/env python3
"""
HITL Implementation Validation Script

Quick validation script to check if the current codebase has all the required
components for the General-Purpose HITL Node as specified in the PRD.

This script validates:
1. 3-Field Architecture exists in AgentState
2. Dedicated HITL request tools are available
3. HITL node implementation exists
4. Routing functions are implemented
5. Core tools that use HITL are available

Run this before the comprehensive evaluation to ensure all components are present.
"""

import sys
import traceback
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


def validate_3field_architecture():
    """Validate that AgentState has the 3-field HITL architecture."""
    print("\n🔍 Validating 3-Field Architecture...")
    
    try:
        from agents.tobi_sales_copilot.state import AgentState
        
        # Check if AgentState is a TypedDict with expected fields
        required_fields = ["hitl_phase", "hitl_prompt", "hitl_context"]
        annotations = getattr(AgentState, '__annotations__', {})
        
        missing_fields = []
        for field in required_fields:
            if field not in annotations:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"   ❌ Missing HITL fields in AgentState: {missing_fields}")
            return False
        
        print(f"   ✅ All 3 HITL fields present in AgentState: {required_fields}")
        
        # Check field types
        for field in required_fields:
            field_type = annotations[field]
            print(f"   ✅ {field}: {field_type}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import AgentState: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating AgentState: {e}")
        return False


def validate_hitl_request_tools():
    """Validate that dedicated HITL request tools exist."""
    print("\n🔍 Validating HITL Request Tools...")
    
    try:
        from agents.hitl import request_approval, request_input, request_selection
        
        tools = {
            "request_approval": request_approval,
            "request_input": request_input, 
            "request_selection": request_selection
        }
        
        for tool_name, tool_func in tools.items():
            if not callable(tool_func):
                print(f"   ❌ {tool_name} is not callable")
                return False
            
            # Check function signature
            import inspect
            sig = inspect.signature(tool_func)
            params = list(sig.parameters.keys())
            
            print(f"   ✅ {tool_name}({', '.join(params)})")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import HITL request tools: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating HITL request tools: {e}")
        return False


def validate_hitl_node():
    """Validate that the HITL node implementation exists."""
    print("\n🔍 Validating HITL Node Implementation...")
    
    try:
        from agents.hitl import hitl_node, HITLPhase, parse_tool_response
        
        # Check hitl_node is callable
        if not callable(hitl_node):
            print(f"   ❌ hitl_node is not callable")
            return False
        
        print(f"   ✅ hitl_node function available")
        
        # Check HITLPhase enum
        if not hasattr(HITLPhase, '__args__'):
            print(f"   ❌ HITLPhase is not properly defined")
            return False
        
        expected_phases = ["needs_prompt", "awaiting_response", "approved", "denied"]
        actual_phases = list(HITLPhase.__args__)
        
        if set(expected_phases) != set(actual_phases):
            print(f"   ❌ HITLPhase mismatch. Expected: {expected_phases}, Got: {actual_phases}")
            return False
        
        print(f"   ✅ HITLPhase enum: {actual_phases}")
        
        # Check parse_tool_response
        if not callable(parse_tool_response):
            print(f"   ❌ parse_tool_response is not callable")
            return False
        
        print(f"   ✅ parse_tool_response function available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import HITL node components: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating HITL node: {e}")
        return False


def validate_agent_routing():
    """Validate that agent routing functions exist."""
    print("\n🔍 Validating Agent Routing Functions...")
    
    try:
        from agents.tobi_sales_copilot.agent import UnifiedToolCallingRAGAgent
        
        # Create agent instance
        agent = UnifiedToolCallingRAGAgent()
        
        # Check required routing methods
        required_methods = [
            "route_from_employee_agent",
            "route_from_hitl"
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(agent, method_name):
                missing_methods.append(method_name)
            else:
                method = getattr(agent, method_name)
                if not callable(method):
                    missing_methods.append(f"{method_name} (not callable)")
        
        if missing_methods:
            print(f"   ❌ Missing routing methods: {missing_methods}")
            return False
        
        print(f"   ✅ All routing methods available: {required_methods}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import UnifiedToolCallingRAGAgent: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating agent routing: {e}")
        return False


def validate_core_tools():
    """Validate that core tools using HITL are available."""
    print("\n🔍 Validating Core HITL-Enabled Tools...")
    
    try:
        from agents.tools import trigger_customer_message, collect_sales_requirements
        
        tools = {
            "trigger_customer_message": trigger_customer_message,
            "collect_sales_requirements": collect_sales_requirements
        }
        
        for tool_name, tool_func in tools.items():
            if not callable(tool_func):
                print(f"   ❌ {tool_name} is not callable")
                return False
            
            # Check if it's properly decorated
            if not hasattr(tool_func, 'name'):
                print(f"   ⚠️  {tool_name} may not be properly decorated as a LangChain tool")
            
            print(f"   ✅ {tool_name} available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import core tools: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating core tools: {e}")
        return False


def validate_llm_interpretation():
    """Validate that LLM interpretation components exist."""
    print("\n🔍 Validating LLM Interpretation Components...")
    
    try:
        from agents.hitl import _interpret_user_intent_with_llm, _process_hitl_response_llm_driven
        
        components = {
            "_interpret_user_intent_with_llm": _interpret_user_intent_with_llm,
            "_process_hitl_response_llm_driven": _process_hitl_response_llm_driven
        }
        
        for comp_name, comp_func in components.items():
            if not callable(comp_func):
                print(f"   ❌ {comp_name} is not callable")
                return False
            
            print(f"   ✅ {comp_name} available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import LLM interpretation components: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating LLM interpretation: {e}")
        return False


def run_validation():
    """Run all validation checks."""
    
    print("="*100)
    print("HITL IMPLEMENTATION VALIDATION")
    print("Checking PRD requirements compliance before comprehensive evaluation")
    print("="*100)
    
    validation_results = {}
    
    # Run all validations
    validation_checks = [
        ("3-Field Architecture", validate_3field_architecture),
        ("HITL Request Tools", validate_hitl_request_tools),
        ("HITL Node Implementation", validate_hitl_node),
        ("Agent Routing Functions", validate_agent_routing),
        ("Core HITL-Enabled Tools", validate_core_tools),
        ("LLM Interpretation", validate_llm_interpretation)
    ]
    
    all_passed = True
    
    for check_name, validation_func in validation_checks:
        try:
            result = validation_func()
            validation_results[check_name] = result
            if not result:
                all_passed = False
                
        except Exception as e:
            print(f"\n❌ Validation error in {check_name}: {e}")
            if "--verbose" in sys.argv:
                print(f"Traceback: {traceback.format_exc()}")
            validation_results[check_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "="*100)
    print("VALIDATION SUMMARY")
    print("="*100)
    
    passed_count = sum(validation_results.values())
    total_count = len(validation_results)
    
    print(f"\nResults: {passed_count}/{total_count} validation checks passed")
    
    for check_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ HITL implementation has all required components")
        print(f"✅ Ready to run comprehensive robustness evaluation")
        print(f"\nNext step: python tests/run_comprehensive_hitl_evaluation.py")
    else:
        print(f"\n⚠️  {total_count - passed_count} validation(s) failed")
        print(f"❌ Fix missing components before running comprehensive evaluation")
        print(f"\nReview the errors above and ensure all PRD requirements are implemented")
    
    print("\n" + "="*100)
    
    return all_passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)