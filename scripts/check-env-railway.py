#!/usr/bin/env python3
"""
Railway Environment Configuration Checker
Validates that all required environment variables are properly set for deployment
"""

import os
import sys
from typing import Dict, List, Tuple

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.END}")

def print_header(text: str):
    """Print a formatted header"""
    print_colored(f"\n{'='*60}", Colors.BLUE)
    print_colored(f"{text.center(60)}", Colors.BOLD + Colors.BLUE)
    print_colored(f"{'='*60}", Colors.BLUE)

def print_section(text: str):
    """Print a section header"""
    print_colored(f"\n{'-'*40}", Colors.CYAN)
    print_colored(f"{text}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'-'*40}", Colors.CYAN)

# Required environment variables for Railway deployment
REQUIRED_VARS = {
    'OPENAI_API_KEY': {
        'description': 'OpenAI API key for AI services',
        'example': 'sk-...',
        'validation': lambda x: x.startswith('sk-') and len(x) > 20
    },
    'SUPABASE_URL': {
        'description': 'Supabase project URL',
        'example': 'https://your-project-id.supabase.co',
        'validation': lambda x: x.startswith('https://') and '.supabase.co' in x
    },
    'SUPABASE_ANON_KEY': {
        'description': 'Supabase anonymous key',
        'example': 'eyJ...',
        'validation': lambda x: len(x) > 50 and x.startswith('eyJ')
    },
    'SUPABASE_SERVICE_KEY': {
        'description': 'Supabase service role key (keep secret!)',
        'example': 'eyJ...',
        'validation': lambda x: len(x) > 50 and x.startswith('eyJ')
    },
    'SUPABASE_DB_PASSWORD': {
        'description': 'Supabase database password',
        'example': 'your_db_password',
        'validation': lambda x: len(x) > 8
    }
}

# Optional but recommended variables
OPTIONAL_VARS = {
    'LANGCHAIN_API_KEY': {
        'description': 'LangSmith API key for monitoring',
        'example': 'ls__...',
        'validation': lambda x: x.startswith('ls__') if x else True
    },
    'FRONTEND_URL': {
        'description': 'Frontend URL for CORS configuration',
        'example': 'https://tobi-frontend-production.up.railway.app',
        'validation': lambda x: x.startswith('https://') if x else True
    },
    'REDIS_URL': {
        'description': 'Redis URL for caching',
        'example': 'redis://localhost:6379',
        'validation': lambda x: x.startswith('redis://') if x else True
    }
}

# Production-specific variables that should be set
PRODUCTION_VARS = {
    'ENVIRONMENT': {
        'description': 'Environment setting (should be "production")',
        'example': 'production',
        'validation': lambda x: x == 'production' if x else False
    },
    'LOG_LEVEL': {
        'description': 'Logging level',
        'example': 'INFO',
        'validation': lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR'] if x else True
    }
}

def check_variable(var_name: str, var_config: Dict, is_required: bool = True) -> Tuple[bool, str]:
    """Check a single environment variable"""
    value = os.getenv(var_name)
    
    if not value:
        if is_required:
            return False, f"❌ {var_name}: NOT SET (Required)"
        else:
            return True, f"⚠️  {var_name}: Not set (Optional)"
    
    # Validate the value
    try:
        if var_config.get('validation') and not var_config['validation'](value):
            return False, f"❌ {var_name}: INVALID FORMAT"
    except Exception as e:
        return False, f"❌ {var_name}: VALIDATION ERROR - {str(e)}"
    
    # Mask sensitive values in output
    display_value = value
    if any(sensitive in var_name.lower() for sensitive in ['key', 'password', 'secret']):
        if len(value) > 10:
            display_value = f"{value[:6]}...{value[-4:]}"
        else:
            display_value = "***"
    
    return True, f"✅ {var_name}: {display_value}"

def check_all_variables() -> Tuple[int, int, List[str]]:
    """Check all environment variables and return results"""
    total_checks = 0
    passed_checks = 0
    issues = []
    
    print_section("Required Variables")
    for var_name, var_config in REQUIRED_VARS.items():
        total_checks += 1
        passed, message = check_variable(var_name, var_config, is_required=True)
        
        if passed:
            print_colored(message, Colors.GREEN)
            passed_checks += 1
        else:
            print_colored(message, Colors.RED)
            issues.append(f"{var_name}: {var_config['description']}")
            print_colored(f"   Example: {var_config['example']}", Colors.YELLOW)
    
    print_section("Production Variables")
    for var_name, var_config in PRODUCTION_VARS.items():
        total_checks += 1
        passed, message = check_variable(var_name, var_config, is_required=False)
        
        if passed and os.getenv(var_name):
            print_colored(message, Colors.GREEN)
            passed_checks += 1
        elif not os.getenv(var_name):
            print_colored(message, Colors.YELLOW)
            issues.append(f"{var_name}: {var_config['description']} (Recommended for production)")
        else:
            print_colored(message, Colors.RED)
            issues.append(f"{var_name}: {var_config['description']}")
    
    print_section("Optional Variables")
    for var_name, var_config in OPTIONAL_VARS.items():
        passed, message = check_variable(var_name, var_config, is_required=False)
        
        if passed and os.getenv(var_name):
            print_colored(message, Colors.GREEN)
        else:
            print_colored(message, Colors.CYAN)
    
    return total_checks, passed_checks, issues

def print_summary(total_checks: int, passed_checks: int, issues: List[str]):
    """Print summary of the environment check"""
    print_section("Summary")
    
    if passed_checks == total_checks and len(issues) == 0:
        print_colored("🎉 All environment variables are properly configured!", Colors.GREEN + Colors.BOLD)
        print_colored("✅ Ready for Railway deployment!", Colors.GREEN)
    else:
        print_colored(f"⚠️  {passed_checks}/{total_checks} checks passed", Colors.YELLOW)
        
        if issues:
            print_colored(f"\n🔧 Issues to resolve:", Colors.RED + Colors.BOLD)
            for issue in issues:
                print_colored(f"   • {issue}", Colors.RED)
    
    print_colored(f"\n📚 For setup instructions, see:", Colors.BLUE)
    print_colored(f"   • env-template.txt", Colors.BLUE)
    print_colored(f"   • RAILWAY_DEPLOYMENT.md", Colors.BLUE)

def main():
    """Main function"""
    print_header("Railway Environment Configuration Checker")
    print_colored("Checking environment variables for Railway deployment...\n", Colors.WHITE)
    
    total_checks, passed_checks, issues = check_all_variables()
    print_summary(total_checks, passed_checks, issues)
    
    # Exit with appropriate code
    if len(issues) == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
