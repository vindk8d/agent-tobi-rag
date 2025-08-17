"""
Agent Toolbox

Modular, organized collection of tools for the RAG agent system.
This package provides a clean separation of concerns with individual
tool files and shared utilities.
"""

import logging
from typing import List, Dict, Any, Optional

# Import all tool modules
from .generate_quotation import generate_quotation
from .crm_query_tools import simple_query_crm_data, get_detailed_schema
from .rag_tools import simple_rag
from .customer_message_tools import trigger_customer_message


# Import toolbox utilities
from .toolbox import (
    UserContext,
    get_current_user_id,
    get_current_conversation_id,
    get_current_user_type,
    get_current_employee_id,
    get_user_context,
    initialize_toolbox
)

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL REGISTRY AND ACCESS CONTROL
# =============================================================================

def get_all_tools():
    """Get all available tools for the RAG agent."""
    return [
        simple_query_crm_data,      # CRM database queries and analytics
        simple_rag,                 # Document retrieval and knowledge base
        trigger_customer_message,   # Customer outreach and communication (Employee only)
        generate_quotation,         # Professional PDF quotation generation (Employee only)
    ]

def get_tools_for_user_type(user_type: str = "employee") -> List:
    """Get tools filtered by user type for access control."""
    if user_type == "customer":
        # Customers get limited access - no employee-only tools
        return [
            simple_query_crm_data,      # Limited to vehicle and pricing data only
            simple_rag,                 # Full access to knowledge base
        ]
    elif user_type in ["employee", "admin"]:
        # Employees get full access to all tools
        return [
            simple_query_crm_data,      # Full CRM database access
            simple_rag,                 # Full access to knowledge base
            trigger_customer_message,   # Employee only - customer outreach
            generate_quotation,         # Employee only - professional quotation generation
        ]
    else:
        # Unknown users get no tools
        logger.warning(f"[TOOLBOX] Unknown user type requested tools: {user_type}")
        return []

def get_tool_names() -> List[str]:
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]

def get_simple_sql_tools():
    """Get tools that perform SQL database operations."""
    return [simple_query_crm_data, get_detailed_schema]

def get_simple_rag_tools():
    """Get tools that perform RAG operations."""
    return [simple_rag]

def get_employee_only_tools():
    """Get tools that are restricted to employees only."""
    return [trigger_customer_message, generate_quotation]

def get_customer_accessible_tools():
    """Get tools that customers can access."""
    return [simple_query_crm_data, simple_rag]

# =============================================================================
# TOOL VALIDATION AND METADATA
# =============================================================================

def validate_tool_access(tool_name: str, user_type: str) -> bool:
    """Validate if a user type can access a specific tool."""
    user_tools = get_tools_for_user_type(user_type)
    user_tool_names = [tool.name for tool in user_tools]
    return tool_name in user_tool_names

def get_tool_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all tools including descriptions and access levels."""
    return {
        "simple_query_crm_data": {
            "description": "Query CRM database for customer, vehicle, and sales information",
            "access_level": "all_users",
            "category": "data_query",
            "employee_restrictions": False,
            "customer_restrictions": True,  # Limited table access
        },
        "simple_rag": {
            "description": "Search knowledge base and documents for information",
            "access_level": "all_users", 
            "category": "knowledge_retrieval",
            "employee_restrictions": False,
            "customer_restrictions": False,
        },
        "trigger_customer_message": {
            "description": "Send messages to customers with HITL confirmation",
            "access_level": "employee_only",
            "category": "customer_communication",
            "employee_restrictions": False,
            "customer_restrictions": True,  # Completely blocked
        },

        "generate_quotation": {
            "description": "Generate professional PDF quotations with LLM-driven intelligence",
            "access_level": "employee_only",
            "category": "document_generation",
            "employee_restrictions": False,
            "customer_restrictions": True,  # Completely blocked
        },
        "get_detailed_schema": {
            "description": "Get detailed database schema information for specific tables",
            "access_level": "all_users",
            "category": "system_information",
            "employee_restrictions": False,
            "customer_restrictions": True,  # Limited table access
        }
    }

def get_tool_categories() -> Dict[str, List[str]]:
    """Get tools organized by category."""
    metadata = get_tool_metadata()
    categories = {}
    
    for tool_name, info in metadata.items():
        category = info.get("category", "uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(tool_name)
    
    return categories

# =============================================================================
# TOOLBOX INITIALIZATION AND HEALTH CHECKS
# =============================================================================

def initialize_agent_toolbox():
    """Initialize the agent toolbox with all necessary setup."""
    try:
        # Initialize the base toolbox
        initialize_toolbox()
        
        # Log toolbox initialization
        all_tools = get_all_tools()
        logger.info(f"[TOOLBOX] Initialized with {len(all_tools)} tools")
        
        # Log tool categories
        categories = get_tool_categories()
        for category, tools in categories.items():
            logger.info(f"[TOOLBOX] {category.replace('_', ' ').title()}: {len(tools)} tool(s)")
        
        logger.info("[TOOLBOX] ✅ Agent toolbox initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"[TOOLBOX] ❌ Failed to initialize agent toolbox: {e}")
        return False

def health_check_toolbox() -> Dict[str, Any]:
    """Perform health check on the toolbox and all tools."""
    health_status = {
        "status": "healthy",
        "total_tools": 0,
        "available_tools": 0,
        "tool_status": {},
        "categories": {},
        "errors": []
    }
    
    try:
        # Check all tools
        all_tools = get_all_tools()
        health_status["total_tools"] = len(all_tools)
        
        for tool in all_tools:
            tool_name = tool.name
            try:
                # Basic tool validation
                if hasattr(tool, 'func') and callable(tool.func):
                    health_status["tool_status"][tool_name] = "available"
                    health_status["available_tools"] += 1
                else:
                    health_status["tool_status"][tool_name] = "invalid"
                    health_status["errors"].append(f"Tool {tool_name} is not properly callable")
            except Exception as e:
                health_status["tool_status"][tool_name] = "error"
                health_status["errors"].append(f"Tool {tool_name} error: {str(e)}")
        
        # Check categories
        health_status["categories"] = get_tool_categories()
        
        # Determine overall status
        if health_status["errors"]:
            health_status["status"] = "degraded"
        
        if health_status["available_tools"] == 0:
            health_status["status"] = "unhealthy"
        
        logger.info(f"[TOOLBOX] Health check complete - Status: {health_status['status']}")
        return health_status
        
    except Exception as e:
        logger.error(f"[TOOLBOX] Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "total_tools": 0,
            "available_tools": 0
        }

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# For backward compatibility, export commonly used functions at package level
__all__ = [
    # Main tool functions
    'get_all_tools',
    'get_tools_for_user_type', 
    'get_tool_names',
    'get_simple_sql_tools',
    'get_simple_rag_tools',
    
    # Individual tools
    'generate_quotation',
    'simple_query_crm_data',
    'simple_rag',
    'trigger_customer_message',

    'get_detailed_schema',
    
    # Utilities
    'UserContext',
    'get_current_user_id',
    'get_current_conversation_id',
    'get_current_user_type', 
    'get_current_employee_id',
    'get_user_context',
    
    # Toolbox management
    'initialize_agent_toolbox',
    'health_check_toolbox',
    'validate_tool_access',
    'get_tool_metadata',
]

# Auto-initialize on import
logger.info("[TOOLBOX] Loading agent toolbox...")
if initialize_agent_toolbox():
    logger.info("[TOOLBOX] ✅ Agent toolbox loaded successfully")
else:
    logger.error("[TOOLBOX] ❌ Agent toolbox loading failed")
