"""
Universal User Verification Utilities.

These utilities can be used by ANY agent that needs user access verification.
Moved from agent-specific code to improve portability and separation of concerns.

Key Features:
- User access verification with database lookups
- Employee and customer verification
- Admin user handling
- Comprehensive error handling and logging
- Portable across all agent implementations

PORTABLE USAGE FOR ANY AGENT:
===============================

```python
from utils.user_verification import verify_user_access, verify_employee_access, handle_access_denied

class MyAgent:
    async def process_request(self, user_id: str, state: dict):
        # Verify user access - works with any agent
        user_type = await verify_user_access(user_id)
        
        if user_type == "unknown":
            return await handle_access_denied(state, "Invalid user credentials")
        
        # Process based on user type
        if user_type == "employee":
            # Employee-specific processing
            pass
        elif user_type == "customer":
            # Customer-specific processing
            pass
```

This ensures consistent user verification across all agents without code duplication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage

# Import database client
import sys
import pathlib
backend_path = pathlib.Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.database import db_client

logger = logging.getLogger(__name__)


async def verify_user_access(user_id: str) -> str:
    """
    PORTABLE: Verify user access and return user type.
    
    Can be used by any agent that needs user verification.
    Determines if a user is an employee, customer, or unknown based on database records.

    Args:
        user_id: The users.id (UUID)

    Returns:
        str: User type - 'employee', 'customer', or 'unknown'
    """
    try:
        # Ensure database client is initialized
        await db_client._async_initialize_client()

        # Look up the user in the users table
        logger.info(f"[PORTABLE_USER_VERIFICATION] Looking up user by ID: {user_id}")
        user_result = await asyncio.to_thread(
            lambda: db_client.client.table("users")
            .select("id, user_type, employee_id, customer_id, email, display_name, is_active")
            .eq("id", user_id)
            .execute()
        )

        if not user_result.data or len(user_result.data) == 0:
            logger.warning(f"[PORTABLE_USER_VERIFICATION] No user found for user_id: {user_id}")
            return "unknown"

        user_record = user_result.data[0]
        user_type = user_record.get("user_type", "").lower()
        employee_id = user_record.get("employee_id")
        customer_id = user_record.get("customer_id")
        is_active = user_record.get("is_active", False)
        email = user_record.get("email", "")

        logger.info(f"[PORTABLE_USER_VERIFICATION] User found - Type: {user_type}, Active: {is_active}, Email: {email}")

        # Check if user is active
        if not is_active:
            logger.warning(f"[PORTABLE_USER_VERIFICATION] User {user_id} is inactive - denying access")
            return "unknown"

        # Handle employee users
        if user_type == "employee":
            if not employee_id:
                logger.warning(f"[PORTABLE_USER_VERIFICATION] Employee user {user_id} has no employee_id - treating as unknown")
                return "unknown"

            # Verify employee record exists and is active
            logger.info(f"[PORTABLE_USER_VERIFICATION] Verifying employee record for employee_id: {employee_id}")
            employee_result = await asyncio.to_thread(
                lambda: db_client.client.table("employees")
                .select("id, name, email, position, is_active")
                .eq("id", employee_id)
                .eq("is_active", True)
                .execute()
            )

            if employee_result.data and len(employee_result.data) > 0:
                employee = employee_result.data[0]
                logger.info(f"[PORTABLE_USER_VERIFICATION] ✅ Employee access verified - {employee['name']} ({employee['position']})")
                return "employee"
            else:
                logger.warning(f"[PORTABLE_USER_VERIFICATION] No active employee found for employee_id: {employee_id}")
                return "unknown"

        # Handle customer users
        elif user_type == "customer":
            if not customer_id:
                logger.warning(f"[PORTABLE_USER_VERIFICATION] Customer user {user_id} has no customer_id - treating as unknown")
                return "unknown"

            # Verify customer record exists
            logger.info(f"[PORTABLE_USER_VERIFICATION] Verifying customer record for customer_id: {customer_id}")
            customer_result = await asyncio.to_thread(
                lambda: db_client.client.table("customers")
                .select("id, name, email")
                .eq("id", customer_id)
                .execute()
            )

            if customer_result.data and len(customer_result.data) > 0:
                customer = customer_result.data[0]
                logger.info(f"[PORTABLE_USER_VERIFICATION] ✅ Customer access verified - {customer['name']} ({customer['email']})")
                return "customer"
            else:
                logger.warning(f"[PORTABLE_USER_VERIFICATION] No customer found for customer_id: {customer_id}")
                return "unknown"

        # Handle admin users (treat as employees for now)
        elif user_type == "admin":
            logger.info(f"[PORTABLE_USER_VERIFICATION] ✅ Admin user detected - granting employee-level access")
            return "employee"

        # Handle system users (deny access)
        elif user_type == "system":
            logger.warning(f"[PORTABLE_USER_VERIFICATION] System user detected - denying access for security")
            return "unknown"

        # Handle unknown user types
        else:
            logger.warning(f"[PORTABLE_USER_VERIFICATION] Unknown user_type: {user_type} for user {user_id}")
            return "unknown"

    except Exception as e:
        logger.error(f"[PORTABLE_USER_VERIFICATION] Error verifying user access for user_id {user_id}: {str(e)}")
        # On database errors, return unknown for security
        return "unknown"


async def verify_employee_access(user_id: str) -> bool:
    """
    PORTABLE: Legacy method for backward compatibility.
    
    Can be used by any agent that needs boolean employee verification.
    Uses the new verify_user_access function internally but maintains boolean return type.

    Args:
        user_id: The users.id (UUID)

    Returns:
        bool: True if user is an employee (including admin), False otherwise
    """
    logger.info(f"[PORTABLE_LEGACY] verify_employee_access called - delegating to verify_user_access")
    user_type = await verify_user_access(user_id)
    
    # Return True for employees and admins, False for customers and unknown
    is_employee = user_type in ["employee", "admin"]
    logger.info(f"[PORTABLE_LEGACY] User type '{user_type}' -> employee access: {is_employee}")
    return is_employee


async def handle_access_denied(state: Dict[str, Any], conversation_id: Optional[str], reason: str) -> Dict[str, Any]:
    """
    PORTABLE: Handle access denied scenarios with consistent messaging.
    
    Can be used by any agent to provide consistent access denial responses.
    
    Args:
        state: Agent state (any agent's state format)
        conversation_id: Optional conversation ID
        reason: Reason for access denial
        
    Returns:
        Updated state with access denied message
    """
    logger.warning(f"[PORTABLE_ACCESS_DENIED] Access denied: {reason}")
    
    error_message = AIMessage(
        content="""I apologize, but I'm unable to verify your access to the system.

This could be due to:
- Your account may be inactive or suspended
- You may not have the necessary permissions
- There may be a temporary system issue

Please contact your system administrator for assistance, or try again later."""
    )

    return {
        "messages": state.get("messages", []) + [error_message],
        "conversation_id": conversation_id,
        "user_id": state.get("user_id"),
        "conversation_summary": state.get("conversation_summary"),
        "customer_id": None,
        "employee_id": None,
    }
