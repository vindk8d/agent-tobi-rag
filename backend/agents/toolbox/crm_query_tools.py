"""
CRM Query Tools

Tools for querying customer relationship management data, vehicle inventory,
and sales analytics with proper access control.
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Core imports
from langchain_core.tools import tool
from langsmith import traceable

# Toolbox imports
from .toolbox import (
    get_sql_database,
    get_sql_llm,
    get_current_user_type,
    get_minimal_schema_info,
    get_conversation_context
)

logger = logging.getLogger(__name__)

# =============================================================================
# CRM QUERY PARAMETERS AND TOOLS
# =============================================================================

class SimpleCRMQueryParams(BaseModel):
    """Parameters for CRM database information lookup and analysis (NOT for quotation generation)."""
    question: str = Field(..., description="Natural language question about CRM data: customer details, vehicle inventory, sales analytics, employee info, etc. (NOT for creating quotations)")
    time_period: Optional[str] = Field(None, description="Optional time period filter for data analysis: 'last 30 days', 'this quarter', '2024', etc.")

@tool(args_schema=SimpleCRMQueryParams)
@traceable(name="simple_query_crm_data")
async def simple_query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database for information lookup and analysis (NOT for generating quotations).
    
    **Use this tool for:**
    - Looking up customer information and contact details
    - Searching vehicle inventory and specifications  
    - Analyzing sales data, performance metrics, and trends
    - Checking opportunity status and pipeline information
    - Finding employee information and branch details
    - General CRM data exploration and reporting
    
    **Do NOT use this tool for:**
    - Creating customer quotations (use generate_quotation instead)
    - Generating PDF documents or official quotes
    - Processing quotation approvals or HITL workflows

    This tool provides read-only database access with user type-based security:
    - Employees: Full CRM database access
    - Customers: Limited to vehicle and pricing information only
    
    Args:
        question: Natural language question about CRM data
        time_period: Optional time filter for analytics queries
        
    Returns:
        Formatted response with requested CRM information
    """
    try:
        logger.info(f"[CRM_QUERY] Processing question: {question[:100]}...")
        
        # Get user type for access control
        user_type = get_current_user_type() or "customer"
        logger.info(f"[CRM_QUERY] User type: {user_type}")
        
        # Add time period context if provided
        enhanced_question = question
        if time_period:
            enhanced_question = f"{question} (Time period: {time_period})"
        
        # Try direct Supabase approach first (Option C - more reliable)
        direct_result = await _query_crm_data_direct(enhanced_question, user_type)
        if direct_result and not direct_result.startswith("❌"):
            logger.info("[CRM_QUERY] ✅ Direct query completed successfully")
            return direct_result
        
        # Fallback to SQL interface approach (Option A - backward compatibility)
        logger.info("[CRM_QUERY] Falling back to SQL interface approach")
        
        # Get database connection
        db = await get_sql_database()
        if not db:
            return "❌ Unable to connect to the database. Please try again later."
        
        # Generate SQL query using LLM
        sql_query = await _generate_sql_query_simple(enhanced_question, db, user_type)
        if not sql_query:
            return "❌ I couldn't generate a proper query for your question. Please try rephrasing it."
        
        logger.info(f"[CRM_QUERY] Generated SQL: {sql_query}")
        
        # Execute the query safely
        query_result = await _execute_sql_query_simple(sql_query, db)
        if not query_result:
            return "❌ The query didn't return any results. Please try a different question."
        
        # Format the result for user presentation
        formatted_result = await _format_sql_result_simple(enhanced_question, sql_query, query_result)
        
        logger.info("[CRM_QUERY] ✅ SQL query completed successfully")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in simple_query_crm_data: {e}")
        return f"I encountered an issue while processing your question. Please try rephrasing it or ask for help."

async def _query_crm_data_direct(question: str, user_type: str = "employee") -> str:
    """
    Query CRM data directly using Supabase (Option C).
    
    This approach is more reliable than SQL parsing and consistent with
    the working lookup functions in the codebase.
    """
    try:
        from .toolbox import get_db_client
        
        # Get database client
        db_client = get_db_client()
        if not db_client:
            logger.error("[CRM_DIRECT] No database connection available")
            return "❌ Unable to connect to the database."
        
        # Get Supabase client
        supabase = db_client.client
        if not supabase:
            logger.error("[CRM_DIRECT] No Supabase client available")
            return "❌ Database client not properly initialized."
        
        logger.info(f"[CRM_DIRECT] Processing question with direct Supabase queries: {question}")
        
        # Analyze question to determine what data to fetch
        question_lower = question.lower()
        
        # Handle vehicle inventory questions
        if any(word in question_lower for word in ['vehicle', 'car', 'inventory', 'stock', 'available']):
            return await _query_vehicles_direct(supabase, question, user_type)
        
        # Handle customer questions
        elif any(word in question_lower for word in ['customer', 'client', 'buyer']):
            if user_type != "employee":
                return "❌ Access denied. Customer information is only available to employees."
            return await _query_customers_direct(supabase, question)
        
        # Handle employee questions
        elif any(word in question_lower for word in ['employee', 'staff', 'salesperson', 'agent']):
            if user_type != "employee":
                return "❌ Access denied. Employee information is only available to employees."
            return await _query_employees_direct(supabase, question)
        
        # Default to vehicle inventory for general questions
        else:
            return await _query_vehicles_direct(supabase, question, user_type)
            
    except Exception as e:
        logger.error(f"[CRM_DIRECT] Error in direct query: {e}")
        return "❌ Error processing your question with direct database access."

async def _query_vehicles_direct(supabase, question: str, user_type: str) -> str:
    """Query vehicle data directly from Supabase."""
    try:
        logger.info("[CRM_DIRECT] Querying vehicles table")
        
        # Get vehicles data
        result = supabase.table('vehicles').select('*').limit(50).execute()
        
        if not result.data:
            return "No vehicles found in the inventory."
        
        # Format results for user presentation
        vehicles = result.data
        
        # Use LLM to format the response based on the question
        from .toolbox import get_appropriate_llm
        llm = await get_appropriate_llm(question)
        
        # Create a comprehensive summary of the data for LLM processing
        vehicle_summary = []
        for vehicle in vehicles[:20]:  # Limit to 20 for LLM processing
            # Basic info
            summary = f"- {vehicle.get('year', 'N/A')} {vehicle.get('brand', vehicle.get('make', 'N/A'))} {vehicle.get('model', 'N/A')}"
            
            # Color and variant
            if vehicle.get('color'):
                summary += f" in {vehicle.get('color')}"
            if vehicle.get('variant'):
                summary += f" ({vehicle.get('variant')})"
            
            # Availability and stock
            if vehicle.get('stock_quantity') is not None:
                summary += f" - Stock: {vehicle.get('stock_quantity')} units"
            if vehicle.get('is_available') is False:
                summary += " - UNAVAILABLE"
            
            # Engine and performance
            if vehicle.get('engine_type'):
                summary += f" - {vehicle.get('engine_type')}"
            if vehicle.get('power_ps'):
                summary += f" {vehicle.get('power_ps')}PS"
            if vehicle.get('transmission'):
                summary += f" {vehicle.get('transmission')}"
            
            # Price info (if available)
            if vehicle.get('price'):
                summary += f" - ${vehicle.get('price'):,.2f}" if isinstance(vehicle.get('price'), (int, float)) else f" - {vehicle.get('price')}"
            if vehicle.get('status'):
                summary += f" ({vehicle.get('status')})"
                
            vehicle_summary.append(summary)
        
        format_prompt = f"""You are a helpful car sales assistant. A user asked: "{question}"

Here is our current vehicle inventory:
{chr(10).join(vehicle_summary)}

Total vehicles in inventory: {len(vehicles)}

Please provide a helpful response that:
1. Directly answers their question about colors, quantities, or specific details
2. If asked about colors or quantities per color, analyze the data and provide specific counts
3. Highlights relevant vehicles if they're looking for something specific
4. Uses a friendly, conversational tone
5. Includes key details like make, model, year, color, price, and availability
6. If asked about color breakdown, count how many vehicles are available in each color
7. Suggests next steps if appropriate

Format your response in a clear, easy-to-read way. Be specific with numbers and details when the user asks for them."""

        response = await llm.ainvoke([{"role": "user", "content": format_prompt}])
        return response.content
        
    except Exception as e:
        logger.error(f"[CRM_DIRECT] Error querying vehicles: {e}")
        return "❌ Error retrieving vehicle information."

async def _query_customers_direct(supabase, question: str) -> str:
    """Query customer data directly from Supabase."""
    try:
        logger.info("[CRM_DIRECT] Querying customers table")
        
        result = supabase.table('customers').select('*').limit(30).execute()
        
        if not result.data:
            return "No customer records found."
        
        # Format customer data for LLM processing  
        customers = result.data
        customer_summary = []
        for customer in customers[:15]:  # Limit for LLM processing
            summary = f"- {customer.get('name', 'N/A')}"
            if customer.get('company'):
                summary += f" ({customer.get('company')})"
            if customer.get('is_for_business'):
                summary += " [Business Customer]"
            if customer.get('phone') or customer.get('mobile_number'):
                phone = customer.get('mobile_number') or customer.get('phone')
                summary += f" - {phone}"
            if customer.get('email'):
                summary += f" - {customer.get('email')}"
            if customer.get('address'):
                # Show just city/area from address
                addr_parts = customer.get('address', '').split(',')
                if len(addr_parts) >= 2:
                    summary += f" - {addr_parts[-2].strip()}"
            customer_summary.append(summary)
        
        from .toolbox import get_appropriate_llm
        llm = await get_appropriate_llm(question)
        
        format_prompt = f"""You are a CRM assistant. A user asked: "{question}"

Here are our customer records:
{chr(10).join(customer_summary)}

Total customers: {len(customers)}

Please provide a helpful response that:
1. Directly answers their question about customers
2. Summarizes key customer information
3. Maintains customer privacy (don't include sensitive details)
4. Uses a professional tone
5. Suggests next steps if appropriate"""

        response = await llm.ainvoke([{"role": "user", "content": format_prompt}])
        return response.content
        
    except Exception as e:
        logger.error(f"[CRM_DIRECT] Error querying customers: {e}")
        return "❌ Error retrieving customer information."

async def _query_employees_direct(supabase, question: str) -> str:
    """Query employee data directly from Supabase."""
    try:
        logger.info("[CRM_DIRECT] Querying employees table")
        
        result = supabase.table('employees').select('*').limit(20).execute()
        
        if not result.data:
            return "No employee records found."
        
        # Format employee data for LLM processing
        employees = result.data
        employee_summary = []
        for employee in employees:
            summary = f"- {employee.get('name', 'N/A')} - {employee.get('position', 'N/A')}"
            if employee.get('branch_id'):
                # We could join with branches table, but for now show the ID
                summary += f" (Branch: {employee.get('branch_id', 'N/A')[:8]}...)"
            if employee.get('email'):
                summary += f" - {employee.get('email')}"
            if employee.get('phone'):
                summary += f" - {employee.get('phone')}"
            if employee.get('is_active') is False:
                summary += " [INACTIVE]"
            employee_summary.append(summary)
        
        from .toolbox import get_appropriate_llm
        llm = await get_appropriate_llm(question)
        
        format_prompt = f"""You are an HR assistant. A user asked: "{question}"

Here are our employee records:
{chr(10).join(employee_summary)}

Total employees: {len(employees)}

Please provide a helpful response that:
1. Directly answers their question about employees
2. Summarizes key employee information
3. Uses a professional tone
4. Respects employee privacy
5. Suggests next steps if appropriate"""

        response = await llm.ainvoke([{"role": "user", "content": format_prompt}])
        return response.content
        
    except Exception as e:
        logger.error(f"[CRM_DIRECT] Error querying employees: {e}")
        return "❌ Error retrieving employee information."

async def _generate_sql_query_simple(question: str, db, user_type: str = "employee") -> str:
    """Generate SQL query from natural language question."""
    try:
        # Get appropriate schema info based on user type
        if user_type == "customer":
            schema_info = _get_customer_table_info(db)
        else:
            schema_info = get_minimal_schema_info(user_type)
        
        # Get conversation context for better query generation
        conversation_context = await get_conversation_context()
        
        # Create prompt for SQL generation
        prompt = f"""You are a SQL expert. Generate a safe, read-only SQL query based on the user's question.

Database Schema:
{schema_info}

User Question: {question}
User Type: {user_type}
Conversation Context: {conversation_context[:200]}...

IMPORTANT RULES:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Always use LIMIT to prevent large result sets (max 50 rows)
3. For customer users, only query vehicles and pricing tables
4. Use proper JOIN syntax when needed
5. Handle NULL values appropriately
6. Include relevant WHERE clauses for filtering
7. Use LIKE for text searches with proper wildcards

Return ONLY the SQL query, no explanations."""

        # Get SQL-optimized LLM and generate query
        llm = await get_sql_llm(question)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        
        # Clean up the response
        sql_query = response.content.strip()
        
        # Remove markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Basic safety check
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"[CRM_QUERY] Dangerous keyword detected: {keyword}")
                return ""
        
        return sql_query
        
    except Exception as e:
        logger.error(f"[CRM_QUERY] Error generating SQL: {e}")
        return ""

async def _execute_sql_query_simple(query: str, db) -> str:
    """Execute SQL query safely and return results."""
    try:
        # Execute the query
        result = db.run(query)
        
        if not result or not result.strip():
            return ""
        
        # Log successful execution
        lines = result.strip().split('\n')
        row_count = max(0, len(lines) - 1)  # Subtract header row
        logger.info(f"[CRM_QUERY] Query executed successfully, returned {row_count} rows")
        
        return result
        
    except Exception as e:
        logger.error(f"[CRM_QUERY] Error executing query: {e}")
        return ""

async def _format_sql_result_simple(question: str, query: str, result: str) -> str:
    """Format SQL query results for user presentation."""
    try:
        if not result or not result.strip():
            return "No results found for your query."
        
        # Get formatting LLM
        from .toolbox import get_appropriate_llm
        llm = await get_appropriate_llm(question)
        
        # Create formatting prompt
        format_prompt = f"""You are a helpful assistant formatting database query results for users.

Original Question: {question}
SQL Query: {query}

Raw Database Results:
{result}

Please format these results in a clear, user-friendly way:
1. Use a conversational tone
2. Present data in tables or lists as appropriate
3. Highlight key insights or patterns
4. If no data found, explain that clearly
5. Keep technical jargon to a minimum
6. Add relevant context or explanations where helpful

Format the response to directly answer the user's question."""

        # Get formatted response
        response = await llm.ainvoke([{"role": "user", "content": format_prompt}])
        return response.content
        
    except Exception as e:
        logger.error(f"[CRM_QUERY] Error formatting results: {e}")
        # Fallback to raw results if formatting fails
        return f"Here are the results for your query:\n\n{result}"

def _get_customer_table_info(db) -> str:
    """
    Get table information filtered for customer access.
    Only includes vehicles and pricing tables.
    """
    try:
        # Get full table info
        full_table_info = db.get_table_info()
        
        # Filter for customer-allowed tables only
        allowed_tables = ["vehicles", "pricing"]
        filtered_lines = []
        current_table = None
        include_line = False
        
        for line in full_table_info.split('\n'):
            line_lower = line.lower()
            
            # Check if this line starts a new table definition
            if line.startswith('CREATE TABLE'):
                table_name = None
                for allowed in allowed_tables:
                    if f'"{allowed}"' in line_lower or f'`{allowed}`' in line_lower or f' {allowed} ' in line_lower:
                        table_name = allowed
                        break
                
                if table_name:
                    current_table = table_name
                    include_line = True
                    filtered_lines.append(line)
                else:
                    current_table = None
                    include_line = False
            elif include_line:
                filtered_lines.append(line)
                # Stop including lines after the table definition ends
                if line.strip() == ');' or line.strip() == ')':
                    include_line = False
        
        if filtered_lines:
            return '\n'.join(filtered_lines)
        else:
            # Fallback to basic schema
            return """
CREATE TABLE vehicles (
    id TEXT,
    make TEXT,
    model TEXT,
    year INTEGER,
    type TEXT,
    price DECIMAL,
    status TEXT,
    color TEXT,
    mileage INTEGER
);

CREATE TABLE pricing (
    vehicle_id TEXT,
    base_price DECIMAL,
    discount DECIMAL,
    final_price DECIMAL
);
"""
    except Exception as e:
        logger.error(f"Error getting customer table info: {e}")
        return "Limited database access available."

# =============================================================================
# SCHEMA INFORMATION TOOLS
# =============================================================================

@tool
async def get_detailed_schema(table_names: str) -> str:
    """
    Get detailed schema information for specific database tables.
    
    Args:
        table_names: Comma-separated list of table names to get schema for
        
    Returns:
        Detailed schema information including columns, types, and relationships
    """
    try:
        logger.info(f"[SCHEMA] Getting detailed schema for tables: {table_names}")
        
        # Get database connection
        db = await get_sql_database()
        if not db:
            return "❌ Unable to connect to the database."
        
        # Clean up table names
        tables = [name.strip() for name in table_names.split(',')]
        
        # Get user type for access control
        user_type = get_current_user_type() or "customer"
        
        # Filter tables based on user access
        if user_type == "customer":
            allowed_tables = ["vehicles", "pricing"]
            tables = [t for t in tables if t.lower() in allowed_tables]
            if not tables:
                return "❌ Access denied. Customers can only access vehicle and pricing information."
        
        # Get schema information
        schema_info = []
        for table in tables:
            try:
                # Get table info from database
                table_info = db.get_table_info([table])
                if table_info:
                    schema_info.append(f"=== {table.upper()} TABLE ===\n{table_info}")
            except Exception as e:
                schema_info.append(f"=== {table.upper()} TABLE ===\nError retrieving schema: {e}")
        
        if schema_info:
            result = "\n\n".join(schema_info)
            logger.info(f"[SCHEMA] ✅ Schema retrieved for {len(schema_info)} tables")
            return result
        else:
            return "❌ No schema information found for the requested tables."
            
    except Exception as e:
        logger.error(f"[SCHEMA] Error getting detailed schema: {e}")
        return f"❌ Error retrieving schema information: {e}"
