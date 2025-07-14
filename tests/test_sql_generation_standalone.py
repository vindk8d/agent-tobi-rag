#!/usr/bin/env python3
"""
Standalone test script for SQL generation data structures and helper functions.

This script tests the core functionality without requiring LangChain dependencies:
1. Schema documentation structure
2. Business context prompts  
3. Helper function logic

Run with: python3 test_sql_generation_standalone.py
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional, Set, Tuple

# Replicate the essential data structures from tools.py for testing
CRM_SCHEMA_DOCS = {
    "database_overview": {
        "description": "CRM Sales Management System for automotive dealerships",
        "business_domain": "Vehicle sales, customer management, and opportunity tracking",
        "key_metrics": ["sales_performance", "conversion_rates", "revenue", "customer_satisfaction"]
    },
    
    "tables": {
        "branches": {
            "description": "Company branches and dealership locations",
            "business_purpose": "Track performance across different locations and regions",
            "primary_key": "id",
            "columns": {
                "id": {"type": "UUID", "description": "Unique branch identifier"},
                "name": {"type": "VARCHAR(255)", "description": "Branch name (e.g., 'Downtown Toyota', 'North Honda')"},
                "region": {"type": "ENUM", "description": "Geographic region", "values": ["north", "south", "east", "west", "central"]},
                "address": {"type": "TEXT", "description": "Physical address of the branch"},
                "brand": {"type": "VARCHAR(255)", "description": "Vehicle brand sold at this branch (Toyota, Honda, etc.)"},
                "created_at": {"type": "TIMESTAMP", "description": "When branch was added to system"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "performance_metrics": "Analyze sales by region, compare branch performance",
                "common_queries": ["regional_sales", "branch_comparison", "territory_analysis"]
            }
        },
        
        "employees": {
            "description": "Sales staff and management hierarchy",
            "business_purpose": "Track sales performance, manage teams, calculate commissions",
            "primary_key": "id",
            "foreign_keys": {
                "branch_id": "branches(id)",
                "manager_ae_id": "employees(id) - self-referencing for hierarchy"
            },
            "columns": {
                "id": {"type": "UUID", "description": "Unique employee identifier"},
                "branch_id": {"type": "UUID", "description": "Branch where employee works"},
                "name": {"type": "VARCHAR(255)", "description": "Employee full name"},
                "position": {"type": "ENUM", "description": "Job role", "values": ["sales_agent", "account_executive", "manager", "director", "admin"]},
                "manager_ae_id": {"type": "UUID", "description": "Manager's ID for hierarchy (NULL for top-level)"},
                "email": {"type": "VARCHAR(255)", "description": "Work email address"},
                "phone": {"type": "VARCHAR(50)", "description": "Work phone number"},
                "hire_date": {"type": "DATE", "description": "Employment start date"},
                "is_active": {"type": "BOOLEAN", "description": "Whether employee is currently active"},
                "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "hierarchy": "Use manager_ae_id for org chart, performance rollups",
                "performance_metrics": "Sales volume, conversion rates, customer satisfaction",
                "common_queries": ["top_performers", "team_performance", "commission_calculations"]
            }
        },
        
        "customers": {
            "description": "Customer contact information and business details",
            "business_purpose": "Manage customer relationships, track preferences, analyze demographics",
            "primary_key": "id",
            "columns": {
                "id": {"type": "UUID", "description": "Unique customer identifier"},
                "name": {"type": "VARCHAR(255)", "description": "Customer full name or business name"},
                "phone": {"type": "VARCHAR(50)", "description": "Primary phone number"},
                "mobile_number": {"type": "VARCHAR(50)", "description": "Mobile phone number"},
                "email": {"type": "VARCHAR(255)", "description": "Email address"},
                "company": {"type": "VARCHAR(255)", "description": "Company name if business customer"},
                "is_for_business": {"type": "BOOLEAN", "description": "Whether this is a business or individual customer"},
                "address": {"type": "TEXT", "description": "Customer address"},
                "notes": {"type": "TEXT", "description": "Additional notes about customer preferences"},
                "created_at": {"type": "TIMESTAMP", "description": "When customer was added"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "segmentation": "B2B vs B2C customers, analyze buying patterns",
                "preferences": "Track vehicle preferences in notes and opportunities",
                "common_queries": ["customer_lifetime_value", "repeat_customers", "referral_analysis"]
            }
        },
        
        "vehicles": {
            "description": "Vehicle inventory with specifications and availability",
            "business_purpose": "Manage inventory, track popular models, pricing analysis",
            "primary_key": "id",
            "columns": {
                "id": {"type": "UUID", "description": "Unique vehicle identifier"},
                "brand": {"type": "VARCHAR(255)", "description": "Vehicle manufacturer (Toyota, Honda, Ford, etc.)"},
                "year": {"type": "INTEGER", "description": "Model year (1900-2100)"},
                "model": {"type": "VARCHAR(255)", "description": "Vehicle model name (Camry, Civic, F-150, etc.)"},
                "type": {"type": "ENUM", "description": "Vehicle category", "values": ["sedan", "suv", "hatchback", "pickup", "van", "motorcycle", "truck"]},
                "color": {"type": "VARCHAR(100)", "description": "Vehicle color"},
                "acceleration": {"type": "DECIMAL(5,2)", "description": "0-100 km/h acceleration time in seconds"},
                "power": {"type": "INTEGER", "description": "Engine horsepower"},
                "torque": {"type": "INTEGER", "description": "Engine torque in Nm"},
                "fuel_type": {"type": "VARCHAR(50)", "description": "Fuel type (gasoline, hybrid, electric, diesel)"},
                "transmission": {"type": "VARCHAR(50)", "description": "Transmission type (manual, automatic, CVT)"},
                "is_available": {"type": "BOOLEAN", "description": "Whether vehicle is available for sale"},
                "stock_quantity": {"type": "INTEGER", "description": "Number of units in stock"},
                "created_at": {"type": "TIMESTAMP", "description": "When vehicle was added to inventory"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "inventory_management": "Track stock levels, popular models, turn rates",
                "performance_specs": "Use acceleration, power, torque for customer matching",
                "common_queries": ["inventory_analysis", "popular_models", "availability_reports"]
            }
        },
        
        "opportunities": {
            "description": "Sales opportunities and lead tracking through the sales pipeline",
            "business_purpose": "Manage sales pipeline, forecast revenue, track conversion",
            "primary_key": "id",
            "foreign_keys": {
                "customer_id": "customers(id)",
                "vehicle_id": "vehicles(id)",
                "opportunity_salesperson_ae_id": "employees(id)"
            },
            "columns": {
                "id": {"type": "UUID", "description": "Unique opportunity identifier"},
                "customer_id": {"type": "UUID", "description": "Associated customer"},
                "vehicle_id": {"type": "UUID", "description": "Vehicle of interest (can be NULL for general inquiry)"},
                "opportunity_salesperson_ae_id": {"type": "UUID", "description": "Assigned sales representative"},
                "created_date": {"type": "TIMESTAMP", "description": "When opportunity was created"},
                "stage": {"type": "ENUM", "description": "Current sales stage", "values": ["New", "Contacted", "Consideration", "Purchase Intent", "Won", "Lost"]},
                "warmth": {"type": "ENUM", "description": "Lead quality", "values": ["hot", "warm", "cold", "dormant"]},
                "referral_name": {"type": "VARCHAR(255)", "description": "Who referred this customer"},
                "expected_close_date": {"type": "DATE", "description": "Estimated date for closing the sale"},
                "probability": {"type": "INTEGER", "description": "Probability of closing (0-100%)"},
                "estimated_value": {"type": "DECIMAL(12,2)", "description": "Expected sale value"},
                "notes": {"type": "TEXT", "description": "Notes about customer interaction and preferences"},
                "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "pipeline_management": "Track progression through stages, identify bottlenecks",
                "forecasting": "Use probability and estimated_value for revenue forecasting",
                "performance": "Measure conversion rates, time in each stage",
                "common_queries": ["pipeline_analysis", "conversion_rates", "revenue_forecast", "lead_quality"]
            }
        },
        
        "transactions": {
            "description": "Completed sales transactions and financial records",
            "business_purpose": "Track revenue, commissions, payment methods, completed deals",
            "primary_key": "id",
            "foreign_keys": {
                "opportunity_id": "opportunities(id)",
                "opportunity_salesperson_ae_id": "employees(id)",
                "assignee_id": "employees(id)"
            },
            "columns": {
                "id": {"type": "UUID", "description": "Unique transaction identifier"},
                "opportunity_id": {"type": "UUID", "description": "Related opportunity that was converted"},
                "opportunity_salesperson_ae_id": {"type": "UUID", "description": "Sales rep who closed the deal"},
                "assignee_id": {"type": "UUID", "description": "Employee assigned to handle transaction processing"},
                "created_date": {"type": "TIMESTAMP", "description": "When transaction was initiated"},
                "end_date": {"type": "TIMESTAMP", "description": "When transaction was completed"},
                "status": {"type": "ENUM", "description": "Transaction status", "values": ["pending", "processing", "completed", "cancelled", "refunded"]},
                "total_amount": {"type": "DECIMAL(12,2)", "description": "Total transaction amount"},
                "payment_method": {"type": "VARCHAR(100)", "description": "How customer paid (cash, financing, trade-in)"},
                "transaction_reference": {"type": "VARCHAR(255)", "description": "External reference number"},
                "notes": {"type": "TEXT", "description": "Transaction notes and special conditions"},
                "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "revenue_tracking": "Calculate actual revenue, commissions, refunds",
                "payment_analysis": "Analyze payment method preferences, financing rates",
                "common_queries": ["revenue_reports", "commission_calculations", "payment_method_analysis"]
            }
        },
        
        "pricing": {
            "description": "Vehicle pricing with discounts, promotions, and additional services",
            "business_purpose": "Manage pricing strategies, promotions, calculate final prices",
            "primary_key": "id",
            "foreign_keys": {
                "vehicle_id": "vehicles(id)"
            },
            "columns": {
                "id": {"type": "UUID", "description": "Unique pricing record identifier"},
                "vehicle_id": {"type": "UUID", "description": "Associated vehicle"},
                "availability_start_date": {"type": "DATE", "description": "When this pricing becomes effective"},
                "availability_end_date": {"type": "DATE", "description": "When this pricing expires (NULL for ongoing)"},
                "warranty": {"type": "VARCHAR(255)", "description": "Warranty terms and duration"},
                "insurance": {"type": "DECIMAL(12,2)", "description": "Insurance cost"},
                "lto": {"type": "DECIMAL(12,2)", "description": "Land Transport Office fees"},
                "price_discount": {"type": "DECIMAL(12,2)", "description": "Discount on base price"},
                "insurance_discount": {"type": "DECIMAL(12,2)", "description": "Discount on insurance"},
                "lto_discount": {"type": "DECIMAL(12,2)", "description": "Discount on LTO fees"},
                "add_ons": {"type": "JSONB", "description": "Additional accessories and services with prices"},
                "warranty_promo": {"type": "TEXT", "description": "Special warranty promotions"},
                "base_price": {"type": "DECIMAL(12,2)", "description": "Base vehicle price before discounts"},
                "final_price": {"type": "DECIMAL(12,2)", "description": "Final price after all discounts"},
                "is_active": {"type": "BOOLEAN", "description": "Whether this pricing is currently active"},
                "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "pricing_strategy": "Analyze discount effectiveness, promotional impact",
                "profit_margins": "Calculate margins after all discounts and fees",
                "common_queries": ["pricing_analysis", "discount_effectiveness", "margin_analysis"]
            }
        },
        
        "activities": {
            "description": "Sales activities and customer interaction tracking",
            "business_purpose": "Track sales process, measure activity effectiveness, follow-up management",
            "primary_key": "id",
            "foreign_keys": {
                "opportunity_salesperson_ae_id": "employees(id)",
                "customer_id": "customers(id)",
                "opportunity_id": "opportunities(id)"
            },
            "columns": {
                "id": {"type": "UUID", "description": "Unique activity identifier"},
                "opportunity_salesperson_ae_id": {"type": "UUID", "description": "Employee who performed the activity"},
                "customer_id": {"type": "UUID", "description": "Customer involved in the activity"},
                "opportunity_id": {"type": "UUID", "description": "Related opportunity (can be NULL for general activities)"},
                "activity_type": {"type": "ENUM", "description": "Type of activity", "values": ["call", "email", "meeting", "demo", "follow_up", "proposal_sent", "contract_signed"]},
                "subject": {"type": "VARCHAR(255)", "description": "Brief activity description"},
                "description": {"type": "TEXT", "description": "Detailed activity notes"},
                "created_date": {"type": "TIMESTAMP", "description": "When activity was performed"},
                "end_date": {"type": "TIMESTAMP", "description": "When activity was completed"},
                "is_completed": {"type": "BOOLEAN", "description": "Whether activity is finished"},
                "priority": {"type": "VARCHAR(20)", "description": "Activity priority", "values": ["low", "medium", "high", "urgent"]},
                "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                "updated_at": {"type": "TIMESTAMP", "description": "Last update timestamp"}
            },
            "business_logic": {
                "activity_tracking": "Measure sales rep activity levels, effectiveness",
                "follow_up_management": "Track pending activities, overdue follow-ups",
                "common_queries": ["activity_reports", "follow_up_analysis", "sales_process_metrics"]
            }
        }
    },
    
    "relationships": {
        "primary_relationships": [
            "branches → employees (one-to-many): Each employee works at one branch",
            "employees → employees (manager hierarchy): Self-referencing for management structure",
            "customers → opportunities (one-to-many): Each customer can have multiple opportunities",
            "vehicles → opportunities (one-to-many): Each vehicle can be of interest to multiple customers",
            "employees → opportunities (one-to-many): Each sales rep handles multiple opportunities",
            "opportunities → transactions (one-to-one): Each opportunity can convert to one transaction",
            "vehicles → pricing (one-to-many): Each vehicle can have multiple pricing records over time",
            "customers → activities (one-to-many): Each customer can have multiple activities",
            "opportunities → activities (one-to-many): Each opportunity can have multiple activities",
            "employees → activities (one-to-many): Each employee performs multiple activities"
        ],
        "key_joins": {
            "sales_performance": "employees → opportunities → transactions",
            "customer_journey": "customers → opportunities → activities → transactions",
            "inventory_analysis": "vehicles → pricing → opportunities",
            "branch_performance": "branches → employees → opportunities → transactions"
        }
    },
    
    "views": {
        "sales_pipeline": {
            "description": "Active opportunities with customer, vehicle, and salesperson details",
            "purpose": "Get overview of current sales pipeline",
            "includes": ["opportunity details", "customer info", "vehicle info", "salesperson", "branch"]
        },
        "employee_hierarchy": {
            "description": "Recursive view showing organizational structure",
            "purpose": "Understand reporting relationships and team structure",
            "includes": ["employee details", "manager relationships", "hierarchy levels"]
        }
    }
}

# SQL Generation Prompts with Business Context
SQL_GENERATION_PROMPTS = {
    "system_prompt": """You are an expert SQL analyst for an automotive dealership CRM system. Your role is to convert natural language business questions into efficient, secure SQL queries.

BUSINESS CONTEXT:
- This is a vehicle sales CRM system for automotive dealerships
- Focus on sales performance, customer relationships, and inventory management
- Common business metrics: revenue, conversion rates, sales volume, customer satisfaction
- Sales pipeline stages: New → Contacted → Consideration → Purchase Intent → Won/Lost

IMPORTANT RULES:
1. ONLY generate SELECT queries - no INSERT, UPDATE, DELETE, CREATE, DROP, etc.
2. ONLY query these allowed tables: branches, employees, customers, vehicles, opportunities, transactions, pricing, activities
3. Always use proper JOINs to get complete business context
4. Apply business filters (e.g., is_active=true for employees, is_available=true for vehicles)
5. Use meaningful aliases and clear column names in results
6. Include relevant dates/timestamps for time-based analysis
7. Handle NULL values appropriately (especially for optional foreign keys)
8. Use appropriate aggregations for business metrics
9. Limit results to reasonable numbers (use LIMIT when appropriate)
10. Generate efficient queries with proper indexing in mind

COMMON BUSINESS PATTERNS:
- Sales Performance: Analyze by employee, branch, time period, product
- Customer Analysis: Lifetime value, preferences, conversion patterns
- Inventory Insights: Popular models, availability, pricing trends
- Pipeline Management: Stage progression, probability analysis, forecasting
- Activity Tracking: Sales process effectiveness, follow-up management""",

    "common_scenarios": {
        "top_performers": {
            "keywords": ["top", "best", "highest", "performing", "salesperson", "employee"],
            "sql_pattern": "employees → opportunities → transactions with revenue aggregation and ranking"
        },
        "sales_by_region": {
            "keywords": ["region", "branch", "location", "territory"],
            "sql_pattern": "branches → employees → opportunities → transactions grouped by region"
        },
        "customer_preferences": {
            "keywords": ["customer", "prefer", "interest", "popular", "demand"],
            "sql_pattern": "customers → opportunities → vehicles with aggregation by vehicle attributes"
        },
        "pipeline_health": {
            "keywords": ["pipeline", "funnel", "stage", "conversion", "forecast"],
            "sql_pattern": "opportunities analysis by stage with probability and value metrics"
        },
        "inventory_status": {
            "keywords": ["inventory", "stock", "available", "popular", "model"],
            "sql_pattern": "vehicles → pricing with availability and interest metrics"
        },
        "revenue_analysis": {
            "keywords": ["revenue", "sales", "income", "profit", "money", "earnings"],
            "sql_pattern": "transactions analysis with time-based grouping and aggregations"
        }
    }
}


# Helper functions (replicated from tools.py for testing)
def _build_schema_context() -> str:
    """Build formatted schema context for SQL generation."""
    context = "DATABASE TABLES AND RELATIONSHIPS:\n\n"
    
    for table_name, table_info in CRM_SCHEMA_DOCS["tables"].items():
        context += f"TABLE: {table_name}\n"
        context += f"Purpose: {table_info['business_purpose']}\n"
        
        # Key columns
        key_columns = []
        for col_name, col_info in table_info["columns"].items():
            if col_name in ["id", "name"] or "date" in col_name.lower() or col_name.endswith("_id"):
                key_columns.append(f"{col_name} ({col_info['type']}): {col_info['description']}")
        
        context += f"Key Columns: {', '.join(key_columns[:5])}\n"
        
        # Foreign keys
        if "foreign_keys" in table_info:
            fks = [f"{k} → {v}" for k, v in table_info["foreign_keys"].items()]
            context += f"Relationships: {', '.join(fks)}\n"
            
        context += "\n"
    
    return context


def _build_business_patterns_context() -> str:
    """Build business patterns context for SQL generation."""
    context = "COMMON BUSINESS QUERY PATTERNS:\n\n"
    
    for pattern_name, pattern_info in SQL_GENERATION_PROMPTS["common_scenarios"].items():
        context += f"{pattern_name.upper()}: {pattern_info['sql_pattern']}\n"
        context += f"Keywords: {', '.join(pattern_info['keywords'])}\n\n"
    
    return context


def _identify_query_pattern(question: str) -> str:
    """Identify the most relevant query pattern based on question keywords."""
    question_lower = question.lower()
    
    # Count keyword matches for each pattern
    pattern_scores = {}
    for pattern_name, pattern_info in SQL_GENERATION_PROMPTS["common_scenarios"].items():
        score = sum(1 for keyword in pattern_info["keywords"] if keyword in question_lower)
        if score > 0:
            pattern_scores[pattern_name] = score
    
    # Return the pattern with highest score
    if pattern_scores:
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        return f"{best_pattern[0]} (confidence: {best_pattern[1]} keyword matches)"
    
    return "general_analysis (no specific pattern detected)"


def _build_time_filter(time_period: str) -> str:
    """Build time filter clause based on time period description."""
    time_period_lower = time_period.lower()
    
    # Map common time periods to SQL filters
    time_filters = {
        "last 30 days": "created_date >= CURRENT_DATE - INTERVAL '30 days'",
        "last month": "created_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
        "this month": "EXTRACT(MONTH FROM created_date) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "this quarter": "EXTRACT(QUARTER FROM created_date) = EXTRACT(QUARTER FROM CURRENT_DATE) AND EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "this year": "EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "year to date": "EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "last 7 days": "created_date >= CURRENT_DATE - INTERVAL '7 days'",
        "last week": "created_date >= DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')",
        "last 90 days": "created_date >= CURRENT_DATE - INTERVAL '90 days'"
    }
    
    for period, filter_clause in time_filters.items():
        if period in time_period_lower:
            return filter_clause
    
    # Return empty string if no match found
    return ""


def _format_business_results(data: str, question: str) -> str:
    """Format SQL query results for business users with context and insights."""
    if not data or data.strip() == "":
        return f"📊 **Query Results for:** {question}\n\n✨ No data found matching your criteria."
    
    # Parse the data to understand structure
    lines = data.strip().split('\n')
    if len(lines) <= 1:
        return f"📊 **Query Results for:** {question}\n\n✨ No results returned."
    
    # Extract headers and row count
    header_line = lines[0] if lines else ""
    data_lines = lines[1:] if len(lines) > 1 else []
    row_count = len(data_lines)
    
    # Build formatted response
    formatted = f"📊 **CRM Query Results**\n\n"
    formatted += f"**Question:** {question}\n\n"
    formatted += f"**Results:** {row_count} record(s) found\n\n"
    
    # Add the data in a readable format
    formatted += "```\n"
    formatted += data
    formatted += "\n```\n\n"
    
    # Add business insights based on result pattern
    insights = _generate_business_insights(data, question, row_count)
    if insights:
        formatted += f"💡 **Business Insights:**\n{insights}"
    
    return formatted


def _generate_business_insights(data: str, question: str, row_count: int) -> str:
    """Generate business insights based on query results."""
    question_lower = question.lower()
    insights = []
    
    # Performance insights
    if any(word in question_lower for word in ["performance", "top", "best", "sales"]):
        if row_count == 0:
            insights.append("• No performance data found for the specified criteria")
        elif row_count < 5:
            insights.append(f"• Limited dataset with {row_count} record(s) - consider expanding time range")
        else:
            insights.append(f"• Performance analysis based on {row_count} records")
    
    # Revenue insights
    if any(word in question_lower for word in ["revenue", "money", "sales", "income"]):
        insights.append("• Review revenue trends to identify growth opportunities")
        insights.append("• Compare results across different time periods for better context")
    
    # Pipeline insights
    if any(word in question_lower for word in ["pipeline", "opportunities", "conversion"]):
        insights.append("• Monitor conversion rates between pipeline stages")
        insights.append("• Focus on high-probability opportunities for better forecasting")
    
    # Customer insights
    if any(word in question_lower for word in ["customer", "client"]):
        insights.append("• Analyze customer patterns to improve retention strategies")
        if "lifetime" in question_lower:
            insights.append("• High-value customers deserve personalized attention")
    
    return "\n".join(insights) if insights else ""


# Test functions
def test_schema_documentation():
    """Test that schema documentation is properly structured."""
    print("🧪 Testing Schema Documentation...")
    
    # Test that all required tables are present
    required_tables = ["branches", "employees", "customers", "vehicles", 
                      "opportunities", "transactions", "pricing", "activities"]
    
    assert "tables" in CRM_SCHEMA_DOCS, "Schema docs missing 'tables' section"
    
    for table in required_tables:
        assert table in CRM_SCHEMA_DOCS["tables"], f"Missing table: {table}"
        table_info = CRM_SCHEMA_DOCS["tables"][table]
        
        # Check required fields
        assert "description" in table_info, f"Table {table} missing description"
        assert "business_purpose" in table_info, f"Table {table} missing business_purpose"
        assert "columns" in table_info, f"Table {table} missing columns"
        
        # Check that primary key columns exist
        columns = table_info["columns"]
        assert "id" in columns, f"Table {table} missing id column"
        assert "created_at" in columns, f"Table {table} missing created_at column"
        assert "updated_at" in columns, f"Table {table} missing updated_at column"
        
        print(f"  ✅ Table '{table}' - {len(columns)} columns documented")
    
    # Test relationships section
    assert "relationships" in CRM_SCHEMA_DOCS, "Schema docs missing relationships"
    assert "primary_relationships" in CRM_SCHEMA_DOCS["relationships"], "Missing primary_relationships"
    assert "key_joins" in CRM_SCHEMA_DOCS["relationships"], "Missing key_joins"
    
    print("  ✅ All schema documentation tests passed!")


def test_sql_generation_prompts():
    """Test that SQL generation prompts are properly structured."""
    print("\n🧪 Testing SQL Generation Prompts...")
    
    # Test system prompt
    assert "system_prompt" in SQL_GENERATION_PROMPTS, "Missing system_prompt"
    system_prompt = SQL_GENERATION_PROMPTS["system_prompt"]
    
    # Check for key business context elements
    assert "automotive dealership" in system_prompt.lower(), "Missing automotive context"
    assert "select" in system_prompt.lower(), "Missing SELECT instruction"
    assert "join" in system_prompt.lower(), "Missing JOIN guidance"
    
    print("  ✅ System prompt contains required business context")
    
    # Test common scenarios
    assert "common_scenarios" in SQL_GENERATION_PROMPTS, "Missing common_scenarios"
    scenarios = SQL_GENERATION_PROMPTS["common_scenarios"]
    
    for scenario_name, scenario_info in scenarios.items():
        assert "keywords" in scenario_info, f"Scenario {scenario_name} missing keywords"
        assert "sql_pattern" in scenario_info, f"Scenario {scenario_name} missing sql_pattern"
        assert len(scenario_info["keywords"]) > 0, f"Scenario {scenario_name} has no keywords"
    
    print("  ✅ All SQL generation prompt tests passed!")


def test_helper_functions():
    """Test the helper functions for SQL generation."""
    print("\n🧪 Testing Helper Functions...")
    
    # Test schema context building
    schema_context = _build_schema_context()
    assert "DATABASE TABLES AND RELATIONSHIPS" in schema_context, "Missing schema context header"
    assert "branches" in schema_context, "Missing branches table in context"
    assert "employees" in schema_context, "Missing employees table in context"
    assert "Purpose:" in schema_context, "Missing business purpose in context"
    print("  ✅ Schema context building works correctly")
    
    # Test business patterns context
    patterns_context = _build_business_patterns_context()
    assert "COMMON BUSINESS QUERY PATTERNS" in patterns_context, "Missing patterns context header"
    assert "TOP_PERFORMERS" in patterns_context, "Missing top performers pattern"
    assert "Keywords:" in patterns_context, "Missing keywords in patterns context"
    print("  ✅ Business patterns context building works correctly")
    
    # Test query pattern identification
    test_cases = [
        ("Who are the top performing salespeople?", "top_performers"),
        ("Show me sales by region", "sales_by_region"),
        ("What do customers prefer?", "customer_preferences"),
        ("How is our pipeline looking?", "pipeline_health"),
        ("What's in inventory?", "inventory_status"),
        ("Show me revenue data", "revenue_analysis")
    ]
    
    for question, expected_pattern in test_cases:
        result = _identify_query_pattern(question)
        assert expected_pattern in result, f"Failed to identify pattern for '{question}': {result}"
        print(f"  ✅ Pattern identification: '{question}' → {expected_pattern}")
    
    # Test time filter building
    time_test_cases = [
        ("last 30 days", "INTERVAL '30 days'"),
        ("this quarter", "EXTRACT(QUARTER"),
        ("this year", "EXTRACT(YEAR"),
        ("last week", "DATE_TRUNC"),
        ("invalid period", "")  # Should return empty string
    ]
    
    for time_period, expected_fragment in time_test_cases:
        result = _build_time_filter(time_period)
        if expected_fragment:
            assert expected_fragment in result, f"Time filter for '{time_period}' missing '{expected_fragment}': {result}"
        else:
            assert result == "", f"Invalid time period should return empty string: {result}"
        print(f"  ✅ Time filter: '{time_period}' → {'valid' if expected_fragment in result else 'empty'}")
    
    print("  ✅ All helper function tests passed!")


def test_result_formatting():
    """Test result formatting and business insights generation."""
    print("\n🧪 Testing Result Formatting...")
    
    # Test with sample data
    sample_data = """salesperson|branch|total_sales|total_revenue
John Smith|Downtown|15|450000
Sarah Johnson|North|12|380000
Mike Davis|South|8|220000"""
    
    question = "Who are our top performing salespeople?"
    formatted_result = _format_business_results(sample_data, question)
    
    assert "CRM Query Results" in formatted_result, "Missing results header"
    assert question in formatted_result, "Missing original question"
    assert "3 record(s) found" in formatted_result, "Incorrect record count"
    assert "John Smith" in formatted_result, "Missing data content"
    
    print("  ✅ Result formatting works correctly")
    
    # Test business insights generation
    insights = _generate_business_insights(sample_data, question, 3)
    assert len(insights) > 0, "No business insights generated"
    # Check for performance-related insights (multiple possible variations)
    has_performance_insight = any(keyword in insights.lower() for keyword in 
                                 ["dataset", "record", "revenue", "trend", "performance", "analysis"])
    assert has_performance_insight, f"Missing performance-related insights. Got: {insights}"
    
    print("  ✅ Business insights generation works correctly")
    
    # Test empty data handling
    empty_result = _format_business_results("", question)
    assert "No data found" in empty_result, "Empty data not handled correctly"
    
    print("  ✅ Empty data handling works correctly")


def test_data_validation():
    """Test data validation and edge cases."""
    print("\n🧪 Testing Data Validation...")
    
    # Test schema completeness
    tables = CRM_SCHEMA_DOCS["tables"]
    total_columns = sum(len(table["columns"]) for table in tables.values())
    print(f"  📊 Total documented columns: {total_columns}")
    
    scenarios = SQL_GENERATION_PROMPTS["common_scenarios"]
    total_scenarios = len(scenarios)
    total_keywords = sum(len(scenario["keywords"]) for scenario in scenarios.values())
    print(f"  📊 Total scenarios: {total_scenarios}, Keywords: {total_keywords}")
    
    # Validate that all tables have proper relationships documented
    relationships = CRM_SCHEMA_DOCS["relationships"]["primary_relationships"]
    relationship_count = len(relationships)
    print(f"  📊 Total documented relationships: {relationship_count}")
    
    assert total_columns >= 60, f"Too few columns documented: {total_columns}"
    assert total_keywords >= 30, f"Too few keywords: {total_keywords}"
    assert relationship_count >= 8, f"Too few relationships: {relationship_count}"
    
    print("  ✅ All data validation tests passed!")


def main():
    """Run all tests."""
    print("🚀 Starting SQL Generation Tool Standalone Tests\n")
    
    try:
        # Run all test functions
        test_schema_documentation()
        test_sql_generation_prompts()
        test_helper_functions()
        test_result_formatting()
        test_data_validation()
        
        print("\n🎉 All standalone tests passed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Schema documentation complete and structured")
        print("   ✅ SQL generation prompts comprehensive")
        print("   ✅ Helper functions working correctly")
        print("   ✅ Result formatting and insights generation working")
        print("   ✅ Data validation confirmed")
        
        print("\n🔧 Implementation Features Validated:")
        print("   • Comprehensive CRM schema documentation")
        print("   • Business context prompts for SQL generation")
        print("   • Pattern recognition for common query types")
        print("   • Time period filter generation")
        print("   • Result formatting with business insights")
        print("   • Professional query result presentation")
        
        print("\n💡 Ready for Integration:")
        print("   • Schema awareness enables intelligent SQL generation")
        print("   • Business context ensures relevant query patterns")
        print("   • Helper functions provide modular functionality")
        print("   • Result formatting provides user-friendly output")
        print("   • Integration with existing security validation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 