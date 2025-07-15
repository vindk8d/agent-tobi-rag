"""
Tools for the RAG agent using @tool decorators.
These tools will be available for the agent to call dynamically.
"""

import json
import logging
import asyncio
import os
import re
import concurrent.futures
import threading
import html
import urllib.parse
from typing import List, Dict, Any, Optional, Set, Tuple

from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from pydantic.v1 import BaseModel, Field
from langsmith import traceable
import sqlalchemy
from sqlalchemy import create_engine, text
import sqlparse
import sqlparse.tokens
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, DML
import numpy as np

from rag.retriever import SemanticRetriever
from rag.embeddings import OpenAIEmbeddings
from config import get_settings

logger = logging.getLogger(__name__)

# Initialize components that tools will use
_retriever = None
_settings = None
_sql_database = None
_embeddings = None

# Define CRM table names that are allowed for SQL queries
CRM_TABLES = [
    "branches",
    "employees", 
    "customers",
    "vehicles",
    "opportunities",
    "transactions",
    "pricing",
    "activities"
]

# Define high-cardinality columns that need semantic validation
HIGH_CARDINALITY_COLUMNS = {
    "customers": ["name", "company"],
    "employees": ["name"],
    "vehicles": ["brand", "model"],
    "opportunities": [],  # No direct high-cardinality text columns
    "branches": ["name", "brand"],
    "transactions": [],
    "pricing": [],
    "activities": ["subject", "description"]
}

# Sample known values for semantic validation (in production, this would be cached from DB)
KNOWN_VALUES_CACHE = {
    "employee_names": [
        "John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wang", "Carlos Rodriguez",
        "Alex Thompson", "Emma Wilson", "David Chen", "Maria Garcia"
    ],
    "customer_names": [
        "Robert Brown", "Jennifer Lee", "Mark Johnson", "TechCorp Solutions", "Global Industries"
    ],
    "vehicle_brands": ["Toyota", "Honda", "Ford", "Nissan", "BMW", "Mercedes", "Audi"],
    "vehicle_models": ["Camry", "RAV4", "Civic", "CR-V", "F-150", "Altima", "Prius", "Corolla", "Accord"],
    "company_names": ["TechCorp Solutions", "Global Industries", "AutoCorp", "MegaTech Inc"]
}

# Comprehensive CRM Schema Documentation with Business Context
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

CRITICAL APPROACH - THINK IN STEPS:
1. IDENTIFY what information the user is asking for
2. LOCATE which table(s) contain that information
3. CHECK if there are direct columns that answer the question
4. PREFER direct column lookups over complex relationships
5. Only use JOINs when you need data from multiple tables
6. Keep queries as simple as possible

BUSINESS CONTEXT:
- This is a vehicle sales CRM system for automotive dealerships
- Focus on sales performance, customer relationships, and inventory management
- Common business metrics: revenue, conversion rates, sales volume, customer satisfaction
- Sales pipeline stages: New → Contacted → Consideration → Purchase Intent → Won/Lost

CRITICAL RULES FOR QUERY BUILDING:
1. ALWAYS use direct columns when available (e.g., employees.position, not complex inferences)
2. ONLY generate SELECT queries - no INSERT, UPDATE, DELETE, CREATE, DROP, etc.
3. ONLY query these allowed tables: branches, employees, customers, vehicles, opportunities, transactions, pricing, activities
4. START with the simplest possible query, then add complexity only if needed
5. For PRICING questions: ALWAYS join vehicles and pricing tables (vehicles.id = pricing.vehicle_id)
6. For CUSTOMER data across tables: Use appropriate JOINs (customers → opportunities → transactions)
7. For EMPLOYEE performance: Join employees → opportunities → transactions
8. Use WHERE clauses with direct column values (e.g., position = 'manager')
9. Apply business filters (e.g., is_active=true for employees, is_available=true for vehicles, pricing.is_active=true)
10. Use meaningful aliases and clear column names in results
11. Handle NULL values appropriately (especially for optional foreign keys)
12. Use appropriate aggregations for business metrics
13. Limit results to reasonable numbers (use LIMIT when appropriate)

SCHEMA KNOWLEDGE - KEY COLUMNS FOR DIRECT QUERIES:
employees:
  - position: ENUM('sales_agent', 'account_executive', 'manager', 'director', 'admin') - USE THIS for position questions
  - name: VARCHAR(255) - employee full name
  - is_active: BOOLEAN - whether employee is currently active
  - manager_ae_id: UUID - only use for hierarchy relationships, NOT for identifying managers

customers:
  - name: VARCHAR(255) - customer full name
  - company: VARCHAR(255) - company name if business customer
  - is_for_business: BOOLEAN - B2B vs B2C segmentation

vehicles:
  - brand: VARCHAR(255) - vehicle manufacturer
  - model: VARCHAR(255) - vehicle model name
  - type: ENUM('sedan', 'suv', 'hatchback', 'pickup', 'van', 'motorcycle', 'truck')
  - is_available: BOOLEAN - whether vehicle is available for sale

opportunities:
  - stage: ENUM('New', 'Contacted', 'Consideration', 'Purchase Intent', 'Won', 'Lost')
  - warmth: ENUM('hot', 'warm', 'cold', 'dormant')
  - probability: INTEGER (0-100%)

transactions:
  - status: ENUM('pending', 'processing', 'completed', 'cancelled', 'refunded')
  - total_amount: DECIMAL(12,2)

activities:
  - activity_type: ENUM('call', 'email', 'meeting', 'demo', 'follow_up', 'proposal_sent', 'contract_signed')
  - is_completed: BOOLEAN

SIMPLE QUERY EXAMPLES:
- "What is [person]'s position?" → SELECT position FROM employees WHERE name = '[person]'
- "How many managers?" → SELECT COUNT(*) FROM employees WHERE position = 'manager' AND is_active = true
- "What vehicles are available?" → SELECT brand, model FROM vehicles WHERE is_available = true
- "How many won deals?" → SELECT COUNT(*) FROM opportunities WHERE stage = 'Won'

PRICING QUERY EXAMPLES (REQUIRES JOIN):
- "How much is the Prius?" → SELECT v.brand, v.model, p.final_price FROM vehicles v JOIN pricing p ON v.id = p.vehicle_id WHERE v.model = 'Prius' AND p.is_active = true
- "What's the price of Honda Civic?" → SELECT v.brand, v.model, p.final_price FROM vehicles v JOIN pricing p ON v.id = p.vehicle_id WHERE v.brand = 'Honda' AND v.model = 'Civic' AND p.is_active = true
- "Show me Toyota prices" → SELECT v.model, p.final_price FROM vehicles v JOIN pricing p ON v.id = p.vehicle_id WHERE v.brand = 'Toyota' AND p.is_active = true

WHEN TO USE JOINS:
- ALWAYS for pricing questions (vehicles + pricing tables)
- When you need data from multiple tables
- Example: "Sales by employee and branch" needs employees JOIN branches  
- Example: "Customer opportunities" needs customers JOIN opportunities
- Example: "Employee performance with revenue" needs employees JOIN opportunities JOIN transactions

STEP-BY-STEP THINKING PROCESS:
Before generating SQL, mentally go through:
1. What is the user asking for? (specific data, count, list, etc.)
2. Which table has this information?
3. Are there direct columns that answer this?
4. Do I need data from other tables?
5. What filters should I apply?
6. Is this the simplest approach?""",

    "query_patterns": {
        "sales_performance": {
            "description": "Analyze sales performance by various dimensions",
            "template": """
-- Sales performance analysis
SELECT 
    e.name as salesperson,
    b.name as branch,
    COUNT(t.id) as total_sales,
    SUM(t.total_amount) as total_revenue,
    AVG(t.total_amount) as avg_deal_size,
    COUNT(o.id) as total_opportunities,
    ROUND(COUNT(t.id)::decimal / COUNT(o.id) * 100, 2) as conversion_rate
FROM employees e
JOIN branches b ON e.branch_id = b.id
LEFT JOIN opportunities o ON e.id = o.opportunity_salesperson_ae_id
LEFT JOIN transactions t ON o.id = t.opportunity_id AND t.status = 'completed'
WHERE e.is_active = true
    AND o.created_date >= '{start_date}'
    AND o.created_date <= '{end_date}'
GROUP BY e.id, e.name, b.name
ORDER BY total_revenue DESC;""",
            "variables": ["start_date", "end_date"],
            "business_logic": "Shows individual and branch performance with key metrics"
        },

        "pipeline_analysis": {
            "description": "Analyze sales pipeline health and progression",
            "template": """
-- Sales pipeline analysis
SELECT 
    o.stage,
    o.warmth,
    COUNT(*) as opportunity_count,
    SUM(o.estimated_value) as total_estimated_value,
    AVG(o.probability) as avg_probability,
    AVG(EXTRACT(DAYS FROM (COALESCE(o.expected_close_date, CURRENT_DATE) - o.created_date))) as avg_days_in_stage
FROM opportunities o
JOIN customers c ON o.customer_id = c.id
LEFT JOIN vehicles v ON o.vehicle_id = v.id
WHERE o.stage NOT IN ('Won', 'Lost')
GROUP BY o.stage, o.warmth
ORDER BY 
    CASE o.stage 
        WHEN 'New' THEN 1 
        WHEN 'Contacted' THEN 2 
        WHEN 'Consideration' THEN 3 
        WHEN 'Purchase Intent' THEN 4 
    END;""",
            "business_logic": "Analyzes active pipeline by stage and lead quality"
        },

        "customer_analysis": {
            "description": "Analyze customer behavior and value",
            "template": """
-- Customer analysis with transaction history
SELECT 
    c.name as customer_name,
    c.company,
    c.is_for_business,
    COUNT(o.id) as total_opportunities,
    COUNT(t.id) as completed_transactions,
    SUM(t.total_amount) as lifetime_value,
    MAX(t.created_date) as last_purchase_date,
    STRING_AGG(DISTINCT v.brand || ' ' || v.model, ', ') as vehicles_of_interest
FROM customers c
LEFT JOIN opportunities o ON c.id = o.customer_id
LEFT JOIN transactions t ON o.id = t.opportunity_id AND t.status = 'completed'
LEFT JOIN vehicles v ON o.vehicle_id = v.id
GROUP BY c.id, c.name, c.company, c.is_for_business
HAVING COUNT(o.id) > 0
ORDER BY lifetime_value DESC NULLS LAST;""",
            "business_logic": "Provides customer lifetime value and purchase patterns"
        },

        "inventory_insights": {
            "description": "Analyze vehicle inventory and performance",
            "template": """
-- Vehicle inventory analysis with sales performance
SELECT 
    v.brand,
    v.model,
    v.year,
    v.type,
    v.stock_quantity,
    COUNT(o.id) as total_interest,
    COUNT(t.id) as total_sold,
    ROUND(COUNT(t.id)::decimal / NULLIF(COUNT(o.id), 0) * 100, 2) as conversion_rate,
    AVG(p.final_price) as avg_price,
    MAX(p.final_price) as max_price,
    MIN(p.final_price) as min_price
FROM vehicles v
LEFT JOIN opportunities o ON v.id = o.vehicle_id
LEFT JOIN transactions t ON o.id = t.opportunity_id AND t.status = 'completed'
LEFT JOIN pricing p ON v.id = p.vehicle_id AND p.is_active = true
WHERE v.is_available = true
GROUP BY v.id, v.brand, v.model, v.year, v.type, v.stock_quantity
ORDER BY total_interest DESC;""",
            "business_logic": "Shows vehicle popularity and sales conversion"
        },

        "activity_tracking": {
            "description": "Track sales activities and follow-up management",
            "template": """
-- Sales activity tracking and effectiveness
SELECT 
    e.name as salesperson,
    a.activity_type,
    COUNT(*) as activity_count,
    COUNT(CASE WHEN a.is_completed THEN 1 END) as completed_count,
    ROUND(COUNT(CASE WHEN a.is_completed THEN 1 END)::decimal / COUNT(*) * 100, 2) as completion_rate,
    COUNT(DISTINCT a.customer_id) as unique_customers_contacted,
    COUNT(DISTINCT a.opportunity_id) as opportunities_worked
FROM employees e
JOIN activities a ON e.id = a.opportunity_salesperson_ae_id
WHERE a.created_date >= '{start_date}'
    AND a.created_date <= '{end_date}'
    AND e.is_active = true
GROUP BY e.id, e.name, a.activity_type
ORDER BY e.name, activity_count DESC;""",
            "variables": ["start_date", "end_date"],
            "business_logic": "Measures sales activity levels and effectiveness"
        }
    },

    "business_rules": {
        "default_filters": {
            "employees": "is_active = true",
            "vehicles": "is_available = true",
            "pricing": "is_active = true",
            "opportunities_active": "stage NOT IN ('Won', 'Lost')",
            "transactions_completed": "status = 'completed'"
        },
        "date_handling": {
            "current_quarter": "EXTRACT(QUARTER FROM CURRENT_DATE) = EXTRACT(QUARTER FROM date_column)",
            "current_month": "EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM date_column) AND EXTRACT(YEAR FROM CURRENT_DATE) = EXTRACT(YEAR FROM date_column)",
            "last_30_days": "date_column >= CURRENT_DATE - INTERVAL '30 days'",
            "year_to_date": "EXTRACT(YEAR FROM date_column) = EXTRACT(YEAR FROM CURRENT_DATE)"
        },
        "aggregation_patterns": {
            "revenue_metrics": ["SUM(total_amount)", "AVG(total_amount)", "COUNT(DISTINCT transaction_id)"],
            "conversion_metrics": ["COUNT(won_opportunities)::decimal / COUNT(total_opportunities) * 100"],
            "time_metrics": ["AVG(EXTRACT(DAYS FROM (close_date - created_date)))"],
            "ranking": ["ROW_NUMBER() OVER (ORDER BY metric DESC)", "RANK() OVER (PARTITION BY category ORDER BY metric DESC)"]
        }
    },

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
ALLOWED_SQL_OPERATIONS = {"SELECT"}  # Only SELECT operations allowed

# DDL operations to block (Data Definition Language)
BLOCKED_DDL_OPERATIONS = {
    "CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME", "COMMENT",
    "GRANT", "REVOKE", "ANALYZE", "VACUUM", "REINDEX"
}

# DML operations to block (Data Manipulation Language - except SELECT)
BLOCKED_DML_OPERATIONS = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE"
}

# Administrative and system operations to block
BLOCKED_ADMIN_OPERATIONS = {
    "CALL", "EXECUTE", "EXEC", "DO", "COPY", "LOAD", "IMPORT", "EXPORT",
    "BACKUP", "RESTORE", "SET", "RESET", "SHOW", "DESCRIBE", "EXPLAIN"
}

# Combine all blocked operations
BLOCKED_OPERATIONS = BLOCKED_DDL_OPERATIONS | BLOCKED_DML_OPERATIONS | BLOCKED_ADMIN_OPERATIONS

class SQLValidationError(Exception):
    """Custom exception for SQL validation failures."""
    pass


def sanitize_sql_input(input_string: str) -> str:
    """
    Sanitize user input to prevent SQL injection attacks.
    
    Args:
        input_string: Raw user input that might be used in SQL queries
        
    Returns:
        Sanitized input string
        
    Raises:
        SQLValidationError: If input contains dangerous patterns
    """
    if not isinstance(input_string, str):
        raise SQLValidationError("Input must be a string")
    
    # Remove null bytes and control characters
    sanitized = input_string.replace('\x00', '').replace('\r', '').replace('\t', ' ')
    
    # Normalize whitespace but preserve newlines after SQL comments
    # First, protect newlines that follow SQL comments
    lines = sanitized.split('\n')
    normalized_lines = []
    for line in lines:
        # Normalize whitespace within each line
        normalized_line = re.sub(r'[ \t]+', ' ', line.strip())
        if normalized_line:
            normalized_lines.append(normalized_line)
    
    # Join lines back together, preserving the structure
    sanitized = '\n'.join(normalized_lines)
    
    # Check for empty input after sanitization
    if not sanitized:
        raise SQLValidationError("Input is empty after sanitization")
    
    # Check length limits (prevent buffer overflow attacks)
    if len(sanitized) > 10000:
        raise SQLValidationError("Input too long. Maximum 10,000 characters allowed.")
    
    # Detect and block common injection patterns
    _detect_injection_patterns(sanitized)
    
    # HTML decode to prevent encoding-based attacks
    sanitized = html.unescape(sanitized)
    
    # URL decode to prevent URL encoding-based attacks
    sanitized = urllib.parse.unquote(sanitized)
    
    # Check again after decoding
    _detect_injection_patterns(sanitized)
    
    return sanitized


def _detect_injection_patterns(text: str) -> None:
    """
    Detect SQL injection patterns in input text.
    
    Args:
        text: Text to analyze for injection patterns
        
    Raises:
        SQLValidationError: If dangerous patterns are detected
    """
    # Normalize for pattern detection
    normalized = text.upper().replace(' ', '').replace('\n', '').replace('\t', '')
    
    # Classic SQL injection patterns
    injection_patterns = [
        # Union-based attacks
        r"UNION.*SELECT", r"UNION.*ALL.*SELECT",
        # Stacked queries
        r";.*SELECT", r";.*INSERT", r";.*UPDATE", r";.*DELETE", r";.*DROP",
        # Boolean-based blind injection
        r"1=1", r"1=0", r"'='", r"''=''", r"1'='1", r"'OR'", r"'AND'",
        # Time-based blind injection
        r"WAITFOR.*DELAY", r"BENCHMARK\(", r"PG_SLEEP\(", r"SLEEP\(",
        # Error-based injection
        r"CAST\(.*AS.*INT\)", r"CONVERT\(.*,.*INT\)",
        # File operations
        r"LOAD_FILE\(", r"INTO.*OUTFILE", r"INTO.*DUMPFILE",
        # Database enumeration
        r"INFORMATION_SCHEMA", r"SYSOBJECTS", r"SYSCOLUMNS",
        # Privilege escalation
        r"EXEC\(", r"EXECUTE\(", r"SP_", r"XP_",
        # NoSQL injection patterns
        r"\$WHERE", r"\$NE", r"\$GT", r"\$LT", r"\$REGEX",
        # XPath injection
        r"EXTRACTVALUE\(", r"UPDATEXML\(",
        # LDAP injection
        r"\(\|\(", r"\)\)\(", r"\*\)\(",
        # Command injection within SQL context
        r"SHELL\(", r"SYSTEM\(", r"`", r"\$\(",
        # Advanced evasion techniques
        r"CHAR\(", r"CHR\(", r"ASCII\(", r"CONCAT\(",
        # Hex encoding detection
        r"0X[0-9A-F]+",
        # Unicode evasion
        r"\\U[0-9A-F]{4}", r"\\X[0-9A-F]{2}",
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            raise SQLValidationError(f"Potential SQL injection detected: {pattern}")
    
    # Check for malicious comment injection patterns (but allow legitimate comments)
    _check_malicious_comments(text)
    
    # Check for suspicious character sequences
    suspicious_sequences = [
        "''", '""', "())", "((", "))", "||", "&&",
        "<%", "%>", "<?", "?>", "${", "#{", "@{", "\\x", "\\u"
    ]
    
    for sequence in suspicious_sequences:
        if sequence in text:
            raise SQLValidationError(f"Suspicious character sequence detected: {sequence}")
    
    # Check for semicolons in suspicious contexts (not at the end of the query)
    # Allow semicolon at the end of query but not in the middle (stacked queries)
    text_stripped = text.strip()
    if ";" in text_stripped:
        # Find all semicolon positions
        semicolon_positions = [i for i, char in enumerate(text_stripped) if char == ";"]
        for pos in semicolon_positions:
            # Check if there's meaningful content after the semicolon
            remaining = text_stripped[pos + 1:].strip()
            if remaining and not remaining.startswith("--"):  # Allow comments after semicolon
                # Check if remaining content contains SQL keywords (indicating stacked queries)
                remaining_upper = remaining.upper()
                sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "GRANT", "REVOKE"]
                if any(keyword in remaining_upper for keyword in sql_keywords):
                    raise SQLValidationError("Suspicious semicolon usage detected: potential stacked query")


def _check_malicious_comments(text: str) -> None:
    """
    Check for malicious comment injection patterns while allowing legitimate SQL comments.
    
    Args:
        text: Text to analyze for comment-based injection
        
    Raises:
        SQLValidationError: If malicious comment patterns are detected
    """
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            continue
            
        # Allow legitimate comments at the beginning of lines
        if stripped_line.startswith('--'):
            # Check if the comment contains suspicious content
            comment_content = stripped_line[2:].strip().upper()
            
            # Block comments that contain SQL keywords that could be injection attempts
            suspicious_comment_patterns = [
                r"UNION.*SELECT", r"INSERT.*INTO", r"UPDATE.*SET", r"DELETE.*FROM",
                r"DROP.*TABLE", r"ALTER.*TABLE", r"CREATE.*TABLE", r"EXEC",
                r"EXECUTE", r"SP_", r"XP_", r"WAITFOR", r"BENCHMARK", r"SLEEP"
            ]
            
            for pattern in suspicious_comment_patterns:
                if re.search(pattern, comment_content):
                    raise SQLValidationError(f"Suspicious content in comment: {pattern}")
            continue
            
        # Check for comments that appear in the middle of SQL statements (potential injection)
        if '--' in stripped_line and not stripped_line.startswith('--'):
            # Look for patterns like: SELECT * FROM table WHERE id = 1 -- malicious content
            parts = stripped_line.split('--')
            if len(parts) > 1:
                before_comment = parts[0].strip().upper()
                after_comment = ''.join(parts[1:]).strip().upper()
                
                # If there's SQL before the comment and potentially malicious content after
                if before_comment and any(keyword in before_comment for keyword in ['SELECT', 'FROM', 'WHERE', 'UPDATE', 'INSERT']):
                    if any(keyword in after_comment for keyword in ['UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']):
                        raise SQLValidationError("Potential comment-based SQL injection detected")
        
        # Check for block comments that could be malicious
        if '/*' in stripped_line or '*/' in stripped_line:
            # Allow simple block comments but check for suspicious patterns
            comment_pattern = r'/\*.*?\*/'
            matches = re.findall(comment_pattern, stripped_line, re.DOTALL)
            for match in matches:
                comment_content = match[2:-2].strip().upper()  # Remove /* and */
                
                # Block comments containing SQL injection patterns
                if any(keyword in comment_content for keyword in ['UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'EXEC']):
                    raise SQLValidationError("Suspicious content in block comment")


def escape_sql_identifier(identifier: str) -> str:
    """
    Properly escape SQL identifiers (table names, column names).
    
    Args:
        identifier: SQL identifier to escape
        
    Returns:
        Escaped identifier safe for use in SQL queries
        
    Raises:
        SQLValidationError: If identifier is invalid
    """
    if not isinstance(identifier, str):
        raise SQLValidationError("Identifier must be a string")
    
    # Remove whitespace and validate
    identifier = identifier.strip()
    
    if not identifier:
        raise SQLValidationError("Identifier cannot be empty")
    
    if len(identifier) > 63:  # PostgreSQL identifier limit
        raise SQLValidationError("Identifier too long. Maximum 63 characters allowed.")
    
    # Check for valid identifier pattern (letters, numbers, underscores, must start with letter)
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', identifier):
        raise SQLValidationError(f"Invalid identifier format: {identifier}")
    
    # Double-quote the identifier to handle reserved words and case sensitivity
    return f'"{identifier}"'


def escape_sql_literal(value: str) -> str:
    """
    Properly escape SQL string literals.
    
    Args:
        value: String value to escape
        
    Returns:
        Escaped string literal safe for use in SQL queries
    """
    if not isinstance(value, str):
        return str(value)
    
    # Escape single quotes by doubling them (PostgreSQL standard)
    escaped = value.replace("'", "''")
    
    # Wrap in single quotes
    return f"'{escaped}'"


async def _get_embeddings():
    """Get or create the embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings()
    return _embeddings

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0

async def validate_high_cardinality_terms(query: str) -> Dict[str, Any]:
    """
    Validate search terms against high-cardinality columns using semantic similarity.
    
    Args:
        query: The SQL query to validate
        
    Returns:
        Dictionary with validation results and suggestions
    """
    validation_result = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "validated_terms": []
    }
    
    try:
        # Extract WHERE clause conditions that reference high-cardinality columns
        terms_to_validate = _extract_high_cardinality_terms(query)
        
        if not terms_to_validate:
            return validation_result
        
        embeddings = await _get_embeddings()
        
        for table, column, search_term in terms_to_validate:
            # Skip validation for obviously valid patterns (IDs, numbers, etc.)
            if _is_trivial_term(search_term):
                continue
                
            # Get the appropriate known values for this column
            known_values = _get_known_values_for_column(table, column)
            
            if not known_values:
                # No known values to validate against, allow but warn
                validation_result["warnings"].append(
                    f"No validation data available for {table}.{column}"
                )
                continue
            
            # Calculate semantic similarity
            similarity_results = await _calculate_semantic_similarities(
                embeddings, search_term, known_values
            )
            
            best_match, best_score = similarity_results[0] if similarity_results else (None, 0.0)
            
            # Apply validation thresholds
            if best_score < 0.5:  # Low similarity - likely suspicious
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"Search term '{search_term}' for {table}.{column} appears invalid (similarity: {best_score:.2f})"
                )
                
                # Suggest alternatives if there are any reasonable matches
                reasonable_matches = [(val, score) for val, score in similarity_results if score > 0.2]
                if reasonable_matches:
                    suggestions = [f"'{val}' ({score:.2f})" for val, score in reasonable_matches[:3]]
                    validation_result["suggestions"].append(
                        f"Did you mean: {', '.join(suggestions)} for {table}.{column}?"
                    )
                    
            elif best_score < 0.8:  # Moderate similarity - suggest correction
                validation_result["suggestions"].append(
                    f"For {table}.{column}, did you mean '{best_match}' instead of '{search_term}'? (similarity: {best_score:.2f})"
                )
                
            validation_result["validated_terms"].append({
                "table": table,
                "column": column,
                "search_term": search_term,
                "best_match": best_match,
                "similarity_score": best_score
            })
            
    except Exception as e:
        logger.warning(f"Semantic validation failed: {e}")
        validation_result["warnings"].append("Semantic validation unavailable")
    
    return validation_result

def _extract_high_cardinality_terms(query: str) -> List[Tuple[str, str, str]]:
    """Extract search terms for high-cardinality columns from SQL query."""
    terms = []
    seen_terms = set()  # To avoid duplicates
    
    # Look for patterns like: table.column = 'value' or column LIKE '%value%'
    # Process table.column patterns first to be more specific
    table_column_patterns = [
        r'(\w+)\.(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',  # table.column = 'value'
        r'(\w+)\.(\w+)\s*LIKE\s*[\'"]([^\'"]+)[\'"]',  # table.column LIKE 'value'
    ]
    
    # First pass: handle table.column patterns
    for pattern in table_column_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for table, column, value in matches:
            table = table.lower()
            column = column.lower()
            
            # Check if this is a high-cardinality column we should validate
            if table in HIGH_CARDINALITY_COLUMNS and column in HIGH_CARDINALITY_COLUMNS[table]:
                term_key = (table, column, value)
                if term_key not in seen_terms:
                    terms.append(term_key)
                    seen_terms.add(term_key)
    
    # Second pass: handle column-only patterns, but only if not already found
    column_only_patterns = [
        r'(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',  # column = 'value'
        r'(\w+)\s*LIKE\s*[\'"]([^\'"]+)[\'"]'  # column LIKE 'value'
    ]
    
    for pattern in column_only_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for column, value in matches:
            table = _infer_table_from_query(column, query)
            
            table = table.lower()
            column = column.lower()
            
            # Check if this is a high-cardinality column we should validate
            if table in HIGH_CARDINALITY_COLUMNS and column in HIGH_CARDINALITY_COLUMNS[table]:
                term_key = (table, column, value)
                if term_key not in seen_terms:  # Only add if not already found
                    terms.append(term_key)
                    seen_terms.add(term_key)
    
    return terms

def _infer_table_from_query(column: str, query: str) -> str:
    """Infer table name from query context when only column is specified."""
    normalized_query = query.upper()
    
    # Look for FROM clause to identify main table
    from_match = re.search(r'\bFROM\s+(\w+)', normalized_query)
    if from_match:
        table = from_match.group(1).lower()
        if table in HIGH_CARDINALITY_COLUMNS and column.lower() in HIGH_CARDINALITY_COLUMNS[table]:
            return table
    
    # Look for table aliases in JOIN clauses
    join_patterns = [
        r'\bFROM\s+(\w+)\s+(\w+)',  # FROM table alias
        r'\bJOIN\s+(\w+)\s+(\w+)'   # JOIN table alias
    ]
    
    for pattern in join_patterns:
        matches = re.findall(pattern, normalized_query)
        for table, alias in matches:
            table = table.lower()
            if table in HIGH_CARDINALITY_COLUMNS and column.lower() in HIGH_CARDINALITY_COLUMNS[table]:
                return table
    
    # Fallback: check all high-cardinality tables for this column
    for table, columns in HIGH_CARDINALITY_COLUMNS.items():
        if column.lower() in columns:
            return table
            
    return "unknown"

def _is_trivial_term(term: str) -> bool:
    """Check if a search term is trivial and doesn't need semantic validation."""
    # Skip UUIDs, pure numbers, very short terms, etc.
    if not term or len(term) < 2:
        return True
    
    # Skip if it looks like a UUID
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', term, re.IGNORECASE):
        return True
    
    # Skip if it's just numbers
    if term.isdigit():
        return True
        
    # Skip very common SQL patterns
    if term.upper() in ['TRUE', 'FALSE', 'NULL', '%', '_']:
        return True
        
    return False

def _get_known_values_for_column(table: str, column: str) -> List[str]:
    """Get known values for a specific table.column combination."""
    # Map table.column to known values cache keys
    column_mapping = {
        ("employees", "name"): "employee_names",
        ("customers", "name"): "customer_names",  
        ("customers", "company"): "company_names",
        ("vehicles", "brand"): "vehicle_brands",
        ("vehicles", "model"): "vehicle_models",
        ("branches", "name"): "customer_names",  # Branch names similar to customer names
        ("branches", "brand"): "vehicle_brands",  # Branch brands are vehicle brands
    }
    
    cache_key = column_mapping.get((table, column))
    return KNOWN_VALUES_CACHE.get(cache_key, []) if cache_key else []

async def _calculate_semantic_similarities(
    embeddings: OpenAIEmbeddings, 
    search_term: str, 
    known_values: List[str]
) -> List[Tuple[str, float]]:
    """Calculate semantic similarities between search term and known values."""
    try:
        # Embed the search term
        search_embedding = await embeddings.embed_query(search_term)
        
        # Embed all known values
        known_embeddings = []
        for value in known_values:
            try:
                embedding = await embeddings.embed_query(value)
                known_embeddings.append((value, embedding))
            except Exception as e:
                logger.warning(f"Failed to embed '{value}': {e}")
                continue
        
        # Calculate similarities
        similarities = []
        for value, known_embedding in known_embeddings:
            similarity = cosine_similarity(search_embedding, known_embedding)
            similarities.append((value, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:10]  # Return top 10 matches
        
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        return []

def validate_sql_query(query: str) -> str:
    """
    Validate SQL query to ensure it only contains allowed operations with comprehensive injection protection.
    Uses sqlparse for robust SQL parsing and validation.
    
    Args:
        query: The SQL query string to validate
        
    Returns:
        The cleaned query if valid
        
    Raises:
        SQLValidationError: If query contains blocked operations or patterns
    """
    if not query or not query.strip():
        raise SQLValidationError("Empty or whitespace-only query not allowed")
    
    # Apply comprehensive input sanitization first
    try:
        sanitized_query = sanitize_sql_input(query)
    except SQLValidationError as e:
        raise SQLValidationError(f"Input sanitization failed: {e}")
    
    # Clean and normalize the query
    cleaned_query = sanitized_query.strip()
    
    # Use sqlparse to parse and validate the SQL
    try:
        parsed = sqlparse.parse(cleaned_query)
    except Exception as e:
        raise SQLValidationError(f"Invalid SQL syntax: {e}")
    
    if len(parsed) == 0:
        raise SQLValidationError("No valid SQL statements found")
    
    if len(parsed) > 1:
        raise SQLValidationError("Multiple statements not allowed. Only single SELECT queries permitted.")
    
    statement = parsed[0]
    
    # Extract the first meaningful token (should be SELECT)
    first_keyword = None
    for token in statement.flatten():
        # Skip whitespace, comments, and punctuation
        if token.ttype in (None, sqlparse.tokens.Text.Whitespace, sqlparse.tokens.Comment.Single, 
                          sqlparse.tokens.Comment.Multiline, sqlparse.tokens.Text.Whitespace.Newline):
            continue
        # Check if token value matches allowed or blocked operations
        if token.value.upper() in (ALLOWED_SQL_OPERATIONS | BLOCKED_OPERATIONS):
            first_keyword = token.value.upper()
            break
    
    if not first_keyword:
        raise SQLValidationError("No valid SQL operation found in query")
    
    # Check if the operation is allowed
    if first_keyword not in ALLOWED_SQL_OPERATIONS:
        if first_keyword in BLOCKED_OPERATIONS:
            raise SQLValidationError(f"Operation '{first_keyword}' is not allowed. Only SELECT queries are permitted.")
        else:
            raise SQLValidationError(f"Unknown or invalid SQL operation: '{first_keyword}'. Only SELECT queries are permitted.")
    
    # Get normalized query for additional checks
    normalized_query = ' '.join(cleaned_query.split()).upper()
    
    # Enhanced security checks with injection protection
    _check_dangerous_patterns(normalized_query)
    _detect_advanced_injection_patterns(normalized_query)
    _validate_table_access(normalized_query)
    _validate_parsed_statement(statement)
    
    return cleaned_query


def _detect_advanced_injection_patterns(normalized_query: str) -> None:
    """
    Detect advanced SQL injection patterns beyond basic dangerous patterns.
    
    Args:
        normalized_query: Normalized SQL query string
        
    Raises:
        SQLValidationError: If advanced injection patterns are detected
    """
    # Advanced injection patterns specific to PostgreSQL
    advanced_patterns = [
        # PostgreSQL-specific dangerous functions
        r'PG_READ_FILE\s*\(', r'PG_WRITE_FILE\s*\(', r'PG_STAT_FILE\s*\(',
        # Large object functions
        r'LO_CREAT\s*\(', r'LO_UNLINK\s*\(', r'LO_IMPORT\s*\(', r'LO_EXPORT\s*\(',
        # Administrative functions
        r'PG_RELOAD_CONF\s*\(', r'PG_ROTATE_LOGFILE\s*\(',
        # Process functions
        r'PG_CANCEL_BACKEND\s*\(', r'PG_TERMINATE_BACKEND\s*\(',
        # Configuration functions
        r'SET_CONFIG\s*\(', r'CURRENT_SETTING\s*\(',
        # Extension loading
        r'CREATE\s+EXTENSION', r'DROP\s+EXTENSION',
        # Foreign data wrappers
        r'CREATE\s+FOREIGN', r'DROP\s+FOREIGN',
        # Trigger functions
        r'CREATE\s+TRIGGER', r'DROP\s+TRIGGER',
        # Function definitions
        r'CREATE\s+FUNCTION', r'CREATE\s+OR\s+REPLACE\s+FUNCTION',
        # Language constructs
        r'LANGUAGE\s+PLPGSQL', r'LANGUAGE\s+SQL', r'LANGUAGE\s+C',
        # Schema manipulation
        r'CREATE\s+SCHEMA', r'DROP\s+SCHEMA', r'ALTER\s+SCHEMA',
        # Role and privilege manipulation
        r'CREATE\s+ROLE', r'DROP\s+ROLE', r'GRANT\s+', r'REVOKE\s+',
        # Database manipulation
        r'CREATE\s+DATABASE', r'DROP\s+DATABASE', r'ALTER\s+DATABASE',
        # Tablespace operations
        r'CREATE\s+TABLESPACE', r'DROP\s+TABLESPACE',
        # Sequence manipulation beyond allowed operations
        r'CREATE\s+SEQUENCE', r'DROP\s+SEQUENCE', r'ALTER\s+SEQUENCE',
        # View manipulation
        r'CREATE\s+VIEW', r'DROP\s+VIEW', r'ALTER\s+VIEW',
        # Index manipulation
        r'CREATE\s+INDEX', r'DROP\s+INDEX', r'REINDEX',
        # Type manipulation
        r'CREATE\s+TYPE', r'DROP\s+TYPE', r'ALTER\s+TYPE',
        # Domain manipulation
        r'CREATE\s+DOMAIN', r'DROP\s+DOMAIN', r'ALTER\s+DOMAIN',
        # Collation manipulation
        r'CREATE\s+COLLATION', r'DROP\s+COLLATION',
        # Conversion manipulation
        r'CREATE\s+CONVERSION', r'DROP\s+CONVERSION',
        # Operator manipulation
        r'CREATE\s+OPERATOR', r'DROP\s+OPERATOR',
        # Aggregate manipulation
        r'CREATE\s+AGGREGATE', r'DROP\s+AGGREGATE',
        # Rule manipulation
        r'CREATE\s+RULE', r'DROP\s+RULE',
        # Transaction control beyond basic
        r'SAVEPOINT', r'RELEASE\s+SAVEPOINT', r'ROLLBACK\s+TO',
        # Cursor operations
        r'DECLARE\s+.*CURSOR', r'FETCH', r'CLOSE\s+',
        # Prepared statement manipulation
        r'PREPARE\s+', r'EXECUTE\s+', r'DEALLOCATE\s+',
        # Lock operations
        r'LOCK\s+TABLE', r'ADVISORY_LOCK',
        # Clustering operations
        r'CLUSTER\s+', r'ANALYZE\s+', r'VACUUM\s+',
        # Notification operations
        r'NOTIFY\s+', r'LISTEN\s+', r'UNLISTEN\s+',
        # COPY operations (file I/O)
        r'COPY\s+.*FROM', r'COPY\s+.*TO',
    ]
    
    for pattern in advanced_patterns:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            raise SQLValidationError(f"Advanced SQL injection pattern detected: {pattern}")
    
    # Check for suspicious SQL constructs
    suspicious_constructs = [
        # Multiple statements (stacked queries)
        r';\s*SELECT', r';\s*INSERT', r';\s*UPDATE', r';\s*DELETE',
        # Union with suspicious patterns
        r'UNION\s+SELECT\s+NULL', r'UNION\s+SELECT\s+\d+',
        # Information disclosure attempts
        r'VERSION\s*\(\)', r'USER\s*\(\)', r'DATABASE\s*\(\)',
        # Error-based injection attempts
        r'CAST\s*\(\s*.*\s+AS\s+INT\s*\)', r'CONVERT\s*\(',
        # Time-based injection attempts
        r'PG_SLEEP\s*\(\s*\d+\s*\)', r'GENERATE_SERIES\s*\(',
        # Boolean-based injection (specific patterns that indicate malicious intent)
        r'\bAND\s+\d+=\d+', r'\bOR\s+\d+=\d+', 
        r'\bAND\s+[\'"][^\'\"]*[\'\"]\s*=\s*[\'"][^\'\"]*[\'"]', # AND 'x'='x' patterns
        r'\bOR\s+[\'"][^\'\"]*[\'\"]\s*=\s*[\'"][^\'\"]*[\'"]',  # OR 'x'='x' patterns
    ]
    
    for construct in suspicious_constructs:
        if re.search(construct, normalized_query, re.IGNORECASE):
            raise SQLValidationError(f"Suspicious SQL construct detected: {construct}")

async def validate_sql_query_with_semantics(query: str) -> Dict[str, Any]:
    """
    Enhanced SQL validation that includes semantic validation for high-cardinality columns.
    
    Args:
        query: The SQL query string to validate
        
    Returns:
        Dictionary with validation results, warnings, and suggestions
    """
    validation_result = {
        "valid": True,
        "cleaned_query": "",
        "warnings": [],
        "suggestions": [],
        "semantic_validation": {}
    }
    
    try:
        # First, run standard SQL validation
        cleaned_query = validate_sql_query(query)
        validation_result["cleaned_query"] = cleaned_query
        
        # Then run semantic validation for high-cardinality columns
        semantic_result = await validate_high_cardinality_terms(cleaned_query)
        validation_result["semantic_validation"] = semantic_result
        
        # Merge semantic validation results
        if not semantic_result["valid"]:
            validation_result["valid"] = False
            
        validation_result["warnings"].extend(semantic_result["warnings"])
        validation_result["suggestions"].extend(semantic_result["suggestions"])
        
        return validation_result
        
    except SQLValidationError as e:
        validation_result["valid"] = False
        validation_result["warnings"].append(str(e))
        return validation_result

def _validate_parsed_statement(statement: Statement) -> None:
    """Additional validation using parsed SQL statement."""
    
    # Check for nested statements or subqueries that might contain dangerous operations
    for token in statement.flatten():
        if token.ttype is Keyword and token.value.upper() in BLOCKED_OPERATIONS:
            raise SQLValidationError(f"Blocked operation '{token.value.upper()}' found in query")
        
        # Check for suspicious function calls
        if hasattr(token, 'value') and isinstance(token.value, str):
            token_upper = token.value.upper()
            dangerous_functions = ['PG_SLEEP', 'SLEEP', 'WAITFOR', 'DELAY']
            for func in dangerous_functions:
                if func in token_upper:
                    raise SQLValidationError(f"Dangerous function '{func}' detected in query")

def _check_dangerous_patterns(normalized_query: str) -> None:
    """Check for dangerous SQL patterns that should be blocked."""
    
    dangerous_patterns = [
        # Function calls that could be dangerous
        r'\bPG_SLEEP\b', r'\bSLEEP\b', r'\bWAIT\b',
        # File operations
        r'\bLO_IMPORT\b', r'\bLO_EXPORT\b', r'\bCOPY\b',
        # System functions
        r'\bPG_READ_FILE\b', r'\bPG_WRITE_FILE\b',
        # Schema manipulation
        r'\bALTER\s+SYSTEM\b', r'\bSET\s+ROLE\b',
        # Procedural constructs
        r'\bEXECUTE\s+IMMEDIATE\b', r'\bDYNAMIC\s+SQL\b',
        # Subshells or command execution
        r'\$\$', r'\bSHELL\b', r'\bSYSTEM\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            raise SQLValidationError(f"Dangerous SQL pattern detected: {pattern}")

def _validate_table_access(normalized_query: str) -> None:
    """Validate that query only accesses allowed CRM tables."""
    
    # Extract potential table names using regex
    # Look for patterns like "FROM table_name" or "JOIN table_name"
    table_patterns = [
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    referenced_tables = set()
    for pattern in table_patterns:
        matches = re.findall(pattern, normalized_query, re.IGNORECASE)
        referenced_tables.update([match.lower() for match in matches])
    
    # Check if all referenced tables are in the allowed CRM tables
    allowed_tables_lower = {table.lower() for table in CRM_TABLES}
    
    for table in referenced_tables:
        if table not in allowed_tables_lower:
            raise SQLValidationError(f"Access to table '{table}' is not allowed. Only CRM tables are accessible: {', '.join(CRM_TABLES)}")

def _check_query_complexity(query: str) -> None:
    """
    Check if query complexity is within acceptable limits.
    
    Limits:
    - Maximum 10 JOINs
    - Maximum 2-level nested subqueries
    - Maximum 5000 characters
    """
    
    normalized_query = query.upper()
    
    # Count JOINs - use comprehensive pattern to avoid double counting
    join_pattern = r'\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN\b'
    join_count = len(re.findall(join_pattern, normalized_query))
    
    if join_count > 10:
        raise SQLValidationError(f"Query too complex: {join_count} JOINs found. Maximum allowed: 10")
    
    # Check subquery nesting levels using proper SQL parsing
    try:
        nesting_level = _count_subquery_nesting_levels(query)
        if nesting_level > 2:
            raise SQLValidationError(f"Query too complex: {nesting_level}-level nested subqueries found. Maximum allowed: 2")
    except SQLValidationError:
        # Re-raise validation errors from nesting analysis
        raise
    except Exception as e:
        logger.warning(f"Could not parse query for subquery analysis: {e}")
        # Fallback to parentheses counting
        subquery_count = normalized_query.count('(')
        if subquery_count > 10:  # Conservative limit for unparseable queries
            raise SQLValidationError(f"Query too complex: too many nested elements. Simplify the query.")
    
    # Check query length
    if len(query) > 5000:
        raise SQLValidationError("Query too long. Maximum allowed: 5000 characters")


def _count_subquery_nesting_levels(query: str) -> int:
    """
    Count the maximum nesting level of subqueries in a SQL query.
    
    Args:
        query: The SQL query to analyze
        
    Returns:
        Maximum nesting level (0 = no subqueries, 1 = one level, etc.)
    """
    try:
        # Parse the SQL query
        parsed = sqlparse.parse(query)[0]
        return _analyze_token_nesting(parsed, 0)
    except Exception as e:
        logger.warning(f"Failed to parse query for nesting analysis: {e}")
        return 0


def _analyze_token_nesting(token, current_level: int) -> int:
    """
    Recursively analyze SQL tokens to find maximum subquery nesting level.
    
    Args:
        token: SQL token to analyze
        current_level: Current nesting level
        
    Returns:
        Maximum nesting level found
    """
    max_level = current_level
    
    if hasattr(token, 'tokens'):
        for sub_token in token.tokens:
            if sub_token.ttype is None and isinstance(sub_token, sqlparse.sql.Parenthesis):
                # Check if this parenthesis contains a SELECT (indicating a subquery)
                sub_content = str(sub_token).strip()
                if re.search(r'\bSELECT\b', sub_content.upper()):
                    # This is a subquery, increase nesting level
                    nested_level = _analyze_token_nesting(sub_token, current_level + 1)
                    max_level = max(max_level, nested_level)
                else:
                    # Regular parenthesis, continue with same level
                    nested_level = _analyze_token_nesting(sub_token, current_level)
                    max_level = max(max_level, nested_level)
            else:
                # Recursively check other tokens
                nested_level = _analyze_token_nesting(sub_token, current_level)
                max_level = max(max_level, nested_level)
    
    return max_level


async def _execute_query_with_timeout(db: SQLDatabase, query: str, timeout_seconds: int = 30) -> str:
    """
    Execute a SQL query with timeout protection to prevent long-running queries.
    
    Args:
        db: Database connection
        query: SQL query to execute
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Query result as string
        
    Raises:
        Exception: If query times out or execution fails
    """
    def _run_query():
        """Synchronous query execution wrapper."""
        return db.run(query)
    
    # Use ThreadPoolExecutor to run the synchronous db.run in a separate thread
    loop = asyncio.get_event_loop()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the query to thread pool and wait with timeout
            future = loop.run_in_executor(executor, _run_query)
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return str(result)
            
    except asyncio.TimeoutError:
        logger.error(f"Query execution timed out after {timeout_seconds} seconds")
        raise SQLValidationError(f"Query execution timed out after {timeout_seconds} seconds. Query may be too complex.")
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise


async def _execute_parameterized_query(
    db: SQLDatabase, 
    query_template: str, 
    parameters: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30
) -> str:
    """
    Execute a parameterized SQL query using prepared statements for injection protection.
    
    Args:
        db: Database connection
        query_template: SQL query template with parameter placeholders (:param_name)
        parameters: Dictionary of parameters to bind to the query
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Query result as string
        
    Raises:
        Exception: If query times out or execution fails
    """
    if parameters is None:
        parameters = {}
    
    # Sanitize all parameter values
    sanitized_params = {}
    for key, value in parameters.items():
        if isinstance(value, str):
            sanitized_params[key] = sanitize_sql_input(value)
        else:
            sanitized_params[key] = value
    
    def _run_parameterized_query():
        """Synchronous parameterized query execution wrapper."""
        try:
            # Get the underlying SQLAlchemy engine for parameterized queries
            engine = db._engine
            
            # Create a parameterized query using SQLAlchemy text() 
            sql_text = text(query_template)
            
            # Execute with parameters to prevent injection
            with engine.connect() as connection:
                result = connection.execute(sql_text, sanitized_params)
                
                # Format results similar to LangChain's db.run()
                if result.returns_rows:
                    rows = result.fetchall()
                    if not rows:
                        return "No results found."
                    
                    # Format as table
                    columns = list(result.keys())
                    result_lines = ["\t".join(columns)]
                    for row in rows:
                        result_lines.append("\t".join(str(cell) for cell in row))
                    return "\n".join(result_lines)
                else:
                    return f"Query executed successfully. Rows affected: {result.rowcount}"
                    
        except Exception as e:
            logger.error(f"Parameterized query execution failed: {e}")
            raise
    
    # Use ThreadPoolExecutor to run the synchronous query in a separate thread
    loop = asyncio.get_event_loop()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the query to thread pool and wait with timeout
            future = loop.run_in_executor(executor, _run_parameterized_query)
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result
            
    except asyncio.TimeoutError:
        logger.error(f"Parameterized query execution timed out after {timeout_seconds} seconds")
        raise SQLValidationError(f"Query execution timed out after {timeout_seconds} seconds. Query may be too complex.")
    except Exception as e:
        logger.error(f"Parameterized query execution failed: {e}")
        raise


async def execute_safe_sql_query(query: str) -> Dict[str, Any]:
    """
    Execute a SQL query with full security and semantic validation.
    
    Args:
        query: The SQL query to execute
        
    Returns:
        Dictionary with execution results, warnings, and suggestions
        
    Raises:
        SQLValidationError: If query fails validation
        Exception: If query execution fails
    """
    # Run enhanced validation with semantics
    validation_result = await validate_sql_query_with_semantics(query)
    
    if not validation_result["valid"]:
        # Include validation warnings and suggestions in the error
        error_msg = "Query validation failed: " + "; ".join(validation_result["warnings"])
        if validation_result["suggestions"]:
            error_msg += "\nSuggestions: " + "; ".join(validation_result["suggestions"])
        raise SQLValidationError(error_msg)
    
    validated_query = validation_result["cleaned_query"]
    
    # Additional complexity check
    _check_query_complexity(validated_query)
    
    # Get database connection
    db = await _get_sql_database()
    if db is None:
        raise SQLValidationError("Database connection not available. Check SUPABASE_DB_PASSWORD configuration.")
    
    try:
        # Execute the validated query with timeout protection
        logger.info(f"Executing validated SQL query: {validated_query[:100]}...")
        result = await _execute_query_with_timeout(db, validated_query, timeout_seconds=30)
        logger.info("Query executed successfully")
        
        # Return comprehensive result with validation info
        return {
            "success": True,
            "result": result,
            "query": validated_query,
            "warnings": validation_result["warnings"],
            "suggestions": validation_result["suggestions"],
            "semantic_validation": validation_result["semantic_validation"]
        }
        
    except Exception as e:
        logger.error(f"SQL query execution failed: {e}")
        raise Exception(f"Query execution failed: {str(e)}")


def _get_retriever():
    """Get or create the retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever

async def _get_settings():
    """Get or create the settings instance asynchronously."""
    global _settings
    if _settings is None:
        _settings = await get_settings()
    return _settings

def reset_sql_database():
    """Reset the global SQL database connection to force reinitialization."""
    global _sql_database
    _sql_database = None
    logger.info("SQL database connection reset")

async def _get_sql_database():
    """Get or create the SQL database connection with restricted table access."""
    global _sql_database
    if _sql_database is None:
        settings = await _get_settings()
        
        # Extract project reference from Supabase URL
        # Format: https://project-ref.supabase.co
        project_ref = settings.supabase_url.split('//')[1].split('.')[0]
        
        # Get database password from settings (already loaded from .env file)
        db_password = settings.supabase_db_password
        if not db_password:
            logger.warning("SUPABASE_DB_PASSWORD not set. SQL tools will not be available.")
            return None
        
        # Use Supabase transaction pooler for reliable connections (supports IPv4 and IPv6)
        # Format: postgresql://postgres.project-ref:password@aws-0-region.pooler.supabase.com:6543/postgres
        # Using transaction mode (port 6543) with the correct region for this project
        db_url = f"postgresql://postgres.{project_ref}:{db_password}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
        
        try:
            # Create SQLDatabase with restricted table access
            _sql_database = SQLDatabase.from_uri(
                database_uri=db_url,
                include_tables=CRM_TABLES,  # Only allow access to CRM tables
                sample_rows_in_table_info=2,  # Limit sample data for privacy
                custom_table_info=None,
                view_support=True  # Allow views like sales_pipeline and employee_hierarchy
            )
            
            logger.info(f"Connected to SQL database via pooler with restricted access to tables: {CRM_TABLES}")
            logger.info(f"Using connection: postgresql://postgres.{project_ref}:***@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQL database: {e}")
            logger.error("Make sure SUPABASE_DB_PASSWORD is set with your database password")
            logger.error(f"Attempted connection: postgresql://postgres.{project_ref}:***@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres")
            raise
    
    return _sql_database


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="The search query to find relevant documents")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve (optional)")
    threshold: Optional[float] = Field(default=None, description="Similarity threshold for filtering results (optional)")


class SourceFormattingInput(BaseModel):
    """Input schema for source formatting tool."""
    sources: List[Dict[str, Any]] = Field(description="List of source documents with metadata from semantic_search results", default=[])


class ContextBuildingInput(BaseModel):
    """Input schema for context building tool."""
    documents: List[Dict[str, Any]] = Field(description="List of retrieved documents to build context from", default=[])


class ConversationSummaryInput(BaseModel):
    """Input schema for conversation summary tool."""
    conversation_history: List[Dict[str, str]] = Field(description="List of conversation messages with role and content", default=[])


class CRMQueryInput(BaseModel):
    """Input schema for CRM SQL query tool."""
    question: str = Field(description="Natural language business question about sales, customers, inventory, or performance data")
    time_period: Optional[str] = Field(default=None, description="Optional time period filter like 'last 30 days', 'this quarter', 'this year', or specific date range")


@tool("semantic_search")
@traceable(name="semantic_search_tool")
async def semantic_search(query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> str:
    """
    Search for relevant documents using semantic similarity.
    
    Use this tool when you need to find documents related to a user's question.
    Returns a JSON string with search results including content, sources, and similarity scores.
    """
    try:
        retriever = _get_retriever()
        settings = await _get_settings()
        
        # Use provided parameters or defaults from config
        top_k = top_k or settings.rag_max_retrieved_documents
        threshold = threshold or settings.rag_similarity_threshold
        
        logger.info(f"Performing semantic search for: {query}")
        
        # Retrieve documents
        documents = await retriever.retrieve(
            query=query,
            threshold=threshold,
            top_k=top_k
        )
        
        if not documents:
            return json.dumps({
                "query": query,
                "num_results": 0,
                "documents": [],
                "message": "No relevant documents found for the query."
            })
        
        # Format results for agent consumption
        results = {
            "query": query,
            "num_results": len(documents),
            "documents": []
        }
        
        for doc in documents:
            results["documents"].append({
                "content": doc.get("content", ""),
                "source": doc.get("source", "Unknown"),
                "similarity": doc.get("similarity", 0.0),
                "metadata": doc.get("metadata", {})
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return json.dumps({
            "error": f"Error performing semantic search: {str(e)}",
            "query": query,
            "num_results": 0,
            "documents": []
        })


@tool("format_sources")
@traceable(name="format_sources_tool")
def format_sources(sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Format document sources for user-friendly display.
    
    Use this tool to create a formatted list of sources with similarity scores.
    Input should be a list of source dictionaries from semantic_search results.
    
    Expected format: {"sources": [{"source": "source_name", "similarity": 0.85, ...}, ...]}
    """
    try:
        if sources is None:
            sources = []
        
        if not sources or not isinstance(sources, list):
            logger.warning(f"format_sources called with invalid sources: {type(sources)}")
            return "No sources available."
        
        if len(sources) == 0:
            return "No sources found."
        
        formatted_sources = []
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
            if not isinstance(source, dict):
                logger.warning(f"Invalid source format at index {i}: {type(source)}")
                continue
                
            source_name = source.get("source", "Unknown source")
            similarity = source.get("similarity", 0.0)
            
            # Handle different similarity formats
            if isinstance(similarity, str):
                try:
                    similarity = float(similarity)
                except ValueError:
                    similarity = 0.0
            
            formatted_sources.append(f"{i}. {source_name} (similarity: {similarity:.2f})")
        
        if not formatted_sources:
            return "No valid sources to format."
        
        return "\n".join(formatted_sources)
        
    except Exception as e:
        logger.error(f"Error formatting sources: {str(e)}")
        return f"Error formatting sources: {str(e)}"


@tool("build_context")
@traceable(name="build_context_tool")
def build_context(documents: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Build a formatted context string from retrieved documents.
    
    Use this tool to create a context string from search results that can be used
    to answer user questions. Input should be a list of document dictionaries from semantic_search.
    """
    try:
        if documents is None:
            documents = []
        
        if not documents or not isinstance(documents, list):
            logger.info(f"build_context called with: {type(documents)}")
            return "No documents available to build context."
        
        if len(documents) == 0:
            return "No documents provided to build context."
        
        context_parts = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Invalid document format at index {i}: {type(doc)}")
                continue
                
            content = doc.get("content", "").strip()
            source = doc.get("source", "Unknown source")
            
            if content:
                context_parts.append(f"From {source}:\n{content}")
        
        if not context_parts:
            return "No content available from documents."
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error building context: {str(e)}")
        return f"Error building document context: {str(e)}"


@tool("get_conversation_summary")
@traceable(name="get_conversation_summary_tool")
def get_conversation_summary(conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Get a summary of the recent conversation for context.
    
    Use this tool to understand the conversation context before answering questions.
    If no conversation history is provided, returns a default message.
    """
    try:
        if conversation_history is None:
            conversation_history = []
        
        if not conversation_history or not isinstance(conversation_history, list):
            logger.info(f"get_conversation_summary called with: {type(conversation_history)}")
            return "No previous conversation available."
        
        if len(conversation_history) == 0:
            return "No previous conversation available."
        
        # Get last 5 messages for context
        recent_messages = conversation_history[-5:]
        
        summary_parts = []
        for msg in recent_messages:
            if not isinstance(msg, dict):
                logger.warning(f"Invalid message format: {type(msg)}")
                continue
                
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:  # Only include messages with content
                summary_parts.append(f"{role.capitalize()}: {content}")
        
        if not summary_parts:
            return "No meaningful conversation history found."
        
        return "Recent conversation:\n" + "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error getting conversation summary: {str(e)}")
        return f"Error getting conversation summary: {str(e)}"


async def _generate_sql_with_context(question: str, time_period: Optional[str] = None) -> str:
    """
    Generate SQL query from natural language question using schema-aware prompts.
    
    Args:
        question: Natural language business question
        time_period: Optional time period filter
        
    Returns:
        Generated SQL query string
    """
    try:
        from openai import AsyncOpenAI
        
        settings = await _get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Build context with schema information
        schema_context = _build_simple_schema_context()
        
        # Build time period filter if specified
        time_filter = _build_time_filter(time_period) if time_period else ""
        
        # Construct the prompt
        prompt = f"""{SQL_GENERATION_PROMPTS['system_prompt']}

SCHEMA CONTEXT:
{schema_context}

USER QUESTION: "{question}"
TIME PERIOD FILTER: {time_filter}

STEP-BY-STEP ANALYSIS:
Before writing SQL, think through these steps:

1. IDENTIFY: What information does the user want?
2. LOCATE: Which table contains this information?
3. CHECK: Are there direct columns that answer this?
4. VERIFY: Do I need data from other tables?
5. SIMPLIFY: What's the simplest query approach?

EXAMPLES FOR REFERENCE:
- Employee position question → Use employees.position column directly
- Count managers → Use WHERE position = 'manager' 
- Customer info → Use customers table directly
- Vehicle availability → Use vehicles.is_available = true

REQUIREMENTS:
- Generate the SIMPLEST possible PostgreSQL SELECT query
- Use direct column lookups when available
- Only add JOINs if you need data from multiple tables
- Apply business filters (is_active=true for employees, is_available=true for vehicles)
- Include meaningful column aliases
- Limit results if returning many rows

RESPOND WITH ONLY THE SQL QUERY (NO SEMICOLON AT THE END), NO EXPLANATIONS OR COMMENTS."""

        # Generate SQL using OpenAI
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1  # Low temperature for more consistent SQL generation
        )
        
        generated_sql = (response.choices[0].message.content or "").strip()
        
        # Clean up the response (remove markdown formatting if present)
        if generated_sql.startswith("```sql"):
            generated_sql = generated_sql[6:]
        if generated_sql.startswith("```"):
            generated_sql = generated_sql[3:]
        if generated_sql.endswith("```"):
            generated_sql = generated_sql[:-3]
        
        # Remove trailing semicolon to avoid injection detection issues
        generated_sql = generated_sql.strip()
        if generated_sql.endswith(";"):
            generated_sql = generated_sql[:-1].strip()
            
        return generated_sql
        
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise Exception(f"Failed to generate SQL query: {str(e)}")


def _build_simple_schema_context() -> str:
    """Build simplified schema context focusing on direct column usage and key relationships."""
    context = "AVAILABLE TABLES AND KEY COLUMNS:\n\n"
    
    # Simplified schema focusing on direct columns and key relationships
    tables = {
        "vehicles": {
            "purpose": "Vehicle inventory and specifications",
            "key_columns": [
                "id (UUID): unique vehicle identifier - USED TO JOIN WITH PRICING",
                "brand (VARCHAR): manufacturer (Toyota, Honda, etc.)",
                "model (VARCHAR): vehicle model (Prius, Civic, etc.)",
                "type (ENUM): 'sedan', 'suv', 'hatchback', 'pickup', 'van', 'motorcycle', 'truck'",
                "year (INTEGER): model year",
                "is_available (BOOLEAN): whether available for sale"
            ],
            "joins_with": "pricing (vehicles.id = pricing.vehicle_id)"
        },
        "pricing": {
            "purpose": "Vehicle pricing with discounts and final prices",
            "key_columns": [
                "id (UUID): unique pricing record identifier",
                "vehicle_id (UUID): FOREIGN KEY TO vehicles.id",
                "base_price (DECIMAL): base vehicle price before discounts",
                "final_price (DECIMAL): final price after all discounts",
                "price_discount (DECIMAL): discount on base price",
                "is_active (BOOLEAN): whether this pricing is currently active"
            ],
            "joins_with": "vehicles (pricing.vehicle_id = vehicles.id)"
        },
        "employees": {
            "purpose": "Employee information and positions",
            "key_columns": [
                "id (UUID): unique employee identifier",
                "name (VARCHAR): employee full name",
                "position (ENUM): 'sales_agent', 'account_executive', 'manager', 'director', 'admin'",
                "is_active (BOOLEAN): whether employee is currently active",
                "branch_id (UUID): which branch they work at",
                "manager_ae_id (UUID): their manager's ID (for hierarchy only)"
            ],
            "joins_with": "branches (employees.branch_id = branches.id)"
        },
        "customers": {
            "purpose": "Customer contact and business information",
            "key_columns": [
                "id (UUID): unique customer identifier",
                "name (VARCHAR): customer full name",
                "company (VARCHAR): company name if business customer",
                "is_for_business (BOOLEAN): B2B vs B2C customer",
                "email (VARCHAR): customer email",
                "phone (VARCHAR): customer phone"
            ],
            "joins_with": "opportunities (customers.id = opportunities.customer_id)"
        },
        "opportunities": {
            "purpose": "Sales opportunities and pipeline tracking",
            "key_columns": [
                "id (UUID): unique opportunity identifier",
                "customer_id (UUID): FOREIGN KEY TO customers.id",
                "vehicle_id (UUID): FOREIGN KEY TO vehicles.id",
                "opportunity_salesperson_ae_id (UUID): FOREIGN KEY TO employees.id",
                "stage (ENUM): 'New', 'Contacted', 'Consideration', 'Purchase Intent', 'Won', 'Lost'",
                "warmth (ENUM): 'hot', 'warm', 'cold', 'dormant'",
                "probability (INTEGER): 0-100% chance of closing",
                "estimated_value (DECIMAL): expected sale value"
            ],
            "joins_with": "customers, vehicles, employees, transactions"
        },
        "transactions": {
            "purpose": "Completed sales and financial records",
            "key_columns": [
                "id (UUID): unique transaction identifier",
                "opportunity_id (UUID): FOREIGN KEY TO opportunities.id",
                "status (ENUM): 'pending', 'processing', 'completed', 'cancelled', 'refunded'",
                "total_amount (DECIMAL): transaction amount",
                "payment_method (VARCHAR): how customer paid"
            ],
            "joins_with": "opportunities (transactions.opportunity_id = opportunities.id)"
        },
        "branches": {
            "purpose": "Company locations and dealerships",
            "key_columns": [
                "id (UUID): unique branch identifier",
                "name (VARCHAR): branch name",
                "region (ENUM): 'north', 'south', 'east', 'west', 'central'",
                "brand (VARCHAR): vehicle brand sold at this branch"
            ],
            "joins_with": "employees (branches.id = employees.branch_id)"
        },
        "activities": {
            "purpose": "Sales activities and customer interactions",
            "key_columns": [
                "id (UUID): unique activity identifier",
                "customer_id (UUID): FOREIGN KEY TO customers.id",
                "opportunity_id (UUID): FOREIGN KEY TO opportunities.id",
                "activity_type (ENUM): 'call', 'email', 'meeting', 'demo', 'follow_up', 'proposal_sent', 'contract_signed'",
                "is_completed (BOOLEAN): whether activity is finished"
            ],
            "joins_with": "customers, opportunities"
        }
    }
    
    for table_name, table_info in tables.items():
        context += f"{table_name.upper()}:\n"
        context += f"  Purpose: {table_info['purpose']}\n"
        context += f"  Key Columns:\n"
        for column in table_info['key_columns']:
            context += f"    - {column}\n"
        if 'joins_with' in table_info:
            context += f"  Common Joins: {table_info['joins_with']}\n"
        context += "\n"
    
    context += "IMPORTANT RELATIONSHIPS:\n"
    context += "🔗 For PRICING: vehicles.id = pricing.vehicle_id (ALWAYS JOIN for price questions)\n"
    context += "🔗 For SALES: employees → opportunities → transactions\n"
    context += "🔗 For CUSTOMERS: customers → opportunities → transactions\n"
    context += "🔗 For LOCATIONS: branches → employees → opportunities\n\n"
    
    return context


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


@tool("query_crm_data")
@traceable(name="query_crm_data_tool")
async def query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database with natural language questions about sales, customers, inventory, performance, and PRICING.
    
    This tool converts business questions into SQL queries and executes them safely against the CRM database.
    
    **IMPORTANT: Use this tool for ANY question about data that exists in our CRM system, including:**
    
    PRICING & VEHICLE DATA:
    - Vehicle prices (e.g., "How much is the Prius?", "What's the price of the Honda Civic?")
    - Price comparisons between models
    - Vehicle specifications (brand, model, year, type, power, etc.)
    - Inventory availability and stock levels
    
    SALES & BUSINESS DATA:
    - Sales performance by employee, branch, time period
    - Customer analysis and lifetime value  
    - Sales pipeline and conversion rates
    - Activity tracking and follow-up management
    - Revenue and transaction analysis
    
    PEOPLE & ORGANIZATION:
    - Employee information (positions, names, hierarchy)
    - Customer details and business relationships
    - Branch and location data
    
    **The database contains comprehensive information about vehicles, pricing, employees, customers, opportunities, and transactions. Always try this tool first before using semantic_search for factual questions.**
    
    Args:
        question: Natural language business question (e.g., "How much is the Prius?", "Who are our top salespeople?")
        time_period: Optional time filter (e.g., "last 30 days", "this quarter", "this year")
        
    Returns:
        Formatted query results with business insights
    """
    try:
        logger.info(f"Processing CRM query: {question}")
        if time_period:
            logger.info(f"Time period filter: {time_period}")
        
        # Generate SQL from natural language
        generated_sql = await _generate_sql_with_context(question, time_period)
        logger.info(f"Generated SQL: {generated_sql}")
        
        # Execute the query with full security validation
        result = await execute_safe_sql_query(generated_sql)
        
        # Debug logging
        logger.info(f"Query execution result: success={result.get('success')}")
        logger.info(f"Raw result data: '{result.get('result', 'NO_RESULT')}'")
        logger.info(f"Result type: {type(result.get('result'))}")
        
        if result["success"]:
            # Format the results for business users
            formatted_result = _format_business_results(result["result"], question)
            return formatted_result
        else:
            error_msg = f"❌ **Query Failed:** {result.get('error', 'Unknown error')}"
            logger.error(f"CRM query failed: {result.get('error', 'Unknown error')}")
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ **Error processing CRM query:** {str(e)}"
        logger.error(f"Error in query_crm_data: {str(e)}")
        return error_msg


def _format_business_results(data: str, question: str) -> str:
    """Format SQL query results for business users with context and insights."""
    if not data or data.strip() == "":
        return f"📊 **Query Results for:** {question}\n\n✨ No data found matching your criteria."
    
    # Parse the data to understand structure
    lines = data.strip().split('\n')
    
    # Debug logging to understand what we're getting
    logger.info(f"Raw SQL result data: '{data}'")
    logger.info(f"Number of lines: {len(lines)}")
    logger.info(f"Lines: {lines}")
    
    # Handle empty results more carefully
    if len(lines) == 0 or (len(lines) == 1 and not lines[0].strip()):
        return f"📊 **Query Results for:** {question}\n\n✨ Employee not found in database."
    
    # Handle single-line results (like [(9,)] from SQLAlchemy)
    if len(lines) == 1 and lines[0].strip():
        line = lines[0].strip()
        
        # Check for SQLAlchemy tuple format like [(9,)]
        if line.startswith('[') and line.endswith(']'):
            import re
            matches = re.findall(r'\((\d+),?\)', line)
            if matches:
                number = matches[0]
                formatted = f"📊 **CRM Query Results**\n\n"
                formatted += f"**Question:** {question}\n\n"
                formatted += f"**Answer:** {number}\n\n"
                formatted += _generate_business_insights(data, question, 1)
                return formatted
        
        # Check for single value results
        if line and not line.startswith('No') and line != 'None':
            formatted = f"📊 **CRM Query Results**\n\n"
            formatted += f"**Question:** {question}\n\n"
            formatted += f"**Answer:** {line}\n\n"
            formatted += _generate_business_insights(data, question, 1)
            return formatted
    
    # Handle traditional tabular results
    if len(lines) >= 2:
        header_line = lines[0] if lines else ""
        data_lines = lines[1:] if len(lines) > 1 else []
        
        # Filter out empty lines
        data_lines = [line for line in data_lines if line.strip()]
        row_count = len(data_lines)
        
        if row_count == 0:
            return f"📊 **Query Results for:** {question}\n\n✨ Employee not found in database."
        
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
    
    # Fallback for unexpected formats
    return f"📊 **Query Results for:** {question}\n\n✨ Employee not found in database."


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


def get_all_tools():
    """
    Get all available tools for the RAG agent.
    
    Returns a list of tool functions that can be bound to the LLM.
    """
    return [
        semantic_search,
        format_sources,
        build_context,
        get_conversation_summary,
        query_crm_data
    ]


def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()] 