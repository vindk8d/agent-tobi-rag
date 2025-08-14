"""
Shared utility functions and classes for all tools in the toolbox.

This module contains common functionality that can be used across all tools:
- Database connections and queries
- LLM model selection and access
- User context management
- Common data structures
- Shared helper functions
"""

import logging
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

# Core imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

# Backend imports - using minimal dependencies for toolbox
try:
    from core.config import get_settings_sync
    from core.database import get_db_client
    settings = get_settings_sync()
    get_db_connection = get_db_client  # Alias for compatibility
except ImportError:
    try:
        # Fallback for when running from backend directory
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from core.config import get_settings_sync
        from core.database import get_db_client
        settings = get_settings_sync()
        get_db_connection = get_db_client  # Alias for compatibility
    except ImportError:
        # Ultimate fallback - create mock settings for testing
        class MockSettings:
            OPENAI_MODEL_SIMPLE = "gpt-4o-mini"
            OPENAI_MODEL_COMPLEX = "gpt-4o"
            OPENAI_API_KEY = "mock-key"
        settings = MockSettings()
        get_db_connection = lambda: None

# Conversation import with fallback
try:
    from models.conversation import get_conversation_messages
except ImportError:
    try:
        from models.conversation import get_conversation_messages
    except ImportError:
        # Fallback for testing
        async def get_conversation_messages(conversation_id: str, limit: int = 6):
            return []

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================

# Context variables for tracking user session data
_user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_conversation_id_context: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)
_user_type_context: ContextVar[Optional[str]] = ContextVar('user_type', default=None)
_employee_id_context: ContextVar[Optional[str]] = ContextVar('employee_id', default=None)

class UserContext:
    """Context manager for setting user session variables."""
    
    def __init__(self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, 
                 user_type: Optional[str] = None, employee_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_type = user_type
        self.employee_id = employee_id
        self.tokens = []
    
    def __enter__(self):
        if self.user_id is not None:
            self.tokens.append(_user_id_context.set(self.user_id))
        if self.conversation_id is not None:
            self.tokens.append(_conversation_id_context.set(self.conversation_id))
        if self.user_type is not None:
            self.tokens.append(_user_type_context.set(self.user_type))
        if self.employee_id is not None:
            self.tokens.append(_employee_id_context.set(self.employee_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context variable tokens are automatically reset when they go out of scope
        # No manual reset needed in modern Python versions
        pass

def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context."""
    return _user_id_context.get()

def get_current_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context."""
    return _conversation_id_context.get()

def get_current_user_type() -> Optional[str]:
    """Get the current user type from context."""
    return _user_type_context.get()

def get_current_employee_id_from_context() -> Optional[str]:
    """Get the current employee ID from context."""
    return _employee_id_context.get()

def get_user_context() -> dict:
    """Get all current user context as a dictionary."""
    return {
        "user_id": get_current_user_id(),
        "conversation_id": get_current_conversation_id(),
        "user_type": get_current_user_type(),
        "employee_id": get_current_employee_id_from_context()
    }

# =============================================================================
# MODEL SELECTION AND LLM ACCESS
# =============================================================================

class QueryComplexity(Enum):
    """Enumeration for query complexity levels."""
    SIMPLE = "simple"
    COMPLEX = "complex"

class ModelSelector:
    """Intelligent model selection based on query complexity."""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Define complexity indicators
        self.complex_indicators = [
            'analyze', 'compare', 'summarize', 'explain', 'generate', 'create',
            'multiple', 'several', 'various', 'different', 'complex', 'detailed',
            'comprehensive', 'thorough', 'in-depth', 'relationship', 'correlation',
            'trend', 'pattern', 'insight', 'recommendation'
        ]
    
    def classify_query_complexity(self, messages):
        """Classify query complexity based on content."""
        if not messages:
            return QueryComplexity.SIMPLE
        
        # Get the most recent user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user' or msg.get('type') == 'human':
                user_message = msg.get('content', '').lower()
                break
        
        # Check for complex indicators
        for indicator in self.complex_indicators:
            if indicator in user_message:
                return QueryComplexity.COMPLEX
        
        # Check length - longer queries tend to be more complex
        if len(user_message.split()) > 15:
            return QueryComplexity.COMPLEX
        
        return QueryComplexity.SIMPLE
    
    def get_model_for_query(self, messages):
        """Select appropriate model based on query complexity."""
        complexity = self.classify_query_complexity(messages)
        
        if complexity == QueryComplexity.COMPLEX:
            return ChatOpenAI(
                model=self.settings.openai_complex_model,
                temperature=0.1,
                max_tokens=4000
            )
        else:
            return ChatOpenAI(
                model=self.settings.openai_simple_model,
                temperature=0.1,
                max_tokens=2000
            )

# Global model selector instance
model_selector = ModelSelector(settings)

# =============================================================================
# QUOTATION BUSINESS INTELLIGENCE - Task 15.6.5
# =============================================================================

@dataclass
class PricingAnalysis:
    """Analysis of pricing decisions and business intelligence."""
    base_pricing: Dict[str, Any] = field(default_factory=dict)
    customer_tier_pricing: Dict[str, Any] = field(default_factory=dict)
    market_adjustments: Dict[str, Any] = field(default_factory=dict)
    promotional_opportunities: List[str] = field(default_factory=list)
    pricing_recommendations: List[str] = field(default_factory=list)
    business_rationale: str = ""
    confidence_score: float = 0.0

@dataclass
class BusinessContext:
    """Business context for intelligent pricing decisions."""
    customer_profile: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    business_rules: Dict[str, Any] = field(default_factory=dict)
    competitive_landscape: Dict[str, Any] = field(default_factory=dict)
    seasonal_factors: Dict[str, Any] = field(default_factory=dict)

class QuotationBusinessIntelligence:
    """
    Intelligent pricing decisions that consider customer context, business rules, and market conditions.
    
    Task 15.6.5: CREATE QuotationBusinessIntelligence class for intelligent pricing decisions
    that consider customer context, business rules, and market conditions.
    
    Consolidates BRITTLE PROCESS #6: Replace fixed pricing calculations with contextual 
    business intelligence that understands customer profiles and promotional opportunities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def analyze_pricing_context(
        self,
        customer_info: Dict[str, Any],
        vehicle_requirements: Dict[str, Any],
        purchase_preferences: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> PricingAnalysis:
        """
        Analyze pricing context using LLM-driven business intelligence.
        
        Replaces fixed pricing calculations with intelligent analysis that considers:
        - Customer tier and profile
        - Vehicle specifications and market positioning
        - Purchase preferences and urgency
        - Market conditions and competitive landscape
        - Promotional opportunities and business rules
        
        Args:
            customer_info: Customer details and profile information
            vehicle_requirements: Vehicle specifications and requirements
            purchase_preferences: Purchase preferences, timeline, and budget
            market_context: Current market conditions and competitive data
            
        Returns:
            PricingAnalysis with intelligent pricing recommendations
        """
        try:
            self.logger.info("[BUSINESS_INTELLIGENCE] 💼 Starting intelligent pricing analysis")
            
            # Get business intelligence LLM (use complex model for strategic decisions)
            llm = await get_appropriate_llm("comprehensive business analysis and strategic pricing decisions")
            
            # Create comprehensive business intelligence template
            analysis_template = self._create_business_intelligence_template()
            
            # Prepare LLM input with all business context
            llm_input = f"""
{analysis_template}

CUSTOMER INFORMATION:
{self._format_customer_context(customer_info)}

VEHICLE REQUIREMENTS:
{self._format_vehicle_context(vehicle_requirements)}

PURCHASE PREFERENCES:
{self._format_purchase_context(purchase_preferences)}

MARKET CONTEXT:
{self._format_market_context(market_context or {})}

BUSINESS INTELLIGENCE REQUEST:
Provide comprehensive pricing analysis and business recommendations in valid JSON format.
"""
            
            self.logger.info("[BUSINESS_INTELLIGENCE] 🧠 Performing LLM-driven pricing analysis")
            
            # Get LLM analysis
            response = await llm.ainvoke([{"role": "user", "content": llm_input}])
            
            # Parse response into PricingAnalysis
            pricing_analysis = self._parse_pricing_analysis_response(response.content)
            
            self.logger.info(f"[BUSINESS_INTELLIGENCE] ✅ Pricing analysis complete - Confidence: {pricing_analysis.confidence_score:.2f}")
            
            return pricing_analysis
            
        except Exception as e:
            self.logger.error(f"[BUSINESS_INTELLIGENCE] Error in pricing analysis: {e}")
            
            # Fallback to basic pricing analysis
            return PricingAnalysis(
                base_pricing={"status": "fallback", "message": "Basic pricing analysis"},
                business_rationale="Error in advanced analysis, using fallback pricing",
                confidence_score=0.5
            )
    
    def _create_business_intelligence_template(self) -> str:
        """Create comprehensive template for LLM business intelligence analysis."""
        return """You are an expert automotive business analyst and pricing strategist. Analyze the provided information and provide comprehensive JSON output.

BUSINESS INTELLIGENCE ANALYSIS:

1. CUSTOMER TIER ASSESSMENT:
   - Analyze customer profile and purchasing power
   - Determine customer tier (premium, standard, budget-conscious)
   - Assess loyalty potential and lifetime value
   - Identify decision-making factors and preferences

2. PRICING STRATEGY ANALYSIS:
   - Evaluate vehicle positioning and market segment
   - Consider competitive landscape and pricing pressures
   - Assess customer budget alignment and flexibility
   - Identify value proposition and differentiation opportunities

3. PROMOTIONAL OPPORTUNITIES:
   - Identify applicable promotions and incentives
   - Evaluate trade-in opportunities and value
   - Consider financing options and payment preferences
   - Assess seasonal factors and market timing

4. BUSINESS RECOMMENDATIONS:
   - Recommend pricing approach (competitive, value-based, premium)
   - Suggest add-ons, accessories, or service packages
   - Identify upselling and cross-selling opportunities
   - Provide negotiation guidance and flexibility ranges

5. RISK ASSESSMENT:
   - Evaluate deal closure probability
   - Assess competitive threats and alternatives
   - Consider market volatility and timing factors
   - Identify potential objections and mitigation strategies

OUTPUT FORMAT:
Provide analysis in this exact JSON structure:

{
  "base_pricing": {
    "vehicle_msrp": "estimated MSRP or base price",
    "market_adjustment": "market-based pricing adjustment",
    "competitive_position": "positioning vs competitors",
    "value_assessment": "value proposition strength"
  },
  "customer_tier_pricing": {
    "customer_tier": "premium|standard|budget",
    "pricing_approach": "competitive|value_based|premium",
    "discount_eligibility": "available discounts and incentives",
    "financing_recommendations": "recommended financing options"
  },
  "market_adjustments": {
    "demand_factor": "high|medium|low demand impact",
    "seasonal_adjustment": "seasonal pricing considerations",
    "inventory_factor": "inventory availability impact",
    "competitive_pressure": "competitive pricing pressure"
  },
  "promotional_opportunities": [
    "list of applicable promotions",
    "trade-in opportunities",
    "financing incentives",
    "package deals"
  ],
  "pricing_recommendations": [
    "specific pricing strategies",
    "negotiation guidelines",
    "value-add suggestions",
    "closing techniques"
  ],
  "business_rationale": "comprehensive explanation of pricing strategy and business logic",
  "confidence_score": 0.85
}

IMPORTANT: Return ONLY valid JSON, no explanations or additional text."""

    def _format_customer_context(self, customer_info: Dict[str, Any]) -> str:
        """Format customer context for LLM analysis."""
        if not customer_info:
            return "No customer information available."
        
        context_parts = []
        
        # Basic customer info
        if customer_info.get("name"):
            context_parts.append(f"Customer: {customer_info['name']}")
        if customer_info.get("email"):
            context_parts.append(f"Email: {customer_info['email']}")
        if customer_info.get("company"):
            context_parts.append(f"Company: {customer_info['company']}")
        
        # Customer profile indicators
        if customer_info.get("customer_tier"):
            context_parts.append(f"Tier: {customer_info['customer_tier']}")
        if customer_info.get("purchase_history"):
            context_parts.append(f"History: {customer_info['purchase_history']}")
        if customer_info.get("communication_style"):
            context_parts.append(f"Style: {customer_info['communication_style']}")
        
        return "\n".join(context_parts) if context_parts else "Basic customer information available."
    
    def _format_vehicle_context(self, vehicle_requirements: Dict[str, Any]) -> str:
        """Format vehicle context for LLM analysis."""
        if not vehicle_requirements:
            return "No vehicle requirements specified."
        
        context_parts = []
        
        # Vehicle specifications
        if vehicle_requirements.get("make"):
            context_parts.append(f"Make: {vehicle_requirements['make']}")
        if vehicle_requirements.get("model"):
            context_parts.append(f"Model: {vehicle_requirements['model']}")
        if vehicle_requirements.get("year"):
            context_parts.append(f"Year: {vehicle_requirements['year']}")
        if vehicle_requirements.get("type"):
            context_parts.append(f"Type: {vehicle_requirements['type']}")
        
        # Requirements and preferences
        if vehicle_requirements.get("quantity"):
            context_parts.append(f"Quantity: {vehicle_requirements['quantity']}")
        if vehicle_requirements.get("budget_range"):
            context_parts.append(f"Budget: {vehicle_requirements['budget_range']}")
        if vehicle_requirements.get("features"):
            context_parts.append(f"Features: {vehicle_requirements['features']}")
        
        return "\n".join(context_parts) if context_parts else "Basic vehicle requirements specified."
    
    def _format_purchase_context(self, purchase_preferences: Dict[str, Any]) -> str:
        """Format purchase context for LLM analysis."""
        if not purchase_preferences:
            return "No purchase preferences specified."
        
        context_parts = []
        
        # Purchase preferences
        if purchase_preferences.get("timeline"):
            context_parts.append(f"Timeline: {purchase_preferences['timeline']}")
        if purchase_preferences.get("financing"):
            context_parts.append(f"Financing: {purchase_preferences['financing']}")
        if purchase_preferences.get("trade_in"):
            context_parts.append(f"Trade-in: {purchase_preferences['trade_in']}")
        if purchase_preferences.get("delivery"):
            context_parts.append(f"Delivery: {purchase_preferences['delivery']}")
        
        # Decision factors
        if purchase_preferences.get("priority_factors"):
            context_parts.append(f"Priorities: {purchase_preferences['priority_factors']}")
        if purchase_preferences.get("urgency"):
            context_parts.append(f"Urgency: {purchase_preferences['urgency']}")
        
        return "\n".join(context_parts) if context_parts else "Standard purchase preferences."
    
    def _format_market_context(self, market_context: Dict[str, Any]) -> str:
        """Format market context for LLM analysis."""
        if not market_context:
            return "Standard market conditions assumed."
        
        context_parts = []
        
        # Market conditions
        if market_context.get("demand"):
            context_parts.append(f"Demand: {market_context['demand']}")
        if market_context.get("inventory"):
            context_parts.append(f"Inventory: {market_context['inventory']}")
        if market_context.get("competition"):
            context_parts.append(f"Competition: {market_context['competition']}")
        if market_context.get("seasonal"):
            context_parts.append(f"Seasonal: {market_context['seasonal']}")
        
        return "\n".join(context_parts) if context_parts else "Standard market conditions."
    
    def _parse_pricing_analysis_response(self, response_content: str) -> PricingAnalysis:
        """Parse LLM response into PricingAnalysis."""
        try:
            import json
            
            # Extract JSON from response
            json_content = self._extract_json_from_response(response_content)
            analysis_data = json.loads(json_content)
            
            return PricingAnalysis(
                base_pricing=analysis_data.get("base_pricing", {}),
                customer_tier_pricing=analysis_data.get("customer_tier_pricing", {}),
                market_adjustments=analysis_data.get("market_adjustments", {}),
                promotional_opportunities=analysis_data.get("promotional_opportunities", []),
                pricing_recommendations=analysis_data.get("pricing_recommendations", []),
                business_rationale=analysis_data.get("business_rationale", ""),
                confidence_score=float(analysis_data.get("confidence_score", 0.0))
            )
            
        except Exception as e:
            self.logger.error(f"[BUSINESS_INTELLIGENCE] Error parsing pricing analysis: {e}")
            
            # Return fallback analysis
            return PricingAnalysis(
                base_pricing={"status": "parse_error", "message": "Could not parse pricing analysis"},
                business_rationale="Error parsing LLM response, using fallback analysis",
                confidence_score=0.3
            )
    
    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response."""
        import re
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return the response as-is and let JSON parsing handle the error
        return response_content

async def get_appropriate_llm(question: Optional[str] = None) -> ChatOpenAI:
    """Get appropriate LLM based on query complexity."""
    try:
        messages = []
        if question:
            messages = [{"role": "user", "content": question}]
        
        return model_selector.get_model_for_query(messages)
    except Exception as e:
        logger.warning(f"Error in model selection, using default: {e}")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_SIMPLE,
            temperature=0.1,
            max_tokens=2000
        )

async def get_sql_llm(question: Optional[str] = None) -> ChatOpenAI:
    """Get LLM optimized for SQL generation tasks."""
    try:
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_COMPLEX,  # Use complex model for SQL
            temperature=0.0,  # Very deterministic for SQL
            max_tokens=1000   # SQL queries don't need many tokens
        )
    except Exception as e:
        logger.warning(f"Error getting SQL LLM, using default: {e}")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000
        )

# =============================================================================
# DATABASE ACCESS
# =============================================================================

# Global database connection cache
_supabase_database_cache: Optional['SupabaseQueryInterface'] = None
_db_connection_pool = None

class SupabaseQueryInterface:
    """
    Supabase-compatible interface that mimics SQLDatabase for existing tools.
    
    This provides both Option A (native Supabase interface) and backward compatibility
    for existing code that expects SQLDatabase-like behavior.
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.supabase = db_client.client
        logger.info("[DATABASE] SupabaseQueryInterface initialized")
    
    def run(self, query: str) -> str:
        """
        Execute a query and return results in SQLDatabase-compatible format.
        
        For safety, this converts SQL queries to Supabase queries where possible,
        or uses Supabase's raw SQL execution for complex queries.
        """
        try:
            query_upper = query.upper().strip()
            logger.info(f"[DATABASE] Executing query: {query[:100]}...")
            
            # Handle different query patterns
            if "SELECT" not in query_upper:
                logger.warning("[DATABASE] Only SELECT queries are supported for security")
                return "Only SELECT queries are supported for security reasons."
            
            # Extract table name and conditions
            table_name = self._extract_table_name(query)
            if not table_name:
                logger.warning(f"[DATABASE] Could not extract table name from query: {query}")
                return "Could not parse table name from query."
            
            # Execute based on table type
            if table_name == 'vehicles':
                return self._query_vehicles(query)
            elif table_name == 'customers':
                return self._query_customers(query)
            elif table_name == 'employees':
                return self._query_employees(query)
            elif table_name in ['conversations', 'messages']:
                return self._query_conversation_data(query, table_name)
            else:
                logger.warning(f"[DATABASE] Unsupported table: {table_name}")
                return f"Table '{table_name}' is not accessible through this interface."
            
        except Exception as e:
            logger.error(f"[DATABASE] Error executing query: {e}")
            return f"Query execution failed: {str(e)}"
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from SQL query."""
        import re
        query_upper = query.upper()
        
        # Look for "FROM table_name" pattern
        match = re.search(r'FROM\s+(\w+)', query_upper)
        if match:
            return match.group(1).lower()
        return ""
    
    def _query_vehicles(self, query: str) -> str:
        """Execute vehicle-related queries."""
        try:
            query_upper = query.upper()
            
            # Build Supabase query
            supabase_query = self.supabase.table('vehicles').select('*')
            
            # Add basic WHERE conditions if present
            if 'WHERE' in query_upper:
                # For now, handle simple conditions
                if 'STATUS' in query_upper and 'AVAILABLE' in query_upper:
                    supabase_query = supabase_query.eq('status', 'available')
                elif 'PRICE' in query_upper:
                    # Handle price conditions - simplified for now
                    pass
            
            # Add LIMIT
            if 'LIMIT' in query_upper:
                import re
                limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
                if limit_match:
                    limit = int(limit_match.group(1))
                    supabase_query = supabase_query.limit(min(limit, 100))  # Cap at 100
                else:
                    supabase_query = supabase_query.limit(50)
            else:
                supabase_query = supabase_query.limit(50)
            
            result = supabase_query.execute()
            return self._format_results_as_sql_output(
                result.data, 
                ['id', 'make', 'model', 'year', 'type', 'price', 'status', 'color', 'mileage']
            )
            
        except Exception as e:
            logger.error(f"[DATABASE] Error in vehicle query: {e}")
            return ""
    
    def _query_customers(self, query: str) -> str:
        """Execute customer-related queries."""
        try:
            supabase_query = self.supabase.table('customers').select('*').limit(50)
            result = supabase_query.execute()
            return self._format_results_as_sql_output(
                result.data, 
                ['id', 'name', 'email', 'phone', 'address', 'company']
            )
        except Exception as e:
            logger.error(f"[DATABASE] Error in customer query: {e}")
            return ""
    
    def _query_employees(self, query: str) -> str:
        """Execute employee-related queries."""
        try:
            supabase_query = self.supabase.table('employees').select('*').limit(50)
            result = supabase_query.execute()
            return self._format_results_as_sql_output(
                result.data, 
                ['id', 'name', 'position', 'email', 'phone', 'branch']
            )
        except Exception as e:
            logger.error(f"[DATABASE] Error in employee query: {e}")
            return ""
    
    def _query_conversation_data(self, query: str, table_name: str) -> str:
        """Execute conversation/message queries with limited data."""
        try:
            if table_name == 'conversations':
                supabase_query = self.supabase.table('conversations').select('id,user_id,title,created_at,updated_at').limit(20)
                result = supabase_query.execute()
                return self._format_results_as_sql_output(
                    result.data, 
                    ['id', 'user_id', 'title', 'created_at', 'updated_at']
                )
            else:  # messages
                # Limit message queries for privacy
                supabase_query = self.supabase.table('messages').select('id,conversation_id,role,created_at').limit(10)
                result = supabase_query.execute()
                return self._format_results_as_sql_output(
                    result.data, 
                    ['id', 'conversation_id', 'role', 'created_at']
                )
        except Exception as e:
            logger.error(f"[DATABASE] Error in {table_name} query: {e}")
            return ""
    
    def _format_results_as_sql_output(self, data: list, columns: list) -> str:
        """Format Supabase results to look like SQL query output."""
        if not data:
            return ""
        
        # Create header
        header = " | ".join(columns)
        separator = "-" * len(header)
        
        # Create rows
        rows = []
        for item in data:
            row_values = []
            for col in columns:
                value = item.get(col, 'NULL')
                if value is None:
                    value = 'NULL'
                row_values.append(str(value))
            rows.append(" | ".join(row_values))
        
        # Combine all parts
        result_lines = [header, separator] + rows
        return "\n".join(result_lines)
    
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get table information in SQLDatabase-compatible format."""
        # Return schema information for the requested tables
        tables = table_names or ['customers', 'vehicles', 'employees', 'conversations', 'messages']
        
        schema_parts = []
        for table in tables:
            if table == 'vehicles':
                schema_parts.append("""
CREATE TABLE vehicles (
    id TEXT PRIMARY KEY,
    make TEXT,
    model TEXT,
    year INTEGER,
    type TEXT,
    price DECIMAL,
    status TEXT,
    color TEXT,
    mileage INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);""")
            elif table == 'customers':
                schema_parts.append("""
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    address TEXT,
    company TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);""")
            elif table == 'employees':
                schema_parts.append("""
CREATE TABLE employees (
    id TEXT PRIMARY KEY,
    name TEXT,
    position TEXT,
    email TEXT,
    phone TEXT,
    branch TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);""")
        
        return "\n".join(schema_parts)

async def get_sql_database() -> Optional[SupabaseQueryInterface]:
    """Get cached Supabase database interface."""
    global _supabase_database_cache
    
    if _supabase_database_cache is None:
        try:
            connection = get_db_client()
            if connection:
                # Create Supabase query interface
                _supabase_database_cache = SupabaseQueryInterface(connection)
                logger.info("[DATABASE] Supabase database interface established")
            else:
                logger.error("[DATABASE] Failed to get database connection")
                return None
        except Exception as e:
            logger.error(f"[DATABASE] Error creating Supabase database interface: {e}")
            return None
    
    return _supabase_database_cache

async def close_database_connections():
    """Close all database connections."""
    global _supabase_database_cache, _db_connection_pool
    
    try:
        if _supabase_database_cache:
            _supabase_database_cache = None
        
        if _db_connection_pool:
            await _db_connection_pool.close()
            _db_connection_pool = None
        
        logger.info("[DATABASE] All connections closed")
    except Exception as e:
        logger.error(f"[DATABASE] Error closing connections: {e}")

async def get_connection_pool_status():
    """Get database connection pool status."""
    try:
        connection = await get_db_connection()
        if connection and hasattr(connection.engine, 'pool'):
            pool = connection.engine.pool
            return {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalidated": pool.invalidated()
            }
    except Exception as e:
        logger.error(f"[DATABASE] Error getting pool status: {e}")
    
    return {"status": "unknown"}

# =============================================================================
# CONVERSATION CONTEXT
# =============================================================================

async def get_conversation_context(conversation_id: str, limit: int = 6) -> str:
    """Get recent conversation context for a given conversation."""
    try:
        if not conversation_id:
            return "No conversation context available."
        
        messages = await get_conversation_messages(conversation_id, limit=limit)
        if not messages:
            return "No previous conversation found."
        
        # Format messages for context
        formatted_messages = []
        for msg in messages[-limit:]:  # Get most recent messages
            role = "User" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')[:200]  # Truncate long messages
            if content:
                formatted_messages.append(f"{role}: {content}")
        
        if formatted_messages:
            return "Recent conversation:\n" + "\n".join(formatted_messages)
        else:
            return "No conversation context available."
    
    except Exception as e:
        logger.error(f"[CONVERSATION] Error getting context: {e}")
        return "Error retrieving conversation context."

async def get_recent_conversation_context() -> str:
    """Get recent conversation context for current conversation."""
    conversation_id = get_current_conversation_id()
    if not conversation_id:
        return "No active conversation found."
    
    return await get_conversation_context(conversation_id)

# =============================================================================
# EMPLOYEE ACCESS CONTROL
# =============================================================================

async def get_current_employee_id() -> Optional[str]:
    """Get current employee ID, checking context and database."""
    # First try context
    employee_id = get_current_employee_id_from_context()
    if employee_id:
        return employee_id
    
    # Then try to get from user context
    user_id = get_current_user_id()
    user_type = get_current_user_type()
    
    if user_type == "employee" and user_id:
        try:
            db = await get_sql_database()
            if db:
                query = "SELECT id FROM employees WHERE user_id = %s LIMIT 1"
                result = db.run(query, parameters=[user_id])
                if result and result.strip():
                    lines = result.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        employee_id = lines[1].strip()
                        if employee_id and employee_id != 'None':
                            return employee_id
        except Exception as e:
            logger.error(f"[EMPLOYEE] Error getting employee ID: {e}")
    
    return None

def validate_employee_access() -> tuple[bool, str]:
    """Validate that current user has employee access."""
    user_type = get_current_user_type()
    
    if user_type != "employee":
        return False, "⚠️ **Access Restricted**: This tool is only available to employees. Please contact your administrator for access."
    
    return True, ""

# =============================================================================
# DATA STRUCTURES FOR QUOTATION INTELLIGENCE
# =============================================================================

@dataclass
class ExtractedContext:
    """Structured extracted context from conversation analysis."""
    customer_info: Dict[str, Optional[str]] = field(default_factory=dict)
    vehicle_requirements: Dict[str, Optional[Union[str, List[str]]]] = field(default_factory=dict)
    purchase_preferences: Dict[str, Optional[str]] = field(default_factory=dict)
    timeline_info: Dict[str, Optional[Union[str, List[str]]]] = field(default_factory=dict)
    contact_preferences: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class FieldMappings:
    """Structured field mappings for different system integrations."""
    database_fields: Dict[str, Optional[str]] = field(default_factory=dict)
    pdf_fields: Dict[str, Optional[str]] = field(default_factory=dict)
    api_fields: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class CompletenessAssessment:
    """Assessment of information completeness for quotation generation."""
    overall_completeness: str = "low"  # high|medium|low
    quotation_ready: bool = False
    minimum_viable_info: bool = False
    risk_level: str = "high"  # low|medium|high
    business_impact: str = "gather_more"  # proceed|gather_more|escalate
    completion_percentage: int = 0

@dataclass
class MissingInfoAnalysis:
    """Analysis of missing information with business priorities."""
    critical_missing: List[str] = field(default_factory=list)
    important_missing: List[str] = field(default_factory=list)
    helpful_missing: List[str] = field(default_factory=list)
    optional_missing: List[str] = field(default_factory=list)
    inferable_from_context: List[str] = field(default_factory=list)
    alternative_sources: Dict[str, str] = field(default_factory=dict)

@dataclass
class ValidationResults:
    """Results from data validation and business rule checks."""
    data_quality: str = "poor"  # excellent|good|fair|poor
    consistency_check: bool = False
    business_rule_compliance: bool = False
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class BusinessRecommendations:
    """Business intelligence recommendations and next actions."""
    next_action: str = "gather_info"  # generate_quotation|gather_info|escalate|offer_alternatives
    priority_actions: List[str] = field(default_factory=list)
    upsell_opportunities: List[str] = field(default_factory=list)
    customer_tier_assessment: str = "basic"  # premium|standard|basic
    urgency_level: str = "low"  # high|medium|low
    sales_strategy: str = "consultative"  # consultative|transactional|relationship|educational

@dataclass
class ConfidenceScores:
    """Confidence scores for different analysis aspects."""
    extraction_confidence: float = 0.0
    completeness_confidence: float = 0.0
    business_assessment_confidence: float = 0.0

@dataclass
class ContextAnalysisResult:
    """Comprehensive result from QuotationContextIntelligence analysis."""
    extracted_context: ExtractedContext = field(default_factory=ExtractedContext)
    field_mappings: FieldMappings = field(default_factory=FieldMappings)
    completeness_assessment: CompletenessAssessment = field(default_factory=CompletenessAssessment)
    missing_info_analysis: MissingInfoAnalysis = field(default_factory=MissingInfoAnalysis)
    validation_results: ValidationResults = field(default_factory=ValidationResults)
    business_recommendations: BusinessRecommendations = field(default_factory=BusinessRecommendations)
    confidence_scores: ConfidenceScores = field(default_factory=ConfidenceScores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_quotation_ready(self) -> bool:
        """Check if analysis indicates readiness for quotation generation."""
        return (
            self.completeness_assessment.quotation_ready and
            self.completeness_assessment.overall_completeness in ["high", "medium"] and
            self.validation_results.data_quality in ["excellent", "good"] and
            len(self.missing_info_analysis.critical_missing) == 0
        )
    
    def get_next_required_info(self) -> List[str]:
        """Get prioritized list of next required information."""
        next_info = []
        if self.missing_info_analysis.critical_missing:
            next_info.extend(self.missing_info_analysis.critical_missing)
        elif self.missing_info_analysis.important_missing:
            next_info.extend(self.missing_info_analysis.important_missing[:2])  # Top 2 important items
        elif self.missing_info_analysis.helpful_missing:
            next_info.extend(self.missing_info_analysis.helpful_missing[:1])  # Top helpful item
        return next_info
    
    def get_confidence_level(self) -> str:
        """Get overall confidence level based on individual scores."""
        avg_confidence = (
            self.confidence_scores.extraction_confidence +
            self.confidence_scores.completeness_confidence +
            self.confidence_scores.business_assessment_confidence
        ) / 3
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.6:
            return "medium"
        else:
            return "low"

# =============================================================================
# COMMON LOOKUP FUNCTIONS
# =============================================================================

async def lookup_customer(customer_identifier: str) -> Optional[dict]:
    """
    Look up customer information from the database.
    
    Args:
        customer_identifier: Customer UUID, name, email, or other identifier
            
    Returns:
        Customer info dictionary if found, None otherwise
    """
    try:
        if not customer_identifier or not customer_identifier.strip():
            logger.info("[CUSTOMER_LOOKUP] Empty customer identifier provided")
            return None

        search_terms = customer_identifier.lower().strip()
        logger.info(f"[CUSTOMER_LOOKUP] Searching for: '{search_terms}'")

        # Use Supabase client instead of SQL database
        db_client = get_db_client()
        if not db_client:
            logger.error("[CUSTOMER_LOOKUP] No database connection available")
            return None
            
        # Get the actual Supabase client from the DatabaseClient wrapper
        supabase = db_client.client
        if not supabase:
            logger.error("[CUSTOMER_LOOKUP] No Supabase client available")
            return None

        # Try different search strategies using Supabase queries
        try:
            # Try exact email match first (most reliable)
            if '@' in customer_identifier:
                result = supabase.table('customers').select('*').ilike('email', customer_identifier).limit(1).execute()
                if result.data:
                    logger.info(f"[CUSTOMER_LOOKUP] ✅ Found customer by email: {result.data[0].get('name', 'Unknown')}")
                    return result.data[0]
            
            # Try exact name match
            result = supabase.table('customers').select('*').ilike('name', customer_identifier).limit(1).execute()
            if result.data:
                logger.info(f"[CUSTOMER_LOOKUP] ✅ Found customer by name: {result.data[0].get('name', 'Unknown')}")
                return result.data[0]
                
            # Try partial name match
            result = supabase.table('customers').select('*').ilike('name', f'%{customer_identifier}%').limit(1).execute()
            if result.data:
                logger.info(f"[CUSTOMER_LOOKUP] ✅ Found customer by partial name: {result.data[0].get('name', 'Unknown')}")
                return result.data[0]
                
        except Exception as query_error:
            logger.warning(f"[CUSTOMER_LOOKUP] Supabase query failed: {query_error}")

        logger.info(f"[CUSTOMER_LOOKUP] ❌ No customer found for: {customer_identifier}")
        return None
            
    except Exception as e:
        logger.error(f"[CUSTOMER_LOOKUP] Error: {e}")
        return None

async def lookup_employee_details(identifier: str) -> Optional[dict]:
    """
    Look up employee details by identifier.
    
    Returns fields needed for quotations: name, position, email, phone.
    """
    try:
        if not identifier or not identifier.strip():
            return None
            
        # Use Supabase client instead of SQL database
        db_client = get_db_client()
        if not db_client:
            logger.error("[EMPLOYEE_LOOKUP] No database connection available")
            return None
            
        # Get the actual Supabase client from the DatabaseClient wrapper
        supabase = db_client.client
        if not supabase:
            logger.error("[EMPLOYEE_LOOKUP] No Supabase client available")
            return None
        
        logger.info(f"[EMPLOYEE_LOOKUP] Searching for employee: '{identifier}'")
        
        # Try different search strategies using Supabase queries
        try:
            # Try exact ID match first
            result = supabase.table('employees').select('*').eq('id', identifier).limit(1).execute()
            if result.data:
                logger.info(f"[EMPLOYEE_LOOKUP] ✅ Found employee by ID: {result.data[0].get('name', 'Unknown')}")
                return result.data[0]
            
            # Try exact name match
            result = supabase.table('employees').select('*').ilike('name', identifier).limit(1).execute()
            if result.data:
                logger.info(f"[EMPLOYEE_LOOKUP] ✅ Found employee by name: {result.data[0].get('name', 'Unknown')}")
                return result.data[0]
                
            # Try email match
            if '@' in identifier:
                result = supabase.table('employees').select('*').ilike('email', identifier).limit(1).execute()
                if result.data:
                    logger.info(f"[EMPLOYEE_LOOKUP] ✅ Found employee by email: {result.data[0].get('name', 'Unknown')}")
                    return result.data[0]
                
        except Exception as query_error:
            logger.warning(f"[EMPLOYEE_LOOKUP] Supabase query failed: {query_error}")
        
        logger.info(f"[EMPLOYEE_LOOKUP] ❌ No employee found for: {identifier}")
        return None
            
    except Exception as e:
        logger.error(f"[EMPLOYEE_LOOKUP] Error: {e}")
        return None

# =============================================================================
# SCHEMA INFORMATION
# =============================================================================

def get_minimal_schema_info(user_type: str) -> str:
    """Get minimal schema information for SQL generation."""
    base_schema = """
Database Schema:
- customers: id, name, email, phone, address, company
- vehicles: id, make, model, year, type, price, status, color, mileage
- employees: id, name, position, email, phone, branch
- conversations: id, user_id, created_at, updated_at
- messages: id, conversation_id, role, content, created_at
"""
    
    if user_type == "employee":
        return base_schema + """
Additional employee access:
- Full access to all customer and vehicle data
- Can query sales analytics and performance metrics
"""
    else:
        return base_schema + """
Customer access:
- Limited to own conversation and message history
"""

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_toolbox():
    """Initialize the toolbox with necessary setup."""
    logger.info("[TOOLBOX] Initialized shared utility functions")

# Auto-initialize when imported
initialize_toolbox()
