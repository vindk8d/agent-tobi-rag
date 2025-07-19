# SQL Generation Process Documentation

## Overview

This document explains the complete SQL generation process in the Tobi RAG Agent system, from user query reception to database query execution and response delivery.

## Architecture Flow

```
User Query → RAG Agent → Tool Selection → SQL Generation → Database Execution → Response
     ↓            ↓            ↓              ↓               ↓              ↓
LangGraph   Memory Prep   query_crm_data   get_table_info   PostgreSQL   Memory Update
```

## Step-by-Step Process

### 1. User Query Reception
- User submits natural language question: *"How many Toyota vehicles do we have in stock?"*
- Query received through the RAG agent's unified interface
- LangGraph workflow initiates with automatic state persistence

### 2. Agent Workflow (LangGraph)
The agent follows this three-node workflow:
```
Memory Preparation → Agent Processing → Memory Update
```

**Memory Preparation Node:**
- Retrieves conversation context from checkpointed state
- Loads relevant long-term context (max 3 items)
- Stores incoming user message to database
- Preserves conversation continuity

**Agent Processing Node:**
- Unified tool-calling and execution
- Model determines appropriate tool usage
- Handles multiple tool iterations (max 3)
- Manages tool call chains

**Memory Update Node:**
- Stores agent responses to database
- Updates long-term memory with insights
- Maintains conversation state persistence

### 3. Tool Selection & Invocation

**LLM Decision Process:**
- GPT-4o-mini analyzes user query
- Identifies business/CRM-related intent
- Selects `query_crm_data` tool from available tools:
  - `semantic_search` (document retrieval)
  - `build_context` (document processing)
  - `format_sources` (citation formatting)
  - `query_crm_data` (database queries)

**Tool Invocation:**
```python
tool_call = {
    "name": "query_crm_data",
    "args": {"question": "How many Toyota vehicles do we have in stock?"},
    "id": "call_xyz123"
}
```

### 4. Database Schema Filtering (`get_table_info`)

**CRM-Specific Table Filtering:**
The system uses **filtered schema access** - only CRM business tables are exposed:

**Included Tables (Business Data):**
```sql
-- Core Business Tables
branches          -- Regional dealership locations
employees         -- Staff information and hierarchy
customers         -- Customer contact and company details
vehicles          -- Inventory specifications and availability
opportunities     -- Sales pipeline and lead tracking  
transactions      -- Sales records and revenue data
pricing           -- Vehicle prices, discounts, fees
activities        -- Customer interactions and follow-ups
```

**Excluded Tables (System Data):**
```sql
-- RAG System Infrastructure
data_sources, document_chunks, embeddings
conversations, messages, query_logs
system_metrics, response_feedback
proactive_suggestions, source_conflicts
```

**Why Filtering Matters:**
- **Security**: Prevents exposure of system internals
- **Performance**: Reduces prompt size and complexity
- **Focus**: Keeps AI focused on business queries only
- **Maintenance**: Clear separation of concerns

### 5. SQL Generation Process

**Prompt Construction:**
The system builds a comprehensive prompt containing:

```python
prompt_components = {
    "table_schema": get_table_info(),  # CRM tables only
    "user_question": user_query,
    "instructions": sql_generation_guidelines,
    "examples": few_shot_examples
}
```

**Key Prompt Elements:**

**A) Schema Information:**
- Table structures with column names and types
- Relationships between tables
- Enum values and constraints
- Sample data patterns

**B) Generation Guidelines:**
- PostgreSQL-specific syntax requirements
- JOIN optimization strategies  
- WHERE clause best practices
- Security constraints (no DDL, DML restrictions)

**C) Few-Shot Examples:**
- Common query patterns
- Business logic examples
- Complex JOIN scenarios
- Aggregation patterns

### 6. SQL Query Generation

**LangChain SQL Agent Process:**
1. **Parse Intent**: Analyze business question
2. **Select Tables**: Determine relevant tables from schema
3. **Build Query**: Construct optimized PostgreSQL query
4. **Validate**: Check syntax and security constraints

**Example Generation:**
```sql
-- User: "How many Toyota vehicles do we have in stock?"
-- Generated Query:
SELECT COUNT(*) as toyota_count 
FROM vehicles 
WHERE brand = 'Toyota' 
  AND status = 'available';
```

### 7. Security & Validation

**Query Restrictions:**
- **READ-ONLY**: Only SELECT statements allowed
- **Table Filtering**: Access limited to CRM tables only
- **SQL Injection**: Parameterized queries and input sanitization
- **Resource Limits**: Query timeout and result size limits

**Validation Checks:**
- Syntax validation before execution
- Table access permission verification  
- Query complexity analysis
- Result set size monitoring

### 8. Database Execution

**PostgreSQL Connection:**
- Async SQLDatabase connection via SQLAlchemy
- Connection pooling for performance
- Transaction isolation for consistency
- Error handling with graceful fallbacks

**Query Execution:**
```python
# Executed query with safety checks
result = await db.arun(generated_sql_query)
```

### 9. Response Formation

**Result Processing:**
- Query results formatted into natural language
- Business context added for clarity
- Data visualization suggestions when appropriate
- Source attribution for transparency

**Response Structure:**
```python
{
    "answer": "You have 45 Toyota vehicles in stock.",
    "sql_query": "SELECT COUNT(*) FROM vehicles WHERE brand = 'Toyota'...",
    "data": [{"toyota_count": 45}],
    "confidence": 0.95
}
```

## Database Schema Details

### Core CRM Tables

**Vehicles Table:**
- **Brands**: Toyota, Honda, Ford, Nissan, BMW, Mercedes, Audi
- **Types**: sedan, suv, hatchback, pickup, van, motorcycle, truck  
- **Attributes**: model, year, color, power, fuel_type, transmission, status

**Employees Table:**
- **Positions**: sales_agent, account_executive, manager, director, admin
- **Hierarchy**: Reports-to relationships and branch assignments

**Opportunities Table:**
- **Pipeline Stages**: New, Contacted, Consideration, Purchase Intent, Won, Lost
- **Lead Warmth**: hot, warm, cold, dormant

**Pricing Table:**
- Base prices, final prices, discount amounts
- Insurance and LTO fees, warranty information

## Common Query Patterns

### Inventory Queries
```sql
-- Vehicle availability by brand
SELECT brand, COUNT(*) FROM vehicles WHERE status = 'available' GROUP BY brand;

-- Specific model search
SELECT * FROM vehicles WHERE model = 'Camry' AND status = 'available';
```

### Sales Performance
```sql
-- Pipeline status summary
SELECT stage, COUNT(*) FROM opportunities GROUP BY stage;

-- Employee performance
SELECT e.name, COUNT(o.id) as opportunities 
FROM employees e LEFT JOIN opportunities o ON e.id = o.assigned_to 
GROUP BY e.name;
```

### Customer Analytics
```sql
-- Customer segments
SELECT customer_type, COUNT(*) FROM customers GROUP BY customer_type;

-- Recent activities
SELECT activity_type, COUNT(*) FROM activities 
WHERE created_at > NOW() - INTERVAL '30 days' 
GROUP BY activity_type;
```

## Error Handling

**Common Error Scenarios:**
1. **Invalid SQL Syntax**: Return error with suggestion
2. **Table Access Denied**: Redirect to available tables
3. **Query Timeout**: Suggest query optimization
4. **No Results Found**: Confirm query logic with user
5. **Database Connection**: Graceful fallback with retry logic

## Performance Considerations

**Optimization Strategies:**
- Query result caching for repeated patterns
- Connection pooling for database efficiency
- Prompt optimization to reduce token usage
- Table schema caching to minimize metadata calls

**Monitoring:**
- Query execution time tracking
- Token usage monitoring  
- Error rate analysis
- Performance bottleneck identification

## Integration Points

**RAG Agent Integration:**
- Seamless tool calling workflow
- Context-aware query generation
- Memory persistence across conversations
- Multi-modal response formatting

**LangGraph Workflow:**
- Automatic state checkpointing
- Tool call orchestration
- Error recovery and retry logic
- Conversation context management

---

## Quick Reference

**Key Components:**
- **Agent**: `UnifiedToolCallingRAGAgent`
- **Tool**: `query_crm_data` 
- **Database**: PostgreSQL with CRM schema
- **LLM**: GPT-4o-mini for SQL generation
- **Framework**: LangChain + LangGraph

**Available Business Data:**
- 8 CRM tables with business data
- Vehicles, customers, employees, pricing
- Sales pipeline and performance metrics
- Customer interactions and activities 