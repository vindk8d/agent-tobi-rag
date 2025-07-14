# CRM Sales Management System

## Overview

The CRM Sales Management System is a comprehensive solution designed to track and manage automotive sales operations. It integrates with the existing RAG-Tobi system to provide intelligent sales assistance and customer management.

## Database Schema

The system consists of 8 core tables that handle the complete sales lifecycle:

### 1. **Branches** 
Company locations and regional operations
- **Purpose**: Manage multiple sales locations across different regions
- **Key Fields**: name, region, address, brand
- **Relationships**: One-to-Many with Employees

### 2. **Employees**
Staff management with hierarchical structure
- **Purpose**: Track sales team with manager relationships
- **Key Fields**: name, position, manager_ae_id (self-referencing)
- **Positions**: sales_agent, account_executive, manager, director, admin
- **Relationships**: Many-to-One with Branches, Self-referencing for hierarchy

### 3. **Customers**
Customer contact and business information
- **Purpose**: Manage both individual and business customers
- **Key Fields**: name, contact info, company, is_for_business
- **Relationships**: One-to-Many with Opportunities

### 4. **Vehicles**
Vehicle inventory with detailed specifications
- **Purpose**: Manage available vehicles with technical details
- **Key Fields**: brand, model, year, type, specifications (power, torque, etc.)
- **Vehicle Types**: sedan, suv, hatchback, pickup, van, motorcycle, truck
- **Relationships**: One-to-Many with Opportunities and Pricing

### 5. **Opportunities**
Sales opportunities and lead tracking
- **Purpose**: Track sales pipeline from lead to closure
- **Key Fields**: customer_id, vehicle_id, stage, warmth, probability, estimated_value
- **Stages**: New → Contacted → Consideration → Purchase Intent → Won/Lost
- **Lead Warmth**: hot, warm, cold, dormant
- **Relationships**: Many-to-One with Customers, Vehicles, and Employees

### 6. **Transactions**
Financial transaction management
- **Purpose**: Handle payment processing and transaction records
- **Key Fields**: opportunity_id, total_amount, status, payment_method
- **Status Types**: pending, processing, completed, cancelled, refunded
- **Relationships**: Many-to-One with Opportunities and Employees

### 7. **Pricing**
Dynamic pricing with promotions and discounts
- **Purpose**: Manage vehicle pricing with time-based availability
- **Key Fields**: vehicle_id, base_price, discounts, add_ons, availability_dates
- **Features**: Price discounts, insurance discounts, LTO discounts, add-ons (JSONB)
- **Relationships**: Many-to-One with Vehicles

### 8. **Activities**
Sales activities and customer interactions
- **Purpose**: Track all customer touchpoints and follow-ups
- **Key Fields**: customer_id, activity_type, subject, description, completion status
- **Activity Types**: call, email, meeting, demo, follow_up, proposal_sent, contract_signed
- **Relationships**: Many-to-One with Customers, Opportunities, and Employees

## Key Features

### 1. **Hierarchical Employee Management**
- Self-referencing employee table for management structure
- Position-based access control
- Manager assignment and reporting

### 2. **Sales Pipeline Tracking**
- Complete opportunity lifecycle management
- Probability and value estimation
- Stage-based workflow
- Lead warmth assessment (hot, warm, cold, dormant)

### 3. **Dynamic Pricing System**
- Time-based pricing availability
- Multiple discount types (price, insurance, LTO)
- JSONB add-ons for flexible product offerings
- Promotion management

### 4. **Customer Segmentation**
- Individual vs. business customer tracking
- Contact information management
- Customer notes and preferences

### 5. **Activity Management**
- Complete interaction history
- Task scheduling and follow-ups
- Priority-based activity management

### 6. **Performance Analytics** (via Views)
- `sales_pipeline`: Active opportunities overview
- `employee_hierarchy`: Organizational structure visualization

## Migration Files

### `20250115000000_create_crm_sales_tables.sql`
Creates the complete CRM schema including:
- All 8 core tables with proper relationships
- Custom ENUM types for data validation
- Comprehensive indexing for performance
- Triggers for automatic timestamp updates
- Useful views for common queries

### `20250115000001_insert_sample_crm_data.sql`
Populates the system with realistic sample data:
- 5 branches across different regions
- Employee hierarchy with managers and sales staff
- Individual and business customers
- Vehicle inventory with various models
- Active sales opportunities in different stages
- Pricing information with discounts
- Customer activities and interactions

## Usage Examples

### Query Active Sales Pipeline
```sql
SELECT * FROM sales_pipeline 
WHERE stage IN ('Contacted', 'Consideration', 'Purchase Intent')
ORDER BY estimated_value DESC;
```

### Query Hot Leads
```sql
SELECT * FROM sales_pipeline 
WHERE warmth = 'hot' AND stage NOT IN ('Won', 'Lost')
ORDER BY estimated_value DESC;
```

### View Employee Hierarchy
```sql
SELECT * FROM employee_hierarchy 
WHERE branch_name = 'Downtown Branch'
ORDER BY level, name;
```

### Find Available Vehicles with Pricing
```sql
SELECT 
    v.brand, v.model, v.year, v.color,
    p.base_price, p.final_price, p.price_discount
FROM vehicles v
JOIN pricing p ON v.id = p.vehicle_id
WHERE v.is_available = true 
  AND p.is_active = true
  AND p.availability_start_date <= CURRENT_DATE
  AND (p.availability_end_date IS NULL OR p.availability_end_date >= CURRENT_DATE);
```

### Customer Activity Summary
```sql
SELECT 
    c.name as customer_name,
    COUNT(a.id) as total_activities,
    COUNT(CASE WHEN a.is_completed = false THEN 1 END) as pending_activities,
    MAX(a.created_date) as last_activity
FROM customers c
LEFT JOIN activities a ON c.id = a.customer_id
GROUP BY c.id, c.name
ORDER BY last_activity DESC;
```

### Lead Warmth Analysis
```sql
SELECT 
    warmth,
    COUNT(*) as lead_count,
    AVG(probability) as avg_probability,
    SUM(estimated_value) as total_pipeline_value
FROM opportunities 
WHERE stage NOT IN ('Won', 'Lost')
GROUP BY warmth
ORDER BY avg_probability DESC;
```

## Integration with RAG-Tobi

The CRM system is designed to integrate seamlessly with the existing RAG-Tobi architecture:

1. **Document Processing**: Sales documents, contracts, and customer communications can be processed through the existing document pipeline
2. **Intelligent Assistance**: The RAG system can provide contextual information about vehicles, pricing, and customer history
3. **Conversation Management**: Customer interactions can be tracked through both the CRM activity system and the conversation tables
4. **Analytics**: Combined analytics across document insights and sales performance

## Performance Considerations

### Indexing Strategy
- Primary keys on all tables (UUID)
- Foreign key indexes for relationship queries
- Composite indexes on frequently queried date ranges
- Full-text search indexes on customer names and vehicle models

### Optimization Features
- Automatic timestamp updates via triggers
- Enum types for data validation and performance
- JSONB for flexible add-ons storage
- Views for complex but common queries

## Security & Access Control

The system is designed with Row Level Security (RLS) in mind:
- Branch-based data isolation
- Employee role-based permissions
- Customer data privacy protection
- Audit trails through created_at/updated_at timestamps

## Next Steps

1. **Apply the migrations** to your Supabase database
2. **Configure Row Level Security** policies based on your organization's needs
3. **Set up API endpoints** for CRUD operations
4. **Integrate with the frontend** for user interface
5. **Connect with RAG-Tobi** for intelligent sales assistance

## API Integration Endpoints (Suggested)

```
GET /api/crm/branches - List all branches
GET /api/crm/employees - List employees with hierarchy
GET /api/crm/customers - Customer management
GET /api/crm/vehicles - Vehicle inventory
GET /api/crm/opportunities - Sales pipeline
GET /api/crm/activities - Customer activities
GET /api/crm/pricing - Current pricing information
POST /api/crm/opportunities - Create new opportunity
PUT /api/crm/opportunities/:id/stage - Update opportunity stage
``` 