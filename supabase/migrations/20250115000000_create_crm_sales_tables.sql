-- Migration: Create CRM Sales Management Tables
-- This migration creates a comprehensive sales management system with:
-- 1. Branch management
-- 2. Employee management with hierarchy
-- 3. Customer management
-- 4. Vehicle inventory
-- 5. Sales opportunities tracking
-- 6. Transaction management
-- 7. Pricing and promotions
-- 8. Activity tracking

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom ENUM types
CREATE TYPE branch_region AS ENUM ('north', 'south', 'east', 'west', 'central');
CREATE TYPE employee_position AS ENUM ('sales_agent', 'account_executive', 'manager', 'director', 'admin');
CREATE TYPE vehicle_type AS ENUM ('sedan', 'suv', 'hatchback', 'pickup', 'van', 'motorcycle', 'truck');
CREATE TYPE opportunity_stage AS ENUM ('New', 'Contacted', 'Consideration', 'Purchase Intent', 'Won', 'Lost');
CREATE TYPE lead_warmth AS ENUM ('hot', 'warm', 'cold', 'dormant');
CREATE TYPE transaction_status AS ENUM ('pending', 'processing', 'completed', 'cancelled', 'refunded');
CREATE TYPE activity_type AS ENUM ('call', 'email', 'meeting', 'demo', 'follow_up', 'proposal_sent', 'contract_signed');

-- 1. Branch Table
CREATE TABLE branches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    region branch_region NOT NULL,
    address TEXT NOT NULL,
    brand VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Employee Table
CREATE TABLE employees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    branch_id UUID REFERENCES branches(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    position employee_position NOT NULL,
    manager_ae_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(50),
    hire_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Customer Table
CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    mobile_number VARCHAR(50),
    email VARCHAR(255),
    company VARCHAR(255),
    is_for_business BOOLEAN DEFAULT false,
    address TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Vehicle Table
CREATE TABLE vehicles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 1900 AND year <= 2100),
    model VARCHAR(255) NOT NULL,
    type vehicle_type NOT NULL,
    color VARCHAR(100),
    acceleration DECIMAL(5,2), -- 0-100 km/h in seconds
    power INTEGER, -- horsepower
    torque INTEGER, -- Nm
    fuel_type VARCHAR(50),
    transmission VARCHAR(50),
    is_available BOOLEAN DEFAULT true,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Opportunities Table (Sales Opportunities/Leads)
CREATE TABLE opportunities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(id) ON DELETE CASCADE,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    opportunity_salesperson_ae_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    stage opportunity_stage DEFAULT 'New',
    warmth lead_warmth DEFAULT 'cold',
    referral_name VARCHAR(255),
    expected_close_date DATE,
    probability INTEGER CHECK (probability >= 0 AND probability <= 100),
    estimated_value DECIMAL(12,2),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Transaction Table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE CASCADE,
    opportunity_salesperson_ae_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    assignee_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    status transaction_status DEFAULT 'pending',
    total_amount DECIMAL(12,2),
    payment_method VARCHAR(100),
    transaction_reference VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Pricing Table
CREATE TABLE pricing (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE CASCADE,
    availability_start_date DATE NOT NULL,
    availability_end_date DATE,
    warranty VARCHAR(255),
    insurance DECIMAL(12,2),
    lto DECIMAL(12,2), -- Land Transport Office fees
    price_discount DECIMAL(12,2) DEFAULT 0,
    insurance_discount DECIMAL(12,2) DEFAULT 0,
    lto_discount DECIMAL(12,2) DEFAULT 0,
    add_ons JSONB DEFAULT '[]'::jsonb, -- Array of add-on items and prices
    warranty_promo TEXT,
    base_price DECIMAL(12,2) NOT NULL,
    final_price DECIMAL(12,2) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_availability_dates CHECK (availability_end_date IS NULL OR availability_end_date >= availability_start_date)
);

-- 8. Activity Table
CREATE TABLE activities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    opportunity_salesperson_ae_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    customer_id UUID REFERENCES customers(id) ON DELETE CASCADE,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    activity_type activity_type NOT NULL,
    subject VARCHAR(255),
    description TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    is_completed BOOLEAN DEFAULT false,
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_branches_region ON branches(region);
CREATE INDEX idx_branches_brand ON branches(brand);

CREATE INDEX idx_employees_branch_id ON employees(branch_id);
CREATE INDEX idx_employees_position ON employees(position);
CREATE INDEX idx_employees_manager_ae_id ON employees(manager_ae_id);
CREATE INDEX idx_employees_is_active ON employees(is_active);

CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_is_for_business ON customers(is_for_business);
CREATE INDEX idx_customers_name ON customers(name);

CREATE INDEX idx_vehicles_brand ON vehicles(brand);
CREATE INDEX idx_vehicles_type ON vehicles(type);
CREATE INDEX idx_vehicles_year ON vehicles(year);
CREATE INDEX idx_vehicles_is_available ON vehicles(is_available);

CREATE INDEX idx_opportunities_customer_id ON opportunities(customer_id);
CREATE INDEX idx_opportunities_vehicle_id ON opportunities(vehicle_id);
CREATE INDEX idx_opportunities_salesperson_id ON opportunities(opportunity_salesperson_ae_id);
CREATE INDEX idx_opportunities_stage ON opportunities(stage);
CREATE INDEX idx_opportunities_warmth ON opportunities(warmth);
CREATE INDEX idx_opportunities_created_date ON opportunities(created_date);

CREATE INDEX idx_transactions_opportunity_id ON transactions(opportunity_id);
CREATE INDEX idx_transactions_salesperson_id ON transactions(opportunity_salesperson_ae_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_created_date ON transactions(created_date);

CREATE INDEX idx_pricing_vehicle_id ON pricing(vehicle_id);
CREATE INDEX idx_pricing_availability_dates ON pricing(availability_start_date, availability_end_date);
CREATE INDEX idx_pricing_is_active ON pricing(is_active);

CREATE INDEX idx_activities_salesperson_id ON activities(opportunity_salesperson_ae_id);
CREATE INDEX idx_activities_customer_id ON activities(customer_id);
CREATE INDEX idx_activities_opportunity_id ON activities(opportunity_id);
CREATE INDEX idx_activities_activity_type ON activities(activity_type);
CREATE INDEX idx_activities_created_date ON activities(created_date);
CREATE INDEX idx_activities_is_completed ON activities(is_completed);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_branches_updated_at
    BEFORE UPDATE ON branches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_employees_updated_at
    BEFORE UPDATE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vehicles_updated_at
    BEFORE UPDATE ON vehicles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_opportunities_updated_at
    BEFORE UPDATE ON opportunities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pricing_updated_at
    BEFORE UPDATE ON pricing
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_activities_updated_at
    BEFORE UPDATE ON activities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW sales_pipeline AS
SELECT 
    o.id as opportunity_id,
    c.name as customer_name,
    c.company,
    v.brand || ' ' || v.model || ' (' || v.year || ')' as vehicle,
    e.name as salesperson_name,
    b.name as branch_name,
    o.stage,
    o.warmth,
    o.probability,
    o.estimated_value,
    o.expected_close_date,
    o.created_date as opportunity_created
FROM opportunities o
JOIN customers c ON o.customer_id = c.id
LEFT JOIN vehicles v ON o.vehicle_id = v.id
LEFT JOIN employees e ON o.opportunity_salesperson_ae_id = e.id
LEFT JOIN branches b ON e.branch_id = b.id
WHERE o.stage NOT IN ('Won', 'Lost')
ORDER BY o.created_date DESC;

CREATE VIEW employee_hierarchy AS
WITH RECURSIVE emp_hierarchy AS (
    -- Base case: top-level employees (no manager)
    SELECT 
        id,
        name,
        position,
        branch_id,
        manager_ae_id,
        0 as level,
        name::TEXT as hierarchy_path
    FROM employees 
    WHERE manager_ae_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT 
        e.id,
        e.name,
        e.position,
        e.branch_id,
        e.manager_ae_id,
        eh.level + 1,
        eh.hierarchy_path || ' -> ' || e.name
    FROM employees e
    JOIN emp_hierarchy eh ON e.manager_ae_id = eh.id
)
SELECT 
    eh.*,
    b.name as branch_name,
    b.region
FROM emp_hierarchy eh
LEFT JOIN branches b ON eh.branch_id = b.id
ORDER BY eh.level, eh.hierarchy_path;

-- Add comments to tables
COMMENT ON TABLE branches IS 'Company branches/locations for sales operations';
COMMENT ON TABLE employees IS 'Employee information with hierarchical structure';
COMMENT ON TABLE customers IS 'Customer contact and business information';
COMMENT ON TABLE vehicles IS 'Vehicle inventory with specifications';
COMMENT ON TABLE opportunities IS 'Sales opportunities and lead tracking';
COMMENT ON TABLE transactions IS 'Financial transactions for sales';
COMMENT ON TABLE pricing IS 'Vehicle pricing with discounts and promotions';
COMMENT ON TABLE activities IS 'Sales activities and customer interactions'; 