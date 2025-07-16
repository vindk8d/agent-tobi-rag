-- Migration: Add Comprehensive CRM Test Data
-- This migration adds realistic test data for evaluating natural language queries
-- against the CRM system

-- Insert test branches in different regions
INSERT INTO branches (id, name, region, address, brand, created_at, updated_at) VALUES
(uuid_generate_v4(), 'Downtown Toyota', 'central', '123 Main Street, City Center', 'Toyota', NOW(), NOW()),
(uuid_generate_v4(), 'North Honda Center', 'north', '456 Oak Avenue, North District', 'Honda', NOW(), NOW()),
(uuid_generate_v4(), 'South Ford Plaza', 'south', '789 Pine Road, South Side', 'Ford', NOW(), NOW()),
(uuid_generate_v4(), 'East Hyundai Hub', 'east', '321 Elm Street, East End', 'Hyundai', NOW(), NOW()),
(uuid_generate_v4(), 'West Nissan World', 'west', '654 Maple Drive, West Side', 'Nissan', NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Get branch IDs for foreign key references
WITH branch_refs AS (
    SELECT id, name FROM branches WHERE name IN ('Downtown Toyota', 'North Honda Center', 'South Ford Plaza', 'East Hyundai Hub', 'West Nissan World')
)
-- Insert test employees with hierarchical structure
INSERT INTO employees (id, branch_id, name, position, manager_ae_id, email, phone, hire_date, is_active, created_at, updated_at)
SELECT
    uuid_generate_v4(),
    b.id,
    emp_data.name,
    emp_data.position::employee_position,
    NULL,
    emp_data.email,
    emp_data.phone,
    emp_data.hire_date,
    true,
    NOW(),
    NOW()
FROM (
    SELECT 'Downtown Toyota' as branch_name, 'Michael Johnson' as name, 'director' as position, 'michael.johnson@company.com' as email, '555-1001' as phone, '2020-01-15'::date as hire_date
    UNION ALL
    SELECT 'North Honda Center', 'Sarah Wilson', 'manager', 'sarah.wilson@company.com', '555-1002', '2020-03-20'::date
    UNION ALL
    SELECT 'South Ford Plaza', 'David Brown', 'manager', 'david.brown@company.com', '555-1003', '2020-05-10'::date
    UNION ALL
    SELECT 'East Hyundai Hub', 'Lisa Chen', 'manager', 'lisa.chen@company.com', '555-1004', '2020-07-25'::date
    UNION ALL
    SELECT 'West Nissan World', 'Robert Davis', 'manager', 'robert.davis@company.com', '555-1005', '2020-09-30'::date
    UNION ALL
    SELECT 'Downtown Toyota', 'Emily Rodriguez', 'sales_agent', 'emily.rodriguez@company.com', '555-2001', '2021-02-15'::date
    UNION ALL
    SELECT 'Downtown Toyota', 'James Anderson', 'sales_agent', 'james.anderson@company.com', '555-2002', '2021-04-20'::date
    UNION ALL
    SELECT 'North Honda Center', 'Jennifer Taylor', 'sales_agent', 'jennifer.taylor@company.com', '555-2003', '2021-06-10'::date
    UNION ALL
    SELECT 'South Ford Plaza', 'Christopher Lee', 'sales_agent', 'christopher.lee@company.com', '555-2004', '2021-08-25'::date
    UNION ALL
    SELECT 'East Hyundai Hub', 'Amanda White', 'account_executive', 'amanda.white@company.com', '555-3001', '2021-10-30'::date
    UNION ALL
    SELECT 'West Nissan World', 'Kevin Martinez', 'account_executive', 'kevin.martinez@company.com', '555-3002', '2022-01-15'::date
) emp_data
JOIN branch_refs b ON b.name = emp_data.branch_name
ON CONFLICT (id) DO NOTHING;

-- Insert test customers with variety of business and personal accounts
INSERT INTO customers (id, name, phone, mobile_number, email, company, is_for_business, address, notes, created_at, updated_at) VALUES
(uuid_generate_v4(), 'Alice Johnson', '555-1111', '555-1234', 'alice.johnson@email.com', 'Tech Solutions Inc', true, '100 Business Park, Suite 200', 'Corporate fleet manager', NOW(), NOW()),
(uuid_generate_v4(), 'Bob Smith', '555-2222', '555-5678', 'bob.smith@personal.com', NULL, false, '123 Residential St', 'First-time buyer', NOW(), NOW()),
(uuid_generate_v4(), 'Carol Thompson', '555-3333', '555-9999', 'carol.thompson@logistics.com', 'Global Logistics Corp', true, '456 Industrial Ave', 'Looking for delivery vehicles', NOW(), NOW()),
(uuid_generate_v4(), 'Daniel Wilson', '555-4444', '555-1357', 'daniel.wilson@gmail.com', NULL, false, '789 Suburban Lane', 'Repeat customer', NOW(), NOW()),
(uuid_generate_v4(), 'Eva Martinez', '555-5555', '555-2468', 'eva.martinez@startup.com', 'StartUp Dynamics', true, '321 Innovation Drive', 'Young company, price-sensitive', NOW(), NOW()),
(uuid_generate_v4(), 'Frank Davis', '555-6666', '555-3579', 'frank.davis@yahoo.com', NULL, false, '654 Family Circle', 'Large family, needs SUV', NOW(), NOW()),
(uuid_generate_v4(), 'Grace Lee', '555-7777', '555-4680', 'grace.lee@consulting.com', 'Strategic Consulting LLC', true, '987 Executive Plaza', 'Professional services firm', NOW(), NOW()),
(uuid_generate_v4(), 'Henry Clark', '555-8888', '555-5791', 'henry.clark@outlook.com', NULL, false, '147 Retirement Village', 'Senior citizen, simple needs', NOW(), NOW()),
(uuid_generate_v4(), 'Irene Rodriguez', '555-9999', '555-6802', 'irene.rodriguez@medical.com', 'HealthCare Partners', true, '258 Medical Center Dr', 'Medical practice fleet', NOW(), NOW()),
(uuid_generate_v4(), 'Jack Thompson', '555-0000', '555-7913', 'jack.thompson@construction.com', 'BuildRight Construction', true, '369 Industrial Blvd', 'Construction company, needs trucks', NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert test vehicles with variety of types, brands, and availability
INSERT INTO vehicles (id, brand, year, model, type, color, acceleration, power, torque, fuel_type, transmission, is_available, stock_quantity, created_at, updated_at) VALUES
(uuid_generate_v4(), 'Toyota', 2024, 'Camry', 'sedan', 'Midnight Blue', 8.2, 203, 184, 'Gasoline', 'Automatic', true, 5, NOW(), NOW()),
(uuid_generate_v4(), 'Toyota', 2024, 'Highlander', 'suv', 'Silver Metallic', 8.5, 295, 263, 'Gasoline', 'Automatic', true, 3, NOW(), NOW()),
(uuid_generate_v4(), 'Honda', 2024, 'Civic', 'sedan', 'Crystal Black', 7.8, 180, 177, 'Gasoline', 'Manual', true, 8, NOW(), NOW()),
(uuid_generate_v4(), 'Honda', 2024, 'CR-V', 'suv', 'Red', 9.1, 190, 179, 'Gasoline', 'Automatic', true, 4, NOW(), NOW()),
(uuid_generate_v4(), 'Ford', 2024, 'F-150', 'pickup', 'Oxford White', 6.5, 400, 500, 'Gasoline', 'Automatic', true, 2, NOW(), NOW()),
(uuid_generate_v4(), 'Ford', 2023, 'Explorer', 'suv', 'Carbonized Gray', 7.9, 300, 310, 'Gasoline', 'Automatic', false, 0, NOW(), NOW()),
(uuid_generate_v4(), 'Hyundai', 2024, 'Elantra', 'sedan', 'Phantom Black', 8.7, 147, 132, 'Gasoline', 'Automatic', true, 6, NOW(), NOW()),
(uuid_generate_v4(), 'Hyundai', 2024, 'Tucson', 'suv', 'Blue', 9.3, 187, 178, 'Gasoline', 'Automatic', true, 3, NOW(), NOW()),
(uuid_generate_v4(), 'Nissan', 2024, 'Altima', 'sedan', 'Pearl White', 8.9, 188, 180, 'Gasoline', 'Automatic', true, 4, NOW(), NOW()),
(uuid_generate_v4(), 'Nissan', 2024, 'Rogue', 'suv', 'Magnetic Black', 9.2, 181, 181, 'Gasoline', 'Automatic', true, 5, NOW(), NOW()),
(uuid_generate_v4(), 'Toyota', 2024, 'Prius', 'hatchback', 'Hypersonic Red', 10.1, 121, 105, 'Hybrid', 'Automatic', true, 3, NOW(), NOW()),
(uuid_generate_v4(), 'Honda', 2024, 'Pilot', 'suv', 'Modern Steel', 8.8, 280, 262, 'Gasoline', 'Automatic', true, 2, NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert pricing data for vehicles
WITH vehicle_refs AS (
    SELECT v.id, v.brand, v.model, v.year 
    FROM vehicles v
    WHERE v.brand IN ('Toyota', 'Honda', 'Ford', 'Hyundai', 'Nissan')
)
INSERT INTO pricing (id, vehicle_id, availability_start_date, availability_end_date, warranty, insurance, lto, price_discount, base_price, final_price, is_active, created_at, updated_at)
SELECT
    uuid_generate_v4(),
    v.id,
    CURRENT_DATE - INTERVAL '30 days',
    CURRENT_DATE + INTERVAL '90 days',
    CASE v.brand
        WHEN 'Toyota' THEN '3 years/36,000 miles'
        WHEN 'Honda' THEN '3 years/36,000 miles'
        WHEN 'Ford' THEN '3 years/36,000 miles'
        WHEN 'Hyundai' THEN '5 years/60,000 miles'
        WHEN 'Nissan' THEN '3 years/36,000 miles'
    END,
    CASE v.model
        WHEN 'Camry' THEN 1200.00
        WHEN 'Civic' THEN 1000.00
        WHEN 'F-150' THEN 1500.00
        WHEN 'Elantra' THEN 950.00
        WHEN 'Altima' THEN 1100.00
        ELSE 1200.00
    END,
    500.00, -- LTO fees
    CASE v.model
        WHEN 'Camry' THEN 1000.00
        WHEN 'Highlander' THEN 2000.00
        WHEN 'Civic' THEN 500.00
        WHEN 'CR-V' THEN 1500.00
        WHEN 'F-150' THEN 3000.00
        WHEN 'Explorer' THEN 2500.00
        WHEN 'Elantra' THEN 800.00
        WHEN 'Tucson' THEN 1200.00
        WHEN 'Altima' THEN 900.00
        WHEN 'Rogue' THEN 1300.00
        WHEN 'Prius' THEN 1100.00
        WHEN 'Pilot' THEN 1800.00
        ELSE 1000.00
    END,
    CASE v.model
        WHEN 'Camry' THEN 25000.00
        WHEN 'Highlander' THEN 38000.00
        WHEN 'Civic' THEN 22000.00
        WHEN 'CR-V' THEN 28000.00
        WHEN 'F-150' THEN 35000.00
        WHEN 'Explorer' THEN 33000.00
        WHEN 'Elantra' THEN 20000.00
        WHEN 'Tucson' THEN 26000.00
        WHEN 'Altima' THEN 24000.00
        WHEN 'Rogue' THEN 27000.00
        WHEN 'Prius' THEN 27000.00
        WHEN 'Pilot' THEN 36000.00
        ELSE 25000.00
    END,
    CASE v.model
        WHEN 'Camry' THEN 24000.00
        WHEN 'Highlander' THEN 36000.00
        WHEN 'Civic' THEN 21500.00
        WHEN 'CR-V' THEN 26500.00
        WHEN 'F-150' THEN 32000.00
        WHEN 'Explorer' THEN 30500.00
        WHEN 'Elantra' THEN 19200.00
        WHEN 'Tucson' THEN 24800.00
        WHEN 'Altima' THEN 23100.00
        WHEN 'Rogue' THEN 25700.00
        WHEN 'Prius' THEN 25900.00
        WHEN 'Pilot' THEN 34200.00
        ELSE 24000.00
    END,
    true,
    NOW(),
    NOW()
FROM vehicle_refs v
ON CONFLICT (id) DO NOTHING;

-- Insert sales opportunities
WITH customer_refs AS (
    SELECT id, name FROM customers LIMIT 10
),
vehicle_refs AS (
    SELECT id, brand, model FROM vehicles WHERE is_available = true LIMIT 8
),
employee_refs AS (
    SELECT id, name FROM employees WHERE position IN ('sales_agent', 'account_executive') LIMIT 8
)
INSERT INTO opportunities (id, customer_id, vehicle_id, opportunity_salesperson_ae_id, created_date, stage, warmth, expected_close_date, probability, estimated_value, notes, created_at, updated_at)
SELECT
    uuid_generate_v4(),
    c.id,
    v.id,
    e.id,
    NOW() - INTERVAL '30 days' + (RANDOM() * INTERVAL '25 days'),
    stages.stage::opportunity_stage,
    warmth.warmth::lead_warmth,
    CURRENT_DATE + INTERVAL '10 days' + (RANDOM() * INTERVAL '50 days'),
    stages.probability,
    CASE v.model
        WHEN 'Camry' THEN 24000.00
        WHEN 'Highlander' THEN 36000.00
        WHEN 'Civic' THEN 21500.00
        WHEN 'CR-V' THEN 26500.00
        WHEN 'F-150' THEN 32000.00
        WHEN 'Elantra' THEN 19200.00
        WHEN 'Tucson' THEN 24800.00
        WHEN 'Altima' THEN 23100.00
        ELSE 25000.00
    END,
    'Generated test opportunity for ' || c.name || ' interested in ' || v.brand || ' ' || v.model,
    NOW(),
    NOW()
FROM customer_refs c
CROSS JOIN vehicle_refs v
CROSS JOIN employee_refs e
CROSS JOIN (
    SELECT 'New' as stage, 10 as probability
    UNION ALL SELECT 'Contacted', 25
    UNION ALL SELECT 'Consideration', 50
    UNION ALL SELECT 'Purchase Intent', 75
    UNION ALL SELECT 'Won', 100
    UNION ALL SELECT 'Lost', 0
) stages
CROSS JOIN (
    SELECT 'hot' as warmth
    UNION ALL SELECT 'warm'
    UNION ALL SELECT 'cold'
) warmth
WHERE RANDOM() < 0.15  -- Only create ~15% of all possible combinations
ON CONFLICT (id) DO NOTHING;

-- Insert some transactions for completed opportunities
WITH won_opportunities AS (
    SELECT o.id, o.customer_id, o.vehicle_id, o.opportunity_salesperson_ae_id, o.estimated_value
    FROM opportunities o
    WHERE o.stage = 'Won'
    LIMIT 10
),
random_employees AS (
    SELECT id FROM employees WHERE position IN ('sales_agent', 'account_executive') ORDER BY RANDOM() LIMIT 5
)
INSERT INTO transactions (id, opportunity_id, opportunity_salesperson_ae_id, assignee_id, created_date, end_date, status, total_amount, payment_method, transaction_reference, notes, created_at, updated_at)
SELECT
    uuid_generate_v4(),
    o.id,
    o.opportunity_salesperson_ae_id,
    e.id,
    NOW() - INTERVAL '20 days' + (RANDOM() * INTERVAL '15 days'),
    NOW() - INTERVAL '5 days' + (RANDOM() * INTERVAL '10 days'),
    'completed'::transaction_status,
    o.estimated_value,
    payment_methods.method,
    'TXN-' || LPAD(FLOOR(RANDOM() * 999999)::TEXT, 6, '0'),
    'Transaction completed for opportunity',
    NOW(),
    NOW()
FROM won_opportunities o
CROSS JOIN random_employees e
CROSS JOIN (
    SELECT 'Cash' as method
    UNION ALL SELECT 'Financing'
    UNION ALL SELECT 'Lease'
    UNION ALL SELECT 'Credit Card'
) payment_methods
WHERE RANDOM() < 0.6  -- Create transactions for 60% of won opportunities
ON CONFLICT (id) DO NOTHING;

-- Insert customer activities
WITH customer_refs AS (
    SELECT id, name FROM customers LIMIT 10
),
employee_refs AS (
    SELECT id, name FROM employees WHERE position IN ('sales_agent', 'account_executive') LIMIT 8
),
opportunity_refs AS (
    SELECT id, customer_id FROM opportunities LIMIT 20
)
INSERT INTO activities (id, opportunity_salesperson_ae_id, customer_id, opportunity_id, activity_type, subject, description, created_date, end_date, is_completed, priority, created_at, updated_at)
SELECT
    uuid_generate_v4(),
    e.id,
    c.id,
    o.id,
    activity_types.activity_type::activity_type,
    activity_types.subject,
    activity_types.description || ' for customer ' || c.name,
    NOW() - INTERVAL '45 days' + (RANDOM() * INTERVAL '40 days'),
    NOW() - INTERVAL '30 days' + (RANDOM() * INTERVAL '35 days'),
    RANDOM() < 0.7,  -- 70% of activities are completed
    priorities.priority,
    NOW(),
    NOW()
FROM customer_refs c
CROSS JOIN employee_refs e
CROSS JOIN opportunity_refs o
CROSS JOIN (
    SELECT 'call' as activity_type, 'Initial Contact Call' as subject, 'Made initial contact call to discuss vehicle needs' as description
    UNION ALL SELECT 'email', 'Follow-up Email', 'Sent follow-up email with vehicle recommendations'
    UNION ALL SELECT 'meeting', 'In-Person Meeting', 'Scheduled meeting to discuss financing options'
    UNION ALL SELECT 'demo', 'Vehicle Demo', 'Conducted test drive and vehicle demonstration'
    UNION ALL SELECT 'follow_up', 'Follow-up Call', 'Follow-up call to answer questions'
    UNION ALL SELECT 'proposal_sent', 'Proposal Sent', 'Sent detailed proposal with pricing'
) activity_types
CROSS JOIN (
    SELECT 'low' as priority
    UNION ALL SELECT 'medium'
    UNION ALL SELECT 'high'
    UNION ALL SELECT 'urgent'
) priorities
WHERE RANDOM() < 0.3  -- Create activities for 30% of combinations
  AND c.id = o.customer_id  -- Ensure activity matches customer and opportunity
ON CONFLICT (id) DO NOTHING;

-- Add some comments to document the test data
COMMENT ON TABLE branches IS 'Test data: 5 branches across different regions and brands';
COMMENT ON TABLE employees IS 'Test data: Hierarchical employee structure with various positions';
COMMENT ON TABLE customers IS 'Test data: Mix of business and personal customers';
COMMENT ON TABLE vehicles IS 'Test data: Variety of vehicle types, brands, and availability';
COMMENT ON TABLE opportunities IS 'Test data: Sales opportunities in various stages';
COMMENT ON TABLE transactions IS 'Test data: Completed transactions with different payment methods';
COMMENT ON TABLE activities IS 'Test data: Customer interaction activities';
COMMENT ON TABLE pricing IS 'Test data: Vehicle pricing with discounts and promotions';

-- Log successful completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ CRM Test Data Setup Complete!';
    RAISE NOTICE '✅ Added comprehensive test data for natural language testing';
    RAISE NOTICE '✅ Data includes: branches, employees, customers, vehicles, opportunities, transactions, pricing, activities';
END $$; 