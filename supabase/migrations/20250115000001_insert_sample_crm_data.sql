-- Migration: Insert sample data for CRM Sales Management System
-- This migration adds sample data to demonstrate the CRM functionality

-- Insert sample branches
INSERT INTO branches (id, name, region, address, brand) VALUES 
(uuid_generate_v4(), 'Downtown Branch', 'central', '123 Main Street, Business District', 'Toyota'),
(uuid_generate_v4(), 'North Branch', 'north', '456 Oak Avenue, North Hills', 'Toyota'),
(uuid_generate_v4(), 'South Branch', 'south', '789 Pine Road, South Bay', 'Honda'),
(uuid_generate_v4(), 'East Branch', 'east', '321 Maple Drive, East Side', 'Ford'),
(uuid_generate_v4(), 'West Branch', 'west', '654 Cedar Lane, West End', 'Nissan');

-- Insert sample employees with hierarchy
WITH branch_ids AS (
    SELECT id, name FROM branches
),
managers AS (
    INSERT INTO employees (id, branch_id, name, position, email, phone, hire_date, is_active) 
    SELECT 
        uuid_generate_v4(),
        b.id,
        CASE 
            WHEN b.name = 'Downtown Branch' THEN 'John Smith'
            WHEN b.name = 'North Branch' THEN 'Sarah Johnson'
            WHEN b.name = 'South Branch' THEN 'Mike Davis'
            WHEN b.name = 'East Branch' THEN 'Lisa Wang'
            WHEN b.name = 'West Branch' THEN 'Carlos Rodriguez'
        END,
        'manager'::employee_position,
        CASE 
            WHEN b.name = 'Downtown Branch' THEN 'john.smith@company.com'
            WHEN b.name = 'North Branch' THEN 'sarah.johnson@company.com'
            WHEN b.name = 'South Branch' THEN 'mike.davis@company.com'
            WHEN b.name = 'East Branch' THEN 'lisa.wang@company.com'
            WHEN b.name = 'West Branch' THEN 'carlos.rodriguez@company.com'
        END,
        '+1-555-0101',
        '2023-01-15'::date,
        true
    FROM branch_ids b
    RETURNING id, branch_id, name
)
INSERT INTO employees (branch_id, name, position, manager_ae_id, email, phone, hire_date, is_active)
SELECT 
    m.branch_id,
    'Alex Thompson',
    'account_executive'::employee_position,
    m.id,
    'alex.thompson@company.com',
    '+1-555-0102',
    '2023-03-01'::date,
    true
FROM managers m WHERE m.name = 'John Smith'
UNION ALL
SELECT 
    m.branch_id,
    'Emma Wilson',
    'sales_agent'::employee_position,
    m.id,
    'emma.wilson@company.com',
    '+1-555-0103',
    '2023-04-15'::date,
    true
FROM managers m WHERE m.name = 'John Smith'
UNION ALL
SELECT 
    m.branch_id,
    'David Chen',
    'sales_agent'::employee_position,
    m.id,
    'david.chen@company.com',
    '+1-555-0104',
    '2023-02-20'::date,
    true
FROM managers m WHERE m.name = 'Sarah Johnson'
UNION ALL
SELECT 
    m.branch_id,
    'Maria Garcia',
    'account_executive'::employee_position,
    m.id,
    'maria.garcia@company.com',
    '+1-555-0105',
    '2023-01-30'::date,
    true
FROM managers m WHERE m.name = 'Mike Davis';

-- Insert sample customers
INSERT INTO customers (name, phone, mobile_number, email, company, is_for_business, address, notes) VALUES 
('Robert Brown', '+1-555-1001', '+1-555-1001', 'robert.brown@email.com', NULL, false, '101 Elm Street, City Center', 'Interested in hybrid vehicles'),
('Jennifer Lee', '+1-555-1002', '+1-555-1002', 'jennifer.lee@email.com', NULL, false, '202 Birch Avenue, Suburbia', 'First-time buyer'),
('TechCorp Solutions', '+1-555-2001', '+1-555-2001', 'procurement@techcorp.com', 'TechCorp Solutions', true, '300 Business Plaza, Tech District', 'Fleet purchase for company vehicles'),
('Global Industries', '+1-555-2002', '+1-555-2002', 'fleet@globalind.com', 'Global Industries', true, '400 Corporate Way, Industrial Park', 'Looking for delivery vehicles'),
('Mark Johnson', '+1-555-1003', '+1-555-1003', 'mark.johnson@email.com', NULL, false, '505 Willow Lane, Residential Area', 'Upgrading from older vehicle');

-- Insert sample vehicles
INSERT INTO vehicles (brand, year, model, type, color, acceleration, power, torque, fuel_type, transmission, is_available, stock_quantity) VALUES 
('Toyota', 2024, 'Camry', 'sedan', 'Silver', 8.4, 203, 247, 'Gasoline', 'Automatic', true, 5),
('Toyota', 2024, 'RAV4', 'suv', 'Blue', 8.1, 203, 184, 'Hybrid', 'CVT', true, 3),
('Honda', 2024, 'Civic', 'sedan', 'White', 7.5, 180, 177, 'Gasoline', 'Manual', true, 4),
('Honda', 2024, 'CR-V', 'suv', 'Black', 9.2, 190, 179, 'Gasoline', 'CVT', true, 2),
('Ford', 2024, 'F-150', 'pickup', 'Red', 6.1, 400, 410, 'Gasoline', 'Automatic', true, 1),
('Nissan', 2024, 'Altima', 'sedan', 'Gray', 8.9, 188, 180, 'Gasoline', 'CVT', true, 6),
('Toyota', 2024, 'Prius', 'hatchback', 'Green', 10.2, 121, 105, 'Hybrid', 'CVT', true, 8);

-- Insert sample pricing
INSERT INTO pricing (vehicle_id, availability_start_date, availability_end_date, warranty, insurance, lto, price_discount, insurance_discount, lto_discount, add_ons, warranty_promo, base_price, final_price, is_active)
SELECT 
    v.id,
    CURRENT_DATE,
    CURRENT_DATE + INTERVAL '6 months',
    '3 years / 100,000 km',
    15000.00,
    3500.00,
    CASE 
        WHEN v.model = 'Camry' THEN 25000.00
        WHEN v.model = 'RAV4' THEN 30000.00
        WHEN v.model = 'Civic' THEN 20000.00
        WHEN v.model = 'CR-V' THEN 28000.00
        WHEN v.model = 'F-150' THEN 45000.00
        WHEN v.model = 'Altima' THEN 22000.00
        WHEN v.model = 'Prius' THEN 27000.00
        ELSE 0.00
    END,
    1000.00, -- insurance discount
    500.00,  -- lto discount
    '["Tinted Windows: $500", "Floor Mats: $200", "Extended Warranty: $2000"]'::jsonb,
    'Special launch promotion - Extended warranty included',
    CASE 
        WHEN v.model = 'Camry' THEN 1200000.00
        WHEN v.model = 'RAV4' THEN 1450000.00
        WHEN v.model = 'Civic' THEN 1100000.00
        WHEN v.model = 'CR-V' THEN 1350000.00
        WHEN v.model = 'F-150' THEN 2200000.00
        WHEN v.model = 'Altima' THEN 1050000.00
        WHEN v.model = 'Prius' THEN 1300000.00
        ELSE 1000000.00
    END,
    CASE 
        WHEN v.model = 'Camry' THEN 1200000.00 - 25000.00
        WHEN v.model = 'RAV4' THEN 1450000.00 - 30000.00
        WHEN v.model = 'Civic' THEN 1100000.00 - 20000.00
        WHEN v.model = 'CR-V' THEN 1350000.00 - 28000.00
        WHEN v.model = 'F-150' THEN 2200000.00 - 45000.00
        WHEN v.model = 'Altima' THEN 1050000.00 - 22000.00
        WHEN v.model = 'Prius' THEN 1300000.00 - 27000.00
        ELSE 1000000.00
    END,
    true
FROM vehicles v;

-- Insert sample opportunities
WITH customer_vehicle_pairs AS (
    SELECT 
        c.id as customer_id,
        v.id as vehicle_id,
        e.id as salesperson_id,
        ROW_NUMBER() OVER () as rn
    FROM customers c
    CROSS JOIN vehicles v
    JOIN employees e ON e.position IN ('sales_agent', 'account_executive')
    WHERE c.name = 'Robert Brown' AND v.model = 'Camry'
    UNION ALL
    SELECT 
        c.id, v.id, e.id, 2
    FROM customers c
    CROSS JOIN vehicles v  
    JOIN employees e ON e.position IN ('sales_agent', 'account_executive')
    WHERE c.name = 'Jennifer Lee' AND v.model = 'Prius'
    UNION ALL
    SELECT 
        c.id, v.id, e.id, 3
    FROM customers c
    CROSS JOIN vehicles v
    JOIN employees e ON e.position IN ('sales_agent', 'account_executive')
    WHERE c.company = 'TechCorp Solutions' AND v.model = 'CR-V'
    LIMIT 1
)
INSERT INTO opportunities (customer_id, vehicle_id, opportunity_salesperson_ae_id, stage, warmth, probability, estimated_value, expected_close_date, notes)
SELECT 
    customer_id,
    vehicle_id,
    salesperson_id,
    CASE 
        WHEN rn = 1 THEN 'Contacted'::opportunity_stage
        WHEN rn = 2 THEN 'Consideration'::opportunity_stage
        WHEN rn = 3 THEN 'Purchase Intent'::opportunity_stage
        ELSE 'New'::opportunity_stage
    END,
    CASE 
        WHEN rn = 1 THEN 'warm'::lead_warmth
        WHEN rn = 2 THEN 'hot'::lead_warmth
        WHEN rn = 3 THEN 'hot'::lead_warmth
        ELSE 'cold'::lead_warmth
    END,
    CASE 
        WHEN rn = 1 THEN 75
        WHEN rn = 2 THEN 60
        WHEN rn = 3 THEN 85
        ELSE 25
    END,
    CASE 
        WHEN rn = 1 THEN 1175000.00
        WHEN rn = 2 THEN 1273000.00
        WHEN rn = 3 THEN 1322000.00
        ELSE 1000000.00
    END,
    CURRENT_DATE + INTERVAL '30 days',
    CASE 
        WHEN rn = 1 THEN 'Customer very interested, scheduled test drive'
        WHEN rn = 2 THEN 'Proposal sent, waiting for decision'
        WHEN rn = 3 THEN 'Fleet purchase, in final negotiations'
        ELSE 'Initial contact made'
    END
FROM customer_vehicle_pairs;

-- Insert sample activities
WITH opportunity_data AS (
    SELECT 
        o.id as opportunity_id,
        o.opportunity_salesperson_ae_id,
        o.customer_id,
        ROW_NUMBER() OVER () as rn
    FROM opportunities o
    LIMIT 3
)
INSERT INTO activities (opportunity_salesperson_ae_id, customer_id, opportunity_id, activity_type, subject, description, created_date, is_completed, priority)
SELECT 
    opportunity_salesperson_ae_id,
    customer_id,
    opportunity_id,
    CASE 
        WHEN rn = 1 THEN 'call'::activity_type
        WHEN rn = 2 THEN 'meeting'::activity_type
        WHEN rn = 3 THEN 'email'::activity_type
        ELSE 'follow_up'::activity_type
    END,
    CASE 
        WHEN rn = 1 THEN 'Initial consultation call'
        WHEN rn = 2 THEN 'Vehicle demonstration'
        WHEN rn = 3 THEN 'Proposal follow-up'
        ELSE 'General follow-up'
    END,
    CASE 
        WHEN rn = 1 THEN 'Discussed customer needs and preferences for vehicle selection'
        WHEN rn = 2 THEN 'Conducted vehicle demonstration and test drive'
        WHEN rn = 3 THEN 'Sent detailed proposal and pricing information'
        ELSE 'Regular check-in with customer'
    END,
    CURRENT_DATE - INTERVAL '1 day',
    true,
    'medium'
FROM opportunity_data
UNION ALL
SELECT 
    opportunity_salesperson_ae_id,
    customer_id,
    opportunity_id,
    'follow_up'::activity_type,
    'Weekly check-in call',
    'Scheduled follow-up to answer any questions about the proposal',
    CURRENT_DATE + INTERVAL '3 days',
    false,
    'high'
FROM opportunity_data; 