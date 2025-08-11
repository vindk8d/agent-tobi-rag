-- Migration: Create Quotations Table
-- This migration creates a quotations table for tracking AI-generated quotations
-- with proper relationships to customers, employees, vehicles, and pricing data.

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom ENUM type for quotation status
CREATE TYPE quotation_status AS ENUM ('draft', 'pending', 'sent', 'viewed', 'accepted', 'rejected', 'expired');

-- Create quotations table
CREATE TABLE quotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    
    -- Vehicle specifications (stored as JSONB for flexibility)
    vehicle_specs JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Pricing data (stored as JSONB to capture all pricing details)
    pricing_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- PDF storage information
    pdf_url TEXT, -- URL to the generated PDF in Supabase storage
    pdf_filename VARCHAR(255), -- Original filename for tracking
    
    -- Quotation metadata
    quotation_number VARCHAR(50) UNIQUE NOT NULL, -- Human-readable quotation number
    title VARCHAR(255) NOT NULL DEFAULT 'Vehicle Quotation',
    notes TEXT,
    
    -- Validity and status
    status quotation_status DEFAULT 'draft',
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE,
    viewed_at TIMESTAMP WITH TIME ZONE,
    responded_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_expiration CHECK (expires_at > created_at),
    CONSTRAINT valid_response_timing CHECK (responded_at IS NULL OR responded_at >= sent_at),
    CONSTRAINT valid_view_timing CHECK (viewed_at IS NULL OR viewed_at >= sent_at)
);

-- Add indexes for performance
CREATE INDEX idx_quotations_customer_id ON quotations(customer_id);
CREATE INDEX idx_quotations_employee_id ON quotations(employee_id);
CREATE INDEX idx_quotations_created_at ON quotations(created_at DESC);
CREATE INDEX idx_quotations_expires_at ON quotations(expires_at);
CREATE INDEX idx_quotations_status ON quotations(status);
CREATE INDEX idx_quotations_quotation_number ON quotations(quotation_number);

-- Add composite indexes for common queries
CREATE INDEX idx_quotations_employee_status ON quotations(employee_id, status);
CREATE INDEX idx_quotations_customer_status ON quotations(customer_id, status);

-- Enable Row Level Security (RLS)
ALTER TABLE quotations ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Employees can only access quotations they created
CREATE POLICY quotations_employee_access ON quotations
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM employees e 
            WHERE e.id = employee_id 
            AND e.email = (SELECT email FROM users WHERE id = auth.uid())
        )
    );

-- RLS Policy: Allow employees to create quotations
CREATE POLICY quotations_employee_insert ON quotations
    FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM employees e 
            WHERE e.id = employee_id 
            AND e.email = (SELECT email FROM users WHERE id = auth.uid())
        )
    );

-- Function to generate quotation numbers
CREATE OR REPLACE FUNCTION generate_quotation_number()
RETURNS TEXT AS $$
DECLARE
    current_year TEXT;
    sequence_num INTEGER;
    quotation_num TEXT;
BEGIN
    -- Get current year
    current_year := EXTRACT(YEAR FROM NOW())::TEXT;
    
    -- Get the next sequence number for this year
    SELECT COALESCE(MAX(
        CAST(SPLIT_PART(quotation_number, '-', 2) AS INTEGER)
    ), 0) + 1
    INTO sequence_num
    FROM quotations
    WHERE quotation_number LIKE 'Q' || current_year || '-%';
    
    -- Format as Q2025-0001, Q2025-0002, etc.
    quotation_num := 'Q' || current_year || '-' || LPAD(sequence_num::TEXT, 4, '0');
    
    RETURN quotation_num;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-generate quotation numbers
CREATE OR REPLACE FUNCTION set_quotation_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.quotation_number IS NULL OR NEW.quotation_number = '' THEN
        NEW.quotation_number := generate_quotation_number();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_quotation_number
    BEFORE INSERT ON quotations
    FOR EACH ROW
    EXECUTE FUNCTION set_quotation_number();

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_quotation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_quotation_timestamp
    BEFORE UPDATE ON quotations
    FOR EACH ROW
    EXECUTE FUNCTION update_quotation_timestamp();

-- Comments for documentation
COMMENT ON TABLE quotations IS 'AI-generated vehicle quotations with PDF storage and tracking';
COMMENT ON COLUMN quotations.vehicle_specs IS 'JSONB storage for flexible vehicle specification data';
COMMENT ON COLUMN quotations.pricing_data IS 'JSONB storage for complete pricing breakdown including base price, discounts, add-ons, taxes, and final totals';
COMMENT ON COLUMN quotations.quotation_number IS 'Human-readable quotation identifier (e.g., Q2025-0001)';
COMMENT ON COLUMN quotations.pdf_url IS 'Supabase storage URL for the generated PDF document';