-- Migration: Create Customer Warmth Table
-- This migration creates a comprehensive warmth scoring system to measure
-- the likelihood of customers to purchase a car, with detailed metrics,
-- historical tracking, and predictive scoring capabilities.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types for warmth scoring
CREATE TYPE warmth_level AS ENUM ('ice_cold', 'cold', 'cool', 'lukewarm', 'warm', 'hot', 'scorching');
CREATE TYPE engagement_type AS ENUM ('inquiry', 'website_visit', 'brochure_request', 'test_drive', 'financing_inquiry', 'trade_in_evaluation', 'callback_request', 'referral');
CREATE TYPE scoring_algorithm AS ENUM ('behavioral', 'demographic', 'interaction_frequency', 'purchase_timeline', 'budget_qualification', 'composite');

-- Create the main customer_warmth table
CREATE TABLE customer_warmth (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    
    -- Core Warmth Metrics
    overall_warmth_score DECIMAL(5,2) NOT NULL CHECK (overall_warmth_score >= 0 AND overall_warmth_score <= 100),
    warmth_level warmth_level NOT NULL,
    purchase_probability DECIMAL(5,2) CHECK (purchase_probability >= 0 AND purchase_probability <= 100),
    
    -- Behavioral Scoring Components
    engagement_score DECIMAL(5,2) DEFAULT 0 CHECK (engagement_score >= 0 AND engagement_score <= 100),
    interaction_frequency_score DECIMAL(5,2) DEFAULT 0 CHECK (interaction_frequency_score >= 0 AND interaction_frequency_score <= 100),
    response_rate_score DECIMAL(5,2) DEFAULT 0 CHECK (response_rate_score >= 0 AND response_rate_score <= 100),
    
    -- Purchase Intent Indicators
    budget_qualification_score DECIMAL(5,2) DEFAULT 0 CHECK (budget_qualification_score >= 0 AND budget_qualification_score <= 100),
    timeline_urgency_score DECIMAL(5,2) DEFAULT 0 CHECK (timeline_urgency_score >= 0 AND timeline_urgency_score <= 100),
    decision_making_authority_score DECIMAL(5,2) DEFAULT 0 CHECK (decision_making_authority_score >= 0 AND decision_making_authority_score <= 100),
    
    -- Demographic and Profile Scoring
    demographic_fit_score DECIMAL(5,2) DEFAULT 0 CHECK (demographic_fit_score >= 0 AND demographic_fit_score <= 100),
    past_purchase_behavior_score DECIMAL(5,2) DEFAULT 0 CHECK (past_purchase_behavior_score >= 0 AND past_purchase_behavior_score <= 100),
    
    -- Time-based Metrics
    days_since_first_contact INTEGER DEFAULT 0,
    days_since_last_interaction INTEGER DEFAULT 0,
    expected_purchase_timeframe INTEGER, -- days
    
    -- Engagement Tracking
    total_interactions INTEGER DEFAULT 0,
    meaningful_interactions INTEGER DEFAULT 0, -- interactions that indicate purchase intent
    last_engagement_type engagement_type,
    last_engagement_date TIMESTAMP WITH TIME ZONE,
    
    -- Vehicle Interest
    interested_vehicle_types TEXT[], -- array of vehicle types they've shown interest in
    price_range_min DECIMAL(12,2),
    price_range_max DECIMAL(12,2),
    preferred_features JSONB DEFAULT '{}'::jsonb,
    
    -- Scoring Metadata
    scoring_algorithm_used scoring_algorithm DEFAULT 'composite',
    algorithm_version VARCHAR(50) DEFAULT '1.0',
    confidence_level DECIMAL(5,2) DEFAULT 50 CHECK (confidence_level >= 0 AND confidence_level <= 100),
    
    -- Business Intelligence
    predicted_purchase_value DECIMAL(12,2),
    predicted_purchase_date DATE,
    churn_risk_score DECIMAL(5,2) DEFAULT 0 CHECK (churn_risk_score >= 0 AND churn_risk_score <= 100),
    retention_probability DECIMAL(5,2) DEFAULT 50 CHECK (retention_probability >= 0 AND retention_probability <= 100),
    
    -- Notes and Context
    warmth_notes TEXT,
    scoring_rationale TEXT,
    manual_adjustments JSONB DEFAULT '{}'::jsonb, -- store any manual score adjustments and reasons
    
    -- Tracking
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    calculated_by UUID REFERENCES employees(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT true,
    
    -- Standard timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_price_range CHECK (price_range_max IS NULL OR price_range_min IS NULL OR price_range_max >= price_range_min),
    CONSTRAINT meaningful_interactions_check CHECK (meaningful_interactions <= total_interactions)
);

-- Create warmth history table for tracking changes over time
CREATE TABLE customer_warmth_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_warmth_id UUID NOT NULL REFERENCES customer_warmth(id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    
    -- Snapshot of warmth data at a point in time
    warmth_score_snapshot DECIMAL(5,2) NOT NULL,
    warmth_level_snapshot warmth_level NOT NULL,
    purchase_probability_snapshot DECIMAL(5,2),
    
    -- What changed
    change_trigger VARCHAR(255), -- what caused this warmth update
    change_description TEXT,
    score_change DECIMAL(5,2), -- positive or negative change from previous score
    
    -- Tracking
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    recorded_by UUID REFERENCES employees(id) ON DELETE SET NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create warmth triggers table for automated scoring rules
CREATE TABLE warmth_scoring_triggers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trigger_name VARCHAR(255) NOT NULL UNIQUE,
    trigger_type engagement_type NOT NULL,
    
    -- Scoring rules
    base_score_impact DECIMAL(5,2) NOT NULL, -- how much this trigger affects the base score
    decay_rate DECIMAL(5,2) DEFAULT 0, -- how much the impact decreases over time (0-1)
    max_occurrences INTEGER, -- maximum times this trigger can be applied
    time_window_days INTEGER, -- time window for trigger effectiveness
    
    -- Conditions
    conditions JSONB DEFAULT '{}'::jsonb, -- additional conditions for trigger activation
    is_active BOOLEAN DEFAULT true,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_customer_warmth_customer_id ON customer_warmth(customer_id);
CREATE INDEX idx_customer_warmth_warmth_level ON customer_warmth(warmth_level);
CREATE INDEX idx_customer_warmth_warmth_score ON customer_warmth(overall_warmth_score);
CREATE INDEX idx_customer_warmth_purchase_probability ON customer_warmth(purchase_probability);
CREATE INDEX idx_customer_warmth_calculated_at ON customer_warmth(calculated_at);
CREATE INDEX idx_customer_warmth_is_active ON customer_warmth(is_active);
CREATE INDEX idx_customer_warmth_last_engagement ON customer_warmth(last_engagement_date);

CREATE INDEX idx_warmth_history_customer_id ON customer_warmth_history(customer_id);
CREATE INDEX idx_warmth_history_customer_warmth_id ON customer_warmth_history(customer_warmth_id);
CREATE INDEX idx_warmth_history_recorded_at ON customer_warmth_history(recorded_at);

CREATE INDEX idx_warmth_triggers_trigger_type ON warmth_scoring_triggers(trigger_type);
CREATE INDEX idx_warmth_triggers_is_active ON warmth_scoring_triggers(is_active);

-- Create function to automatically determine warmth level from score
CREATE OR REPLACE FUNCTION determine_warmth_level(score DECIMAL(5,2))
RETURNS warmth_level AS $$
BEGIN
    RETURN CASE
        WHEN score >= 90 THEN 'scorching'::warmth_level
        WHEN score >= 75 THEN 'hot'::warmth_level
        WHEN score >= 60 THEN 'warm'::warmth_level
        WHEN score >= 45 THEN 'lukewarm'::warmth_level
        WHEN score >= 30 THEN 'cool'::warmth_level
        WHEN score >= 15 THEN 'cold'::warmth_level
        ELSE 'ice_cold'::warmth_level
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to calculate composite warmth score
CREATE OR REPLACE FUNCTION calculate_composite_warmth_score(
    p_engagement_score DECIMAL(5,2),
    p_interaction_frequency_score DECIMAL(5,2),
    p_response_rate_score DECIMAL(5,2),
    p_budget_qualification_score DECIMAL(5,2),
    p_timeline_urgency_score DECIMAL(5,2),
    p_decision_authority_score DECIMAL(5,2),
    p_demographic_fit_score DECIMAL(5,2),
    p_past_behavior_score DECIMAL(5,2)
)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    weighted_score DECIMAL(5,2);
BEGIN
    -- Weighted scoring algorithm
    weighted_score := (
        (COALESCE(p_engagement_score, 0) * 0.20) +
        (COALESCE(p_interaction_frequency_score, 0) * 0.15) +
        (COALESCE(p_response_rate_score, 0) * 0.10) +
        (COALESCE(p_budget_qualification_score, 0) * 0.20) +
        (COALESCE(p_timeline_urgency_score, 0) * 0.15) +
        (COALESCE(p_decision_authority_score, 0) * 0.10) +
        (COALESCE(p_demographic_fit_score, 0) * 0.05) +
        (COALESCE(p_past_behavior_score, 0) * 0.05)
    );
    
    RETURN LEAST(100.0, GREATEST(0.0, weighted_score));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create trigger function to update warmth level when score changes
CREATE OR REPLACE FUNCTION update_warmth_level()
RETURNS TRIGGER AS $$
BEGIN
    NEW.warmth_level = determine_warmth_level(NEW.overall_warmth_score);
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger function to log warmth changes to history
CREATE OR REPLACE FUNCTION log_warmth_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Only log if the warmth score actually changed
    IF OLD.overall_warmth_score IS DISTINCT FROM NEW.overall_warmth_score THEN
        INSERT INTO customer_warmth_history (
            customer_warmth_id,
            customer_id,
            warmth_score_snapshot,
            warmth_level_snapshot,
            purchase_probability_snapshot,
            change_trigger,
            change_description,
            score_change,
            recorded_by
        ) VALUES (
            NEW.id,
            NEW.customer_id,
            NEW.overall_warmth_score,
            NEW.warmth_level,
            NEW.purchase_probability,
            'score_update',
            'Automated warmth score update',
            NEW.overall_warmth_score - COALESCE(OLD.overall_warmth_score, 0),
            NEW.calculated_by
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_warmth_level
    BEFORE INSERT OR UPDATE OF overall_warmth_score ON customer_warmth
    FOR EACH ROW
    EXECUTE FUNCTION update_warmth_level();

CREATE TRIGGER trigger_update_warmth_updated_at
    BEFORE UPDATE ON customer_warmth
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_log_warmth_change
    AFTER UPDATE OF overall_warmth_score ON customer_warmth
    FOR EACH ROW
    EXECUTE FUNCTION log_warmth_change();

CREATE TRIGGER trigger_update_warmth_triggers_updated_at
    BEFORE UPDATE ON warmth_scoring_triggers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default scoring triggers
INSERT INTO warmth_scoring_triggers (trigger_name, trigger_type, base_score_impact, decay_rate, max_occurrences, time_window_days, conditions) VALUES
('Initial Inquiry', 'inquiry', 10.0, 0.1, 1, 30, '{"first_time": true}'),
('Website Visit', 'website_visit', 2.0, 0.05, 10, 7, '{"pages_viewed": ">= 3"}'),
('Brochure Request', 'brochure_request', 15.0, 0.1, 2, 60, '{}'),
('Test Drive Scheduled', 'test_drive', 25.0, 0.05, 3, 90, '{}'),
('Financing Inquiry', 'financing_inquiry', 20.0, 0.1, 2, 60, '{}'),
('Trade-in Evaluation', 'trade_in_evaluation', 18.0, 0.1, 2, 60, '{}'),
('Callback Request', 'callback_request', 12.0, 0.1, 5, 30, '{}'),
('Referral', 'referral', 22.0, 0.05, 1, 180, '{}');

-- Create view for warmth analytics
CREATE VIEW warmth_analytics AS
SELECT 
    cw.customer_id,
    c.name as customer_name,
    c.company,
    cw.overall_warmth_score,
    cw.warmth_level,
    cw.purchase_probability,
    cw.predicted_purchase_value,
    cw.predicted_purchase_date,
    cw.total_interactions,
    cw.meaningful_interactions,
    cw.days_since_first_contact,
    cw.days_since_last_interaction,
    cw.last_engagement_type,
    cw.last_engagement_date,
    CASE 
        WHEN cw.days_since_last_interaction <= 7 THEN 'Recent'
        WHEN cw.days_since_last_interaction <= 30 THEN 'Active'
        WHEN cw.days_since_last_interaction <= 90 THEN 'Dormant'
        ELSE 'Inactive'
    END as engagement_status,
    cw.calculated_at,
    cw.updated_at
FROM customer_warmth cw
JOIN customers c ON cw.customer_id = c.id
WHERE cw.is_active = true
ORDER BY cw.overall_warmth_score DESC;

-- Create view for warmth trends
CREATE VIEW warmth_trends AS
SELECT 
    cwh.customer_id,
    c.name as customer_name,
    DATE(cwh.recorded_at) as trend_date,
    AVG(cwh.warmth_score_snapshot) as avg_daily_score,
    MAX(cwh.warmth_score_snapshot) as max_daily_score,
    MIN(cwh.warmth_score_snapshot) as min_daily_score,
    COUNT(*) as daily_updates,
    STRING_AGG(DISTINCT cwh.change_trigger, ', ') as change_triggers
FROM customer_warmth_history cwh
JOIN customers c ON cwh.customer_id = c.id
WHERE cwh.recorded_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY cwh.customer_id, c.name, DATE(cwh.recorded_at)
ORDER BY trend_date DESC, avg_daily_score DESC;

-- Add table comments
COMMENT ON TABLE customer_warmth IS 'Comprehensive customer warmth scoring system to measure purchase likelihood';
COMMENT ON TABLE customer_warmth_history IS 'Historical tracking of warmth score changes over time';
COMMENT ON TABLE warmth_scoring_triggers IS 'Automated scoring rules for different customer engagement types';

-- Add column comments for key fields
COMMENT ON COLUMN customer_warmth.overall_warmth_score IS 'Composite score (0-100) indicating likelihood to purchase';
COMMENT ON COLUMN customer_warmth.warmth_level IS 'Categorical warmth level derived from overall score';
COMMENT ON COLUMN customer_warmth.purchase_probability IS 'Statistical probability (0-100%) of purchase within expected timeframe';
COMMENT ON COLUMN customer_warmth.scoring_algorithm_used IS 'Algorithm version used to calculate this score';
COMMENT ON COLUMN customer_warmth.confidence_level IS 'Confidence in the accuracy of the warmth score (0-100%)'; 