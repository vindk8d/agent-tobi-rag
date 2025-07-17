# Customer Warmth Scoring System

## Overview

The Customer Warmth Scoring System is a comprehensive solution for measuring the likelihood of customers to purchase a car. It provides data-driven insights into customer behavior, engagement patterns, and purchase intent through a sophisticated scoring algorithm that tracks multiple behavioral and demographic factors.

## Table of Contents

- [System Architecture](#system-architecture)
- [Core Tables](#core-tables)
- [Scoring Algorithm](#scoring-algorithm)
- [Usage Examples](#usage-examples)
- [Analytics & Reporting](#analytics--reporting)
- [API Integration](#api-integration)
- [Best Practices](#best-practices)

## System Architecture

The warmth system consists of three main tables:

```
customers (existing)
    ↓
customer_warmth (main scoring table)
    ↓
customer_warmth_history (historical tracking)
    ↓
warmth_scoring_triggers (automation rules)
```

---

## Core Tables

### 1. `customer_warmth` - Main Scoring Table

This is the primary table that stores current warmth scores and customer purchase likelihood data.

#### Core Warmth Metrics

| Field | Type | Description | Usage Example |
|-------|------|-------------|---------------|
| `overall_warmth_score` | DECIMAL(5,2) | Composite score (0-100) indicating purchase likelihood | `85.50` = High likelihood to purchase |
| `warmth_level` | ENUM | Categorical level: ice_cold, cold, cool, lukewarm, warm, hot, scorching | `hot` = Customer ready to buy soon |
| `purchase_probability` | DECIMAL(5,2) | Statistical probability (0-100%) of purchase | `78.5` = 78.5% chance of purchase |

#### Behavioral Scoring Components

These fields break down the overall score into specific behavioral areas:

| Field | Type | Range | Weight | Usage |
|-------|------|-------|--------|-------|
| `engagement_score` | DECIMAL(5,2) | 0-100 | 20% | How actively customer engages (calls, emails, visits) |
| `interaction_frequency_score` | DECIMAL(5,2) | 0-100 | 15% | How often customer initiates contact |
| `response_rate_score` | DECIMAL(5,2) | 0-100 | 10% | How quickly customer responds to outreach |
| `budget_qualification_score` | DECIMAL(5,2) | 0-100 | 20% | Customer's financial readiness and budget alignment |
| `timeline_urgency_score` | DECIMAL(5,2) | 0-100 | 15% | How soon customer needs to make purchase |
| `decision_making_authority_score` | DECIMAL(5,2) | 0-100 | 10% | Customer's ability to make final purchase decision |
| `demographic_fit_score` | DECIMAL(5,2) | 0-100 | 5% | How well customer matches target demographics |
| `past_purchase_behavior_score` | DECIMAL(5,2) | 0-100 | 5% | Historical purchase patterns and loyalty |

#### Time-based Metrics

| Field | Type | Description | Business Logic |
|-------|------|-------------|----------------|
| `days_since_first_contact` | INTEGER | Days since initial customer contact | Longer periods may indicate cooling interest |
| `days_since_last_interaction` | INTEGER | Days since last meaningful interaction | >30 days = dormant, >90 days = at risk |
| `expected_purchase_timeframe` | INTEGER | Expected days until purchase decision | Used for sales forecasting and follow-up timing |

#### Engagement Tracking

| Field | Type | Description | Usage |
|-------|------|-------------|-------|
| `total_interactions` | INTEGER | Total number of customer touchpoints | Measures overall engagement level |
| `meaningful_interactions` | INTEGER | Interactions indicating purchase intent | Quality over quantity metric |
| `last_engagement_type` | ENUM | Type of last customer interaction | inquiry, website_visit, test_drive, etc. |
| `last_engagement_date` | TIMESTAMP | When last interaction occurred | For follow-up scheduling |

#### Vehicle Interest Profile

| Field | Type | Description | Sales Application |
|-------|------|-------------|-------------------|
| `interested_vehicle_types` | TEXT[] | Array of vehicle types customer likes | `['suv', 'sedan']` - Focus sales efforts |
| `price_range_min` | DECIMAL(12,2) | Minimum budget range | Qualify customer financially |
| `price_range_max` | DECIMAL(12,2) | Maximum budget range | Present appropriate options |
| `preferred_features` | JSONB | Customer's desired vehicle features | `{"color": "red", "fuel_type": "hybrid"}` |

#### Business Intelligence

| Field | Type | Description | Strategic Use |
|-------|------|-------------|---------------|
| `predicted_purchase_value` | DECIMAL(12,2) | Expected transaction value | Revenue forecasting |
| `predicted_purchase_date` | DATE | Estimated purchase date | Sales pipeline management |
| `churn_risk_score` | DECIMAL(5,2) | Risk of customer leaving (0-100) | Retention campaign targeting |
| `retention_probability` | DECIMAL(5,2) | Likelihood to remain engaged | Resource allocation |

#### Scoring Metadata

| Field | Type | Description | Technical Use |
|-------|------|-------------|---------------|
| `scoring_algorithm_used` | ENUM | Algorithm version used | Version control and A/B testing |
| `algorithm_version` | VARCHAR(50) | Specific version number | Track algorithm improvements |
| `confidence_level` | DECIMAL(5,2) | Confidence in score accuracy | Weight recommendations |
| `manual_adjustments` | JSONB | Any manual score overrides | Audit trail for manual changes |

---

### 2. `customer_warmth_history` - Historical Tracking

Tracks all changes to customer warmth scores over time for trend analysis.

| Field | Type | Description | Usage |
|-------|------|-------------|-------|
| `customer_warmth_id` | UUID | Reference to main warmth record | Links to current state |
| `warmth_score_snapshot` | DECIMAL(5,2) | Score at time of change | Track score evolution |
| `warmth_level_snapshot` | ENUM | Warmth level at time of change | Identify level transitions |
| `change_trigger` | VARCHAR(255) | What caused the score change | `test_drive_completed`, `financing_approved` |
| `change_description` | TEXT | Detailed description of change | Context for score movement |
| `score_change` | DECIMAL(5,2) | +/- change from previous score | Measure impact of activities |

---

### 3. `warmth_scoring_triggers` - Automation Rules

Defines automated scoring rules for different customer engagement types.

| Field | Type | Description | Configuration |
|-------|------|-------------|---------------|
| `trigger_name` | VARCHAR(255) | Human-readable trigger name | "Test Drive Scheduled" |
| `trigger_type` | ENUM | Type of engagement | inquiry, test_drive, financing_inquiry |
| `base_score_impact` | DECIMAL(5,2) | Score points to add/subtract | +25 for test drive, +10 for inquiry |
| `decay_rate` | DECIMAL(5,2) | How impact decreases over time | 0.1 = 10% decay per time window |
| `max_occurrences` | INTEGER | Maximum times trigger can fire | Prevent score inflation |
| `time_window_days` | INTEGER | Days trigger remains effective | 90 days for test drive impact |

---

## Scoring Algorithm

### Composite Score Calculation

The overall warmth score uses a weighted algorithm:

```sql
overall_score = (
    engagement_score * 0.20 +           -- 20% weight
    budget_qualification * 0.20 +       -- 20% weight  
    interaction_frequency * 0.15 +      -- 15% weight
    timeline_urgency * 0.15 +          -- 15% weight
    response_rate * 0.10 +             -- 10% weight
    decision_authority * 0.10 +        -- 10% weight
    demographic_fit * 0.05 +           -- 5% weight
    past_behavior * 0.05               -- 5% weight
)
```

### Warmth Level Mapping

| Score Range | Warmth Level | Description | Sales Action |
|-------------|--------------|-------------|--------------|
| 90-100 | scorching | Ready to buy immediately | Close the deal |
| 75-89 | hot | High purchase intent | Aggressive follow-up |
| 60-74 | warm | Genuine interest | Regular engagement |
| 45-59 | lukewarm | Some interest | Nurture relationship |
| 30-44 | cool | Minimal interest | Educational content |
| 15-29 | cold | Low engagement | Re-engagement campaign |
| 0-14 | ice_cold | No current interest | Long-term nurturing |

---

## Usage Examples

### 1. Creating a New Warmth Record

```sql
INSERT INTO customer_warmth (
    customer_id,
    overall_warmth_score,
    engagement_score,
    budget_qualification_score,
    timeline_urgency_score,
    predicted_purchase_value,
    price_range_min,
    price_range_max,
    interested_vehicle_types,
    calculated_by
) VALUES (
    'customer-uuid-here',
    75.5,                          -- High warmth score
    80.0,                          -- Very engaged customer
    85.0,                          -- Budget qualified
    70.0,                          -- Moderate urgency
    1200000.00,                    -- ₱1.2M expected purchase
    1000000.00,                    -- ₱1M minimum budget
    1500000.00,                    -- ₱1.5M maximum budget
    ARRAY['suv', 'sedan'],         -- Interested in SUVs and sedans
    'employee-uuid-here'           -- Calculated by sales agent
);
```

### 2. Updating Warmth After Customer Activity

```sql
UPDATE customer_warmth 
SET 
    engagement_score = engagement_score + 15,
    total_interactions = total_interactions + 1,
    meaningful_interactions = meaningful_interactions + 1,
    last_engagement_type = 'test_drive',
    last_engagement_date = NOW(),
    days_since_last_interaction = 0
WHERE customer_id = 'customer-uuid-here';
```

### 3. Finding Hot Prospects

```sql
SELECT 
    c.name,
    c.email,
    cw.overall_warmth_score,
    cw.warmth_level,
    cw.predicted_purchase_value,
    cw.predicted_purchase_date
FROM customer_warmth cw
JOIN customers c ON cw.customer_id = c.id
WHERE cw.warmth_level IN ('hot', 'scorching')
  AND cw.is_active = true
ORDER BY cw.overall_warmth_score DESC;
```

---

## Analytics & Reporting

### 1. Using the `warmth_analytics` View

```sql
SELECT * FROM warmth_analytics 
WHERE engagement_status = 'Recent'
ORDER BY overall_warmth_score DESC
LIMIT 10;
```

**Returns:** Top 10 recently active customers by warmth score

### 2. Using the `warmth_trends` View

```sql
SELECT 
    customer_name,
    trend_date,
    avg_daily_score,
    change_triggers
FROM warmth_trends 
WHERE customer_id = 'specific-customer-uuid'
ORDER BY trend_date DESC;
```

**Returns:** 30-day warmth trend for a specific customer

### 3. Sales Pipeline Analysis

```sql
SELECT 
    warmth_level,
    COUNT(*) as customer_count,
    AVG(predicted_purchase_value) as avg_deal_size,
    SUM(predicted_purchase_value) as total_pipeline_value
FROM customer_warmth 
WHERE is_active = true
GROUP BY warmth_level
ORDER BY 
    CASE warmth_level 
        WHEN 'scorching' THEN 7
        WHEN 'hot' THEN 6
        WHEN 'warm' THEN 5
        WHEN 'lukewarm' THEN 4
        WHEN 'cool' THEN 3
        WHEN 'cold' THEN 2
        WHEN 'ice_cold' THEN 1
    END DESC;
```

---

## API Integration

### REST API Endpoints

**Get Customer Warmth**
```http
GET /api/customers/{customer_id}/warmth
```

**Update Warmth Score**
```http
PUT /api/customers/{customer_id}/warmth
Content-Type: application/json

{
    "engagement_score": 75.0,
    "budget_qualification_score": 80.0,
    "notes": "Customer completed test drive"
}
```

**Get Warmth Analytics**
```http
GET /api/analytics/warmth?level=hot&limit=50
```

### Function Examples

**Calculate Composite Score**
```sql
SELECT calculate_composite_warmth_score(
    75.0,  -- engagement_score
    60.0,  -- interaction_frequency_score
    80.0,  -- response_rate_score
    85.0,  -- budget_qualification_score
    70.0,  -- timeline_urgency_score
    75.0,  -- decision_authority_score
    60.0,  -- demographic_fit_score
    70.0   -- past_behavior_score
);
-- Returns: 73.25
```

---

## Best Practices

### 1. Data Quality

- **Regular Updates**: Update warmth scores after every meaningful customer interaction
- **Accurate Inputs**: Ensure behavioral scores reflect actual customer behavior
- **Validation**: Use confidence levels to indicate data quality

### 2. Scoring Guidelines

- **Be Conservative**: Better to under-score than over-score customers
- **Document Changes**: Always provide context in `warmth_notes` for manual adjustments
- **Monitor Trends**: Use history table to identify patterns and anomalies

### 3. Sales Process Integration

- **Daily Reviews**: Check `warmth_analytics` view daily for hot prospects
- **Follow-up Scheduling**: Use `days_since_last_interaction` for outreach timing
- **Resource Allocation**: Prioritize high-warmth customers for sales efforts

### 4. Performance Optimization

- **Index Usage**: Leverage existing indexes for common queries
- **Batch Updates**: Use bulk operations for score recalculations
- **Archive Old Data**: Regularly archive historical records older than 2 years

---

## Maintenance

### Regular Tasks

1. **Weekly**: Review and update scoring triggers based on performance
2. **Monthly**: Analyze warmth trends and adjust algorithm weights
3. **Quarterly**: Audit manual adjustments and scoring accuracy
4. **Annually**: Review and update algorithm version

### Monitoring Queries

**Check Score Distribution**
```sql
SELECT 
    warmth_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM customer_warmth 
WHERE is_active = true
GROUP BY warmth_level;
```

**Identify Stale Records**
```sql
SELECT customer_id, days_since_last_interaction
FROM customer_warmth 
WHERE days_since_last_interaction > 90
  AND warmth_level NOT IN ('ice_cold', 'cold');
```

---

## Conclusion

The Customer Warmth Scoring System provides a comprehensive framework for measuring and tracking customer purchase likelihood. By leveraging behavioral data, engagement patterns, and predictive analytics, sales teams can:

- **Prioritize efforts** on the most promising prospects
- **Optimize timing** for follow-ups and outreach
- **Predict revenue** more accurately
- **Improve conversion rates** through data-driven insights

For technical support or questions about implementation, refer to the migration file `20250128000000_create_customer_warmth_table.sql` or contact the development team. 