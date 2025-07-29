-- Migration: Remove lead_warmth column from opportunities table
-- This separates warmth (customer-level) from opportunity stages (opportunity-level)
-- 
-- Background:
-- - Warmth should be measured at customer level (customer_warmth table)
-- - Opportunity stage should be measured at opportunity level (opportunities.stage)
-- - The warmth column in opportunities was conceptually incorrect

-- Step 1: Drop the sales_pipeline view that references the warmth column
DROP VIEW IF EXISTS sales_pipeline;

-- Step 2: Drop the index on the warmth column
DROP INDEX IF EXISTS idx_opportunities_warmth;

-- Step 3: Drop the warmth column from opportunities table
ALTER TABLE opportunities DROP COLUMN IF EXISTS warmth;

-- Step 4: Drop the lead_warmth enum type since it's no longer used
DROP TYPE IF EXISTS lead_warmth;

-- Step 5: Recreate the sales_pipeline view without the warmth column
-- This view now focuses purely on opportunity stages, not customer warmth
CREATE VIEW sales_pipeline AS 
SELECT 
    o.id AS opportunity_id,
    c.name AS customer_name,
    c.company,
    (v.brand || ' ' || v.model || ' (' || v.year || ')') AS vehicle,
    e.name AS salesperson_name,
    b.name AS branch_name,
    o.stage,
    -- Note: Warmth is now tracked at customer level in customer_warmth table
    -- To get customer warmth, join with customer_warmth table on customer_id
    o.probability,
    o.estimated_value,
    o.expected_close_date,
    o.created_date AS opportunity_created
FROM opportunities o
JOIN customers c ON o.customer_id = c.id
LEFT JOIN vehicles v ON o.vehicle_id = v.id
LEFT JOIN employees e ON o.opportunity_salesperson_ae_id = e.id
LEFT JOIN branches b ON e.branch_id = b.id
WHERE o.stage NOT IN ('Won', 'Lost')
ORDER BY o.created_date DESC;

-- Step 6: Create an enhanced view that includes customer warmth data
-- This properly separates concerns: opportunity data + customer warmth data
CREATE VIEW sales_pipeline_with_warmth AS 
SELECT 
    sp.*,
    cw.overall_warmth_score,
    cw.warmth_level,
    cw.purchase_probability,
    cw.last_engagement_date,
    cw.last_engagement_type
FROM sales_pipeline sp
LEFT JOIN customer_warmth cw ON sp.opportunity_id IN (
    SELECT o2.id 
    FROM opportunities o2 
    WHERE o2.customer_id = cw.customer_id
) AND cw.is_active = true;

-- Step 7: Add helpful comment explaining the new structure
COMMENT ON VIEW sales_pipeline IS 'Sales pipeline view focusing on opportunity stages. For customer warmth data, use sales_pipeline_with_warmth view or join directly with customer_warmth table.';

COMMENT ON VIEW sales_pipeline_with_warmth IS 'Enhanced sales pipeline view that combines opportunity stages with customer warmth metrics. This properly separates opportunity-level data from customer-level warmth scoring.'; 