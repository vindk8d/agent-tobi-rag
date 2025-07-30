-- Migration to convert USD pricing to Philippine Peso
-- File: 20250130142000_convert_usd_to_php_pricing.sql
-- Description: Convert vehicle pricing from mixed USD/PHP to consistent PHP pricing
-- Exchange Rate Used: 1 USD = 58 PHP (as of January 2025)

BEGIN;

-- Add a column to track the currency used
ALTER TABLE pricing ADD COLUMN IF NOT EXISTS currency_code VARCHAR(3) DEFAULT 'PHP';

-- First, let's see what we're converting (for logging purposes)
-- Records that appear to be in USD (base_price < 100000) will be converted

-- Update records that appear to be in USD to PHP
UPDATE pricing 
SET 
    base_price = ROUND(base_price * 58, 2),
    final_price = ROUND(final_price * 58, 2),
    insurance = ROUND(insurance * 58, 2),
    lto = ROUND(lto * 58, 2),
    price_discount = ROUND(price_discount * 58, 2),
    insurance_discount = ROUND(insurance_discount * 58, 2),
    lto_discount = ROUND(lto_discount * 58, 2),
    currency_code = 'PHP',
    updated_at = NOW()
WHERE base_price < 100000;

-- Ensure all existing PHP records also have the currency code set
UPDATE pricing 
SET 
    currency_code = 'PHP',
    updated_at = NOW()
WHERE currency_code IS NULL OR currency_code != 'PHP';

-- Add index for currency lookups (optional performance improvement)
CREATE INDEX IF NOT EXISTS idx_pricing_currency_code ON pricing(currency_code);

-- Add comments to document the changes
COMMENT ON COLUMN pricing.currency_code IS 'Currency code for all pricing fields. All prices standardized to PHP as of Jan 2025 migration.';
COMMENT ON TABLE pricing IS 'Test data: Vehicle pricing with discounts and promotions. All prices in Philippine Peso (PHP) as of Jan 2025 migration.';

-- Log the changes made
DO $$
DECLARE 
    converted_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO converted_count 
    FROM pricing 
    WHERE currency_code = 'PHP' AND updated_at >= NOW() - INTERVAL '1 minute';
    
    RAISE NOTICE 'Migration completed: % pricing records standardized to PHP currency', converted_count;
END $$;

COMMIT;