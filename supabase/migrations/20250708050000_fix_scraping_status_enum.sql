-- Fix scraping_status enum to match code expectations
-- Migration: 20250708050000_fix_scraping_status_enum.sql

-- Add missing enum values to scraping_status
-- Note: In PostgreSQL, new enum values can only be used after the transaction commits
ALTER TYPE scraping_status ADD VALUE IF NOT EXISTS 'pending';
ALTER TYPE scraping_status ADD VALUE IF NOT EXISTS 'error';

-- Add comment to document the change
COMMENT ON TYPE scraping_status IS 'Status enum for data source scraping operations: active, inactive, failed, pending, error'; 