-- Add missing columns to data_sources table for enhanced management
-- Migration: 20250708040000_add_data_source_columns.sql

-- Skip scraping frequency enum - column already exists as INTERVAL type

-- Add missing columns to data_sources table (skip scraping_frequency - already exists)
ALTER TABLE data_sources 
ADD COLUMN IF NOT EXISTS description TEXT,
ADD COLUMN IF NOT EXISTS last_success TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS document_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_error TEXT,
ADD COLUMN IF NOT EXISTS configuration JSONB DEFAULT '{}'::jsonb;

-- Update existing data_sources to have default values
UPDATE data_sources 
SET document_count = 0, error_count = 0
WHERE document_count IS NULL OR error_count IS NULL;
-- Skip setting scraping_frequency since it's already an interval type

-- Add indexes for new columns (skip scraping_frequency - already exists)
CREATE INDEX IF NOT EXISTS idx_data_sources_last_success ON data_sources(last_success);
CREATE INDEX IF NOT EXISTS idx_data_sources_document_count ON data_sources(document_count);
CREATE INDEX IF NOT EXISTS idx_data_sources_error_count ON data_sources(error_count);

-- Add updated_at trigger for data_sources if not exists
DROP TRIGGER IF EXISTS update_data_sources_updated_at ON data_sources;
CREATE TRIGGER update_data_sources_updated_at
    BEFORE UPDATE ON data_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a function to update document count for data sources
CREATE OR REPLACE FUNCTION update_data_source_document_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE data_sources 
        SET document_count = (
            SELECT COUNT(*) 
            FROM documents 
            WHERE data_source_id = NEW.data_source_id
        )
        WHERE id = NEW.data_source_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE data_sources 
        SET document_count = (
            SELECT COUNT(*) 
            FROM documents 
            WHERE data_source_id = OLD.data_source_id
        )
        WHERE id = OLD.data_source_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update document count
DROP TRIGGER IF EXISTS update_data_source_count_on_insert ON documents;
CREATE TRIGGER update_data_source_count_on_insert
    AFTER INSERT ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_document_count();

DROP TRIGGER IF EXISTS update_data_source_count_on_delete ON documents;
CREATE TRIGGER update_data_source_count_on_delete
    AFTER DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_document_count();

-- Add comment to document the schema changes
COMMENT ON COLUMN data_sources.description IS 'Optional description of the data source';
-- Skip scraping_frequency comment - column already exists
COMMENT ON COLUMN data_sources.last_success IS 'Last successful scraping timestamp';
COMMENT ON COLUMN data_sources.document_count IS 'Number of documents processed from this source';
COMMENT ON COLUMN data_sources.error_count IS 'Number of errors encountered';
COMMENT ON COLUMN data_sources.last_error IS 'Last error message encountered';
COMMENT ON COLUMN data_sources.configuration IS 'JSON configuration for scraping parameters'; 