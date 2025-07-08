-- Migration: Add Document Metadata Columns
-- Adds proper columns to documents table instead of storing everything in metadata JSON
-- This improves querying performance, enables indexing, and provides type safety

-- Add new columns to documents table (safe - will not fail if columns already exist)
DO $$ 
BEGIN
    -- Add processed_at column for tracking when document processing completed
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'processed_at') THEN
        ALTER TABLE documents ADD COLUMN processed_at TIMESTAMP WITH TIME ZONE;
    END IF;

    -- Add storage_path column for file system location
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'storage_path') THEN
        ALTER TABLE documents ADD COLUMN storage_path TEXT;
    END IF;

    -- Add embedding_count column for analytics
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'embedding_count') THEN
        ALTER TABLE documents ADD COLUMN embedding_count INTEGER DEFAULT 0;
    END IF;

    -- Add original_filename column for user display
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'original_filename') THEN
        ALTER TABLE documents ADD COLUMN original_filename TEXT;
    END IF;

    -- Add file_size column for storage analytics
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'file_size') THEN
        ALTER TABLE documents ADD COLUMN file_size BIGINT;
    END IF;
END $$;

-- Migrate existing data from metadata JSON to proper columns
-- This will extract data from metadata and populate the new columns
UPDATE documents 
SET 
    processed_at = CASE 
        WHEN metadata->>'processed_at' IS NOT NULL 
        THEN to_timestamp((metadata->>'processed_at')::numeric)
        ELSE NULL 
    END,
    storage_path = metadata->>'storage_path',
    embedding_count = CASE 
        WHEN metadata->>'embedding_count' IS NOT NULL 
        THEN (metadata->>'embedding_count')::integer
        ELSE 0 
    END,
    original_filename = metadata->>'original_filename',
    file_size = CASE 
        WHEN metadata->>'file_size' IS NOT NULL 
        THEN (metadata->>'file_size')::bigint
        ELSE NULL 
    END
WHERE metadata IS NOT NULL;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at);
CREATE INDEX IF NOT EXISTS idx_documents_storage_path ON documents(storage_path);
CREATE INDEX IF NOT EXISTS idx_documents_embedding_count ON documents(embedding_count);
CREATE INDEX IF NOT EXISTS idx_documents_original_filename ON documents(original_filename);
CREATE INDEX IF NOT EXISTS idx_documents_file_size ON documents(file_size);

-- Add comments to document the new columns
COMMENT ON COLUMN documents.processed_at IS 'Timestamp when document processing completed';
COMMENT ON COLUMN documents.storage_path IS 'File system path or storage identifier for the document';
COMMENT ON COLUMN documents.embedding_count IS 'Number of embeddings generated for this document';
COMMENT ON COLUMN documents.original_filename IS 'Original filename as uploaded by user';
COMMENT ON COLUMN documents.file_size IS 'File size in bytes';

-- Optional: Clean up migrated metadata fields from JSONB 
-- (Uncomment if you want to remove the fields from metadata after migration)
-- UPDATE documents 
-- SET metadata = metadata - 'processed_at' - 'storage_path' - 'embedding_count' - 'original_filename' - 'file_size'
-- WHERE metadata IS NOT NULL; 