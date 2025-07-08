-- Migration: Add document_type column to documents table
-- Handles existing tables gracefully and ensures enum type exists

-- Create document_type enum if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'document_type') THEN
        CREATE TYPE document_type AS ENUM ('pdf', 'word', 'text', 'html', 'markdown', 'web_page');
    END IF;
END $$;

-- Add document_type column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'documents' AND column_name = 'document_type') THEN
        ALTER TABLE documents ADD COLUMN document_type document_type DEFAULT 'text';
    END IF;
END $$;

-- Create index for document_type if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_documents_document_type ON documents(document_type);

-- Add comment to document the column
COMMENT ON COLUMN documents.document_type IS 'Type of document (pdf, word, text, html, markdown, web_page)';

-- Update any existing documents that have document_type in metadata but not in the column
UPDATE documents 
SET document_type = (metadata->>'document_type')::document_type
WHERE document_type IS NULL 
  AND metadata->>'document_type' IS NOT NULL
  AND metadata->>'document_type' IN ('pdf', 'word', 'text', 'html', 'markdown', 'web_page'); 