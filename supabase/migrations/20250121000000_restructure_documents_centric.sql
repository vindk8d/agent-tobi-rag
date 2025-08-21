-- Migration: Restructure to Documents-Centric Architecture
-- This migration eliminates redundancy between data_sources and documents tables
-- by making documents the primary table for file uploads and vehicle specifications

-- Step 1: Add document_id column to document_chunks table
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS document_id UUID REFERENCES documents(id) ON DELETE CASCADE;

-- Step 2: Create index for performance
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

-- Step 3: Migrate existing file-based data_sources to documents table
-- First, let's identify file-based data sources (type = 'document')
INSERT INTO documents (
    id,
    data_source_id,
    content,
    metadata,
    document_type,
    created_at,
    updated_at,
    processed_at,
    storage_path,
    original_filename,
    file_size,
    vehicle_id
)
SELECT 
    uuid_generate_v4() as id,
    ds.id as data_source_id,
    COALESCE(ds.description, 'Document content will be populated from chunks') as content, -- Provide non-null content
    ds.metadata,
    ds.document_type,
    ds.created_at,
    ds.updated_at,
    ds.last_success_at as processed_at,
    NULL as storage_path, -- Will be updated by backend
    NULL as original_filename, -- Will be updated by backend
    NULL as file_size, -- Will be updated by backend
    NULL as vehicle_id -- Will be updated for vehicle specs
FROM data_sources ds
WHERE ds.type = 'document' 
AND NOT EXISTS (
    SELECT 1 FROM documents d WHERE d.data_source_id = ds.id
);

-- Step 4: Update document_chunks to reference documents for file-based content
-- Create a temporary mapping between data_source_id and new document_id
UPDATE document_chunks 
SET document_id = d.id
FROM documents d
WHERE document_chunks.data_source_id = d.data_source_id
AND document_chunks.document_id IS NULL;

-- Step 5: Update documents table with aggregated content from chunks
-- This helps maintain the document content field
UPDATE documents 
SET content = COALESCE(
    (
        SELECT string_agg(dc.content, E'\n\n' ORDER BY dc.chunk_index)
        FROM document_chunks dc 
        WHERE dc.document_id = documents.id
    ),
    content -- Keep existing content if no chunks found
)
WHERE EXISTS (
    SELECT 1 FROM document_chunks dc WHERE dc.document_id = documents.id
);

-- Step 6: For vehicle specifications, update vehicle_id based on data source name pattern
-- This is a best-effort migration for existing Ford Bronco specs
UPDATE documents 
SET vehicle_id = v.id
FROM vehicles v, data_sources ds
WHERE documents.data_source_id = ds.id
AND ds.document_type = 'vehicle_specification'
AND ds.name ILIKE '%ford%bronco%'
AND v.brand ILIKE 'ford'
AND v.model ILIKE '%bronco%';

-- Step 7: Update document_chunks with vehicle_id for consistency
UPDATE document_chunks
SET vehicle_id = d.vehicle_id
FROM documents d
WHERE document_chunks.document_id = d.id
AND document_chunks.vehicle_id IS NULL
AND d.vehicle_id IS NOT NULL;

-- Step 8: Create function to maintain data consistency
-- This function ensures document_chunks always have the same vehicle_id as their parent document
CREATE OR REPLACE FUNCTION sync_document_chunk_vehicle_id()
RETURNS TRIGGER AS $$
BEGIN
    -- When document vehicle_id changes, update all its chunks
    IF TG_OP = 'UPDATE' AND OLD.vehicle_id IS DISTINCT FROM NEW.vehicle_id THEN
        UPDATE document_chunks 
        SET vehicle_id = NEW.vehicle_id 
        WHERE document_id = NEW.id;
    END IF;
    
    -- When inserting a new chunk, inherit vehicle_id from document
    IF TG_OP = 'INSERT' AND NEW.document_id IS NOT NULL THEN
        SELECT vehicle_id INTO NEW.vehicle_id 
        FROM documents 
        WHERE id = NEW.document_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 9: Create triggers to maintain consistency
DROP TRIGGER IF EXISTS trigger_sync_document_vehicle_id ON documents;
CREATE TRIGGER trigger_sync_document_vehicle_id
    AFTER UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION sync_document_chunk_vehicle_id();

DROP TRIGGER IF EXISTS trigger_inherit_document_vehicle_id ON document_chunks;
CREATE TRIGGER trigger_inherit_document_vehicle_id
    BEFORE INSERT ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION sync_document_chunk_vehicle_id();

-- Step 10: Update vector search functions to work with new structure
-- Update the existing vector search function to handle both data_source_id and document_id
CREATE OR REPLACE FUNCTION search_embeddings_with_vehicle_filter(
    query_embedding vector(1536),
    vehicle_filter_id uuid DEFAULT NULL,
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 10
)
RETURNS TABLE (
    chunk_id uuid,
    content text,
    similarity float,
    vehicle_id uuid,
    document_type text,
    metadata jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id as chunk_id,
        dc.content,
        (1 - (e.embedding <=> query_embedding)) as similarity,
        dc.vehicle_id,
        dc.document_type::text,
        dc.metadata
    FROM document_chunks dc
    JOIN embeddings e ON e.document_chunk_id = dc.id
    WHERE 
        (vehicle_filter_id IS NULL OR dc.vehicle_id = vehicle_filter_id)
        AND (1 - (e.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Step 11: Add helpful views for debugging and monitoring
CREATE OR REPLACE VIEW vehicle_documents_summary AS
SELECT 
    v.id as vehicle_id,
    v.brand,
    v.model,
    v.year,
    COUNT(d.id) as document_count,
    COUNT(dc.id) as chunk_count,
    COUNT(e.id) as embedding_count,
    MAX(d.updated_at) as last_updated
FROM vehicles v
LEFT JOIN documents d ON d.vehicle_id = v.id
LEFT JOIN document_chunks dc ON dc.vehicle_id = v.id
LEFT JOIN embeddings e ON e.document_chunk_id = dc.id
GROUP BY v.id, v.brand, v.model, v.year
ORDER BY v.brand, v.model, v.year;

-- Step 12: Create indexes for optimal performance
CREATE INDEX IF NOT EXISTS idx_documents_vehicle_id_type ON documents(vehicle_id, document_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_vehicle_id_type ON document_chunks(vehicle_id, document_type);

-- Step 13: Add comments for documentation
COMMENT ON TABLE documents IS 'Primary table for file uploads, vehicle specifications, and direct content. Use data_sources for web scraping and external APIs only.';
COMMENT ON TABLE data_sources IS 'External sources like web scraping and APIs. File uploads should use documents table.';
COMMENT ON COLUMN document_chunks.document_id IS 'References documents table for file-based content. Use data_source_id for web-scraped content.';
COMMENT ON COLUMN document_chunks.vehicle_id IS 'Denormalized vehicle reference for fast filtering. Automatically synced with parent document.';
