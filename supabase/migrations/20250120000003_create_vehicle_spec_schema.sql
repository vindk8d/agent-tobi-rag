-- Migration: Create Vehicle Specification Upload Schema
-- This migration:
-- 1. Applies the document_chunks restructure (from 20250109000000_restructure_for_clarity.sql)
-- 2. Adds vehicle_id foreign keys to support vehicle specification uploads
-- 3. Adds vehicle_specification to document_type enum
-- 4. Creates necessary indexes for vehicle-document relationships

-- Step 1: Add vehicle_specification to document_type enum
ALTER TYPE document_type ADD VALUE IF NOT EXISTS 'vehicle_specification';

-- Step 2: Add new columns to data_sources table (from restructure migration)
ALTER TABLE data_sources 
ADD COLUMN IF NOT EXISTS document_type document_type DEFAULT 'text',
ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0;

-- Step 3: Populate document_type and chunk_count in data_sources from documents data
UPDATE data_sources 
SET document_type = (
    SELECT d.document_type 
    FROM documents d 
    WHERE d.data_source_id = data_sources.id 
    LIMIT 1
),
chunk_count = (
    SELECT COUNT(*) 
    FROM documents d 
    WHERE d.data_source_id = data_sources.id
);

-- Step 4: Create document_chunks table with vehicle_id support
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_source_id UUID REFERENCES data_sources(id) ON DELETE CASCADE,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE CASCADE,
    title TEXT,
    content TEXT NOT NULL,
    document_type document_type DEFAULT 'text',
    chunk_index INTEGER DEFAULT 0,
    status document_status DEFAULT 'pending',
    word_count INTEGER,
    character_count INTEGER,
    processed_at TIMESTAMP WITH TIME ZONE,
    storage_path TEXT,
    original_filename TEXT,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Step 5: Add vehicle_id to existing documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vehicle_id UUID REFERENCES vehicles(id) ON DELETE CASCADE;

-- Step 6: Copy data from documents to document_chunks (matching actual documents table schema)
INSERT INTO document_chunks (
    id, data_source_id, vehicle_id, content, document_type, 
    processed_at, storage_path, original_filename, file_size, 
    created_at, updated_at, metadata
)
SELECT 
    id, data_source_id, NULL as vehicle_id, content, document_type,
    processed_at, storage_path, original_filename, file_size,
    created_at, updated_at, metadata
FROM documents
ON CONFLICT (id) DO NOTHING;

-- Step 7: Create new embeddings table structure with document_chunk_id
CREATE TABLE IF NOT EXISTS embeddings_new (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1536),
    model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Step 8: Copy embeddings data with new column name (only if embeddings table exists with document_id)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'embeddings' AND column_name = 'document_id'
    ) THEN
        INSERT INTO embeddings_new (id, document_chunk_id, embedding, model_name, created_at, metadata)
        SELECT id, document_id, embedding, model_name, created_at, metadata
        FROM embeddings
        ON CONFLICT (id) DO NOTHING;
    END IF;
END $$;

-- Step 9: Drop old triggers and functions before table operations
DROP TRIGGER IF EXISTS update_data_source_count_on_insert ON documents;
DROP TRIGGER IF EXISTS update_data_source_count_on_delete ON documents;
DROP FUNCTION IF EXISTS update_data_source_document_count();
DROP VIEW IF EXISTS system_dashboard;

-- Step 10: Replace embeddings table if it exists with document_id column
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'embeddings' AND column_name = 'document_id'
    ) THEN
        DROP TABLE embeddings;
        ALTER TABLE embeddings_new RENAME TO embeddings;
    ELSE
        -- If embeddings already has document_chunk_id, just drop the new table
        DROP TABLE embeddings_new;
    END IF;
END $$;

-- Step 11: Create all necessary indexes
CREATE INDEX IF NOT EXISTS idx_document_chunks_data_source_id ON document_chunks(data_source_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_vehicle_id ON document_chunks(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_status ON document_chunks(status);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_type ON document_chunks(document_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_created_at ON document_chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_document_chunks_processed_at ON document_chunks(processed_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_storage_path ON document_chunks(storage_path);
CREATE INDEX IF NOT EXISTS idx_document_chunks_original_filename ON document_chunks(original_filename);
CREATE INDEX IF NOT EXISTS idx_document_chunks_file_size ON document_chunks(file_size);

CREATE INDEX IF NOT EXISTS idx_documents_vehicle_id ON documents(vehicle_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_document_chunk_id ON embeddings(document_chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);

CREATE INDEX IF NOT EXISTS idx_data_sources_document_type ON data_sources(document_type);
CREATE INDEX IF NOT EXISTS idx_data_sources_chunk_count ON data_sources(chunk_count);

-- Step 12: Create vector similarity search index
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Step 13: Update updated_at trigger for document_chunks
CREATE OR REPLACE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Step 14: Create function to update chunk count in data_sources
CREATE OR REPLACE FUNCTION update_data_source_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE data_sources 
        SET chunk_count = (
            SELECT COUNT(*) 
            FROM document_chunks 
            WHERE data_source_id = NEW.data_source_id
        )
        WHERE id = NEW.data_source_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE data_sources 
        SET chunk_count = (
            SELECT COUNT(*) 
            FROM document_chunks 
            WHERE data_source_id = OLD.data_source_id
        )
        WHERE id = OLD.data_source_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Step 15: Create triggers to automatically update chunk count
CREATE OR REPLACE TRIGGER update_data_source_chunk_count_on_insert
    AFTER INSERT ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_chunk_count();

CREATE OR REPLACE TRIGGER update_data_source_chunk_count_on_delete
    AFTER DELETE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_chunk_count();

-- Step 16: Update vector search functions to use new table structure
DROP FUNCTION IF EXISTS similarity_search(vector, float, int);
DROP FUNCTION IF EXISTS hybrid_search(text, vector, float, int);

CREATE OR REPLACE FUNCTION similarity_search(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10,
    filter_vehicle_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_chunk_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    vehicle_id UUID
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.document_chunk_id,
        dc.content,
        1 - (e.embedding <=> query_embedding) AS similarity,
        dc.metadata,
        dc.vehicle_id
    FROM embeddings e
    JOIN document_chunks dc ON e.document_chunk_id = dc.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
        AND (filter_vehicle_id IS NULL OR dc.vehicle_id = filter_vehicle_id)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10,
    filter_vehicle_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_chunk_id UUID,
    content TEXT,
    similarity FLOAT,
    keyword_rank FLOAT,
    combined_score FLOAT,
    metadata JSONB,
    vehicle_id UUID
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.document_chunk_id,
        dc.content,
        (1 - (e.embedding <=> query_embedding))::FLOAT AS similarity,
        ts_rank_cd(to_tsvector('english', dc.content), plainto_tsquery('english', query_text))::FLOAT AS keyword_rank,
        ((1 - (e.embedding <=> query_embedding)) * 0.7 + ts_rank_cd(to_tsvector('english', dc.content), plainto_tsquery('english', query_text)) * 0.3)::FLOAT AS combined_score,
        dc.metadata,
        dc.vehicle_id
    FROM embeddings e
    JOIN document_chunks dc ON e.document_chunk_id = dc.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
        OR to_tsvector('english', dc.content) @@ plainto_tsquery('english', query_text)
    AND (filter_vehicle_id IS NULL OR dc.vehicle_id = filter_vehicle_id)
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Step 17: Update system dashboard view to use new structure
CREATE OR REPLACE VIEW system_dashboard AS
SELECT
    (SELECT COUNT(*) FROM data_sources WHERE status = 'active') as active_sources,
    (SELECT COUNT(*) FROM document_chunks WHERE status = 'completed') as processed_chunks,
    (SELECT COUNT(*) FROM embeddings) as total_embeddings,
    (SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '24 hours') as daily_conversations,
    (SELECT AVG(response_time_ms) FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours') as avg_response_time,
    (SELECT AVG(rating) FROM response_feedback WHERE created_at > NOW() - INTERVAL '7 days') as avg_user_rating,
    (SELECT SUM(chunk_count) FROM data_sources) as total_chunks,
    (SELECT COUNT(*) FROM document_chunks WHERE vehicle_id IS NOT NULL) as vehicle_documents;

-- Step 18: Add comments to document the vehicle integration
COMMENT ON TABLE document_chunks IS 'Stores individual document chunks ready for embedding and search, with optional vehicle association';
COMMENT ON COLUMN document_chunks.vehicle_id IS 'Optional foreign key to vehicles table for vehicle-specific documents';
COMMENT ON COLUMN documents.vehicle_id IS 'Optional foreign key to vehicles table for vehicle-specific documents';

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Vehicle specification schema migration complete!';
    RAISE NOTICE '✅ document_chunks table created with vehicle_id support';
    RAISE NOTICE '✅ documents table updated with vehicle_id column';
    RAISE NOTICE '✅ embeddings table restructured to use document_chunk_id';
    RAISE NOTICE '✅ vehicle_specification added to document_type enum';
    RAISE NOTICE '✅ Vector search functions updated with vehicle filtering';
    RAISE NOTICE '✅ All indexes and triggers created';
END
$$;
