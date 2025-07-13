-- Migration: Restructure database for clarity and explicitness
-- This migration restructures the database to make it more understandable:
-- 1. Rename 'documents' table to 'document_chunks' 
-- 2. Remove embedding_count from document_chunks
-- 3. Add document_type and chunk_count to data_sources
-- 4. Change document_id to document_chunk_id in embeddings table
-- 5. Update all related functions and triggers

-- Step 1: Add new columns to data_sources table
ALTER TABLE data_sources 
ADD COLUMN IF NOT EXISTS document_type document_type DEFAULT 'text',
ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0;

-- Step 2: Populate document_type and chunk_count in data_sources from documents data
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

-- Step 3: Create new document_chunks table by copying from documents table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_source_id UUID REFERENCES data_sources(id) ON DELETE CASCADE,
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

-- Step 4: Copy data from documents to document_chunks (excluding embedding_count)
INSERT INTO document_chunks (
    id, data_source_id, title, content, document_type, chunk_index, status,
    word_count, character_count, processed_at, storage_path, original_filename,
    file_size, created_at, updated_at, metadata
)
SELECT 
    id, data_source_id, title, content, document_type, chunk_index, status,
    word_count, character_count, processed_at, storage_path, original_filename,
    file_size, created_at, updated_at, metadata
FROM documents;

-- Step 5: Create new embeddings table structure with document_chunk_id
CREATE TABLE embeddings_new (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1536),
    model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Step 6: Copy embeddings data with new column name
INSERT INTO embeddings_new (id, document_chunk_id, embedding, model_name, created_at, metadata)
SELECT id, document_id, embedding, model_name, created_at, metadata
FROM embeddings;

-- Step 7: Drop old triggers that referenced the documents table before dropping the table
DROP TRIGGER IF EXISTS update_data_source_count_on_insert ON documents;
DROP TRIGGER IF EXISTS update_data_source_count_on_delete ON documents;
DROP FUNCTION IF EXISTS update_data_source_document_count();

-- Drop system_dashboard view first to avoid dependency issues
DROP VIEW IF EXISTS system_dashboard;

-- Drop old tables and rename new ones
DROP TABLE embeddings;
ALTER TABLE embeddings_new RENAME TO embeddings;

DROP TABLE documents;

-- Step 8: Create all necessary indexes
CREATE INDEX IF NOT EXISTS idx_document_chunks_data_source_id ON document_chunks(data_source_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_status ON document_chunks(status);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_type ON document_chunks(document_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_created_at ON document_chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_document_chunks_processed_at ON document_chunks(processed_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_storage_path ON document_chunks(storage_path);
CREATE INDEX IF NOT EXISTS idx_document_chunks_original_filename ON document_chunks(original_filename);
CREATE INDEX IF NOT EXISTS idx_document_chunks_file_size ON document_chunks(file_size);

CREATE INDEX IF NOT EXISTS idx_embeddings_document_chunk_id ON embeddings(document_chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);

CREATE INDEX IF NOT EXISTS idx_data_sources_document_type ON data_sources(document_type);
CREATE INDEX IF NOT EXISTS idx_data_sources_chunk_count ON data_sources(chunk_count);

-- Step 9: Create vector similarity search index
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Step 10: Update updated_at trigger for document_chunks
CREATE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Step 11: Create function to update chunk count in data_sources
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

-- Step 12: Create triggers to automatically update chunk count
CREATE TRIGGER update_data_source_chunk_count_on_insert
    AFTER INSERT ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_chunk_count();

CREATE TRIGGER update_data_source_chunk_count_on_delete
    AFTER DELETE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_data_source_chunk_count();

-- Step 13: Drop existing functions first to avoid signature conflicts
DROP FUNCTION IF EXISTS similarity_search(vector, float, int);
DROP FUNCTION IF EXISTS hybrid_search(text, vector, float, int);

-- Update vector search functions to use new table structure
CREATE OR REPLACE FUNCTION similarity_search(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_chunk_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB
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
        dc.metadata
    FROM embeddings e
    JOIN document_chunks dc ON e.document_chunk_id = dc.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_chunk_id UUID,
    content TEXT,
    similarity FLOAT,
    keyword_rank FLOAT,
    combined_score FLOAT,
    metadata JSONB
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
        dc.metadata
    FROM embeddings e
    JOIN document_chunks dc ON e.document_chunk_id = dc.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
        OR to_tsvector('english', dc.content) @@ plainto_tsquery('english', query_text)
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Step 14: Update system dashboard view to use new structure
CREATE OR REPLACE VIEW system_dashboard AS
SELECT
    (SELECT COUNT(*) FROM data_sources WHERE status = 'active') as active_sources,
    (SELECT COUNT(*) FROM document_chunks WHERE status = 'completed') as processed_chunks,
    (SELECT COUNT(*) FROM embeddings) as total_embeddings,
    (SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '24 hours') as daily_conversations,
    (SELECT AVG(response_time_ms) FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours') as avg_response_time,
    (SELECT AVG(rating) FROM response_feedback WHERE created_at > NOW() - INTERVAL '7 days') as avg_user_rating,
    (SELECT SUM(chunk_count) FROM data_sources) as total_chunks;

-- Step 15: (These triggers were already dropped in Step 7)

-- Step 16: Add comments to document the new structure
COMMENT ON TABLE document_chunks IS 'Stores individual document chunks ready for embedding and search';
COMMENT ON TABLE data_sources IS 'Manages document sources with aggregate information like chunk counts';
COMMENT ON COLUMN data_sources.document_type IS 'Type of document this data source represents';
COMMENT ON COLUMN data_sources.chunk_count IS 'Number of chunks generated from this data source';
COMMENT ON COLUMN embeddings.document_chunk_id IS 'Foreign key to document_chunks table';

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Database restructuring complete!';
    RAISE NOTICE '✅ documents table renamed to document_chunks';
    RAISE NOTICE '✅ embedding_count moved to data_sources as chunk_count';
    RAISE NOTICE '✅ document_type added to data_sources';
    RAISE NOTICE '✅ embeddings.document_id renamed to document_chunk_id';
    RAISE NOTICE '✅ All functions and triggers updated';
END
$$; 