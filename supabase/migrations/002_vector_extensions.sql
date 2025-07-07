-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for storing document embeddings
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(1536), -- OpenAI text-embedding-3-large dimension
    model_name VARCHAR(100) DEFAULT 'text-embedding-3-large',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for vector similarity search using cosine distance
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create additional indexes for performance
CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX idx_embeddings_model_name ON embeddings(model_name);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION similarity_search(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
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
        e.document_id,
        d.content,
        1 - (e.embedding <=> query_embedding) AS similarity,
        d.metadata
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create function for hybrid search (combining keyword and vector search)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
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
        e.document_id,
        d.content,
        (1 - (e.embedding <=> query_embedding))::FLOAT AS similarity,
        ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text))::FLOAT AS keyword_rank,
        ((1 - (e.embedding <=> query_embedding)) * 0.7 + ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) * 0.3)::FLOAT AS combined_score,
        d.metadata
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE (1 - (e.embedding <=> query_embedding)) > match_threshold
        OR to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Create full text search index for hybrid search
CREATE INDEX idx_documents_content_fts ON documents USING gin(to_tsvector('english', content));

-- Create function to get document statistics
CREATE OR REPLACE FUNCTION get_embedding_stats()
RETURNS TABLE (
    total_documents BIGINT,
    total_embeddings BIGINT,
    avg_similarity FLOAT,
    embedding_model VARCHAR(100)
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT d.id) AS total_documents,
        COUNT(e.id) AS total_embeddings,
        AVG(1 - (e.embedding <=> e.embedding)) AS avg_similarity, -- Self-similarity for quality check
        e.model_name AS embedding_model
    FROM documents d
    LEFT JOIN embeddings e ON d.id = e.document_id
    GROUP BY e.model_name;
END;
$$; 