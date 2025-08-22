-- Migration: Standardize similarity_search function schema
-- This ensures all similarity_search functions use the current document_chunks schema

-- Drop any existing similarity_search functions to avoid conflicts
DROP FUNCTION IF EXISTS similarity_search(vector, float, int);
DROP FUNCTION IF EXISTS similarity_search(vector, float, int, uuid);

-- Create the standardized similarity_search function (without vehicle filter)
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

-- Create the vehicle-filtered version for vehicle-specific searches
CREATE OR REPLACE FUNCTION similarity_search_with_vehicle(
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

-- Update hybrid_search function to match the same schema
DROP FUNCTION IF EXISTS hybrid_search(text, vector, float, int);
DROP FUNCTION IF EXISTS hybrid_search(text, vector, float, int, uuid);

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

-- Create vehicle-filtered hybrid search
CREATE OR REPLACE FUNCTION hybrid_search_with_vehicle(
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
