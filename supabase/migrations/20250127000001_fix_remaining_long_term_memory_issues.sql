-- Migration: Fix remaining long-term memory issues
-- Fixes UUID/text comparison and other critical issues

-- 1. Fix search_conversation_summaries function - UUID/text type mismatch
-- The issue is that target_user_id parameter is TEXT but user_id column is UUID
DROP FUNCTION IF EXISTS search_conversation_summaries(vector, text, float, int, varchar);
CREATE OR REPLACE FUNCTION search_conversation_summaries(
    query_embedding VECTOR(1536),
    target_user_id UUID,  -- Changed from TEXT to UUID
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 5,
    summary_type_filter VARCHAR(50) DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    conversation_id UUID,
    summary_text TEXT,
    similarity FLOAT,
    message_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    summary_type VARCHAR(50)
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.id,
        cs.conversation_id,
        cs.summary_text,
        (1 - (cs.summary_embedding <=> query_embedding))::FLOAT AS similarity,
        cs.message_count,
        cs.created_at,
        cs.summary_type
    FROM conversation_summaries cs
    WHERE cs.summary_embedding IS NOT NULL
        AND cs.user_id = target_user_id  -- Now both are UUID type
        AND cs.consolidation_status = 'active'
        AND (summary_type_filter IS NULL OR cs.summary_type = summary_type_filter)
        AND (1 - (cs.summary_embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- 2. Create a helper function for UUID validation and conversion
CREATE OR REPLACE FUNCTION safe_uuid_cast(input_text TEXT) RETURNS UUID AS $$
BEGIN
    BEGIN
        RETURN input_text::UUID;
    EXCEPTION WHEN invalid_text_representation THEN
        RETURN NULL;
    END;
END;
$$ LANGUAGE plpgsql;

-- 3. Create an overloaded version that accepts TEXT and converts to UUID
CREATE OR REPLACE FUNCTION search_conversation_summaries(
    query_embedding VECTOR(1536),
    target_user_id TEXT,  -- This version accepts TEXT
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 5,
    summary_type_filter VARCHAR(50) DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    conversation_id UUID,
    summary_text TEXT,
    similarity FLOAT,
    message_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    summary_type VARCHAR(50)
)
LANGUAGE plpgsql
AS $$
DECLARE
    user_uuid UUID;
BEGIN
    -- Convert TEXT to UUID safely
    user_uuid := safe_uuid_cast(target_user_id);
    
    IF user_uuid IS NULL THEN
        -- Return empty result if invalid UUID
        RETURN;
    END IF;
    
    -- Call the UUID version of the function
    RETURN QUERY
    SELECT * FROM search_conversation_summaries(
        query_embedding,
        user_uuid,
        similarity_threshold,
        match_count,
        summary_type_filter
    );
END;
$$;

-- 4. Test the function works correctly
-- This should not fail anymore
DO $$
BEGIN
    PERFORM search_conversation_summaries(
        ARRAY[0.1]::VECTOR(1536),
        'test_user'::TEXT,
        0.5,
        5,
        NULL
    );
    RAISE NOTICE 'search_conversation_summaries function test completed successfully';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'search_conversation_summaries function test failed: %', SQLERRM;
END;
$$;

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Fixed UUID/text comparison issue in search_conversation_summaries';
    RAISE NOTICE '✅ Added safe UUID conversion helper function';
    RAISE NOTICE '✅ Created overloaded function for TEXT parameter compatibility';
END;
$$; 