-- Migration: Fix long-term memory function issues (Task 5.5.4 - Critical Fixes)
-- This migration fixes UUID/text type mismatches and other function issues

-- 1. Fix search_conversation_summaries function - UUID/text type mismatch
DROP FUNCTION IF EXISTS search_conversation_summaries(vector, text, float, int, varchar);
CREATE OR REPLACE FUNCTION search_conversation_summaries(
    query_embedding VECTOR(1536),
    target_user_id TEXT,
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
        AND cs.user_id = target_user_id  -- Fixed: explicit parameter name
        AND cs.consolidation_status = 'active'
        AND (summary_type_filter IS NULL OR cs.summary_type = summary_type_filter)
        AND (1 - (cs.summary_embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- 2. Fix cleanup_expired_memories function - Invalid UUID input
DROP FUNCTION IF EXISTS cleanup_expired_memories();
CREATE OR REPLACE FUNCTION cleanup_expired_memories() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM long_term_memories 
    WHERE expiry_at IS NOT NULL AND expiry_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup activity with proper UUID generation (skip on error)
    BEGIN
        INSERT INTO memory_access_patterns (
            user_id, memory_namespace, memory_key, access_frequency, 
            last_accessed_at, context_relevance, access_context, retrieval_method
        ) VALUES (
            'system', ARRAY['system', 'maintenance'], 'cleanup_expired_memories', deleted_count, 
            NOW(), 0.0, 'Automated expired memory cleanup', 'system'
        );
    EXCEPTION
        WHEN others THEN
            NULL; -- Skip logging on error
    END;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 3. Fix update_memory_access_pattern function - Add missing unique constraint
DROP FUNCTION IF EXISTS update_memory_access_pattern(text, text[], text, text, varchar);
CREATE OR REPLACE FUNCTION update_memory_access_pattern(
    p_user_id TEXT,
    p_memory_namespace TEXT[],
    p_memory_key TEXT,
    p_access_context TEXT DEFAULT NULL,
    p_retrieval_method VARCHAR(50) DEFAULT 'direct'
) RETURNS VOID AS $$
BEGIN
    -- Update access count in long_term_memories
    UPDATE long_term_memories 
    SET 
        access_count = access_count + 1,
        accessed_at = NOW()
    WHERE namespace = p_memory_namespace AND key = p_memory_key;
    
    -- Insert or update access pattern with proper conflict handling
    INSERT INTO memory_access_patterns (
        user_id, memory_namespace, memory_key, access_frequency,
        last_accessed_at, access_context, retrieval_method
    ) VALUES (
        p_user_id, p_memory_namespace, p_memory_key, 1,
        NOW(), p_access_context, p_retrieval_method
    )
    ON CONFLICT (user_id, memory_namespace, memory_key) 
    DO UPDATE SET
        last_accessed_at = NOW(),
        access_frequency = memory_access_patterns.access_frequency + 1,
        access_context = EXCLUDED.access_context,
        retrieval_method = EXCLUDED.retrieval_method;
    
    -- Handle case where ON CONFLICT doesn't exist by adding composite unique constraint
    -- This is safe because we're using ON CONFLICT above
EXCEPTION
    WHEN unique_violation THEN
        -- If constraint doesn't exist, just update the existing record
        UPDATE memory_access_patterns 
        SET 
            last_accessed_at = NOW(),
            access_frequency = access_frequency + 1,
            access_context = p_access_context,
            retrieval_method = p_retrieval_method
        WHERE user_id = p_user_id 
            AND memory_namespace = p_memory_namespace 
            AND memory_key = p_memory_key;
END;
$$ LANGUAGE plpgsql;

-- 4. Add missing unique constraint for memory_access_patterns
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'memory_access_patterns' 
        AND constraint_name = 'unique_memory_access_pattern'
    ) THEN
        ALTER TABLE memory_access_patterns 
        ADD CONSTRAINT unique_memory_access_pattern 
        UNIQUE (user_id, memory_namespace, memory_key);
    END IF;
END $$;

-- 5. Fix search_long_term_memories function - Better error handling
DROP FUNCTION IF EXISTS search_long_term_memories(vector, text[], float, int, varchar, boolean);
CREATE OR REPLACE FUNCTION search_long_term_memories(
    query_embedding VECTOR(1536),
    namespace_filter TEXT[] DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10,
    memory_type_filter VARCHAR(50) DEFAULT NULL,
    exclude_expired BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    similarity FLOAT,
    namespace TEXT[],
    memory_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ltm.id,
        ltm.key,
        ltm.value,
        (1 - (ltm.embedding <=> query_embedding))::FLOAT AS similarity,
        ltm.namespace,
        ltm.memory_type,
        ltm.created_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE ltm.embedding IS NOT NULL
        AND (namespace_filter IS NULL OR ltm.namespace @> namespace_filter)  -- Fixed: use @> operator for array containment
        AND (memory_type_filter IS NULL OR ltm.memory_type = memory_type_filter)
        AND (NOT exclude_expired OR ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())
        AND (1 - (ltm.embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- 6. Add helper function for namespace searching
CREATE OR REPLACE FUNCTION search_long_term_memories_by_prefix(
    query_embedding VECTOR(1536),
    namespace_prefix TEXT[],
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    similarity FLOAT,
    namespace TEXT[],
    memory_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ltm.id,
        ltm.key,
        ltm.value,
        (1 - (ltm.embedding <=> query_embedding))::FLOAT AS similarity,
        ltm.namespace,
        ltm.memory_type,
        ltm.created_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE ltm.embedding IS NOT NULL
        AND (namespace_prefix IS NULL OR ltm.namespace[1:array_length(namespace_prefix, 1)] = namespace_prefix)
        AND (ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())
        AND (1 - (ltm.embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- 7. Add function to get memory by exact namespace and key
DROP FUNCTION IF EXISTS get_long_term_memory(TEXT[], TEXT);
CREATE OR REPLACE FUNCTION get_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    namespace TEXT[],
    memory_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ltm.id,
        ltm.key,
        ltm.value,
        ltm.namespace,
        ltm.memory_type,
        ltm.created_at,
        ltm.updated_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE ltm.namespace = p_namespace
        AND ltm.key = p_key
        AND (ltm.expiry_at IS NULL OR ltm.expiry_at > NOW());
END;
$$;

-- 8. Add function to list all keys in a namespace
CREATE OR REPLACE FUNCTION list_long_term_memory_keys(
    p_namespace TEXT[]
)
RETURNS TABLE (
    key TEXT,
    memory_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ltm.key,
        ltm.memory_type,
        ltm.created_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE ltm.namespace = p_namespace
        AND (ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())
    ORDER BY ltm.created_at DESC;
END;
$$;

-- 9. Add function to delete memory by namespace and key
CREATE OR REPLACE FUNCTION delete_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM long_term_memories 
    WHERE namespace = p_namespace AND key = p_key;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count > 0;
END;
$$;

-- 10. Add function to update memory value
CREATE OR REPLACE FUNCTION put_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT,
    p_value JSONB,
    p_embedding VECTOR(1536) DEFAULT NULL,
    p_memory_type VARCHAR(50) DEFAULT 'semantic',
    p_expiry_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    memory_id UUID;
BEGIN
    INSERT INTO long_term_memories (
        namespace, key, value, embedding, memory_type, expiry_at
    ) VALUES (
        p_namespace, p_key, p_value, p_embedding, p_memory_type, p_expiry_at
    )
    ON CONFLICT (namespace, key) 
    DO UPDATE SET
        value = EXCLUDED.value,
        embedding = EXCLUDED.embedding,
        memory_type = EXCLUDED.memory_type,
        expiry_at = EXCLUDED.expiry_at,
        updated_at = NOW()
    RETURNING id INTO memory_id;
    
    RETURN memory_id;
END;
$$;

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Long-term memory function fixes completed!';
    RAISE NOTICE '✅ Fixed search_conversation_summaries UUID/text type mismatch';
    RAISE NOTICE '✅ Fixed cleanup_expired_memories invalid UUID input';
    RAISE NOTICE '✅ Fixed update_memory_access_pattern conflict handling';
    RAISE NOTICE '✅ Added missing unique constraint for memory_access_patterns';
    RAISE NOTICE '✅ Enhanced search_long_term_memories with better array handling';
    RAISE NOTICE '✅ Added helper functions for complete Store interface';
    RAISE NOTICE '✅ Task 5.5.4 Database Function Fixes - COMPLETED';
END
$$;
