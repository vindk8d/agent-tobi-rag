-- Fix function overloading issues and improve store operations
-- This migration addresses all remaining long-term memory issues

-- 1. Drop existing functions to resolve overloading
DROP FUNCTION IF EXISTS search_conversation_summaries(vector, text, float, int, varchar);
DROP FUNCTION IF EXISTS search_conversation_summaries(vector, uuid, float, int, varchar);

-- 2. Create a single, properly typed function
CREATE OR REPLACE FUNCTION search_conversation_summaries(
    query_embedding VECTOR(1536),
    target_user_id UUID,
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
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.id,
        cs.conversation_id,
        cs.summary_text,
        (cs.embedding <=> query_embedding) AS similarity,
        cs.message_count,
        cs.created_at,
        cs.updated_at
    FROM conversation_summaries cs
    WHERE 
        cs.user_id = target_user_id
        AND (cs.embedding <=> query_embedding) >= similarity_threshold
        AND (summary_type_filter IS NULL OR cs.summary_type = summary_type_filter)
    ORDER BY (cs.embedding <=> query_embedding) ASC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- 3. Drop and recreate get_long_term_memory function for better retrieval
DROP FUNCTION IF EXISTS get_long_term_memory(text[], text);
CREATE OR REPLACE FUNCTION get_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT
) RETURNS TABLE (
    namespace TEXT[],
    key TEXT,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ltm.namespace,
        ltm.key,
        ltm.value,
        ltm.created_at,
        ltm.updated_at
    FROM long_term_memories ltm
    WHERE 
        ltm.namespace = p_namespace
        AND ltm.key = p_key
        AND (ltm.expires_at IS NULL OR ltm.expires_at > NOW())
    ORDER BY ltm.updated_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- 4. Drop and recreate put_long_term_memory function to ensure proper storage
DROP FUNCTION IF EXISTS put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone);
CREATE OR REPLACE FUNCTION put_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT,
    p_value JSONB,
    p_embedding VECTOR(1536),
    p_memory_type VARCHAR(50) DEFAULT 'general',
    p_expiry_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
BEGIN
    -- Insert or update the memory
    INSERT INTO long_term_memories (
        namespace,
        key,
        value,
        embedding,
        memory_type,
        expires_at
    ) VALUES (
        p_namespace,
        p_key,
        p_value,
        p_embedding,
        p_memory_type,
        p_expiry_at
    )
    ON CONFLICT (namespace, key) DO UPDATE SET
        value = p_value,
        embedding = p_embedding,
        memory_type = p_memory_type,
        expires_at = p_expiry_at,
        updated_at = NOW()
    RETURNING id INTO memory_id;
    
    -- Update access pattern
    PERFORM update_memory_access_pattern(
        p_namespace,
        p_key,
        'write',
        1,
        NOW()
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- 5. Add a function to check if memory exists
CREATE OR REPLACE FUNCTION memory_exists(
    p_namespace TEXT[],
    p_key TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    result BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM long_term_memories 
        WHERE namespace = p_namespace 
        AND key = p_key
        AND (expires_at IS NULL OR expires_at > NOW())
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- 6. Grant necessary permissions
GRANT EXECUTE ON FUNCTION search_conversation_summaries(vector, uuid, float, int, varchar) TO authenticated;
GRANT EXECUTE ON FUNCTION get_long_term_memory(text[], text) TO authenticated;
GRANT EXECUTE ON FUNCTION put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone) TO authenticated;
GRANT EXECUTE ON FUNCTION memory_exists(text[], text) TO authenticated; 