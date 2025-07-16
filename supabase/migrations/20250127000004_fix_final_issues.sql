-- Fix final issues with column names and function parameters

-- 1. Fix search_conversation_summaries to use correct column name
DROP FUNCTION IF EXISTS search_conversation_summaries(vector, uuid, float, int, varchar);
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
        (cs.summary_embedding <=> query_embedding) AS similarity,  -- Fixed: use summary_embedding
        cs.message_count,
        cs.created_at,
        cs.created_at as updated_at  -- Use created_at as updated_at since there's no updated_at column
    FROM conversation_summaries cs
    WHERE 
        cs.user_id = target_user_id
        AND (cs.summary_embedding <=> query_embedding) >= similarity_threshold  -- Fixed: use summary_embedding
        AND (summary_type_filter IS NULL OR cs.summary_type = summary_type_filter)
    ORDER BY (cs.summary_embedding <=> query_embedding) ASC  -- Fixed: use summary_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- 2. Update put_long_term_memory to use correct update_memory_access_pattern call
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
    user_id TEXT;
BEGIN
    -- Extract user_id from namespace (first element)
    user_id := p_namespace[1];
    
    -- Insert or update the memory
    INSERT INTO long_term_memories (
        namespace,
        key,
        value,
        embedding,
        memory_type,
        expiry_at
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
        expiry_at = p_expiry_at,
        updated_at = NOW()
    RETURNING id INTO memory_id;
    
    -- Update access pattern with correct parameters
    PERFORM update_memory_access_pattern(
        user_id,        -- p_user_id
        p_namespace,    -- p_memory_namespace
        p_key,          -- p_memory_key
        'store',        -- p_access_context
        'direct'        -- p_retrieval_method
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- 3. Update get_long_term_memory to use correct update_memory_access_pattern call
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
DECLARE
    user_id TEXT;
BEGIN
    -- Extract user_id from namespace (first element)
    user_id := p_namespace[1];
    
    -- Update access pattern with correct parameters
    PERFORM update_memory_access_pattern(
        user_id,        -- p_user_id
        p_namespace,    -- p_memory_namespace
        p_key,          -- p_memory_key
        'read',         -- p_access_context
        'direct'        -- p_retrieval_method
    );
    
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
        AND (ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())
    ORDER BY ltm.updated_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- 4. Grant permissions
GRANT EXECUTE ON FUNCTION search_conversation_summaries(vector, uuid, float, int, varchar) TO authenticated;
GRANT EXECUTE ON FUNCTION get_long_term_memory(text[], text) TO authenticated;
GRANT EXECUTE ON FUNCTION put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone) TO authenticated; 