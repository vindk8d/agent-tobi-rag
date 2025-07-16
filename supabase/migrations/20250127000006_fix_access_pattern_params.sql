-- Fix parameter types for update_memory_access_pattern function calls

-- 1. Fix put_long_term_memory function - convert UUID back to text for access pattern
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
    user_id_text TEXT;
    user_id_uuid UUID;
BEGIN
    -- Extract user_id from namespace (first element) and cast to UUID
    user_id_text := p_namespace[1];
    user_id_uuid := user_id_text::UUID;
    
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
    
    -- Update access pattern with correct parameters (convert UUID back to text)
    PERFORM update_memory_access_pattern(
        user_id_text,   -- p_user_id (text)
        p_namespace,    -- p_memory_namespace (text[])
        p_key,          -- p_memory_key (text)
        'store',        -- p_access_context (text)
        'direct'        -- p_retrieval_method (varchar)
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- 2. Fix get_long_term_memory function - convert UUID back to text for access pattern  
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
    user_id_text TEXT;
    user_id_uuid UUID;
BEGIN
    -- Extract user_id from namespace (first element) and cast to UUID
    user_id_text := p_namespace[1];
    user_id_uuid := user_id_text::UUID;
    
    -- Update access pattern with correct parameters (convert UUID back to text)
    PERFORM update_memory_access_pattern(
        user_id_text,   -- p_user_id (text)
        p_namespace,    -- p_memory_namespace (text[])
        p_key,          -- p_memory_key (text)
        'read',         -- p_access_context (text)
        'direct'        -- p_retrieval_method (varchar)
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

-- 3. Grant permissions
GRANT EXECUTE ON FUNCTION get_long_term_memory(text[], text) TO authenticated;
GRANT EXECUTE ON FUNCTION put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone) TO authenticated; 